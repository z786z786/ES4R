"""
Minimal reference for the project's adapter implementations.

Original code:
- src/modeling_adapter.py
"""

import torch
from torch import nn


class Conv1dSubsampler(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_sizes=(3, 3)):
        super().__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def forward(self, src_tokens):
        x = src_tokens.transpose(1, 2).contiguous()
        for conv in self.conv_layers:
            x = nn.functional.glu(conv(x), dim=1)
        return x.transpose(1, 2).contiguous()


class Subsampler(nn.Module):
    """
    Project role:
    - compress long speech sequence
    - transform features before later attention / LLM bridge
    """

    def __init__(self, in_dim, mid_dim, out_dim, kernel_sizes=(5, 5, 5)):
        super().__init__()
        self.subsampler = Conv1dSubsampler(in_dim, 2 * in_dim, out_dim, kernel_sizes)
        self.fc1 = nn.Linear(out_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, out_dim, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.subsampler(x)
        residual = x
        x = self.fc2(self.activation(self.fc1(x))) + residual
        return self.norm(x)


class CFormerSketch(nn.Module):
    """
    Sketch of the project CFormer:
    1. pre-CIF context encoding
    2. alpha prediction
    3. CIF aggregation
    4. post-CIF encoding
    5. projection to LLM hidden space
    """

    def __init__(self, in_dim, out_dim, vocab_size):
        super().__init__()
        self.cif_proj = nn.Linear(in_dim - 1, in_dim)
        self.token_embed_proj = nn.Linear(in_dim, out_dim)
        self.lm_head = nn.Linear(in_dim, vocab_size, bias=False)

    def get_alphas(self, hidden_states, attention_mask):
        alphas = torch.sigmoid(hidden_states[:, :, -1])
        return alphas * attention_mask.float()

    def forward(self, hidden_states, attention_mask):
        alphas = self.get_alphas(hidden_states, attention_mask)
        # In the real project, CIF aggregation transforms frame sequence
        # into shorter token-like units before projection.
        hidden_states = self.cif_proj(hidden_states[:, :, :-1])
        logits = self.lm_head(hidden_states)
        hidden_states = self.token_embed_proj(hidden_states)
        return hidden_states, attention_mask, logits, alphas


ADAPTER_ROLE = """
Adapter role in the project:
- bridge Whisper hidden states to Qwen hidden space
- compress long speech sequence
- produce token-like speech embeddings for later cross-attention and generation
"""
