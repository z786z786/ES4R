"""
Reference math for attention and softmax used in the project.

Project mappings:
- Double-level self-attention: src/modeling_blsp2.py
- Cross-attention: src/modeling_blsp2.py
- Softmax in Qwen attention: src/modeling_qwen.py
"""

import math
import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax.
    """
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Shapes:
    - q: (..., Lq, d)
    - k: (..., Lk, d)
    - v: (..., Lk, dv)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return torch.matmul(attn, v)


DOUBLE_LEVEL_SELF_ATTENTION_NOTE = """
Project implementation uses nn.MultiheadAttention rather than a handwritten kernel.

Level 1:
- input per sample: (num_utterances, T_sub, 80)
- meaning: self-attention inside each utterance sequence

Level 2:
- reshape and pad to: (batch_size, total_audio_tokens, 80)
- meaning: self-attention across all historical speech tokens within the sample
"""


CROSS_ATTENTION_NOTE = """
Project cross-attention:

fused_text_to_audio, _ = self.cross_attn(
    query=speech_input_embeds,
    key=input_embeds,
    value=input_embeds,
    key_padding_mask=~input_mask.to(torch.bool)
)

Meaning:
- speech acts as query
- text history acts as key/value
- current speech state retrieves relevant history semantics
"""
