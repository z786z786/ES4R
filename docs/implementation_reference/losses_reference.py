"""
Reference implementations for the project's main losses.

Original code:
- src/modeling_blsp2.py
"""

import torch
import torch.nn.functional as F


def response_ce_loss(speech_logits: torch.Tensor, speech_labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = speech_logits[..., :-1, :].contiguous()
    shifted_labels = speech_labels[..., 1:].contiguous()
    return F.cross_entropy(
        shifted_logits[shifted_labels != -100],
        shifted_labels[shifted_labels != -100],
        reduction="mean",
    )


def response_kl_loss(
    speech_logits: torch.Tensor,
    text_logits: torch.Tensor,
    speech_response_kd_labels: torch.Tensor,
    text_response_kd_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(speech_logits[speech_response_kd_labels == 1] / temperature, dim=-1),
        F.softmax(text_logits[text_response_kd_labels == 1] / temperature, dim=-1).detach(),
        reduction="batchmean",
    )


def input_kl_loss(
    speech_logits: torch.Tensor,
    text_logits: torch.Tensor,
    speech_kd_labels: torch.Tensor,
    input_kd_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(speech_logits[speech_kd_labels == 1] / temperature, dim=-1),
        F.softmax(text_logits[input_kd_labels == 1] / temperature, dim=-1),
        reduction="batchmean",
    )


def input_er_loss(
    speech_input_embeds: torch.Tensor,
    speech_input_mask: torch.Tensor,
    emotion_labels: torch.Tensor,
    hidden2emotion,
) -> torch.Tensor:
    hidden_states = speech_input_embeds.clone()
    hidden_states[speech_input_mask == 0] = 0.0
    pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
    er_logits = hidden2emotion(pooled_output)
    return F.cross_entropy(er_logits.view(-1, er_logits.size(-1)), emotion_labels.view(-1))


LOSS_ROLE_NOTE = """
response_ce:
- hard-label supervision
- teaches speech path to generate the gold response

response_kl:
- soft-label distillation
- makes speech path follow text path response distribution

input_kl:
- aligns speech path and text path on context positions

input_er:
- explicit emotion supervision on pooled speech representation
"""
