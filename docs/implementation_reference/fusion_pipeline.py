"""
Reference excerpts for the project's fusion pipeline.

Original code:
- src/modeling_blsp2.py
"""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def double_level_self_attention(self, input_audio_features, input_audio_mask):
    """
    Simplified mirror of the project's main speech-context pipeline.
    """
    input_audio_attns, input_audio_masks = [], []
    for input_audio_feat, input_audio_msk in zip(input_audio_features, input_audio_mask):
        processed_feats = []
        processed_masks = []
        for audio_feat, audio_msk in zip(input_audio_feat, input_audio_msk):
            audio_feat_sub, audio_msk_sub, _, _, _ = self.subsample(
                audio_feat.transpose(1, 2), audio_msk
            )
            processed_feats.append(audio_feat_sub.squeeze(0))
            processed_masks.append(audio_msk_sub.squeeze(0))

        input_audio_feat_sub = pad_sequence(processed_feats, batch_first=True, padding_value=0.0)
        input_audio_msk_sub = pad_sequence(processed_masks, batch_first=True, padding_value=0)

        input_audio_attn_out, _ = self.self_attn(
            query=input_audio_feat_sub,
            key=input_audio_feat_sub,
            value=input_audio_feat_sub,
            key_padding_mask=~input_audio_msk_sub.to(torch.bool),
        )
        input_audio_attns.append(input_audio_attn_out)
        input_audio_masks.append(input_audio_msk_sub)

    input_audio_attns_reshaped = [
        x.reshape(x.shape[0] * x.shape[1], 80).unsqueeze(0) for x in input_audio_attns
    ]
    input_audio_mask_reshaped = [
        x.reshape(x.shape[0] * x.shape[1]).unsqueeze(0) for x in input_audio_masks
    ]

    max_len = max(x.shape[1] for x in input_audio_attns_reshaped)
    attn_padded = [F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in input_audio_attns_reshaped]
    mask_padded = [F.pad(x, (0, max_len - x.shape[1])) for x in input_audio_mask_reshaped]

    merged_attn = torch.cat(attn_padded, dim=0)
    merged_mask = torch.cat(mask_padded, dim=0)

    input_turn_attn_out, _ = self.self_attn(
        query=merged_attn,
        key=merged_attn,
        value=merged_attn,
        key_padding_mask=~merged_mask.to(torch.bool),
    )
    return input_turn_attn_out, merged_mask


GET_SPEECH_FEATURES_REFERENCE = """
speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
    self.get_speech_features(
        input_turn_attn_out.transpose(1, 2),
        input_audio_mask_reshaped_padded_merged.to(torch.int32),
        input_mask.sum(-1)
    )

Inside get_speech_features:
1. pad sequence to fixed length
2. run WhisperEncoder
3. build attention mask from output lengths
4. run adapter (default: CFormer)
"""


CROSS_MODAL_FUSION_REFERENCE = """
fused_text_to_audio, _ = self.cross_attn(
    query=speech_input_embeds,
    key=input_embeds,
    value=input_embeds,
    key_padding_mask=~input_mask.to(torch.bool)
)

speech_embeds = torch.cat(
    [start_embeds, audio_instruction_embeds, fused_text_to_audio, suffix_embeds],
    dim=1
)
"""
