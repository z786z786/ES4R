import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperConfig

from .plora import LoraConfig, LoraModel
from .modeling_adapter import Subsampler, CFormer
from .configuration_blsp2 import Blsp2Config
from .configuration_qwen import QWenConfig
from .modeling_utils import length_to_attention_mask, check_shape
from .modeling_whisper_encoder import WhisperEncoder
from .modeling_qwen import QWenLMHeadModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

text_llm_related_losses = {"response_kl", "input_kl"}
speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er"}
lm_related_losses = text_llm_related_losses | speech_llm_related_losses


class Blsp2Model(PreTrainedModel):
    config_class = Blsp2Config
    base_model_prefix = "caes"

    def __init__(self, config: Blsp2Config):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.qwen_config = QWenConfig(**config.qwen_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)
        self.qwen_model = QWenLMHeadModel(self.qwen_config)

        if config.lora_config:
            self.lora_config = LoraConfig(**config.lora_config)
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")

        if config.adapter_type == "subsampler":
            self.adapter = Subsampler(self.whisper_config.d_model, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)

        elif config.adapter_type == "cformer":
            self.adapter = CFormer(self.whisper_config, self.qwen_config.hidden_size,
                                   self.qwen_config.vocab_size,
                                   num_pre_cif_layers=config.num_pre_cif_layers,
                                   num_post_cif_layers=config.num_post_cif_layers)
        else:
            raise ValueError(f"unsupported adapter type: {config.adapter_type}")
        
        self.subsample = Subsampler(
            in_dim=80,  # Whisper's feature dimension
            mid_dim=80, 
            out_dim=80,
            num_hidden_layers=0,          # 不用 transformer 层就设为 0
            whisper_config=self.whisper_config,
            conv_kernel_sizes="5,5,5"
        )
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=80,
            num_heads=16,
            batch_first=True,
        )
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.qwen_config.hidden_size,
            num_heads=self.qwen_config.num_attention_heads,
            batch_first=True,
        )
        # self.cross_attn_layernorm = nn.LayerNorm(self.qwen_config.hidden_size)
        
        self.hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.loss_names = [] # must be a list of loss names:  seq_kd, token_kd, or others before training

    def set_loss_names(self, names):
        self.loss_names = names

    def forward(
        self,
        start_ids: torch.LongTensor,
        start_mask: torch.Tensor,
        start_labels: torch.LongTensor,
        instruction_ids: torch.LongTensor,
        instruction_mask: torch.Tensor,
        instruction_labels: torch.LongTensor,
        audio_instruction_ids: torch.LongTensor,
        audio_instruction_mask: torch.Tensor,
        audio_instruction_labels: torch.LongTensor,
        input_ids: torch.LongTensor,
        input_mask: torch.Tensor,
        input_labels: torch.LongTensor,
        input_audio_features: List[List[torch.FloatTensor]],
        input_audio_mask: List[List[torch.LongTensor]],
        suffix_ids: torch.LongTensor,
        suffix_mask: torch.Tensor,
        suffix_labels: torch.LongTensor,
        suffix_audio_features: List[torch.FloatTensor],
        suffix_audio_mask: List[torch.LongTensor],
        emotion_labels: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        assert len(self.loss_names) > 0, "self.loss_names cannot be empty"
        if not any ("response" in loss_name for loss_name in self.loss_names):
            batch_size = start_ids.size(0)
            instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            audio_instruction_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            audio_instruction_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            audio_instruction_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
            suffix_ids = torch.zeros(batch_size, 0, dtype=start_ids.dtype, device=start_ids.device)
            suffix_mask = torch.zeros(batch_size, 0, dtype=start_mask.dtype, device=start_mask.device)
            suffix_labels = torch.zeros(batch_size, 0, dtype=start_labels.dtype, device=start_labels.device)
        start_embeds = self.qwen_model.get_input_embeddings()(start_ids)
        instruction_embeds = self.qwen_model.get_input_embeddings()(instruction_ids)
        audio_instruction_embeds = self.qwen_model.get_input_embeddings()(audio_instruction_ids)
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_ids)

        # print("#################################################################################")
        # print("input_ids.shape", input_ids.shape)
        # print("instruction_embeds.shape", start_embeds.shape)
        # print("#################################################################################")
        # print("input_embeds.shape", input_embeds.shape)
        # print("instruction_embeds.shape", instruction_embeds.shape)
        # print("#################################################################################")
        
        #DOUBLE LEVEL SELF-ATTENTION
        input_audio_attns, input_audio_masks = [], []
        for input_audio_feat, input_audio_msk in zip(input_audio_features, input_audio_mask):
            # (B, D, T) -> (B, T, D)
            processed_feats = []
            processed_masks = []
            for audio_feat, audio_msk in zip(input_audio_feat, input_audio_msk):

                audio_feat_sub, audio_msk_sub, _, _, _ = self.subsample(audio_feat.transpose(1, 2), audio_msk)

                processed_feats.append(audio_feat_sub.squeeze(0))
                processed_masks.append(audio_msk_sub.squeeze(0))
            
            input_audio_feat_sub = pad_sequence(processed_feats, batch_first=True, padding_value=0.0)
            input_audio_msk_sub = pad_sequence(processed_masks, batch_first=True, padding_value=0)
           
            #In-Speech Self-attention
            input_audio_attn_out, _ = self.self_attn(
                query=input_audio_feat_sub,
                key=input_audio_feat_sub,
                value=input_audio_feat_sub,
                key_padding_mask=~input_audio_msk_sub.to(torch.bool)
            )

            input_audio_attns.append(input_audio_attn_out)  # List[List[(1, Tᵢ, D)]]
            input_audio_masks.append(input_audio_msk_sub)  # List[(Tᵢ, D)]
        input_audio_mask = input_audio_masks
        
        # reshape (speech_num, Tᵢ, D) → (1, speech_num*Tᵢ, D)
        input_audio_attns_reshaped = [input_audio_attn.reshape(input_audio_attn.shape[0]*input_audio_attn.shape[1], 80).unsqueeze(0) for input_audio_attn in input_audio_attns]
        input_audio_mask_reshaped = [input_audio_msk.reshape(input_audio_msk.shape[0]*input_audio_msk.shape[1]).unsqueeze(0) for input_audio_msk in input_audio_mask]
        
        # padding to max speech_num
        max_speech_num = max(input_audio_attn_reshaped.shape[1] for input_audio_attn_reshaped in input_audio_attns_reshaped)
        input_audio_attns_reshaped_padded = [F.pad(input_audio_attn_reshaped, (0, 0, 0, max_speech_num - input_audio_attn_reshaped.shape[1])) for input_audio_attn_reshaped in input_audio_attns_reshaped]
        input_audio_mask_reshaped_padded = [F.pad(input_audio_msk_reshaped, (0, max_speech_num - input_audio_msk_reshaped.shape[1])) for input_audio_msk_reshaped in input_audio_mask_reshaped]
        input_audio_attns_reshaped_padded_merged = torch.cat(input_audio_attns_reshaped_padded, dim=0)
        input_audio_mask_reshaped_padded_merged = torch.cat(input_audio_mask_reshaped_padded, dim=0)
   
        # print(f"Type: {type(input_audio_attns_reshaped_padded_merged)}")
        # # print(f"Shape: {input_audio_attns_reshaped_padded_merged.shape}")
        # print(f"Dtype: {input_audio_attns_reshaped_padded_merged.dtype}")
        
        input_turn_attn_out, _ = self.self_attn(
            query=input_audio_attns_reshaped_padded_merged,  # (turn_num, speech_num*Tᵢ, D)
            key=input_audio_attns_reshaped_padded_merged,
            value=input_audio_attns_reshaped_padded_merged,
            key_padding_mask=~input_audio_mask_reshaped_padded_merged.to(torch.bool)  # (turn_num, speech_num*Tᵢ)
        )

        # print("[DEBUG] input_turn_attn_out.transpose(1, 2):")
        # # print(input_turn_attn_out.transpose(1, 2))
        # print("shape:", input_turn_attn_out.transpose(1, 2).shape)

        # print("\n[DEBUG] input_audio_mask_reshaped_padded_merged.to(torch.int32):")
        # # print(input_audio_mask_reshaped_padded_merged.to(torch.int32))
        # print("shape:", input_audio_mask_reshaped_padded_merged.shape)

        # print("\n[DEBUG] input_mask.sum(-1):")
        # # print(input_mask.sum(-1))
        # print("shape:", input_mask.sum(-1).shape)
        
        speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.get_speech_features(input_turn_attn_out.transpose(1, 2), input_audio_mask_reshaped_padded_merged.to(torch.int32), input_mask.sum(-1))
        speech_input_labels = speech_input_mask.new_ones(speech_input_embeds.size(0), speech_input_embeds.size(1),
                                                            dtype=torch.int64).fill_(-100)

        if input_embeds is not None:
            #CROSS-ATTENTION
            fused_text_to_audio, _ = self.cross_attn(
                query=speech_input_embeds,
                key=input_embeds,
                value=input_embeds,
                key_padding_mask=~input_mask.to(torch.bool)
            )
            
            # fused_audio_to_text, _ = self.cross_attn(
            #     query=input_embeds,
            #     key=speech_input_embeds,
            #     value=speech_input_embeds,
            #     key_padding_mask=~speech_input_mask.to(torch.bool)
            # )

        # self_sentence_attns = []
        # for input_audio_sentence_feat, input_audio_sentence_msk in zip(input_audio_features, input_audio_mask):
        #     input_sentence_attn = []
        #     for feat, mask in zip(input_audio_sentence_feat, input_audio_sentence_msk):
        #         input_sentence_attn_out, _ = self.cross_attn(
        #             query=feat,
        #             key=feat,
        #             value=feat,
        #             key_padding_mask=~mask.to(torch.bool)  # (1, Tᵢ)
        #         )
        #         input_sentence_attn.append(input_sentence_attn_out.squeeze(0))  # (Tᵢ, D)
        #     self_sentence_attns.append(input_sentence_attn)    
        
        # input_sentence_attns, input_sentence_mask = [], []
        # for self_sentence_attn, input_sentence_mask in zip(self_sentence_attns, input_sentence_mask):
        #     input_sentence_attns.append(torch.cat(self_sentence_attn, dim=-1))
        #     input_sentence_mask.append(torch.cat(input_sentence_mask, dim=-1))
        # input_sentence_attns = input_sentence_attns.transpose(1, 2)
        # input_sentence_mask = input_sentence_mask.transpose(1, 2)
        
        # input_audio_attns = []
        # for input_audio_turn_feat, input_audio_turn_msk in zip(input_sentence_attns, input_sentence_mask):
        #     input_turn_attn_out, _ = self.cross_attn(
        #         query=input_audio_turn_feat.unsqueeze(0),  # (1, Tᵢ, D)
        #         key=input_audio_turn_feat.unsqueeze(0),
        #         value=input_audio_turn_feat.unsqueeze(0),
        #         key_padding_mask=~input_audio_turn_msk.to(torch.bool).unsqueeze(0)  # (1, Tᵢ)
        #     )
        #     input_audio_attns.append(input_turn_attn_out.squeeze(0))
        
        # #CROSS-ATTENTION
        # fused_text_to_audio, _ = self.cross_attn(
        #     query=input_ids,  # (R, D)
        #     key=input_audio_attns,     # (R, D)
        #     value=input_audio_attns,   # (R, D)
        #     key_padding_mask=~input_audio_mask.unsqueeze(0)
        # )
        # fused_audio_to_text, _ = self.cross_attn(
        #     query=fused_text_to_audio,       # (R, D)
        #     key=input_ids,    # (R, D)
        #     value=input_ids,   # (R, D)
        #     key_padding_mask=~input_mask.unsqueeze(0) 
        # )
        

        # speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
        #     self.get_speech_features(speech_values, speech_mask, input_mask.sum(-1))
        # speech_input_labels = speech_input_mask.new_ones(speech_input_embeds.size(0), speech_input_embeds.size(1),
        #                                                  dtype=torch.int64).fill_(-100)

        # speech_embeds = torch.cat([start_embeds, audio_instruction_embeds, speech_input_embeds, suffix_embeds], dim=1)
        # speech_mask = torch.cat([start_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
        # speech_labels = torch.cat([start_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)


        #############################################################################################
        # text_llm_related_losses = {"response_kl", "input_kl"}
        # speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er"}
        # lm_related_losses = text_llm_related_losses | speech_llm_related_losses
        #############################################################################################
        
        if any(loss_name in text_llm_related_losses for loss_name in self.loss_names):
            text_embeds = torch.cat([start_embeds, instruction_embeds, input_embeds, suffix_embeds], dim=1)
            text_mask = torch.cat([start_mask, instruction_mask, input_mask, suffix_mask], dim=1)
            text_labels = torch.cat([start_labels, instruction_labels, input_labels, suffix_labels], dim=1)
            
            input_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                         torch.zeros_like(instruction_labels),
                                         input_mask,
                                         torch.zeros_like(suffix_labels)], dim=1)
            speech_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                          torch.zeros_like(audio_instruction_labels),
                                          input_mask,
                                          torch.zeros_like(suffix_labels)], dim=1)
            text_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                 torch.zeros_like(instruction_labels),
                                                 torch.zeros_like(input_labels),
                                                 (suffix_labels != -100).long()], dim=1)
            speech_response_kd_labels = torch.cat([torch.zeros_like(start_labels),
                                                   torch.zeros_like(audio_instruction_labels),
                                                   torch.zeros_like(speech_input_labels),
                                                   (suffix_labels != -100).long()], dim=1)
            
            lora_audio_mask = torch.zeros_like(text_labels)
            self.update_lora_mask(lora_audio_mask, False)

            # with torch.no_grad():
            text_output = self.qwen_model(inputs_embeds=text_embeds, attention_mask=text_mask,
                                            position_ids=text_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            text_logits = text_output.logits
        if any(loss_name in speech_llm_related_losses for loss_name in self.loss_names):
            lora_audio_mask = torch.cat([torch.zeros_like(start_mask),
                                            torch.zeros_like(audio_instruction_mask),
                                            torch.ones_like(speech_input_mask),
                                            torch.zeros_like(suffix_mask)], dim=1)
            self.update_lora_mask(lora_audio_mask, False)
            speech_embeds = torch.cat([start_embeds, audio_instruction_embeds, fused_text_to_audio, suffix_embeds], dim=1)
            speech_mask = torch.cat([start_mask, audio_instruction_mask, speech_input_mask, suffix_mask], dim=1)
            speech_labels = torch.cat([start_labels, audio_instruction_labels, speech_input_labels, suffix_labels], dim=1)
            
            speech_output = self.qwen_model(inputs_embeds=speech_embeds, attention_mask=speech_mask,
                                            position_ids=speech_mask.cumsum(dim=-1) - 1, output_hidden_states=True,
                                            return_dict=True)
            speech_logits = speech_output.logits
            
        total_loss = input_embeds.new_zeros(())
        for loss_name in self.loss_names:
            if loss_name == "response_ce":
                shifted_logits = speech_logits[..., :-1, :].contiguous()
                shifted_labels = speech_labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(shifted_logits[shifted_labels != -100],
                                       shifted_labels[shifted_labels != -100], reduction="mean")
                
                total_loss += loss
                
            elif loss_name == "response_kl":
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_response_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[text_response_kd_labels == 1] / self.config.kd_temperature, dim=-1).detach(),
                    reduction="batchmean"
                )
               
                total_loss += loss
            elif loss_name == "input_kl":
                check_shape(input_labels, speech_input_labels)
                loss = F.kl_div(
                    F.log_softmax(speech_logits[speech_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    F.softmax(text_logits[input_kd_labels == 1] / self.config.kd_temperature, dim=-1),
                    reduction="batchmean"
                )
                
                total_loss += loss
            elif loss_name == "cif":
                if speech_pred_num_tokens is None:
                    raise RuntimeError("predicted_num_tokens not set but cif_loss is requested")
                loss = F.l1_loss(speech_pred_num_tokens/input_mask.sum(-1), torch.ones_like(speech_pred_num_tokens),
                                  reduction="mean")
                total_loss += loss
            elif loss_name == "input_er":
                hidden_states = speech_input_embeds.clone()
                hidden_states[speech_input_mask == 0] = 0.0
                pooled_output = hidden_states.sum(dim=1) / speech_input_mask.sum(dim=1).view(-1, 1)
                er_logits = self.hidden2emotion(pooled_output)
                loss = F.cross_entropy(er_logits.view(-1, self.config.num_emotions), emotion_labels.view(-1))
                total_loss += loss
            else:
                raise RuntimeError(f"Unsupported loss name: {loss_name}")

        return {"loss": total_loss}


    def add_lora(self, lora_config, lora_scope="global"):
        if self.config.lora_config:
            logger.warning(f"add_lora ignored as model already has lora enabled")
        else:
            self.lora_config = lora_config
            self.config.lora_config = lora_config.to_dict()
            self.qwen_model = LoraModel(self.qwen_model, self.lora_config, "default")
            self.config.lora_scope = lora_scope

    def update_lora_mask(self, audio_mask, inference_mode: bool):
        if not self.config.lora_config or self.config.lora_scope == "global":
            return

        self.qwen_model.update_inference_mode(inference_mode)
        if self.config.lora_scope == "audio":
            self.qwen_model.update_lora_mask("default", audio_mask)
        elif self.config.lora_scope == "text":
            self.qwen_model.update_lora_mask("default", torch.ones_like(audio_mask) - audio_mask)
        elif self.config.lora_scope == "global":
            pass # do nonthing as official peft uses global lora
        else:
            raise ValueError(f"The scope value {self.config.lora_scope} for lora adapter 'default' is not supported")

    def merge_lora(self):
        if hasattr(self, 'lora_config'):
            if self.config.lora_scope != "global":
                raise ValueError(f"cannot call merge_lora when the lora_scope is not global ("
                                 f"{self.config.lora_scope})")
            self.qwen_model = self.qwen_model.merge_and_unload()
            self.config.lora_config = {}
            del self.lora_config
        else:
            raise ValueError("cannot call merge_lora when no self.lora_config is set")

    def get_speech_features(self, speech_values, speech_attention_mask, num_tokens=None):
        
        pad_len = 3000 - speech_values.size(-1)
        speech_values = F.pad(speech_values, (0, pad_len))
        speech_attention_mask = F.pad(speech_attention_mask, pad=(0, pad_len))
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)
        # print("speech_embeds:")
        # print("  dtype:", speech_embeds.dtype)
        # print("  device:", speech_embeds.device)
        # print("  shape:", speech_embeds.shape)
        # print("  sample values:", speech_embeds[0, :2, :5])  # 取前两个时间步的前5维

        # print("attention_mask:")
        # print("  dtype:", attention_mask.dtype)
        # print("  shape:", attention_mask.shape)
        # print("  sample values:", attention_mask[0, :20])
        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.adapter(speech_embeds, attention_mask, num_tokens)
        return speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens

    @torch.no_grad()
    def generate(
        self,
        start_ids,
        start_mask,
        input_ids,
        input_mask,
        instruction_ids,
        instruction_mask,
        suffix_ids,
        suffix_mask,
        input_audio_features,
        input_audio_mask,
        generation_config=None,
        stop_words_ids=None
    ):
        
        start_embeds = self.qwen_model.get_input_embeddings()(start_ids)
        input_embeds = self.qwen_model.get_input_embeddings()(input_ids)
        suffix_embeds = self.qwen_model.get_input_embeddings()(suffix_ids)
        instruction_embeds = self.qwen_model.get_input_embeddings()(instruction_ids)
        # if speech_values is not None:
        
        #DOUBLE LEVEL SELF-ATTENTION
        input_audio_attns, input_audio_masks = [], []
        for input_audio_feat, input_audio_msk in zip(input_audio_features, input_audio_mask):
            # (B, D, T) -> (B, T, D)
            processed_feats = []
            processed_masks = []
            for audio_feat, audio_msk in zip(input_audio_feat, input_audio_msk):
                # print("audio_feat dtype:", audio_feat.dtype)
                # # print("audio_msk dtype:", audio_msk.dtype)
                # print("audio_feat shape:", audio_feat.shape)
                # print("audio_msk shape:", audio_msk.shape)
                # print("audio_feat:", audio_feat)
                # print("audio_msk:", audio_msk)
                audio_feat = audio_feat.to(torch.bfloat16)
                audio_feat_sub, audio_msk_sub, _, _, _ = self.subsample(audio_feat.transpose(1, 2), audio_msk)
                

                processed_feats.append(audio_feat_sub.squeeze(0))
                processed_masks.append(audio_msk_sub.squeeze(0))
            
            input_audio_feat_sub = pad_sequence(processed_feats, batch_first=True, padding_value=0.0)
            input_audio_msk_sub = pad_sequence(processed_masks, batch_first=True, padding_value=0)
           
            #In-Speech Self-attention
            input_audio_attn_out, _ = self.self_attn(
                query=input_audio_feat_sub,
                key=input_audio_feat_sub,
                value=input_audio_feat_sub,
                key_padding_mask=~input_audio_msk_sub.to(torch.bool)
            )

            input_audio_attns.append(input_audio_attn_out)  # List[List[(1, Tᵢ, D)]]
            input_audio_masks.append(input_audio_msk_sub)  # List[(Tᵢ, D)]
        input_audio_mask = input_audio_masks
        
        # reshape (speech_num, Tᵢ, D) → (1, speech_num*Tᵢ, D)
        input_audio_attns_reshaped = [input_audio_attn.reshape(input_audio_attn.shape[0]*input_audio_attn.shape[1], 80).unsqueeze(0) for input_audio_attn in input_audio_attns]
        input_audio_mask_reshaped = [input_audio_msk.reshape(input_audio_msk.shape[0]*input_audio_msk.shape[1]).unsqueeze(0) for input_audio_msk in input_audio_mask]
        
        # padding to max speech_num
        max_speech_num = max(input_audio_attn_reshaped.shape[1] for input_audio_attn_reshaped in input_audio_attns_reshaped)
        input_audio_attns_reshaped_padded = [F.pad(input_audio_attn_reshaped, (0, 0, 0, max_speech_num - input_audio_attn_reshaped.shape[1])) for input_audio_attn_reshaped in input_audio_attns_reshaped]
        input_audio_mask_reshaped_padded = [F.pad(input_audio_msk_reshaped, (0, max_speech_num - input_audio_msk_reshaped.shape[1])) for input_audio_msk_reshaped in input_audio_mask_reshaped]
        input_audio_attns_reshaped_padded_merged = torch.cat(input_audio_attns_reshaped_padded, dim=0)
        input_audio_mask_reshaped_padded_merged = torch.cat(input_audio_mask_reshaped_padded, dim=0)
        
        input_turn_attn_out, _ = self.self_attn(
            query=input_audio_attns_reshaped_padded_merged,  # (turn_num, speech_num*Tᵢ, D)
            key=input_audio_attns_reshaped_padded_merged,
            value=input_audio_attns_reshaped_padded_merged,
            key_padding_mask=~input_audio_mask_reshaped_padded_merged.to(torch.bool)  # (turn_num, speech_num*Tᵢ)
        )

        speech_input_embeds, speech_input_mask, speech_input_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.get_speech_features(input_turn_attn_out.transpose(1, 2), input_audio_mask_reshaped_padded_merged.to(torch.int32), input_mask.sum(-1))        
        
        #CROSS-ATTENTION
        fused_text_to_audio, _ = self.cross_attn(   
            query=speech_input_embeds,
            key=input_embeds,
            value=input_embeds,
            key_padding_mask=~input_mask.to(torch.bool)
        )
        
        # fused_audio_to_text, _ = self.cross_attn(
        #     query=input_embeds,
        #     key=fused_text_to_audio,
        #     value=fused_text_to_audio,
        #     key_padding_mask=~speech_input_mask.to(torch.bool)
        # )

        inputs_embeds, inputs_mask, lora_audio_mask = [], [], []
        
        inputs_embeds.append(start_embeds)
        inputs_mask.append(start_mask)
        lora_audio_mask.append(torch.zeros_like(start_mask))
        inputs_embeds.append(instruction_embeds)
        inputs_mask.append(instruction_mask)
        lora_audio_mask.append(torch.zeros_like(instruction_mask))
        inputs_embeds.append(input_embeds)
        inputs_mask.append(input_mask)
        lora_audio_mask.append(torch.zeros_like(input_mask))
        inputs_embeds.append(fused_text_to_audio)
        inputs_mask.append(speech_input_mask)
        lora_audio_mask.append(torch.ones_like(speech_input_mask))
        inputs_embeds.append(suffix_embeds)
        inputs_mask.append(suffix_mask)
        lora_audio_mask.append(torch.ones_like(suffix_mask))

        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        input_attention_mask = torch.cat(inputs_mask, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)
        
        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=input_attention_mask,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        )
    
    @torch.no_grad()
    def chat(
        self,
        history,
        generation_config,
        stop_words_ids,
        device,
    ):
        inputs_embeds = []
        lora_audio_mask = []

        for h in history:
            if len(h) == 1:
                ### text
                input_ids = h[0].to(device)
                embeds = self.qwen_model.get_input_embeddings()(input_ids)
                inputs_embeds.append(embeds)
                lora_audio_mask.append(torch.zeros_like(input_ids))
            elif len(h) == 2:
                ### speech
                speech_values, speech_attention_mask = h[0].to(device), h[1].to(device)
                speech_embeds, speech_attention_mask, _, _, _= self.get_speech_features(speech_values, speech_attention_mask)
                inputs_embeds.append(speech_embeds)
                lora_audio_mask.append(speech_attention_mask)
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.cat(inputs_embeds, dim=1)
        lora_audio_mask = torch.cat(lora_audio_mask, dim=1)
        self.update_lora_mask(lora_audio_mask, True)

        return self.qwen_model.generate(
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
            stop_words_ids=stop_words_ids
        )
