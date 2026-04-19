"""
Reference notes for LoRA usage in the project.

Original code:
- train.py
- src/modeling_blsp2.py
- src/plora.py
"""

LORA_CONFIG_REFERENCE = """
Default training arguments:

lora_r = 16
lora_alpha = 16
lora_dropout = 0.1
lora_target_modules = "c_attn,c_proj,w1,w2"
lora_scope = "audio"
"""


LORA_INSERTION_REFERENCE = """
When unfreeze_qwen=True in train.py:

lora_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    target_modules=model_args.lora_target_modules.split(","),
    lora_dropout=model_args.lora_dropout,
    bias="none"
)
model.add_lora(lora_config, model_args.lora_scope)
"""


LORA_SCOPE_REFERENCE = """
Project-specific behavior:
- LoRA is inserted into Qwen linear layers
- scope controls where LoRA is active along the input sequence

For speech path:
lora_audio_mask = torch.cat([
    torch.zeros_like(start_mask),
    torch.zeros_like(audio_instruction_mask),
    torch.ones_like(speech_input_mask),
    torch.zeros_like(suffix_mask)
], dim=1)

self.update_lora_mask(lora_audio_mask, False)

Meaning:
- LoRA is not physically attached to Whisper
- LoRA stays inside Qwen
- but it is activated mainly on speech-related token positions
"""


WHY_THESE_MODULES = """
Target modules:
- c_attn
- c_proj
- w1
- w2

Meaning:
- attention projections
- MLP projections

This is the project default, not the only possible LoRA target setting.
"""
