# Implementation Reference

This directory reorganizes the most important implementation ideas into topic-oriented reference files that are easier to read in isolation during code walkthroughs, paper reproduction, and maintenance.

## Notes

- The files here are reference-oriented excerpts and minimal rewrites intended to improve readability.
- The authoritative training and inference logic still lives in `src/` and `train.py`.
- If you need to change repository behavior, update the original source files first rather than only editing this reference directory.

## File Index

- [audio_frontend.py](audio_frontend.py)
  - `get_waveform`
  - `convert_waveform`
  - how `WhisperFeatureExtractor` is used

- [attention_and_softmax.py](attention_and_softmax.py)
  - scaled dot-product attention
  - a minimal softmax implementation
  - how these pieces relate to dual-level self-attention and cross-attention in the repository

- [adapter_reference.py](adapter_reference.py)
  - `Subsampler`
  - `CFormer`
  - speech-to-LLM bridging logic

- [fusion_pipeline.py](fusion_pipeline.py)
  - dual-level self-attention
  - `get_speech_features`
  - cross-modal fusion

- [losses_reference.py](losses_reference.py)
  - `response_ce`
  - `response_kl`
  - `input_kl`
  - `input_er`

- [lora_reference.py](lora_reference.py)
  - LoRA integration points
  - `lora_scope=audio`
  - target module notes

## Source Mapping

- Main model: [src/modeling_blsp2.py](../../src/modeling_blsp2.py)
- Adapter: [src/modeling_adapter.py](../../src/modeling_adapter.py)
- Audio processing: [src/instruction_dataset.py](../../src/instruction_dataset.py)
- LoRA: [src/plora.py](../../src/plora.py)
- Qwen attention softmax: [src/modeling_qwen.py](../../src/modeling_qwen.py)
