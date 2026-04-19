# Implementation Reference

这个目录把项目中最关键的实现按主题整理成了可单独阅读的参考文件，方便代码走读、论文复现和项目维护。

说明：

- 这里的文件是“实现摘录 + 最小化重写 + 原文件定位”，目的是便于理解。
- 原始训练逻辑仍以 `src/` 和 `train.py` 为准。
- 如果你要改项目功能，请优先改原始代码，不要只改这个目录里的参考文件。

## 文件索引

- [audio_frontend.py](audio_frontend.py)
  - `get_waveform`
  - `convert_waveform`
  - `WhisperFeatureExtractor` 使用方式

- [attention_and_softmax.py](attention_and_softmax.py)
  - scaled dot-product attention 数学实现
  - softmax 最小实现
  - 项目中双层 self-attention / cross-attention 的对应关系

- [adapter_reference.py](adapter_reference.py)
  - `Subsampler`
  - `CFormer`
  - speech-to-LLM 桥接逻辑

- [fusion_pipeline.py](fusion_pipeline.py)
  - 双层 self-attention
  - `get_speech_features`
  - cross-attention 融合

- [losses_reference.py](losses_reference.py)
  - `response_ce`
  - `response_kl`
  - `input_kl`
  - `input_er`

- [lora_reference.py](lora_reference.py)
  - LoRA 接入点
  - `lora_scope=audio`
  - target modules 说明

## 原始代码定位

- 主模型：[src/modeling_blsp2.py](../../src/modeling_blsp2.py)
- Adapter：[src/modeling_adapter.py](../../src/modeling_adapter.py)
- 音频处理：[src/instruction_dataset.py](../../src/instruction_dataset.py)
- LoRA：[src/plora.py](../../src/plora.py)
- Qwen 注意力中的 softmax：[src/modeling_qwen.py](../../src/modeling_qwen.py)
