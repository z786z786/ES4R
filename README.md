# ES4R

English: Open-source research code for the paper [ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation](https://arxiv.org/abs/2601.16225).

中文：本仓库开源了论文 [ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation](https://arxiv.org/abs/2601.16225) 的研究代码，重点覆盖语音共情回复生成主链路。

Project Page | 在线页面: https://z786z786.github.io/ES4R/

## Overview | 项目概览

English:

- multi-turn speech dialogue preprocessing
- Qwen-based intermediate text label generation
- dual-path training
- batch inference
- local Gradio demo

中文：

- 多轮语音对话预处理
- 基于 Qwen 的中间文本标签生成
- 双路径训练
- 批量推理
- 本地 Gradio 演示

This is a research-code release, not a production SDK.

这是研究型代码仓库，不是生产级 SDK。

## Repository Layout | 仓库结构

```text
.
├── chat_demo.py
├── emotion_text_generation.py
├── generate.py
├── response_metrics.py
├── train.py
├── config/
├── data_process/
├── docs/
├── examples/
├── figures/
├── scripts/
└── src/
```

Key documents:

- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Architecture Deep Dive](docs/KEY_MODULES.md)
- [Data Format](docs/DATA_FORMAT.md)
- [Implementation Reference](docs/implementation_reference/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Release Checklist](RELEASE_CHECKLIST.md)

## Environment Setup | 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data Format | 数据格式

English: The training and inference pipeline expects each sample to contain `dialogue_history`, `response`, per-turn `audio_path`, and optional `speaker_emotion`.

中文：训练与推理默认要求样本包含 `dialogue_history`、`response`、逐轮 `audio_path`，以及可选的 `speaker_emotion`。

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the full schema and examples.

完整 schema 和示例见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

## Workflow | 使用流程

### 1. Intermediate Label Generation | 中间标签生成

```bash
python3 emotion_text_generation.py generate \
  --qwen_path /path/to/qwen \
  --manifest examples/train/train_iemocap.jsonl \
  --lab_dir examples/train/emotion_labels \
  --nshard 1 \
  --rank 0 \
  --use_emotion True
```

### 2. Offline Preprocessing | 离线预处理

```bash
python3 src/instruction_dataset.py offline \
  --data_root examples/train/emotion_labels \
  --manifest_files "*_clean.jsonl" \
  --lm_path /path/to/qwen \
  --save_dir examples/train/emotion_labels/processed \
  --input_field text \
  --output_field output \
  --use_emotion True
```

### 3. Dual-Path Training | 双路径训练

```bash
bash scripts/train_dual_path.sh
```

Required environment variables:

- `DATA_ROOT`
- `SAVE_ROOT`
- `BLSP_MODEL`, or both `QWEN_PATH` and `WHISPER_PATH`

Optional environment variables:

- `DEEPSPEED_CONFIG`
- `LOSS_NAMES`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`

English: The public training entry now uses a single dual-path script that jointly optimizes the text path and speech path, instead of exposing a staged shell workflow.

中文：公开仓库现在只保留一份双路径训练脚本，用于联合优化文本路径和语音路径，不再拆分公开训练入口。

### 4. Batch Inference | 批量推理

```bash
python3 generate.py \
  --input_file examples/test/test_iemocap.jsonl \
  --output_file predictions.jsonl \
  --blsp_model /path/to/checkpoint \
  --input_field dialogue_history \
  --output_field response \
  --use_emotion
```

### 5. Local Demo | 本地演示

```bash
python3 chat_demo.py --blsp_model /path/to/checkpoint --use_emotion
```

## Main Entry Points | 主要入口

- `train.py`: unified training entry
- `generate.py`: batch inference
- `chat_demo.py`: Gradio demo
- `emotion_text_generation.py`: intermediate label generation
- `response_metrics.py`: automatic evaluation
- `src/`: model and dataset implementation

## Citation | 引用

If this repository or the associated paper is useful in your work, please cite:

如果本仓库或对应论文对你的工作有帮助，请引用：

```bibtex
@article{gao2026es4r,
  title={ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation},
  author={Gao, Zhuoyue and Wang, Xiaohui and Yang, Xiaocui and Zhang, Wen and Wang, Daling and Feng, Shi and Zhang, Yifei},
  journal={arXiv preprint arXiv:2601.16225},
  year={2026},
  doi={10.48550/arXiv.2601.16225}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## Notes | 说明

English:

- the current release focuses on the response-generation path
- datasets and pretrained checkpoints are not bundled
- some paper-level components may depend on external assets not included here

中文：

- 当前开源版本重点覆盖回复生成主链路
- 数据集和预训练 checkpoint 不包含在仓库内
- 论文中的部分扩展组件可能仍依赖仓库外资源

## Contributing | 贡献

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations.

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## License | 许可证

See [LICENSE](LICENSE).
