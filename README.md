# ES4R

[English](README.md) | [简体中文](README.zh-CN.md)

Open-source research code for the paper [ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation](https://arxiv.org/abs/2601.16225).

Project Page: https://z786z786.github.io/ES4R/

## Overview

- multi-turn speech dialogue preprocessing
- Qwen-based intermediate text label generation
- dual-path training
- batch inference
- local Gradio demo

This is a research-code release, not a production SDK.

## Repository Layout

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

## Key Documents

- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Architecture Deep Dive](docs/KEY_MODULES.md)
- [Data Format](docs/DATA_FORMAT.md)
- [Implementation Reference](docs/implementation_reference/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Release Checklist](RELEASE_CHECKLIST.md)

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data Format

The training and inference pipeline expects each sample to contain `dialogue_history`, `response`, per-turn `audio_path`, and optional `speaker_emotion`.

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the full schema and examples.

## Workflow

### 1. Intermediate Label Generation

```bash
python3 emotion_text_generation.py generate \
  --qwen_path /path/to/qwen \
  --manifest examples/train/train_iemocap.jsonl \
  --lab_dir examples/train/emotion_labels \
  --nshard 1 \
  --rank 0 \
  --use_emotion True
```

### 2. Offline Preprocessing

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

### 3. Dual-Path Training

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

The public training entry now uses a single dual-path script that jointly optimizes the text path and speech path, instead of exposing a staged shell workflow.

### 4. Batch Inference

```bash
python3 generate.py \
  --input_file examples/test/test_iemocap.jsonl \
  --output_file predictions.jsonl \
  --blsp_model /path/to/checkpoint \
  --input_field dialogue_history \
  --output_field response \
  --use_emotion
```

### 5. Local Demo

```bash
python3 chat_demo.py --blsp_model /path/to/checkpoint --use_emotion
```

## Main Entry Points

- `train.py`: unified training entry
- `generate.py`: batch inference
- `chat_demo.py`: Gradio demo
- `emotion_text_generation.py`: intermediate label generation
- `response_metrics.py`: automatic evaluation
- `src/`: model and dataset implementation

## Citation

If this repository or the associated paper is useful in your work, please cite:

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

## Notes

- the current release focuses on the response-generation path
- datasets and pretrained checkpoints are not bundled
- some paper-level components may depend on external assets not included here

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations.

## License

See [LICENSE](LICENSE).
