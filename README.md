# ES4R / CAES

Open-source research code for the paper [ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation](https://arxiv.org/abs/2601.16225).

This repository focuses on the response-generation side of the project: multi-turn speech dialogue preprocessing, two-stage training, batch inference, and a local demo built on top of `Whisper + Qwen`.

## What Is Included

- Offline preprocessing for multi-turn speech dialogue manifests
- Qwen-based text label generation for intermediate data construction
- Two-stage training scripts
- Batch inference and automatic response evaluation
- Gradio local demo for speech-based empathetic dialogue

## Release Scope

This is a research code release, not a production SDK.

- The repository keeps the main training and inference path aligned with the paper.
- It does not bundle datasets, pretrained checkpoints, or large experiment artifacts.
- The codebase still reflects research iteration in a few core modules, but unnecessary private files and obvious debug noise have been removed for public release.

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

Key documentation:

- [Project Overview](/Users/z786/Workspace/caes_original/docs/PROJECT_OVERVIEW.md)
- [Architecture Deep Dive](/Users/z786/Workspace/caes_original/docs/KEY_MODULES.md)
- [Data Format](/Users/z786/Workspace/caes_original/docs/DATA_FORMAT.md)
- [Implementation Reference](/Users/z786/Workspace/caes_original/docs/implementation_reference/README.md)
- [Contributing Guide](/Users/z786/Workspace/caes_original/CONTRIBUTING.md)
- [Release Checklist](/Users/z786/Workspace/caes_original/RELEASE_CHECKLIST.md)

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Expected Data Format

The training and inference pipeline expects each sample to contain:

- `dialogue_history`: a list of multi-turn utterances
- `response`: the target response object
- `audio_path`: for every speech turn used by the model
- `speaker_emotion`: optional emotion label used by the empathy-oriented pipeline

See [docs/DATA_FORMAT.md](/Users/z786/Workspace/caes_original/docs/DATA_FORMAT.md) for the full schema and JSON examples.

## Workflow

### 1. Generate Intermediate Text Labels

```bash
python3 emotion_text_generation.py generate \
  --qwen_path /path/to/qwen \
  --manifest examples/train/train_iemocap.jsonl \
  --lab_dir examples/train/emotion_labels \
  --nshard 1 \
  --rank 0 \
  --use_emotion True
```

### 2. Offline Preprocess the Manifest

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

### 3. Stage-1 Training

```bash
bash scripts/train_stage1.sh
```

Required environment variables:

- `QWEN_PATH`
- `WHISPER_PATH`
- `DATA_ROOT`
- `SAVE_ROOT`

### 4. Stage-2 Training

```bash
bash scripts/train_stage2.sh
```

Required environment variables:

- `BLSP_MODEL`
- `DATA_ROOT`
- `SAVE_ROOT`

### 5. Batch Inference

```bash
python3 generate.py \
  --input_file examples/test/test_iemocap.jsonl \
  --output_file predictions.jsonl \
  --blsp_model /path/to/checkpoint \
  --input_field dialogue_history \
  --output_field response \
  --use_emotion
```

### 6. Local Demo

```bash
python3 chat_demo.py --blsp_model /path/to/checkpoint --use_emotion
```

## Main Entry Points

- `train.py`: main training entry
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

Machine-readable citation metadata is provided in [CITATION.cff](/Users/z786/Workspace/caes_original/CITATION.cff).

## Contributing

External improvements are welcome, especially around reproducibility, documentation, and bug fixes in the public training and inference path.

See [CONTRIBUTING.md](/Users/z786/Workspace/caes_original/CONTRIBUTING.md) for contribution expectations.

## Notes

- The current release assumes the input manifest already contains aligned dialogue turns and audio paths.
- The repository is optimized for code clarity and reproducibility of the response-generation pipeline, not for generic plug-and-play deployment.
- Some paper-level components may depend on data preparation or downstream synthesis assets that are not included in this repository.

## License

See [LICENSE](/Users/z786/Workspace/caes_original/LICENSE).
