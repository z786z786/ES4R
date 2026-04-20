# Project Overview

This document is the fastest way to understand what the public ES4R repository includes, how the main pipeline is organized, and where the boundaries of the open-source release currently sit.

## 1. Repository Scope

This repository focuses on the core response-generation path of the paper [ES4R](https://arxiv.org/abs/2601.16225), with emphasis on:

- multi-turn speech dialogue sample organization
- affect-aware speech encoding and fusion
- dual-path training
- batch inference and a local demo

This is research code, not a general speech SDK and not a production-ready end-to-end product. The public release is intentionally scoped around the main experimental path needed for paper understanding and partial reproduction.

## 2. Top-Level Layout

```text
.
├── train.py                     # training entry point
├── generate.py                  # batch inference entry point
├── chat_demo.py                 # local Gradio demo
├── emotion_text_generation.py   # intermediate text label generation
├── response_metrics.py          # automatic evaluation
├── config/                      # DeepSpeed and adapter configs
├── data_process/                # text cleaning and normalization
├── docs/                        # public documentation
├── examples/                    # example manifests
├── figures/                     # paper figures and supporting visuals
├── scripts/                     # training and utility scripts
└── src/                         # core models and data pipeline
```

During repository cleanup, the public path was stripped down to remove:

- private interview or non-project materials
- historical cleanup backups and caches
- obviously local temporary files
- hard-coded conversion scripts that were not part of the main workflow

## 3. Entry Points

### `train.py`

The primary training entry point. It is responsible for:

- parsing model, data, and training arguments
- loading `Blsp2Model`
- building or loading the offline dataset through `src/instruction_dataset.py`
- launching training through the Hugging Face `Trainer`

### `generate.py`

The offline inference entry point. It is responsible for:

- reading JSON or JSONL test manifests
- assembling text tokens and audio features for each turn
- calling `model.generate`
- writing prediction files

### `chat_demo.py`

A local Gradio demo intended for qualitative inspection and live demonstrations.

### `emotion_text_generation.py`

Generates intermediate text labels with Qwen for downstream data preparation.

### `response_metrics.py`

Runs automatic evaluation and reports metrics such as BLEU, ROUGE, METEOR, BERTScore, and Distinct.

## 4. Core Source Files

### `src/modeling_blsp2.py`

The main model implementation. It connects Whisper, the adapter stack, dual-level attention, cross-modal fusion, and Qwen into the core training path.

### `src/instruction_dataset.py`

The dataset builder and batch collator. It handles:

- reading raw manifests
- assembling multi-turn dialogue samples
- loading audio and extracting Whisper features
- saving offline Hugging Face datasets

### `src/modeling_adapter.py`

Implements temporal compression and speech-to-LLM bridging modules, mainly `Subsampler` and `CFormer`.

### `src/modeling_whisper_encoder.py`

A lightweight wrapper around the Whisper encoder that supports direct loading from Whisper checkpoints.

### `src/modeling_qwen.py`

Qwen model adaptation and generation logic.

## 5. Recommended Reading Order

For a first pass through the repository, read in this order:

1. [README.md](../README.md)
2. [docs/DATA_FORMAT.md](DATA_FORMAT.md)
3. [train.py](../train.py)
4. [src/instruction_dataset.py](../src/instruction_dataset.py)
5. [src/modeling_blsp2.py](../src/modeling_blsp2.py)
6. [docs/KEY_MODULES.md](KEY_MODULES.md)

## 6. Pipeline Summary

### Data Preparation

1. A raw manifest provides multi-turn `dialogue_history`, `response`, and audio paths.
2. `emotion_text_generation.py` can generate intermediate text labels.
3. `src/instruction_dataset.py offline` converts samples into the offline dataset expected by training.

### Training

The public repository now exposes a single dual-path training flow:

1. one script optimizes the text path and speech path together
2. losses such as `response_ce` and `response_kl` jointly constrain generation quality and cross-path alignment
3. training can continue from an existing `BLSP_MODEL`, or initialize from `QWEN_PATH + WHISPER_PATH`

### Inference

1. `generate.py` reads the test manifest
2. audio features and dialogue context are prepared turn by turn
3. the model produces empathetic text responses

## 7. What Was Cleaned Up

The open-source preparation included:

- rewriting the repository entry documentation
- removing clearly non-public materials
- trimming noisy debug output in the main path
- tightening the repository boundary around the research implementation

## 8. Current Technical Debt

The codebase is usable, but several obvious maintenance improvements remain:

- `src/modeling_blsp2.py` is still too large and should be split by responsibility
- the data format is task-specific rather than formalized as a reusable protocol
- there is no minimal automated test suite yet
- training is still driven mostly by shell templates rather than a more explicit config layer
