# ES4R

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库开源了论文 [ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation](https://arxiv.org/abs/2601.16225) 的研究代码，重点覆盖语音共情回复生成主链路。

在线页面：https://z786z786.github.io/ES4R/

## 项目概览

- 多轮语音对话预处理
- 基于 Qwen 的中间文本标签生成
- 双路径训练
- 批量推理
- 本地 Gradio 演示

这是研究型代码仓库，不是生产级 SDK。

## 仓库结构

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

## 关键文档

- [项目总览](docs/PROJECT_OVERVIEW.md)
- [架构深挖](docs/KEY_MODULES.md)
- [数据格式](docs/DATA_FORMAT.md)
- [实现参考](docs/implementation_reference/README.md)
- [贡献指南](CONTRIBUTING.md)
- [发布检查清单](RELEASE_CHECKLIST.md)

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 数据格式

训练与推理默认要求样本包含 `dialogue_history`、`response`、逐轮 `audio_path`，以及可选的 `speaker_emotion`。

完整 schema 和示例见 [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)。

## 使用流程

### 1. 中间标签生成

```bash
python3 emotion_text_generation.py generate \
  --qwen_path /path/to/qwen \
  --manifest examples/train/train_iemocap.jsonl \
  --lab_dir examples/train/emotion_labels \
  --nshard 1 \
  --rank 0 \
  --use_emotion True
```

### 2. 离线预处理

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

### 3. 双路径训练

```bash
bash scripts/train_dual_path.sh
```

必需环境变量：

- `DATA_ROOT`
- `SAVE_ROOT`
- `BLSP_MODEL`，或者同时提供 `QWEN_PATH` 和 `WHISPER_PATH`

可选环境变量：

- `DEEPSPEED_CONFIG`
- `LOSS_NAMES`
- `LEARNING_RATE`
- `NUM_TRAIN_EPOCHS`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `GRADIENT_ACCUMULATION_STEPS`

公开仓库现在只保留一份双路径训练脚本，用于联合优化文本路径和语音路径，不再拆分公开训练入口。

### 4. 批量推理

```bash
python3 generate.py \
  --input_file examples/test/test_iemocap.jsonl \
  --output_file predictions.jsonl \
  --blsp_model /path/to/checkpoint \
  --input_field dialogue_history \
  --output_field response \
  --use_emotion
```

### 5. 本地演示

```bash
python3 chat_demo.py --blsp_model /path/to/checkpoint --use_emotion
```

## 主要入口

- `train.py`：统一训练入口
- `generate.py`：批量推理
- `chat_demo.py`：Gradio 演示
- `emotion_text_generation.py`：中间标签生成
- `response_metrics.py`：自动评测
- `src/`：模型和数据集实现

## 引用

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

机器可读引用元数据见 [CITATION.cff](CITATION.cff)。

## 说明

- 当前开源版本重点覆盖回复生成主链路
- 数据集和预训练 checkpoint 不包含在仓库内
- 论文中的部分扩展组件可能仍依赖仓库外资源

## 贡献

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

见 [LICENSE](LICENSE)。
