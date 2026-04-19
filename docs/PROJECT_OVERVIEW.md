# Project Overview

这份文档面向开源读者，目标是先回答三个问题：

1. 这个仓库到底实现了论文中的哪一部分。
2. 代码主流程从哪里进、怎么跑、输出什么。
3. 仓库整理后应该如何理解目录边界。

## 1. 仓库定位

当前仓库对应论文 [ES4R](https://arxiv.org/abs/2601.16225) 的语音共情回复生成主链路，重点在：

- 多轮语音对话样本组织
- 情感上下文驱动的语音编码与融合
- 两阶段训练
- 批量推理与本地 Demo

这不是一个通用语音 SDK，也不是完整产品化系统。开源版本优先保留论文复现所需的主流程与关键实现。

## 2. 顶层目录

```text
.
├── train.py                     # 训练入口
├── generate.py                  # 批量推理入口
├── chat_demo.py                 # Gradio 本地演示
├── emotion_text_generation.py   # 中间文本标签生成
├── response_metrics.py          # 自动评测
├── config/                      # DeepSpeed / adapter 配置
├── data_process/                # 文本清洗与规范化
├── docs/                        # 开源说明与深挖文档
├── examples/                    # 示例 manifest
├── figures/                     # 论文或说明图
├── scripts/                     # 训练、推理、演示脚本
└── src/                         # 核心模型与数据管线
```

本次整理中，已经把以下内容移出开源主路径：

- 私有面试材料
- 历史清理备份与缓存
- 明显本地化的临时文件
- 未接入主流程的硬编码数据转换脚本

## 3. 运行入口

### `train.py`

训练主入口，负责：

- 解析模型、数据、训练参数
- 加载 `Blsp2Model`
- 调用 `src/instruction_dataset.py` 构建或读取离线数据集
- 使用 Hugging Face `Trainer` 执行训练

### `generate.py`

离线推理入口，负责：

- 读取 JSON / JSONL 格式测试集
- 为每个 turn 组装文本 token 和音频特征
- 调用 `model.generate`
- 输出预测结果文件

### `chat_demo.py`

Gradio 本地 Demo，适合做定性展示或答辩现场演示。

### `emotion_text_generation.py`

使用 Qwen 为训练数据生成中间文本标签，通常服务于阶段化数据构建。

### `response_metrics.py`

对推理结果做自动评测，输出 BLEU、ROUGE、METEOR、BERTScore、Distinct 等指标。

## 4. 关键源文件

### `src/modeling_blsp2.py`

项目主模型，负责把 Whisper、Adapter、双层注意力、跨模态融合和 Qwen 串成一条完整训练链路。

### `src/instruction_dataset.py`

数据组织与 batch collator，负责：

- 读取原始 manifest
- 组装多轮对话训练样本
- 加载音频并提取 Whisper 特征
- 保存离线处理后的 Hugging Face dataset

### `src/modeling_adapter.py`

音频时序压缩与桥接模块，核心类是 `Subsampler` 和 `CFormer`。

### `src/modeling_whisper_encoder.py`

对 Whisper encoder 做轻量封装，支持直接从 Whisper checkpoint 中抽取编码器权重。

### `src/modeling_qwen.py`

Qwen 模型适配与生成逻辑。

## 5. 推荐阅读顺序

如果你第一次接手这个仓库，建议按下面顺序看：

1. [README.md](/Users/z786/Workspace/caes_original/README.md)
2. [docs/DATA_FORMAT.md](/Users/z786/Workspace/caes_original/docs/DATA_FORMAT.md)
3. [train.py](/Users/z786/Workspace/caes_original/train.py)
4. [src/instruction_dataset.py](/Users/z786/Workspace/caes_original/src/instruction_dataset.py)
5. [src/modeling_blsp2.py](/Users/z786/Workspace/caes_original/src/modeling_blsp2.py)
6. [docs/KEY_MODULES.md](/Users/z786/Workspace/caes_original/docs/KEY_MODULES.md)

## 6. 主流程概览

### 数据准备

1. 原始 manifest 提供多轮 `dialogue_history`、`response` 和音频路径。
2. `emotion_text_generation.py` 可生成中间文本标签。
3. `src/instruction_dataset.py offline` 把样本转换为训练所需的离线数据集。

### 训练

1. Stage 1 侧重语义对齐和桥接模块学习。
2. Stage 2 在共情对话场景上继续训练，强化情绪一致性与回复质量。

### 推理

1. `generate.py` 读取测试 manifest。
2. 逐轮提取音频特征并构造历史上下文。
3. 模型输出共情文本回复。

## 7. 这次开源整理做了什么

- 重写了 README 和项目文档入口
- 清理了明显不应公开的私有材料
- 删掉了主链路中的高噪声调试输出
- 保留了研究实现，但收敛了仓库边界

## 8. 仍然存在的技术债

- `src/modeling_blsp2.py` 仍然偏大，后续适合拆模块
- 数据格式依然依赖具体任务 schema，而不是通用协议
- 还没有配套的最小化自动化测试
- 训练脚本仍以 shell 模板为主，配置管理可继续增强
