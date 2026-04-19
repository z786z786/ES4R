# Architecture Deep Dive

这份文档从“论文方法”映射到“当前代码实现”，重点回答：

1. 代码里的核心创新模块是什么。
2. 多轮语音上下文如何进入模型。
3. 双路径训练到底在优化什么。
4. 开源仓库当前覆盖了论文链路中的哪些部分。

## 1. 方法主线

根据论文 [ES4R](https://arxiv.org/abs/2601.16225) 的摘要，方法主线可以概括为：

- 在语音编码前后显式建模结构化情感上下文
- 用双层注意力捕获 turn-level 和 dialogue-level affective dynamics
- 用 speech-guided cross-modal attention 把语音情感线索注入文本生成

当前仓库和这条主线的对应关系是明确的：

- 双层注意力在 [src/modeling_blsp2.py](../src/modeling_blsp2.py)
- 语音桥接模块在 [src/modeling_adapter.py](../src/modeling_adapter.py)
- 数据结构化组织在 [src/instruction_dataset.py](../src/instruction_dataset.py)
- 训练入口在 [train.py](../train.py)

## 2. 端到端链路

从输入到输出，主流程可以画成：

```text
multi-turn dialogue manifest
    -> text tokenization + audio path collection
    -> WhisperFeatureExtractor
    -> WhisperEncoder
    -> Subsampler / CFormer
    -> intra-turn self-attention
    -> inter-turn self-attention
    -> speech-guided cross-modal fusion
    -> Qwen decoder
    -> empathetic response text
```

这条链路的关键点不在“把语音喂给 LLM”，而在“先把多轮语音上下文组织成可学习的情感结构，再送入语言模型”。

## 3. 数据层为什么重要

### 3.1 输入不是单轮语音

数据预处理阶段保留的是完整多轮历史，而不是单句音频或纯 ASR 文本。

每条样本至少包含：

- `dialogue_history`
- `response`
- 每轮 `audio_path`

代码位置：

- [src/instruction_dataset.py](../src/instruction_dataset.py)

### 3.2 样本会被拆成几个逻辑段

预处理后，模型读入的不是原始 JSON，而是几段结构化输入：

- `start_*`: system prompt
- `instruction_*`: 文本 instruction
- `audio_instruction_*`: 音频侧 instruction
- `input_*`: 多轮历史文本 token
- `input_audio_paths`: 多轮历史音频路径
- `suffix_*`: 目标回复
- `suffix_audio_path`: 目标回复音频路径

这意味着“情感上下文建模”其实从数据层就已经开始，而不是等到模型内部才处理。

## 4. Whisper 不是 ASR 中间站

### 4.1 仓库里的 Whisper 用法

Whisper 在这里承担语音编码器角色，而不是先转写再丢给 LLM。

使用方式是：

1. `WhisperFeatureExtractor` 负责把原始波形转成特征。
2. `WhisperEncoder` 负责输出连续时序表示。

代码位置：

- [src/modeling_whisper_encoder.py](../src/modeling_whisper_encoder.py)

### 4.2 为什么这一步和论文目标一致

论文关注的是保留语音中的副语言信息，例如：

- prosody
- tone
- intensity
- rhythm

如果先做 ASR，再把文字给 LLM，这些信息会被压缩掉。当前实现保留连续语音表示，正是为了避免这一步的早期损失。

## 5. Adapter 是语音到 LLM 的桥

### 5.1 两种桥接方式

仓库里有两个桥接模块：

- `Subsampler`
- `CFormer`

默认配置见 [src/configuration_blsp2.py](../src/configuration_blsp2.py)。

### 5.2 `Subsampler`

`Subsampler` 的核心是：

- 1D 卷积降采样
- 线性投影
- 可选隐藏层
- LayerNorm

作用是把长时序语音帧压缩到 LLM 更容易处理的表示长度。

### 5.3 `CFormer`

`CFormer` 更接近结构化 token 聚合，包含：

- pre-CIF layers
- alpha 预测
- CIF 聚合
- post-CIF layers
- token projection

这条路径更强调“把语音表示整理成 token-like 序列”，为后面的语言模型解码做准备。

代码位置：

- [src/modeling_adapter.py](../src/modeling_adapter.py)

## 6. 双层注意力是这份代码最关键的研究实现

### 6.1 第一层：单轮内部语音注意力

在 [src/modeling_blsp2.py](../src/modeling_blsp2.py) 中，模型先对每一轮语音做压缩与 self-attention。

作用：

- 在单个 turn 内捕获局部情感线索
- 聚合同一轮中的韵律、节奏和强度变化

### 6.2 第二层：跨轮历史注意力

第一层输出会被 reshape 和 padding，再进入第二层 self-attention。

作用：

- 把不同轮次的语音状态串起来
- 建模 dialogue-level affective dynamics
- 让模型不只看当前一句，而是看情绪如何在多轮对话中延续和变化

这一点和论文摘要里的 dual-level attention 是高度一致的，也是当前仓库里最值得在开源说明中强调的部分。

## 7. Speech-Guided Cross-Modal Fusion

双层语音注意力输出之后，代码会把语音侧表示送入 `get_speech_features(...)`，再通过 cross-attention 与文本历史融合。

作用：

- 让语音情感表征影响文本生成条件
- 不是简单把文本和语音拼接，而是让语音侧成为文本语义检索和生成的引导信号

对应文件：

- [src/modeling_blsp2.py](../src/modeling_blsp2.py)

这部分可以理解为“情感上下文不是附加标签，而是生成时真正参与条件化的表示”。

## 8. 双路径训练在优化什么

公开仓库中，训练入口已经统一为单一的 dual-path 流程，而不是分阶段脚本。

脚本：

- [scripts/train_dual_path.sh](../scripts/train_dual_path.sh)

双路径训练的核心是：

- text path 保留语言模型的文本生成能力
- speech path 注入语音情感上下文与跨模态融合表示
- 通过 CE 与 KL 类损失联合约束两条路径

这样做的目标不是把训练拆成多个公开阶段，而是在同一个可复现入口里稳定优化：

- 语音到文本生成的可用性
- speech/text 两条路径的表示对齐
- 共情回复的内容质量与情绪一致性

## 9. 损失函数设计

训练中可见的核心损失包括：

- `response_ce`
- `response_kl`
- `input_kl`
- `input_er`
- `cif`

可以把它们粗分成三类：

- 直接生成损失：保证回复文本可训练
- 蒸馏类损失：稳定 speech/text 双路径对齐
- 结构约束损失：约束语音桥接输出长度或情感判别

这些损失的组合逻辑集中在 [src/modeling_blsp2.py](../src/modeling_blsp2.py)。

## 10. 论文和当前仓库的边界

当前仓库最完整的是“语音到共情文本回复”的主链路。

需要明确的是：

- 仓库里最成熟的是 response generation，而不是完整产品级 speech-to-speech 系统
- 论文里如果涉及更完整的语音输出策略，开源仓库未必全部包含
- 因此文档应该准确表述为“论文核心响应生成代码”和“主实验链路实现”，不要写成通用平台

## 11. 开源读者最该关注的文件

如果只读五个文件，建议优先看：

1. [train.py](../train.py)
2. [src/instruction_dataset.py](../src/instruction_dataset.py)
3. [src/modeling_blsp2.py](../src/modeling_blsp2.py)
4. [src/modeling_adapter.py](../src/modeling_adapter.py)
5. [generate.py](../generate.py)

## 12. 后续如果继续增强

如果后续继续做开源维护，最值得投入的方向是：

1. 把 `src/modeling_blsp2.py` 按 encoder、fusion、losses 拆分。
2. 把 manifest schema 固化成独立数据协议。
3. 增加最小可运行测试，至少覆盖 preprocess、collator、single forward、single generation。
4. 把训练脚本参数从 shell 环境变量继续抽象成配置文件。
