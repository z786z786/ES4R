# Data Format

这份文档定义当前开源仓库默认使用的数据格式。训练和推理代码都围绕这个 schema 组织。

## 1. 单条样本结构

最常见的 JSONL 记录包含这些字段：

```json
{
  "conversation_id": "0001",
  "turn_id": 3,
  "dialogue_history": [
    {
      "index": 0,
      "role": "speaker",
      "utterance": "I feel exhausted today.",
      "audio_path": "audio/0001_0_speaker.wav"
    },
    {
      "index": 1,
      "role": "listener",
      "utterance": "That sounds like a rough day.",
      "audio_path": "audio/0001_1_listener.wav"
    }
  ],
  "speaker_emotion": "sad",
  "response": {
    "index": 2,
    "role": "listener",
    "utterance": "I'm sorry you're dealing with that. Do you want to talk about what made it so draining?",
    "audio_path": "audio/0001_2_listener.wav"
  },
  "last_turn": false
}
```

## 2. 字段说明

### 顶层字段

- `conversation_id`: 对话标识
- `turn_id`: 当前样本在整段对话中的 turn 编号
- `dialogue_history`: 当前轮之前的历史对话
- `speaker_emotion`: 当前样本使用的情绪标签
- `response`: 目标回复
- `last_turn`: 是否为该对话的最后一个样本

### `dialogue_history` 中的字段

- `index`: 当前 utterance 在整段对话中的顺序
- `role`: `speaker` 或 `listener`
- `utterance`: 文本内容
- `audio_path`: 音频文件路径

### `response` 中的字段

- `index`: 回复所在位置
- `role`: 角色名，通常是 `listener`
- `utterance`: 目标回复文本
- `audio_path`: 目标回复音频路径

## 3. 代码默认依赖

下列文件默认依赖该 schema：

- [src/instruction_dataset.py](../src/instruction_dataset.py)
- [generate.py](../generate.py)

## 4. 角色约定

当前实现中，角色映射逻辑基本是：

- `speaker` -> `user`
- `listener` 或非 `speaker` -> `assistant`

因此如果你的数据集角色命名不同，需要在预处理阶段先统一。

## 5. 音频约定

- 所有参与训练和推理的 turn 都应提供可读取的 `audio_path`
- 默认采样率以 Whisper 特征提取配置为准
- 代码中通常会过滤持续时间过短或过长的音频

## 6. `last_turn` 的作用

`last_turn` 会影响目标回复后缀是否追加终止标记。为了避免训练目标不一致，建议在构建数据集时显式给出该字段。

如果你的数据没有这个字段，建议在预处理前先补齐。

## 7. 中间数据格式

离线预处理后，数据会被转换为 Hugging Face dataset，并包含：

- `start_ids`
- `instruction_ids`
- `audio_instruction_ids`
- `input_ids`
- `input_audio_paths`
- `suffix_ids`
- `suffix_audio_path`
- `emotion_label`

这些字段不要求你手写；它们由 [src/instruction_dataset.py](../src/instruction_dataset.py) 自动生成。

## 8. 最小建议

如果你准备复现实验，建议先保证三件事：

1. 每轮历史都带 `audio_path`
2. `role` 只使用一套稳定命名
3. `last_turn` 在同一数据集里定义一致
