# Data Format

This document defines the default data schema assumed by the public ES4R repository. The training and inference code are both organized around this structure.

## 1. Sample Structure

A typical JSONL record looks like this:

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

## 2. Field Definitions

### Top-level fields

- `conversation_id`: conversation identifier
- `turn_id`: the current sample’s turn index in the full conversation
- `dialogue_history`: the dialogue history before the current prediction point
- `speaker_emotion`: emotion label associated with the current sample
- `response`: the target response object
- `last_turn`: whether this sample is the last one in the conversation

### Fields inside `dialogue_history`

- `index`: position of the utterance inside the conversation
- `role`: typically `speaker` or `listener`
- `utterance`: text content
- `audio_path`: path to the utterance audio

### Fields inside `response`

- `index`: position of the response turn
- `role`: role name, usually `listener`
- `utterance`: target response text
- `audio_path`: path to the target response audio

## 3. Code Paths That Depend on This Schema

The following files assume this schema directly:

- [src/instruction_dataset.py](../src/instruction_dataset.py)
- [generate.py](../generate.py)

## 4. Role Conventions

The current implementation largely maps roles as follows:

- `speaker` -> `user`
- `listener` or any non-`speaker` role -> `assistant`

If your dataset uses different role names, normalize them during preprocessing before feeding them into the repository pipeline.

## 5. Audio Conventions

- every turn used for training or inference should provide a readable `audio_path`
- the expected sampling behavior follows the Whisper feature extraction configuration
- the code may filter out clips that are too short or too long

## 6. What `last_turn` Controls

`last_turn` affects whether an end marker is appended after the target response suffix. To keep training targets consistent, it is better to set this field explicitly during dataset construction.

If your source data does not include `last_turn`, add it before preprocessing.

## 7. Intermediate Offline Dataset Format

After offline preprocessing, the data is converted into a Hugging Face dataset with fields such as:

- `start_ids`
- `instruction_ids`
- `audio_instruction_ids`
- `input_ids`
- `input_audio_paths`
- `suffix_ids`
- `suffix_audio_path`
- `emotion_label`

These fields are generated automatically by [src/instruction_dataset.py](../src/instruction_dataset.py). They are not meant to be hand-authored in raw manifests.

## 8. Minimum Practical Recommendations

If you want a stable reproduction starting point, make sure of three things first:

1. every dialogue-history turn has an `audio_path`
2. `role` uses one consistent naming scheme across the dataset
3. `last_turn` is defined consistently within the same dataset
