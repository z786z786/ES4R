# Architecture Deep Dive

This document maps the main method claims in the paper to the code that exists in the current repository. It is meant to answer four practical questions:

1. What are the core research modules in the code?
2. How does multi-turn speech context enter the model?
3. What exactly does dual-path training optimize?
4. Which parts of the paper are covered by the public repository?

## 1. Method Backbone

From the paper [ES4R](https://arxiv.org/abs/2601.16225), the main method can be summarized as:

- explicitly modeling structured affective context around speech encoding
- using dual-level attention to capture turn-level and dialogue-level affective dynamics
- injecting speech affect cues into text generation through speech-guided cross-modal attention

The code mapping is direct:

- dual-level attention: [src/modeling_blsp2.py](../src/modeling_blsp2.py)
- speech bridging modules: [src/modeling_adapter.py](../src/modeling_adapter.py)
- structured data preparation: [src/instruction_dataset.py](../src/instruction_dataset.py)
- training entry: [train.py](../train.py)

## 2. End-to-End Flow

From raw input to output, the core path is:

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

The important point is not simply “feed speech into an LLM.” The repository is organized around first converting multi-turn speech context into a learnable affective structure, then conditioning generation on that structure.

## 3. Why the Data Layer Matters

### 3.1 The input is not a single utterance

Preprocessing preserves the full multi-turn history rather than collapsing everything into a single audio clip or plain ASR text.

Each sample minimally contains:

- `dialogue_history`
- `response`
- per-turn `audio_path`

See [src/instruction_dataset.py](../src/instruction_dataset.py).

### 3.2 Samples are decomposed into multiple logical segments

After preprocessing, the model does not consume the raw JSON directly. Instead it receives structured segments such as:

- `start_*`: system prompt
- `instruction_*`: text instruction
- `audio_instruction_*`: audio-side instruction
- `input_*`: text tokens from multi-turn history
- `input_audio_paths`: audio paths from multi-turn history
- `suffix_*`: target response
- `suffix_audio_path`: target response audio path

That means affective context modeling already begins at the data layer before the model stack itself runs.

## 4. Whisper Is Used as an Encoder, Not an ASR Detour

### 4.1 How Whisper is used here

In this repository, Whisper is used as the speech encoder rather than a transcription step before an LLM.

The flow is:

1. `WhisperFeatureExtractor` converts raw waveform input into acoustic features.
2. `WhisperEncoder` produces continuous sequence representations.

See [src/modeling_whisper_encoder.py](../src/modeling_whisper_encoder.py).

### 4.2 Why that matches the paper

The paper needs to preserve paralinguistic information such as:

- prosody
- tone
- intensity
- rhythm

If the system first runs ASR and only passes text to the LLM, much of this signal is compressed away too early. The current implementation avoids that early loss by preserving continuous speech representations.

## 5. The Adapter Bridges Speech to the LLM

The repository exposes two main bridging modules:

- `Subsampler`
- `CFormer`

Their default configuration is defined in [src/configuration_blsp2.py](../src/configuration_blsp2.py).

### 5.1 `Subsampler`

`Subsampler` mainly performs:

- 1D convolutional downsampling
- linear projection
- optional hidden layers
- LayerNorm

Its job is to compress long speech sequences into lengths the language model can handle more effectively.

### 5.2 `CFormer`

`CFormer` is closer to structured token aggregation. It includes:

- pre-CIF layers
- alpha prediction
- CIF aggregation
- post-CIF layers
- token projection

This path is more explicitly about reshaping speech features into a token-like sequence before decoding.

See [src/modeling_adapter.py](../src/modeling_adapter.py).

## 6. Dual-Level Attention Is the Core Research Mechanism

### 6.1 First level: intra-turn speech attention

In [src/modeling_blsp2.py](../src/modeling_blsp2.py), the model first compresses each turn and applies self-attention within the turn.

This stage is responsible for:

- capturing local affective cues within a single turn
- aggregating prosodic, rhythmic, and intensity changes within that turn

### 6.2 Second level: inter-turn history attention

The first-level outputs are reshaped and padded before entering a second self-attention stage.

This stage is responsible for:

- connecting speech states across turns
- modeling dialogue-level affective dynamics
- allowing the model to track how affect evolves across the conversation, not just in the current utterance

This is the clearest implementation counterpart to the paper’s dual-level attention claim.

## 7. Speech-Guided Cross-Modal Fusion

After dual-level speech attention, the code routes speech-side representations through `get_speech_features(...)`, then fuses them with text history using cross-attention.

This stage is responsible for:

- making speech affect representations influence the text generation condition
- using speech as a guidance signal rather than merely concatenating text and audio features

See [src/modeling_blsp2.py](../src/modeling_blsp2.py).

A practical way to describe this module is: affective context is not an auxiliary label here; it is an active conditional representation used at generation time.

## 8. What Dual-Path Training Optimizes

The public repository now exposes one dual-path training flow rather than staged shell scripts.

Script:

- [scripts/train_dual_path.sh](../scripts/train_dual_path.sh)

The core idea is:

- the text path preserves the language model’s text generation capability
- the speech path injects speech affect context and cross-modal fusion features
- CE- and KL-style losses jointly constrain the two paths

The goal is not to present multiple public training stages. The goal is to provide one reproducible entry that jointly optimizes:

- speech-to-text generation usability
- representation alignment between the speech and text paths
- content quality and affective consistency of empathetic responses

## 9. Loss Design

The main visible losses include:

- `response_ce`
- `response_kl`
- `input_kl`
- `input_er`
- `cif`

They can be grouped into three practical categories:

- direct generation losses: keep target response generation trainable
- distillation-style losses: stabilize alignment between the speech and text paths
- structural losses: constrain bridge output length or auxiliary affective behavior

The composition logic lives mainly in [src/modeling_blsp2.py](../src/modeling_blsp2.py).

## 10. Boundary of the Public Release

The most complete part of the repository is the speech-to-empathetic-text response path.

A few boundaries are important:

- the strongest public support is for response generation, not a product-ready speech-to-speech system
- paper-level speech output strategies may not all be included in the repository
- documentation should therefore describe the code as the core research response-generation path, not as a general platform

## 11. Files Worth Reading First

If you only read five files, start with:

1. [train.py](../train.py)
2. [src/instruction_dataset.py](../src/instruction_dataset.py)
3. [src/modeling_blsp2.py](../src/modeling_blsp2.py)
4. [src/modeling_adapter.py](../src/modeling_adapter.py)
5. [generate.py](../generate.py)

## 12. Good Next Maintenance Targets

If open-source maintenance continues, the highest-value next steps are:

1. split `src/modeling_blsp2.py` by encoder, fusion, and loss responsibilities
2. formalize the manifest schema as a standalone data contract
3. add a minimal test path covering preprocess, collator, single forward, and single generation
4. move training configuration away from shell-only environment variables toward an explicit config layer
