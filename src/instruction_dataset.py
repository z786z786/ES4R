import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import soundfile as sf
import mmap
import io

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import random
import datasets
from datasets import Features, Sequence, Value

from dataclasses import dataclass

from transformers import WhisperFeatureExtractor
try:
    from .tokenization_qwen import QWenTokenizer
except:
    from tokenization_qwen import QWenTokenizer

logger = logging.getLogger(__name__)

emotion2idx = {
    "afraid": 1,
    "proud": 2,
    "faithful": 3,
    "terrified": 4,
    "joyful": 5,
    "angry": 6,
    "sad": 7,
    "jealous": 8,
    "grateful": 9,
    "prepared": 10,
    "embarrassed": 11,
    "excited": 12,
    "annoyed": 13,
    "lonely": 14,
    "ashamed": 15,
    "guilty": 16,
    "surprised": 17,
    "nostalgic": 18,
    "confident": 19,
    "furious": 20,
    "disappointed": 21,
    "caring": 22,
    "trusting": 23,
    "disgusted": 24,
    "anticipating": 25,
    "anxious": 26,
    "content": 27,
    "impressed": 28,
    "hopeful": 29,
    "apprehensive": 30,
    "devastated": 31,
    "sentimental": 32
}

feature_schema = Features({
    "start_ids": Sequence(Value("int64")),
    "start_mask": Sequence(Value("int64")),
    "start_labels": Sequence(Value("int64")),
    "instruction_ids": Sequence(Value("int64")),
    "instruction_mask": Sequence(Value("int64")),
    "instruction_labels": Sequence(Value("int64")),
    "audio_instruction_ids": Sequence(Value("int64")),
    "audio_instruction_mask": Sequence(Value("int64")),
    "audio_instruction_labels": Sequence(Value("int64")),
    "input_ids": Sequence(Sequence(Value("int64"))),
    "input_mask": Sequence(Sequence(Value("int64"))),
    "input_labels": Sequence(Sequence(Value("int64"))),
    "suffix_ids": Sequence(Value("int64")),
    "suffix_mask": Sequence(Value("int64")),
    "suffix_labels": Sequence(Value("int64")),
    # "input_audio_paths": Sequence(Struct({
    #     "prefix": Sequence(Value("int64")),
    #     "audio_path": Value("string"),
    # })),
    "suffix_audio_path": {
        "prefix": Sequence(Value("int64")),
        "audio_path": Value("string"),
        "suffix": Sequence(Value("int64"))
    },
    "input_audio_paths": Sequence(Value("string")),
    # "suffix_audio_path": Value("string"),
    "emotion_label": Value("int64"),
    "to_keep": Value("bool"),
})

def process_dataset(
    batch,
    tokenizer,
    _tokenize_str,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="input",
    output_field="output",
    max_length=1024,
    min_duration=1.0,
    max_duration=30.0,
    check_audio=True,
    use_emotion=False,
):

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")
    eod_tokens = [tokenizer.eod_id]

    if use_emotion:
        system_prompt = "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
    else:
        system_prompt = "You are a helpful assistant."

    instruction_ids, instruction_mask, instruction_labels = [], [], []
    if instruction:
        instruction_ids = _tokenize_str(content=instruction)
        instruction_mask = [1] * len(instruction_ids)
        instruction_labels = [-100] * len(instruction_ids)

    audio_instruction_ids, audio_instruction_mask, audio_instruction_labels = instruction_ids, instruction_mask, instruction_labels
    if audio_instruction:
        audio_instruction_ids = _tokenize_str(content=audio_instruction)
        audio_instruction_mask = [1] * len(audio_instruction_ids)
        audio_instruction_labels = [-100] * len(audio_instruction_ids)
    
        # def _tokenize_str(role="", content=""):
        # tokens = []
        # if role:
        #     tokens += tokenizer.encode(role, allowed_special=set()) + tokenizer.encode("\n")
        # if content:
        #     tokens += tokenizer.encode(content, allowed_special=set())
        # return tokens
    
    # 构造 start 段（即 system prompt 开头）
    start_ids, start_mask, start_labels = [], [], []
    start_ids += im_start_tokens + _tokenize_str(role="system", content=f"{system_prompt}")
    start_mask = [1] * len(start_ids)
    start_labels = [-100] * len(start_ids)

    # input_field="dialogue_history"
    input = batch[input_field]


    emotion_label = 0
    to_keep = True
    input_ids, input_mask, input_labels = [], [], []
    input_audio_paths = []

    for speech in input:
        speech_role = speech["role"]
        speech_utterance = speech["utterance"]
        
        input_id, input_masks, input_label = [], [], []
        input_id += im_end_tokens + nl_tokens + im_start_tokens + _tokenize_str(role=("user" if speech_role=="speaker" else "assistant"), content=speech_utterance)
        input_masks += [1] * len(input_id)
        input_label += [-100] * len(input_id)
        input_ids.append(input_id)
        input_mask.append(input_masks)
        input_labels.append(input_label)
        
        if check_audio:
            audio_path = speech.get("audio_path", "")
            if audio_path:
                try:
                    waveform = get_waveform(audio_path)
                    duration = 1.0 * waveform.shape[0] / 16000.0
                    if duration < min_duration or duration > max_duration:
                        to_keep = False
                    else:
                        
                        #  def _tokenize_str(role="", content=""):
                        #     tokens = []
                        #     if role:
                        #         tokens += tokenizer.encode(role, allowed_special=set()) + tokenizer.encode("\n")
                        #     if content:
                        #         tokens += tokenizer.encode(content, allowed_special=set())
                        #     return tokens
                        
                        # role_tokens = tokenizer.encode(role=("user" if speech_role=="speaker" else "assistant"), allowed_special=set())
                        # audio_dict = {
                        #     "prefix": im_end_tokens + nl_tokens + im_start_tokens +  _tokenize_str(role=("user" if speech_role=="speaker" else "assistant"), content=""),
                        #     "audio_path": audio_path,
                        # }
                        # input_audio_paths.append(audio_dict)
                        input_audio_paths.append(audio_path)
                except:
                    to_keep = False
            else:
                to_keep = False
        
    response = batch[output_field]
    response_role = response["role"]
    response_utterance = response["utterance"]
    # def _tokenize_str(role="", content=""):
    #     tokens = []
    #     if role:
    #         tokens += tokenizer.encode(role, allowed_special=set()) + tokenizer.encode("\n")
    #     if content:
    #         tokens += tokenizer.encode(content, allowed_special=set())
    #     return tokens

    suffix_audio_path = {}
    suffix_ids, suffix_mask, suffix_labels = [], [], []
    suffix_ids += im_end_tokens + nl_tokens + im_start_tokens + _tokenize_str(role=("user" if response_role=="speaker" else "assistant"))
    # suffix_mask += [1] * len(suffix_ids)
    suffix_labels += [-100] * len(suffix_ids)
    
    
    if batch.get("last_turn", False):
        suffix_ids += _tokenize_str(content=response_utterance) + im_end_tokens + nl_tokens + eod_tokens
        suffix_mask += [1] * len(suffix_ids)
        suffix_labels += _tokenize_str(content=response_utterance) + im_end_tokens + nl_tokens + eod_tokens
    else:               
        suffix_ids += _tokenize_str(content=response_utterance) + im_end_tokens + nl_tokens
        suffix_mask += [1] * len(suffix_ids)
        suffix_labels += _tokenize_str(content=response_utterance) + im_end_tokens + nl_tokens
    if check_audio:
        response_audio_path = response.get("audio_path", "")
        if response_audio_path:
            try:
                waveform = get_waveform(response_audio_path)
                duration = 1.0 * waveform.shape[0] / 16000.0
                if duration < min_duration or duration > max_duration:
                    to_keep = False
                else:
                    audio_dict = {
                        "prefix": im_end_tokens + nl_tokens + im_start_tokens +  _tokenize_str(role=("user" if response_role=="speaker" else "assistant"), content=""),
                        "audio_path": response_audio_path,
                        "suffix": im_end_tokens + nl_tokens + eod_tokens
                    }
                    suffix_audio_path = audio_dict
            except:
                to_keep = False
        else:
            to_keep = False
    
    if use_emotion:
        emotion = batch.get("speaker_emotion", "")
        if emotion:
            emotion_label = emotion2idx[emotion]
        else:
            to_keep = False
    # if to_keep == True:
    #     assert len(audio_paths) == len(input) + 1
    # total_len = len(start_ids) + len(instruction_ids) + len(audio_instruction_ids) + len(input_ids) + len(suffix_ids)
    # if total_len > max_length:
    #     to_keep = False

    batch["start_ids"] = start_ids
    batch["start_mask"] = start_mask
    batch["start_labels"] = start_labels
    batch["instruction_ids"] = instruction_ids
    batch["instruction_mask"] = instruction_mask
    batch["instruction_labels"] = instruction_labels
    batch["audio_instruction_ids"] = audio_instruction_ids
    batch["audio_instruction_mask"] = audio_instruction_mask
    batch["audio_instruction_labels"] = audio_instruction_labels
    batch["input_ids"] = input_ids
    batch["input_mask"] = input_mask
    batch["input_labels"] = input_labels
    batch["suffix_ids"] = suffix_ids
    batch["suffix_mask"] = suffix_mask
    batch["suffix_labels"] = suffix_labels
    batch["input_audio_paths"] = input_audio_paths
    batch["suffix_audio_path"] = suffix_audio_path
    batch["emotion_label"] = emotion_label
    batch["to_keep"] = to_keep

    return batch



def load_instruction_dataset(
    manifest_dir="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    output_field="",
    max_length=1024,
    min_duration=1.0,
    max_duration=30.0,
    num_proc=1,
    use_emotion=True,
):
    if not manifest_files:
        logger.warning(f"loading processed dataset from {manifest_dir}")
        dataset = datasets.load_from_disk(manifest_dir)
        return dataset
    
    logger.warning(f"load dataset from scratch from {manifest_dir}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        manifest_dir, data_files=manifest_files_list, split="train", streaming=False
    )

    def _tokenize_str(role="", content=""):
        tokens = []
        if role:
            tokens += tokenizer.encode(role, allowed_special=set()) + tokenizer.encode("\n")
        if content:
            tokens += tokenizer.encode(content, allowed_special=set())
        return tokens

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "_tokenize_str": _tokenize_str,
            "instruction": instruction,
            "instruction_field": instruction_field,
            "audio_instruction": audio_instruction,
            "audio_instruction_field": audio_instruction_field,
            "input_field": input_field,
            "output_field": output_field,
            "max_length": max_length,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "use_emotion": use_emotion,
        },
        # 定义 map 后输出的 schema（必须是 datasets.Features 类型）
        features=feature_schema,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc,
    )

    def to_keep(flag):
        return flag

    dataset = dataset.filter(
        to_keep,
        input_columns=["to_keep"]
    )

    return dataset

def load_instruction_datasets(data_args, tokenizer=None, num_proc=8):
    if os.path.exists(data_args.dataset_save_dir) and os.listdir(data_args.dataset_save_dir):
        logger.warning(f"loading processed dataset from {data_args.dataset_save_dir}")
        dataset = datasets.load_from_disk(data_args.dataset_save_dir)
        return dataset

    manifest_keys = ["manifest_dirs", "manifest_files", "instructions", "instruction_fields",
                     "audio_instructions", "audio_instruction_fields", "input_fields",
                     "audio_fields", "output_fields"]
    if data_args.dataset_dirs:
        dataset_dirs = data_args.dataset_dirs.split("|")
        all_datasets = [load_instruction_dataset(manifest_dir=dataset_dir) for dataset_dir in dataset_dirs]
        num_datasets = len(all_datasets)
    else:
        manifest_values = [(getattr(data_args, key)).split("|") for key in manifest_keys]
        num_datasets = len(manifest_values[0])
        if num_datasets == 0:
            raise ValueError("no datasets specified")
        for i, key in enumerate(manifest_keys):
            if len(manifest_values[i]) != num_datasets:
                raise ValueError(f"unexpected number of {key} in {data_args}")
        all_datasets = [load_instruction_dataset(manifest_dir=manifest_values[0][i],
                                                 manifest_files=manifest_values[1][i],
                                                 instruction=manifest_values[2][i],
                                                 instruction_field=manifest_values[3][i],
                                                 audio_instruction=manifest_values[4][i],
                                                 audio_instruction_field=manifest_values[5][i],
                                                 input_field=manifest_values[6][i],
                                                #  audio_field=manifest_values[7][i],
                                                 output_field=manifest_values[8][i],
                                                 tokenizer=tokenizer,
                                                 num_proc=num_proc)
                        for i in range(num_datasets)]
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        sample_probs = data_args.sample_probs.split("|")
        if len(sample_probs) == num_datasets:
            sample_probs = [float(prob) for prob in sample_probs]
        else:
            if data_args.sample_probs == "None":
                sample_probs = None
            else:
                raise ValueError(f"unexpected number of probabilities in {data_args}")
        dataset = datasets.interleave_datasets(all_datasets, stopping_strategy=data_args.interleave_stopping_strategy,
                                               probabilities=sample_probs)

    
    if data_args.dataset_save_dir and (not dist.is_initialized() or dist.get_rank() == 0):
        dataset.save_to_disk(data_args.dataset_save_dir)

    return dataset


def collate_tokens(
    values: List[List[int]],
    pad_id: int,
    left_pad: bool = False
):
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            copy_tensor(torch.LongTensor(v), res[i][-len(v): ])
        else:
            copy_tensor(torch.LongTensor(v), res[i][: len(v)])
    return res

def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
            data = mmap_o[offset : offset + length]
    return data

def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)

def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg

def get_waveform(
    path_or_fp: str,
    normalization=True,
    mono=True,
    frames=-1,
    start=0,
    always_2d=False,
    output_sample_rate=16000,
) -> Tuple[np.ndarray, int]:
    meta = path_or_fp.split(":")
    if len(meta) == 3:
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])
    else:
        path_or_fp = path_or_fp
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLACC/OGG/MP3/OPUS audios")

    ext = Path(path_or_fp).suffix
    if ext in [".wav", ".flac", ".ogg", ".mp3", ".opus"]:
        waveform, sample_rate = sf.read(
            path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
        )
    elif ext in [".zip"]:
        data = read_from_stored_zip(path_or_fp, start, frames)
        assert is_sf_audio_data(data)
        f = io.BytesIO(data)
        waveform, sample_rate = sf.read(
            f, dtype="float32", always_2d=True
        )
    else:
        raise ValueError(f"Unsupported audio format: {ext}")
    
    waveform = waveform.T

    waveform, sample_rate = convert_waveform(waveform, sample_rate, to_mono=mono, to_sample_rate=output_sample_rate)
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)

    return waveform

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization
    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate

def process_dialogues(input_ids, input_mask, input_labels, pad_id=0):
    max_dialogue_length = max([len(dialogue) for dialogue in input_ids])  # 找到最大的对话轮次长度
    pad_input_ids = [
        dialogue + [[pad_id]] * (max_dialogue_length - len(dialogue)) if len(dialogue) < max_dialogue_length else dialogue[:max_dialogue_length]
     for dialogue in input_ids]
    pad_input_masks = [
        dialogue + [[0]] * (max_dialogue_length - len(dialogue)) if len(dialogue) < max_dialogue_length else dialogue[:max_dialogue_length]
     for dialogue in input_mask]
    pad_input_labels = [
        dialogue + [[-100]] * (max_dialogue_length - len(dialogue)) if len(dialogue) < max_dialogue_length else dialogue[:max_dialogue_length]
     for dialogue in input_labels]
    
    max_speech_length = max([len(speech) for dialogue in input_ids for speech in dialogue])  # 找到最大的对话轮次长度
    padded_input_ids = [[
        speech + [pad_id] * (max_speech_length - len(speech)) if len(speech) < max_speech_length else speech[:max_speech_length]
        for speech in dialogue
    ] for dialogue in pad_input_ids]
    padded_input_masks = [[
        speech + [0] * (max_speech_length - len(speech)) if len(speech) < max_speech_length else speech[:max_speech_length]
        for speech in dialogue
    ] for dialogue in pad_input_masks]
    padded_input_labels = [[
        speech + [-100] * (max_speech_length - len(speech)) if len(speech) < max_speech_length else speech[:max_speech_length]
        for speech in dialogue
    ] for dialogue in pad_input_labels]

    return padded_input_ids, padded_input_masks, padded_input_labels

def process_suffixes(suffix_ids, suffix_masks, suffix_labels, pad_id=0):
    max_suffix_length = max([len(suffix) for suffix in suffix_ids])
    padded_suffix_ids = [suffix + [pad_id] * (max_suffix_length - len(suffix)) if len(suffix) < max_suffix_length else suffix[:max_suffix_length] for suffix in suffix_ids]
    padded_suffix_masks = [suffix + [0] * (max_suffix_length - len(suffix)) if len(suffix) < max_suffix_length else suffix[:max_suffix_length] for suffix in suffix_masks]
    padded_suffix_labels = [suffix + [-100] * (max_suffix_length - len(suffix)) if len(suffix) < max_suffix_length else suffix[:max_suffix_length] for suffix in suffix_labels]

    return padded_suffix_ids, padded_suffix_masks, padded_suffix_labels

# @dataclass
# class InstructionDataCollator:
#     """
#     Data collator that dynamically pads text fields and handles multiple audio segments
#     per sample (e.g. dialogue history utterances), each with its corresponding emotion label.
#     """
#     pad_id: int = 0
#     sampling_rate: int = 16000
#     extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

#     def __call__(self, samples: List[Dict]):
#         # gzy------------------------------
#         print("实际 batch 样本数:", len(samples))
#         # ------------------------------
#         start_ids = [sample["start_ids"] for sample in samples]
#         start_mask = [sample["start_mask"] for sample in samples]
#         start_labels = [sample["start_labels"] for sample in samples]
#         instruction_ids = [sample["instruction_ids"] for sample in samples]
#         instruction_mask = [sample["instruction_mask"] for sample in samples]
#         instruction_labels = [sample["instruction_labels"] for sample in samples]
#         audio_instruction_ids = [sample["audio_instruction_ids"] for sample in samples]
#         audio_instruction_mask = [sample["audio_instruction_mask"] for sample in samples]
#         audio_instruction_labels = [sample["audio_instruction_labels"] for sample in samples]
        
#         input_ids = [sample["input_ids"] for sample in samples]
#         input_mask = [sample["input_mask"] for sample in samples]
#         input_labels = [sample["input_labels"] for sample in samples]
        
#         suffix_ids = [sample["suffix_ids"] for sample in samples]
#         suffix_mask = [sample["suffix_mask"] for sample in samples]
#         suffix_labels = [sample["suffix_labels"] for sample in samples]
#         emotion_labels = [sample["emotion_label"] for sample in samples]
        
#         padded_input_ids, padded_input_masks, padded_input_labels = process_dialogues(input_ids, input_mask, input_labels, self.pad_id)
#         padded_suffix_id, padded_suffix_mask, padded_suffix_label = process_suffixes(suffix_ids, suffix_mask, suffix_labels, self.pad_id)
        
#         start_ids = collate_tokens(start_ids, self.pad_id)
#         start_mask = collate_tokens(start_mask, 0)
#         start_labels = collate_tokens(start_labels, -100)
#         instruction_ids = collate_tokens(instruction_ids, self.pad_id)
#         instruction_mask = collate_tokens(instruction_mask, 0)
#         instruction_labels = collate_tokens(instruction_labels, -100)
#         audio_instruction_ids = collate_tokens(audio_instruction_ids, self.pad_id)
#         audio_instruction_mask = collate_tokens(audio_instruction_mask, 0)
#         audio_instruction_labels = collate_tokens(audio_instruction_labels, -100)
#         input_ids = [collate_tokens(padded_input_id, self.pad_id) for padded_input_id in padded_input_ids]
#         input_mask = [collate_tokens(padded_input_mask, 0) for padded_input_mask in padded_input_masks]
#         input_labels = [collate_tokens(padded_input_label, -100) for padded_input_label in padded_input_labels]
#         suffix_ids = torch.unbind(collate_tokens(suffix_ids, self.pad_id),dim=0)
#         suffix_mask = torch.unbind(collate_tokens(suffix_mask, 0),dim=0)
#         suffix_labels = torch.unbind(collate_tokens(suffix_labels, -100),dim=0)
#         emotion_labels = torch.LongTensor(emotion_labels)

#         all_input_audio_paths = [sample["audio_paths"][:-1] for sample in samples]
#         max_input_audio_num = max([len(input_audio_paths) for input_audio_paths in all_input_audio_paths])
#         all_pad_input_audio_paths = [
#             input_audio_paths + [""] * (max_input_audio_num - len(input_audio_paths)) if len(input_audio_paths) < max_input_audio_num else input_audio_paths[:max_input_audio_num]
#         for input_audio_paths in all_input_audio_paths
#         ]
#         all_input_audio_waveforms = [[np.array([0]) if pad_input_audio_path == "" else get_waveform(pad_input_audio_path, 16000) for pad_input_audio_path in pad_input_audio_paths ] for pad_input_audio_paths in all_pad_input_audio_paths]

#         all_suffix_audio_path = [sample["audio_paths"][-1] for sample in samples]
#         all_suffix_audio_waveform = [get_waveform(suffix_audio_path, 16000) for suffix_audio_path in all_suffix_audio_path]

#         assert len(all_input_audio_waveforms) == len(all_suffix_audio_waveform), \
#             f"Expected input audio waveforms and suffix audio waveforms to have the same length, but got {len(all_input_audio_waveforms)} and {len(all_suffix_audio_waveform)}."
            
#         input_audio_features = []
#         input_audio_mask = []
#         suffix_audio_features = []
#         suffix_audio_mask = []

#         for i in range(len(all_input_audio_waveforms)):
#             input_audio = self.extractor(
#                 all_input_audio_waveforms[i],
#                 sampling_rate=self.sampling_rate, 
#                 return_attention_mask=True,
#                 return_tensors="pt"
#             )
#             suffix_audio = self.extractor(
#                 all_suffix_audio_waveform[i],
#                 sampling_rate=self.sampling_rate, 
#                 return_attention_mask=True,
#                 return_tensors="pt"
#             )
#             input_audio_feature = input_audio.input_features
#             input_audio_masks = input_audio.attention_mask
#             suffix_audio_feature = suffix_audio.input_features
#             suffix_audio_masks = suffix_audio.attention_mask
#             input_audio_features.append(input_audio_feature)
#             input_audio_mask.append(input_audio_masks)
#             suffix_audio_features.append(suffix_audio_feature)
#             suffix_audio_mask.append(suffix_audio_masks)
        
#         return {
#             "start_ids": start_ids,
#             "start_mask": start_mask,
#             "start_labels": start_labels,
#             "input_ids": input_ids,
#             "input_mask": input_mask,
#             "input_labels": input_labels,
#             "input_audio_features": input_audio_features,
#             "input_audio_mask": input_audio_mask,
#             "suffix_ids": suffix_ids,
#             "suffix_mask": suffix_mask,
#             "suffix_labels": suffix_labels,
#             "suffix_audio_features": suffix_audio_features,
#             "suffix_audio_mask": suffix_audio_mask,
#             "emotion_labels": emotion_labels,
#             "instruction_ids": instruction_ids,
#             "instruction_mask": instruction_mask,
#             "instruction_labels": instruction_labels,
#             "audio_instruction_ids": audio_instruction_ids,
#             "audio_instruction_mask": audio_instruction_mask,
#             "audio_instruction_labels": audio_instruction_labels,
#         }

@dataclass
class InstructionDataCollator:
    # 负责对多种文本字段做动态 padding、
    # 把多段音频（每条样本可能有多个 utterance 音频段）提取成特征并收集对应的 attention mask，
    # 同时把情感标签转为张量，最终返回一个可直接送入模型的 batch 字典
    """
    Data collator that dynamically pads text fields and handles multiple audio segments
    per sample (e.g. dialogue history utterances), each with its corresponding emotion label.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()
    
    # __call__ 的输入（samples） —— 期望的单条样本结构
    def __call__(self, samples: List[Dict]):
        
        start_ids = [sample["start_ids"] for sample in samples]
        start_mask = [sample["start_mask"] for sample in samples]
        start_labels = [sample["start_labels"] for sample in samples]
        instruction_ids = [sample["instruction_ids"] for sample in samples]
        instruction_mask = [sample["instruction_mask"] for sample in samples]
        instruction_labels = [sample["instruction_labels"] for sample in samples]
        audio_instruction_ids = [sample["audio_instruction_ids"] for sample in samples]
        audio_instruction_mask = [sample["audio_instruction_mask"] for sample in samples]
        audio_instruction_labels = [sample["audio_instruction_labels"] for sample in samples]
        
        # input_ids = [sample["input_ids"] for sample in samples]
        # input_mask = [sample["input_mask"] for sample in samples]
        # input_labels = [sample["input_labels"] for sample in samples]
        input_ids = [sum(sample["input_ids"], []) for sample in samples]
        input_mask = [sum(sample["input_mask"], []) for sample in samples]
        input_labels = [sum(sample["input_labels"], []) for sample in samples]
        suffix_ids = [sample["suffix_ids"] for sample in samples]
        suffix_mask = [sample["suffix_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]
        emotion_labels = [sample["emotion_label"] for sample in samples]
        
        # padded_input_ids, padded_input_masks, padded_input_labels = process_dialogues(input_ids, input_mask, input_labels, self.pad_id)
        # padded_suffix_id, padded_suffix_mask, padded_suffix_label = process_suffixes(suffix_ids, suffix_mask, suffix_labels, self.pad_id)
        
        start_ids = collate_tokens(start_ids, self.pad_id)
        start_mask = collate_tokens(start_mask, 0)
        start_labels = collate_tokens(start_labels, -100)
        instruction_ids = collate_tokens(instruction_ids, self.pad_id)
        instruction_mask = collate_tokens(instruction_mask, 0)
        instruction_labels = collate_tokens(instruction_labels, -100)
        audio_instruction_ids = collate_tokens(audio_instruction_ids, self.pad_id)
        audio_instruction_mask = collate_tokens(audio_instruction_mask, 0)
        audio_instruction_labels = collate_tokens(audio_instruction_labels, -100)
        input_ids = collate_tokens(input_ids, self.pad_id)
        input_mask = collate_tokens(input_mask, 0)
        input_labels = collate_tokens(input_labels, -100)
        suffix_ids = collate_tokens(suffix_ids, self.pad_id)
        suffix_mask = collate_tokens(suffix_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)
        # input_ids = [collate_tokens(padded_input_id, self.pad_id) for padded_input_id in padded_input_ids]
        # input_mask = [collate_tokens(padded_input_mask, 0) for padded_input_mask in padded_input_masks]
        # input_labels = [collate_tokens(padded_input_label, -100) for padded_input_label in padded_input_labels]
        # suffix_ids = torch.unbind(collate_tokens(suffix_ids, self.pad_id),dim=0)
        # suffix_mask = torch.unbind(collate_tokens(suffix_mask, 0),dim=0)
        # suffix_labels = torch.unbind(collate_tokens(suffix_labels, -100),dim=0)
        emotion_labels = torch.LongTensor(emotion_labels)

        # all_input_audio_paths = [sample["audio_paths"][:-1] for sample in samples]
        # # max_input_audio_num = max([len(input_audio_paths) for input_audio_paths in all_input_audio_paths])
        # # all_pad_input_audio_paths = [
        # #     input_audio_paths + [""] * (max_input_audio_num - len(input_audio_paths)) if len(input_audio_paths) < max_input_audio_num else input_audio_paths[:max_input_audio_num]
        # # for input_audio_paths in all_input_audio_paths
        # # ]
        # all_input_audio_waveforms = [[np.array([0]) if pad_input_audio_path == "" else get_waveform(pad_input_audio_path, 16000) for pad_input_audio_path in pad_input_audio_paths ] for pad_input_audio_paths in all_pad_input_audio_paths]


        # all_suffix_audio_path = [sample["audio_paths"][-1] for sample in samples]
        # all_suffix_audio_waveform = [get_waveform(suffix_audio_path, 16000) for suffix_audio_path in all_suffix_audio_path]

        # assert len(all_input_audio_waveforms) == len(all_suffix_audio_waveform), \
        #     f"Expected input audio waveforms and suffix audio waveforms to have the same length, but got {len(all_input_audio_waveforms)} and {len(all_suffix_audio_waveform)}."
        
        input_audio_features, input_audio_mask = [], []
        for sample in samples:
            input_audio_paths = sample["input_audio_paths"]
            audio_features, audio_mask = [], []
            for input_audio_path in input_audio_paths:
                audio_waveform = get_waveform(input_audio_path, output_sample_rate=self.sampling_rate)
                audio_feature = self.extractor(
                    audio_waveform, 
                    sampling_rate=self.sampling_rate,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                #NOTE   'input_features': tensor of shape (batch_size, feature_dim, time_frames),
                #       'attention_mask': tensor of shape (batch_size, time_frames)
                audio_feat = audio_feature.input_features
                audio_msk = audio_feature.attention_mask
                # tolist = speech_values.tolist()
                audio_features.append(audio_feat)
                audio_mask.append(audio_msk)
            input_audio_features.append(audio_features)
            input_audio_mask.append(audio_mask)            
        
        suffix_audio_features, suffix_audio_mask = [], []
        for sample in samples:
            suffix_audio_path = sample["suffix_audio_path"]
            audio_waveform = get_waveform(suffix_audio_path["audio_path"], output_sample_rate=self.sampling_rate)
            audio_feature = self.extractor(
                audio_waveform, 
                sampling_rate=self.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt"
            )
            audio_feat = audio_feature.input_features
            audio_msk = audio_feature.attention_mask
            suffix_audio_features.append(audio_feat)
            suffix_audio_mask.append(audio_msk)


        # for i in range(len(all_input_audio_waveforms)):
        #     input_audio = self.extractor(
        #         all_input_audio_waveforms[i],
        #         sampling_rate=self.sampling_rate, 
        #         return_attention_mask=True,
        #         return_tensors="pt"
        #     )
        #     suffix_audio = self.extractor(
        #         all_suffix_audio_waveform[i],
        #         sampling_rate=self.sampling_rate, 
        #         return_attention_mask=True,
        #         return_tensors="pt"
        #     )
        #     input_audio_feature = input_audio.input_features
        #     input_audio_masks = input_audio.attention_mask
        #     suffix_audio_feature = suffix_audio.input_features
        #     suffix_audio_masks = suffix_audio.attention_mask
        #     input_audio_features.append(input_audio_feature)
        #     input_audio_mask.append(input_audio_masks)
        #     suffix_audio_features.append(suffix_audio_feature)
        #     suffix_audio_mask.append(suffix_audio_masks)
        # return {
        #     "start_ids": start_ids,
        #     "start_mask": start_mask,
        #     "start_labels": start_labels,
        #     "instruction_ids": instruction_ids,
        #     "instruction_mask": instruction_mask,
        #     "instruction_labels": instruction_labels,
        #     "audio_instruction_ids": audio_instruction_ids,
        #     "audio_instruction_mask": audio_instruction_mask,
        #     "audio_instruction_labels": audio_instruction_labels,
        #     "input_ids": input_ids,
        #     "input_mask": input_mask,
        #     "input_labels": input_labels,
        #     "input_audio_features": input_audio_features,
        #     "input_audio_mask": input_audio_mask,
        #     "suffix_ids": suffix_ids,
        #     "suffix_mask": suffix_mask,
        #     "suffix_labels": suffix_labels,
        #     "suffix_audio_features": suffix_audio_features,
        #     "suffix_audio_mask": suffix_audio_mask,
        #     "emotion_labels": emotion_labels,
        # }
        features = {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "start_labels": start_labels,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "instruction_labels": instruction_labels,
            "audio_instruction_ids": audio_instruction_ids,
            "audio_instruction_mask": audio_instruction_mask,
            "audio_instruction_labels": audio_instruction_labels,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "input_labels": input_labels,
            "input_audio_features": input_audio_features,
            "input_audio_mask": input_audio_mask,
            "suffix_ids": suffix_ids,
            "suffix_mask": suffix_mask,
            "suffix_labels": suffix_labels,
            "suffix_audio_features": suffix_audio_features,
            "suffix_audio_mask": suffix_audio_mask,
            "emotion_labels": emotion_labels,
        }

        # 打印 key + shape（避免太大）
        # print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in features.items()})
        
        # print("111111111111111111111111111111111111111111111111111111111111")
        # print("input_audio_features type:", type(input_audio_features))
        # print("input_audio_features[0] type:", type(input_audio_features[0]))
        # print("input_audio_features[0][0].dtype:", input_audio_features[0][0].dtype)
        # print("input_audio_features[0][0].shape:", input_audio_features[0][0].shape)
        return features



def offline_process(
    data_root="",
    manifest_files="",
    lm_path="",
    instruction="",
    instruction_field="",
    audio_instruction="",
    audio_instruction_field="",
    input_field="",
    # audio_field="audio",
    output_field="",
    save_dir="",
    max_length=1024,
    min_duration=1.0,
    max_duration=60.0,
    num_proc=64,
    use_emotion=True,
):
    text_tokenizer = QWenTokenizer.from_pretrained(lm_path)

    dataset = load_instruction_dataset(
        data_root,
        manifest_files,
        text_tokenizer,
        instruction,
        instruction_field,
        audio_instruction,
        audio_instruction_field,
        input_field,
        # audio_field,
        output_field,
        max_length,
        min_duration,
        max_duration,
        num_proc,
        use_emotion,
    )
    print(len(dataset))
    for key in dataset[0].keys():
        if key != "input_audio_paths" and key != "suffix_audio_path" and key != "to_keep" and key != "emotion_label":
            print(key, len(dataset[0][key]))
        else:
            print(key, dataset[0][key])
    
    if save_dir:
        dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
