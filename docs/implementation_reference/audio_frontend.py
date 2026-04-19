"""
Reference implementations for the audio frontend used in this project.

Original code:
- src/instruction_dataset.py
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import io
import mmap

import numpy as np
import torch


def mmap_read(path: str, offset: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            return mapped[offset : offset + length]


def read_from_stored_zip(zip_path: str, offset: int, length: int) -> bytes:
    return mmap_read(zip_path, offset, length)


def is_sf_audio_data(data: bytes) -> bool:
    is_wav = data[0] == 82 and data[1] == 73 and data[2] == 70
    is_flac = data[0] == 102 and data[1] == 76 and data[2] == 97
    is_ogg = data[0] == 79 and data[1] == 103 and data[2] == 103
    return is_wav or is_flac or is_ogg


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """
    Convert waveform by:
    - optional volume normalization
    - resampling
    - channel conversion
    """
    import torchaudio.sox_effects as ta_sox

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])

    if not effects:
        return waveform, sample_rate

    is_np_input = isinstance(waveform, np.ndarray)
    tensor_waveform = torch.from_numpy(waveform) if is_np_input else waveform
    converted, converted_sample_rate = ta_sox.apply_effects_tensor(
        tensor_waveform, sample_rate, effects
    )
    if is_np_input:
        converted = converted.numpy()
    return converted, converted_sample_rate


def get_waveform(
    path_or_fp: str,
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = False,
    output_sample_rate: int = 16000,
):
    """
    Project implementation:
    - parse path or path:start:frames
    - read audio from common formats or packed zip
    - transpose to (channels, time)
    - convert to mono / target sample rate
    """
    import soundfile as sf

    meta = path_or_fp.split(":")
    if len(meta) == 3:
        path_or_fp = meta[0]
        start = int(meta[1])
        frames = int(meta[2])

    ext = Path(path_or_fp).suffix
    if ext in [".wav", ".flac", ".ogg", ".mp3", ".opus"]:
        waveform, sample_rate = sf.read(
            path_or_fp,
            dtype="float32",
            always_2d=True,
            frames=frames,
            start=start,
        )
    elif ext == ".zip":
        data = read_from_stored_zip(path_or_fp, start, frames)
        assert is_sf_audio_data(data)
        waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    else:
        raise ValueError(f"Unsupported audio format: {ext}")

    waveform = waveform.T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )
    if not normalization:
        waveform *= 2 ** 15
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform


WHISPER_FEATURE_EXTRACTOR_USAGE = """
audio_feature = extractor(
    audio_waveform,
    sampling_rate=16000,
    return_attention_mask=True,
    return_tensors="pt"
)

input_features = audio_feature.input_features   # shape: (1, 80, T)
attention_mask = audio_feature.attention_mask   # shape: (1, T)
"""
