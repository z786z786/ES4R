import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import GenerationConfig, WhisperFeatureExtractor

from src.instruction_dataset import get_waveform
from src.modeling_blsp2 import Blsp2Model
from src.qwen_generation_utils import decode_tokens, get_stop_words_ids
from src.tokenization_qwen import QWenTokenizer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def collate_tokens(values: List[List[int]], pad_id: int):
    size = max(len(v) for v in values)
    batch_size = len(values)
    result = torch.LongTensor(batch_size, size).fill_(pad_id)
    for i, value in enumerate(values):
        result[i, -len(value):] = torch.LongTensor(value)
    return result


def load_examples(path: str) -> List[Dict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        if file_path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


@dataclass
class DataCollator:
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        start_ids = collate_tokens([sample["start_ids"] for sample in samples], self.pad_id)
        start_mask = collate_tokens([sample["start_mask"] for sample in samples], 0)
        input_ids = collate_tokens([sum(sample["input_ids"], []) for sample in samples], self.pad_id)
        input_mask = collate_tokens([sum(sample["input_mask"], []) for sample in samples], 0)
        instruction_ids = collate_tokens([sample["instruction_ids"] for sample in samples], self.pad_id)
        instruction_mask = collate_tokens([sample["instruction_mask"] for sample in samples], 0)
        suffix_ids = collate_tokens([sample["suffix_ids"] for sample in samples], self.pad_id)
        suffix_mask = collate_tokens([sample["suffix_mask"] for sample in samples], 0)
        references = [sample["reference"] for sample in samples]

        input_audio_features, input_audio_masks = [], []
        for sample in samples:
            turn_features, turn_masks = [], []
            for audio_path in sample["input_audio_paths"]:
                waveform = get_waveform(audio_path, output_sample_rate=self.sampling_rate)
                features = self.extractor(
                    waveform,
                    sampling_rate=self.sampling_rate,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                turn_features.append(features.input_features)
                turn_masks.append(features.attention_mask)
            input_audio_features.append(turn_features)
            input_audio_masks.append(turn_masks)

        return {
            "start_ids": start_ids,
            "start_mask": start_mask,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "suffix_ids": suffix_ids,
            "suffix_mask": suffix_mask,
            "input_audio_features": input_audio_features,
            "input_audio_mask": input_audio_masks,
            "response": references,
        }


def main():
    parser = argparse.ArgumentParser(description="Batch inference for CAES checkpoints.")
    parser.add_argument("--input_file", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--output_file", required=True, help="Output JSONL file.")
    parser.add_argument("--model_path", default=None, help="Checkpoint path.")
    parser.add_argument("--blsp_model", default=None, help="Alias of --model_path.")
    parser.add_argument("--instruction", default="", help="Instruction prompt.")
    parser.add_argument("--audio_field", default="", help="Reserved for compatibility.")
    parser.add_argument("--input_field", default="", help="Dialogue history field name.")
    parser.add_argument("--output_field", default="", help="Reference response field name.")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--use_emotion", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--check_audio", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()

    model_path = args.blsp_model or args.model_path
    if not model_path:
        raise ValueError("One of --blsp_model or --model_path must be provided.")

    tokenizer = QWenTokenizer.from_pretrained(model_path)
    extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    stop_words_ids = get_stop_words_ids(generation_config.chat_format, tokenizer)

    dataset = Dataset.from_list(load_examples(args.input_file))

    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def tokenize_segment(role="", content=""):
        tokens = []
        if role:
            tokens += tokenizer.encode(role, allowed_special=set()) + nl_tokens
        if content:
            tokens += tokenizer.encode(content, allowed_special=set())
        return tokens

    def process_dataset(batch):
        min_duration = 1.0
        max_duration = 30.0

        system_prompt = (
            "You are a helpful assistant. Your response should fulfill requests with empathy toward user's emotion tone."
            if args.use_emotion
            else "You are a helpful assistant."
        )

        input_history = batch.get(args.input_field, "")
        response = batch.get(args.output_field, "")

        start_ids = im_start_tokens + tokenize_segment(role="system", content=system_prompt)
        start_mask = [1] * len(start_ids)
        instruction_ids = tokenize_segment(content=args.instruction)
        instruction_mask = [1] * len(instruction_ids)

        input_ids, input_mask, input_audio_paths = [], [], []
        to_keep = True

        for turn in input_history:
            role = "user" if turn["role"] == "speaker" else "assistant"
            utterance_tokens = im_end_tokens + nl_tokens + im_start_tokens + tokenize_segment(role=role, content=turn["utterance"])
            input_ids.append(utterance_tokens)
            input_mask.append([1] * len(utterance_tokens))

            if args.check_audio:
                audio_path = turn.get("audio_path", "")
                if not audio_path:
                    to_keep = False
                    continue
                try:
                    waveform = get_waveform(audio_path)
                    duration = waveform.shape[0] / 16000.0
                    if duration < min_duration or duration > max_duration:
                        to_keep = False
                    else:
                        input_audio_paths.append(audio_path)
                except Exception:
                    to_keep = False

        response_role = "user" if response["role"] == "speaker" else "assistant"
        suffix_ids = im_end_tokens + nl_tokens + im_start_tokens + tokenize_segment(role=response_role)
        suffix_mask = [1] * len(suffix_ids)

        batch["start_ids"] = start_ids
        batch["start_mask"] = start_mask
        batch["input_ids"] = input_ids
        batch["input_mask"] = input_mask
        batch["instruction_ids"] = instruction_ids
        batch["instruction_mask"] = instruction_mask
        batch["suffix_ids"] = suffix_ids
        batch["suffix_mask"] = suffix_mask
        batch["reference"] = response["utterance"]
        batch["input_audio_paths"] = input_audio_paths
        batch["to_keep"] = to_keep
        return batch

    dataset = dataset.map(
        process_dataset,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        num_proc=1,
    )
    dataset = dataset.filter(lambda flag: flag, input_columns=["to_keep"])

    model = Blsp2Model.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    data_collator = DataCollator(generation_config.pad_token_id, extractor.sampling_rate, extractor)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        collate_fn=data_collator,
        batch_size=args.batch_size,
    )

    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_beams": 1,
            "num_return_sequences": 1,
            "bos_token_id": nl_tokens[0],
        }
    )

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for batch in tqdm(dataloader):
            input_audio_features = [[tensor.cuda() for tensor in sample] for sample in batch["input_audio_features"]]
            input_audio_mask = [[tensor.cuda() for tensor in sample] for sample in batch["input_audio_mask"]]

            outputs = model.generate(
                start_ids=batch["start_ids"].cuda(),
                start_mask=batch["start_mask"].cuda(),
                input_ids=batch["input_ids"].cuda(),
                input_mask=batch["input_mask"].cuda(),
                instruction_ids=batch["instruction_ids"].cuda(),
                instruction_mask=batch["instruction_mask"].cuda(),
                suffix_ids=batch["suffix_ids"].cuda(),
                suffix_mask=batch["suffix_mask"].cuda(),
                input_audio_features=input_audio_features,
                input_audio_mask=input_audio_mask,
                generation_config=generation_config,
                stop_words_ids=stop_words_ids,
            )

            decoded = [
                decode_tokens(
                    output,
                    tokenizer,
                    raw_text_len=0,
                    context_length=0,
                    chat_format=generation_config.chat_format,
                    verbose=False,
                    errors="replace",
                )
                for output in outputs
            ]

            for reference, response in zip(batch["response"], decoded):
                fout.write(json.dumps({"response": response, "reference": reference}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
