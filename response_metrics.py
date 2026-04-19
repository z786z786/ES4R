import argparse
import json
from pathlib import Path

import pandas as pd
from bert_score import score
from evaluate import load
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from tqdm.auto import tqdm


def calc_distinct_n(n, candidates):
    seen = {}
    total = 0
    tokenized_candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in tokenized_candidates:
        for i in range(len(sentence) - n + 1):
            seen[tuple(sentence[i : i + n])] = 1
            total += 1
    score_value = len(seen) / (total + 1e-16)
    return {f"distinct_{n}": score_value}


def calc_distinct(predictions, references):
    results = {}
    for n in range(1, 3):
        results.update(calc_distinct_n(n, predictions))
    return results


def calculate_rouge_score(predictions, references):
    rouge = load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def calculate_meteor_score(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(meteor_score([word_tokenize(ref)], word_tokenize(pred)))
    return {"meteor-score": sum(scores) / len(scores)}


def calculate_bert_score(predictions, references):
    precision, recall, f1 = score(predictions, references, model_type="roberta-large", lang="en", verbose=False)
    return {
        "bert-score-precision": precision.mean().item(),
        "bert-score-recall": recall.mean().item(),
        "bert-score-f1": f1.mean().item(),
    }


def calculate_corpus_bleu_score(predictions, references):
    tokenized_predictions = [word_tokenize(pred) for pred in predictions]
    tokenized_references = [[word_tokenize(ref)] for ref in references]
    smooth = SmoothingFunction()
    return {
        "BLEU-1": corpus_bleu(tokenized_references, tokenized_predictions, weights=(1, 0, 0, 0), smoothing_function=smooth.method1),
        "BLEU-2": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1),
        "BLEU-3": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1),
        "BLEU-4": corpus_bleu(tokenized_references, tokenized_predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1),
    }


def evaluate_predictions(predictions, references):
    results = {}
    tasks = [
        (calculate_rouge_score, "ROUGE"),
        (calculate_bert_score, "BERTScore"),
        (calculate_corpus_bleu_score, "BLEU"),
        (calculate_meteor_score, "METEOR"),
        (calc_distinct, "Distinct"),
    ]
    for func, label in tqdm(tasks, desc="Evaluating metrics"):
        tqdm.write(f"Calculating {label}")
        results.update(func(predictions=predictions, references=references))
    return results


def load_pairs(path: str, prediction_field: str, reference_field: str):
    predictions = []
    references = []
    empty_predictions = 0

    with open(path, "r", encoding="utf-8") as handle:
        if Path(path).suffix == ".jsonl":
            records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = json.load(handle)

    for record in records:
        reference = record.get(reference_field, "").strip()
        prediction = (record.get(prediction_field) or "").strip()
        if not prediction:
            empty_predictions += 1
            continue
        predictions.append(prediction)
        references.append(reference)

    return predictions, references, empty_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses.")
    parser.add_argument("--data_path", required=True, help="Input JSON or JSONL file.")
    parser.add_argument("--prediction_field", default="response")
    parser.add_argument("--reference_field", default="reference")
    parser.add_argument("--output_csv", default="")
    args = parser.parse_args()

    predictions, references, empty_predictions = load_pairs(
        args.data_path,
        args.prediction_field,
        args.reference_field,
    )
    results = evaluate_predictions(predictions, references)
    results["empty_prediction_count"] = empty_predictions

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results]).to_csv(output_path, index=False)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
