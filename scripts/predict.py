#!/usr/bin/env python3
"""CLI inference for a single tweet."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import TFIDF_VECTORIZER, get_model_path  # noqa: E402
from cyberbullying.inference import load_checkpoint, load_sklearn_baseline, predict_text, predict_text_sklearn  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Predict cyberbullying label for tweet text")
    parser.add_argument("text", type=str, help="Tweet text to classify")
    parser.add_argument("--model", choices=["bilstm", "logistic", "rf"], default="bilstm", help="Model type to use")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary", help="Task type to use")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = get_model_path(args.model, args.task)
    
    if args.model == "bilstm":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}. Run scripts/train.py first.")
        model, vocab, label_map = load_checkpoint(model_path, device)
        result = predict_text(args.text, model, vocab, label_map, device)
    else:
        if not model_path.exists() or not TFIDF_VECTORIZER.exists():
            raise FileNotFoundError(f"Baseline models not found. Run scripts/train_baseline.py first.")
        model, vectorizer, label_map = load_sklearn_baseline(model_path, TFIDF_VECTORIZER)
        result = predict_text_sklearn(args.text, model, vectorizer, label_map)

    print(f"model: {args.model}")
    print(f"label: {result['label']}")
    print(f"confidence: {result['confidence']}")
    print(f"task: {result['task']}")


if __name__ == "__main__":
    main()
