#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out test set."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import MAX_SEQ_LEN, OUTPUTS_DIR, get_vocab_path, get_model_path, get_processed_data_path  # noqa: E402
from cyberbullying.dataset import TweetDataset, collate_batch  # noqa: E402
from cyberbullying.inference import load_checkpoint  # noqa: E402
from cyberbullying.metrics import (  # noqa: E402
    classification_report_dict,
    compute_binary_metrics,
    compute_multiclass_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics,
)
from cyberbullying.model import BiLSTMClassifier  # noqa: E402
from cyberbullying.preprocessing import Vocab, load_label_map, load_splits  # noqa: E402
from cyberbullying.train_utils import evaluate_model, get_loss_fn  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cyberbullying classifier")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override default model load path")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_path = get_processed_data_path(args.task)
    if not processed_path.exists():
        print(f"Processed data not found for task {args.task}. Please run preprocessing first.")
        sys.exit(1)

    if args.checkpoint is None:
        args.checkpoint = get_model_path("bilstm", args.task)

    df = pd.read_csv(processed_path)
    splits = load_splits(args.task)
    vocab = Vocab.load(get_vocab_path(args.task))
    label_map = load_label_map(args.task)
    labels = label_map["labels"]

    test_subset = df.iloc[splits["test"]]
    test_loader = DataLoader(
        TweetDataset(
            test_subset["clean_text"].tolist(),
            test_subset["label"].tolist(),
            vocab,
            MAX_SEQ_LEN,
            task=args.task,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    num_classes = 1 if args.task == "binary" else len(labels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=num_classes,
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = get_loss_fn(args.task)
    _, y_true, y_scores = evaluate_model(model, test_loader, criterion, device, args.task)

    out_dir = OUTPUTS_DIR / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "binary":
        metrics = compute_binary_metrics(y_true, y_scores)
        y_pred = (y_scores >= 0.5).astype(int)
        metrics["classification_report"] = classification_report_dict(y_true, y_pred, labels)
        plot_confusion_matrix(y_true, y_pred, labels, out_dir / "confusion_matrix.png", "Binary Confusion Matrix")
        if len(np.unique(y_true)) > 1:
            plot_roc_curve(y_true, y_scores, out_dir / "roc_curve.png")
            plot_pr_curve(y_true, y_scores, out_dir / "pr_curve.png")
    else:
        y_pred = y_scores.astype(int)
        metrics = compute_multiclass_metrics(y_true, y_pred)
        metrics["classification_report"] = classification_report_dict(y_true, y_pred, labels)
        plot_confusion_matrix(y_true, y_pred, labels, out_dir / "confusion_matrix.png", "Multi-class Confusion Matrix")

    save_metrics(metrics, out_dir / "metrics.json")

    print(f"Test metrics ({args.task}):")
    for key, value in metrics.items():
        if key != "classification_report":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"Saved plots and metrics to {out_dir}")


if __name__ == "__main__":
    main()
