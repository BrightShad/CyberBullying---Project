#!/usr/bin/env python3
"""Train BiLSTM classifier on preprocessed data."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import (  # noqa: E402
    BATCH_SIZE,
    get_model_path,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    EMBED_DIM,
    EPOCHS,
    GLOVE_PATH,
    HIDDEN_DIM,
    LEARNING_RATE,
    MAX_SEQ_LEN,
    get_processed_data_path,
    RANDOM_SEED,
    get_splits_path,
    get_vocab_path,
)
from cyberbullying.dataset import TweetDataset, collate_batch  # noqa: E402
from cyberbullying.model import BiLSTMClassifier  # noqa: E402
from cyberbullying.preprocessing import Vocab, load_label_map, load_splits, load_glove_embeddings  # noqa: E402
from cyberbullying.train_utils import TrainConfig, compute_pos_weight, compute_class_weights, train_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM cyberbullying classifier")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--output", type=Path, default=None, help="Override default model save path")
    return parser.parse_args()


def make_loader(df, indices, vocab, batch_size, shuffle, task):
    subset = df.iloc[indices]
    dataset = TweetDataset(
        subset["clean_text"].tolist(),
        subset["label"].tolist(),
        vocab,
        MAX_SEQ_LEN,
        task=task,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.output is None:
        args.output = get_model_path("bilstm", args.task)

    label_map = load_label_map(args.task)
    splits = load_splits(args.task)
    vocab_path = get_vocab_path(args.task)
    
    if not get_processed_data_path(args.task).exists():
        print(f"Processed data not found for task {args.task}. Please run preprocessing first.")
        sys.exit(1)

    df = pd.read_csv(get_processed_data_path(args.task))
    vocab = Vocab.load(vocab_path)

    train_loader = make_loader(df, splits["train"], vocab, args.batch_size, shuffle=True, task=args.task)
    val_loader = make_loader(df, splits["val"], vocab, args.batch_size, shuffle=False, task=args.task)

    num_classes = 1 if args.task == "binary" else len(label_map["labels"])
    
    pretrained_embeddings = None
    if GLOVE_PATH.exists():
        pretrained_embeddings = load_glove_embeddings(vocab, GLOVE_PATH, EMBED_DIM)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=DROPOUT,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    pos_weight = None
    train_labels = df.iloc[splits["train"]]["label"].tolist()
    if args.task == "binary":
        pos_weight = compute_pos_weight(train_labels)
        print(f"pos_weight for BCE loss: {pos_weight:.3f}")
    else:
        pos_weight = compute_class_weights(train_labels, num_classes)
        print(f"class_weights for CE loss: {pos_weight}")

    config = TrainConfig(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=device,
    )

    history, best_state = train_model(model, train_loader, val_loader, config, pos_weight=pos_weight)

    if best_state is None:
        best_state = model.state_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "vocab_path": str(vocab_path),
        "label_map": label_map,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "task": args.task,
    }
    torch.save(checkpoint, args.output)
    print(f"Saved checkpoint to {args.output}")

    from cyberbullying.config import OUTPUTS_DIR
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUTS_DIR / f"training_history_{args.task}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
