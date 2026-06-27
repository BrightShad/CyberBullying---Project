#!/usr/bin/env python3
"""Preprocess raw CSV into train/val/test splits and vocabulary."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import DATA_PATH  # noqa: E402
from cyberbullying.preprocessing import run_preprocessing  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess cyberbullying tweet dataset")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to input CSV")
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass"],
        default="binary",
        help="Classification task",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)
    summary = run_preprocessing(df, args.task)

    print(f"Preprocessing complete ({args.task})")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
