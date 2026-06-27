#!/usr/bin/env python3
"""Train Scikit-Learn baseline models (Logistic Regression & Random Forest) for comparison."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.config import (  # noqa: E402
    MODELS_DIR,
    OUTPUTS_DIR,
    get_processed_data_path,
    get_tfidf_vectorizer_path,
    get_model_path,
    MAX_VOCAB_SIZE,
    MIN_WORD_FREQ,
)
from cyberbullying.metrics import (  # noqa: E402
    classification_report_dict,
    compute_binary_metrics,
    compute_multiclass_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    save_metrics,
)
from cyberbullying.preprocessing import load_label_map, load_splits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train Scikit-Learn baseline models")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading data for task: {args.task}...")

    processed_path = get_processed_data_path(args.task)
    if not processed_path.exists():
        print(f"Processed data not found for task {args.task}. Please run preprocessing first.")
        sys.exit(1)

    df = pd.read_csv(processed_path)
    splits = load_splits(args.task)
    label_map = load_label_map(args.task)
    labels = label_map["labels"]

    train_subset = df.iloc[splits["train"]]
    test_subset = df.iloc[splits["test"]]

    X_train_raw = train_subset["clean_text"].tolist()
    y_train = train_subset["label"].tolist()

    X_test_raw = test_subset["clean_text"].tolist()
    y_test = test_subset["label"].tolist()

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=MAX_VOCAB_SIZE, min_df=MIN_WORD_FREQ)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print("Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("Evaluating models...")
    # Logistic Regression Evaluation & Plotting
    lr_out_dir = OUTPUTS_DIR / "logistic" / args.task
    lr_out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "binary":
        y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
        lr_metrics = compute_binary_metrics(y_test, y_prob_lr)
        y_pred_lr = (y_prob_lr >= 0.5).astype(int)
        lr_metrics["classification_report"] = classification_report_dict(y_test, y_pred_lr, labels)
        plot_confusion_matrix(y_test, y_pred_lr, labels, lr_out_dir / "confusion_matrix.png", "Logistic Regression Confusion Matrix")
        plot_roc_curve(y_test, y_prob_lr, lr_out_dir / "roc_curve.png")
        plot_pr_curve(y_test, y_prob_lr, lr_out_dir / "pr_curve.png")
    else:
        y_pred_lr = log_reg.predict(X_test)
        lr_metrics = compute_multiclass_metrics(y_test, y_pred_lr)
        lr_metrics["classification_report"] = classification_report_dict(y_test, y_pred_lr, labels)
        plot_confusion_matrix(y_test, y_pred_lr, labels, lr_out_dir / "confusion_matrix.png", "Logistic Regression Confusion Matrix")
    save_metrics(lr_metrics, lr_out_dir / "metrics.json")

    # Random Forest Evaluation & Plotting
    rf_out_dir = OUTPUTS_DIR / "rf" / args.task
    rf_out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "binary":
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        rf_metrics = compute_binary_metrics(y_test, y_prob_rf)
        y_pred_rf = (y_prob_rf >= 0.5).astype(int)
        rf_metrics["classification_report"] = classification_report_dict(y_test, y_pred_rf, labels)
        plot_confusion_matrix(y_test, y_pred_rf, labels, rf_out_dir / "confusion_matrix.png", "Random Forest Confusion Matrix")
        plot_roc_curve(y_test, y_prob_rf, rf_out_dir / "roc_curve.png")
        plot_pr_curve(y_test, y_prob_rf, rf_out_dir / "pr_curve.png")
    else:
        y_pred_rf = rf.predict(X_test)
        rf_metrics = compute_multiclass_metrics(y_test, y_pred_rf)
        rf_metrics["classification_report"] = classification_report_dict(y_test, y_pred_rf, labels)
        plot_confusion_matrix(y_test, y_pred_rf, labels, rf_out_dir / "confusion_matrix.png", "Random Forest Confusion Matrix")
    save_metrics(rf_metrics, rf_out_dir / "metrics.json")

    print(f"\nTest metrics ({args.task}) - Logistic Regression:")
    for key, value in lr_metrics.items():
        if key != "classification_report":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nTest metrics ({args.task}) - Random Forest:")
    for key, value in rf_metrics.items():
        if key != "classification_report":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, get_tfidf_vectorizer_path(args.task))
    joblib.dump(log_reg, get_model_path("logistic", args.task))
    joblib.dump(rf, get_model_path("rf", args.task))
    print(f"\nSaved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
