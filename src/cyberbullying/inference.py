from pathlib import Path

import joblib
import numpy as np
import torch

from cyberbullying.config import MAX_SEQ_LEN
from cyberbullying.model import BiLSTMClassifier
from cyberbullying.preprocessing import Vocab, clean_text, load_label_map


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[BiLSTMClassifier, Vocab, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = Vocab.load(Path(checkpoint["vocab_path"]))
    label_map = checkpoint.get("label_map")
    if label_map is None:
        task_str = checkpoint.get("task", "binary")
        label_map = load_label_map(task_str)
        
    task = label_map["task"]
    num_classes = 1 if task == "binary" else len(label_map["labels"])

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=num_classes,
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, label_map


def predict_text(text: str, model: BiLSTMClassifier, vocab: Vocab, label_map: dict, device: torch.device) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning")

    input_ids = torch.tensor([vocab.encode(cleaned, MAX_SEQ_LEN)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)

    task = label_map["task"]
    labels = label_map["labels"]

    if task == "binary":
        prob = torch.sigmoid(logits.squeeze()).item()
        pred_idx = int(prob >= 0.5)
        confidence = prob if pred_idx == 1 else 1.0 - prob
    else:
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "label": labels[pred_idx],
        "confidence": round(confidence, 4),
        "task": task,
    }


def load_sklearn_baseline(model_path: Path, vectorizer_path: Path, task: str) -> tuple[any, any, dict]:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_map = load_label_map(task)
    return model, vectorizer, label_map


def predict_text_sklearn(text: str, model: any, vectorizer: any, label_map: dict) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Text is empty after cleaning")

    X = vectorizer.transform([cleaned])
    task = label_map["task"]
    labels = label_map["labels"]

    if task == "binary":
        prob = model.predict_proba(X)[0, 1]
        pred_idx = int(prob >= 0.5)
        confidence = prob if pred_idx == 1 else 1.0 - prob
    else:
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    return {
        "label": labels[pred_idx],
        "confidence": round(float(confidence), 4),
        "task": task,
    }

