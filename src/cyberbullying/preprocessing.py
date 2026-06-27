import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from cyberbullying.config import (
    ARTIFACTS_DIR,
    get_label_map_path,
    MAX_SEQ_LEN,
    MAX_VOCAB_SIZE,
    MIN_WORD_FREQ,
    MULTICLASS_LABELS,
    PAD_TOKEN,
    get_processed_data_path,
    RANDOM_SEED,
    get_splits_path,
    TEST_RATIO,
    TRAIN_RATIO,
    UNK_TOKEN,
    VAL_RATIO,
    get_vocab_path,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9#_\s]+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, keep hashtags as signal."""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


def to_binary_label(label: str) -> int:
    return 0 if label == "not_cyberbullying" else 1


def to_multiclass_label(label: str) -> int:
    return MULTICLASS_LABELS.index(label)


def build_label_map(task: str) -> dict:
    if task == "binary":
        return {"task": task, "labels": ["not_cyberbullying", "cyberbullying"]}
    if task == "multiclass":
        return {"task": task, "labels": MULTICLASS_LABELS}
    raise ValueError(f"Unknown task: {task}")


class Vocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, texts: list[str], min_freq: int = MIN_WORD_FREQ, max_size: int = MAX_VOCAB_SIZE):
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        itos = [PAD_TOKEN, UNK_TOKEN]
        for word, freq in counter.most_common():
            if freq < min_freq:
                continue
            if len(itos) >= max_size:
                break
            itos.append(word)

        stoi = {word: idx for idx, word in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        tokens = tokenize(text)
        ids = [self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.stoi[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        itos = data["itos"]
        stoi = {word: idx for idx, word in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)


def prepare_dataframe(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = df.copy()
    df["tweet_text"] = df["tweet_text"].astype(str)
    df["clean_text"] = df["tweet_text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    if task == "binary":
        df["label"] = df["cyberbullying_type"].map(to_binary_label)
    elif task == "multiclass":
        df["label"] = df["cyberbullying_type"].map(to_multiclass_label)
    else:
        raise ValueError(f"Unknown task: {task}")

    return df


def stratified_splits(labels: list[int], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    indices = list(range(len(labels)))
    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_y,
    )
    return train_idx, val_idx, test_idx


def run_preprocessing(df: pd.DataFrame, task: str) -> dict:
    """Clean data, split, build vocab from train set, save artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    processed = prepare_dataframe(df, task)
    train_idx, val_idx, test_idx = stratified_splits(
        processed["label"].tolist(),
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
    )

    train_texts = processed.iloc[train_idx]["clean_text"].tolist()
    vocab = Vocab.build(train_texts)

    label_map = build_label_map(task)
    processed.to_csv(get_processed_data_path(task), index=False)

    with open(get_splits_path(task), "wb") as f:
        pickle.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

    vocab.save(get_vocab_path(task))
    with open(get_label_map_path(task), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    return {
        "num_samples": len(processed),
        "vocab_size": len(vocab),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }


def load_splits(task: str) -> dict[str, list[int]]:
    with open(get_splits_path(task), "rb") as f:
        return pickle.load(f)


def load_label_map(task: str) -> dict:
    with open(get_label_map_path(task), encoding="utf-8") as f:
        return json.load(f)


def load_glove_embeddings(vocab: Vocab, glove_path: Path, embed_dim: int) -> torch.Tensor:
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab.stoi:
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs

    embedding_matrix = torch.zeros((len(vocab), embed_dim))
    hit = 0
    for word, i in vocab.stoi.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = torch.tensor(embedding_vector)
            hit += 1
        else:
            embedding_matrix[i] = torch.randn(embed_dim) * 0.1

    print(f"Found {hit}/{len(vocab)} words in GloVe.")
    return embedding_matrix
