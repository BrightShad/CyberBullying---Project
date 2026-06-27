import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    task: str
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    device: torch.device


def get_loss_fn(task: str, weight: float | torch.Tensor | None = None) -> nn.Module:
    if task == "binary":
        if weight is not None:
            w_val = weight if isinstance(weight, float) else weight.item()
            w = torch.tensor([w_val], dtype=torch.float32)
        else:
            w = None
        return nn.BCEWithLogitsLoss(pos_weight=w)
    return nn.CrossEntropyLoss(weight=weight if isinstance(weight, torch.Tensor) else None)


def compute_pos_weight(labels: list[int]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        return 1.0
    return negatives / positives

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        if task == "binary":
            loss = criterion(logits.squeeze(-1), labels)
        else:
            loss = criterion(logits, labels.long())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels: list[float | int] = []
    all_scores: list[float | int] = []

    for inputs, labels in tqdm(loader, desc="Eval", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)

        if task == "binary":
            loss = criterion(logits.squeeze(-1), labels)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_scores.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
        else:
            loss = criterion(logits, labels.long())
            preds = torch.argmax(logits, dim=1)
            all_scores.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.array(all_labels), np.array(all_scores)


def score_for_early_stopping(task: str, y_true: np.ndarray, y_scores: np.ndarray) -> float:
    if task == "binary":
        from cyberbullying.metrics import compute_binary_metrics

        return compute_binary_metrics(y_true, y_scores)["f1"]

    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_scores, average="macro", zero_division=0))


class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False
        self.best_state: dict | None = None

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    pos_weight: float | torch.Tensor | None = None,
) -> tuple[dict, dict | None]:
    criterion = get_loss_fn(config.task, pos_weight)
    if pos_weight is not None:
        if hasattr(criterion, 'pos_weight') and criterion.pos_weight is not None:
            criterion.pos_weight = criterion.pos_weight.to(config.device)
        elif hasattr(criterion, 'weight') and criterion.weight is not None:
            criterion.weight = criterion.weight.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    early_stopping = EarlyStopping(patience=config.patience)

    history = {"train_loss": [], "val_loss": [], "val_score": []}

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device, config.task)
        val_loss, y_true, y_scores = evaluate_model(model, val_loader, criterion, config.device, config.task)
        val_score = score_for_early_stopping(config.task, y_true, y_scores)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_score={val_score:.4f}"
        )

        early_stopping.step(val_score, model)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    return history, early_stopping.best_state
