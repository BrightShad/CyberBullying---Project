import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cyberbullying.model import BiLSTMClassifier
from cyberbullying.preprocessing import Vocab, clean_text, tokenize


class TestPreprocessing:
    def test_clean_text_removes_urls(self):
        text = "check this out https://example.com bad"
        assert "https" not in clean_text(text)

    def test_clean_text_removes_mentions(self):
        text = "@user hello there"
        assert "@" not in clean_text(text)

    def test_clean_text_keeps_hashtags(self):
        text = "#bullying is bad"
        assert "#bullying" in clean_text(text)

    def test_tokenize_splits_words(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_vocab_encode_unknown_word(self):
        vocab = Vocab.build(["hello world hello"], min_freq=1, max_size=100)
        encoded = vocab.encode("hello xyz")
        unk_id = vocab.stoi["<unk>"]
        assert unk_id in encoded

    def test_vocab_padding(self):
        vocab = Vocab.build(["hello world"], min_freq=1, max_size=100)
        encoded = vocab.encode("hello", max_len=5)
        assert len(encoded) == 5
        assert encoded.count(vocab.stoi["<pad>"]) == 4


class TestModel:
    def test_binary_forward_shape(self):
        model = BiLSTMClassifier(vocab_size=100, embed_dim=32, hidden_dim=16, num_classes=1)
        x = torch.randint(1, 50, (4, 10))
        out = model(x)
        assert out.shape == (4, 1)

    def test_multiclass_forward_shape(self):
        model = BiLSTMClassifier(vocab_size=100, embed_dim=32, hidden_dim=16, num_classes=6)
        x = torch.randint(1, 50, (4, 10))
        out = model(x)
        assert out.shape == (4, 6)

    def test_overfit_single_batch(self):
        torch.manual_seed(42)
        model = BiLSTMClassifier(vocab_size=50, embed_dim=32, hidden_dim=16, num_classes=1)
        x = torch.randint(1, 40, (8, 12))
        y = torch.randint(0, 2, (8,)).float()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for _ in range(30):
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(model(x).squeeze(-1)) >= 0.5).float()
        assert preds.equal(y)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import api.main as api_module

    vocab = Vocab.build(["you are ugly", "have a nice day"], min_freq=1, max_size=100)
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=32, hidden_dim=16, num_classes=1)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_path": str(vocab_path),
            "label_map": {"task": "binary", "labels": ["not_cyberbullying", "cyberbullying"]},
            "embed_dim": 32,
            "hidden_dim": 16,
            "dropout": 0.3,
        },
        checkpoint_path,
    )

    monkeypatch.setattr(api_module, "DEFAULT_CHECKPOINT", checkpoint_path)
    with TestClient(api_module.app) as client:
        yield client


class TestAPI:
    def test_health(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_valid_text(self, api_client):
        response = api_client.post("/predict", json={"text": "you are worthless"})
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ["not_cyberbullying", "cyberbullying"]
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text_rejected(self, api_client):
        response = api_client.post("/predict", json={"text": "   "})
        assert response.status_code == 422
