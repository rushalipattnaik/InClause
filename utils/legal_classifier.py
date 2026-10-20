"""Local Legal-BERT risk classifier (Safe, Risky, Suspicious).

The model is fine-tuned once with scripts/train_legal_bert.py and then loaded from disk, so nothing is sent to an
outside service at inference time.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

LABELS = ["Safe", "Risky", "Suspicious"]
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = "models/legal-bert-risk"
CONFIG_NAME = "inclause_config.json"
DEFAULT_MAX_LENGTH = 256
TRAIN_COMMANDS = "python -m scripts.prepare_training_data  then  python -m scripts.train_legal_bert"


class ModelNotTrainedError(RuntimeError):
    """Raised when the fine-tuned model directory does not exist yet."""


def default_model_dir() -> Path:
    path = Path(os.environ.get("LEGAL_BERT_DIR", DEFAULT_MODEL_DIR))
    return path if path.is_absolute() else ROOT / path


def model_available(model_dir: str | os.PathLike | None = None) -> bool:
    return (Path(model_dir) if model_dir else default_model_dir()).joinpath("config.json").exists()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def column_order(id2label: dict) -> list[int]:
    """Indices that reorder the model's output columns into LABELS order, tolerating generic LABEL_n names."""
    names = {int(i): str(name) for i, name in id2label.items()}
    if not all(label in names.values() for label in LABELS):
        return list(range(len(LABELS)))
    by_name = {name: i for i, name in names.items()}
    return [by_name[label] for label in LABELS]


class LegalBertRiskClassifier:
    """Batched, lazily loaded, thread-safe wrapper around a fine-tuned sequence classifier."""

    def __init__(self, model_dir: str | os.PathLike | None = None, device: str = "cpu",
                 batch_size: int = 16, max_length: int | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.RLock()
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._order = list(range(len(LABELS)))

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not model_available(self.model_dir):
                raise ModelNotTrainedError(
                    f"No trained model found in {self.model_dir}. Train it first: {TRAIN_COMMANDS}")
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            config_path = self.model_dir / CONFIG_NAME
            if self.max_length is None:
                saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
                self.max_length = int(saved.get("max_length", DEFAULT_MAX_LENGTH))
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(self.device).eval()
            self._order = column_order(self._model.config.id2label)
            self._loaded = True

    def _forward(self, texts: list[str]) -> np.ndarray:
        """Return raw logits, shape (len(texts), 3), for one batch."""
        import torch

        encoded = self._tokenizer(texts, truncation=True, max_length=self.max_length, padding=True,
                                  return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self._model(**encoded).logits.detach().cpu().numpy()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Class probabilities, shape (n, 3), columns in the order Safe, Risky, Suspicious."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, len(LABELS)))
        with self._lock:
            self._ensure_loaded()
            chunks = [softmax(self._forward(texts[i:i + self.batch_size])[:, self._order])
                      for i in range(0, len(texts), self.batch_size)]
        return np.vstack(chunks)

    def predict(self, texts: list[str]) -> list[str]:
        return [LABELS[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def predict_with_confidence(self, texts: list[str]) -> list[tuple[str, float]]:
        probs = self.predict_proba(texts)
        return [(LABELS[int(row.argmax())], float(row.max())) for row in probs]
