import csv
import os
import threading

import numpy as np

from utils.classifier import WEAK_REASONS

LABELS = ["Safe", "Risky", "Suspicious"]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = [os.path.join(BASE_DIR, "eval", "clauses.csv"), os.path.join(BASE_DIR, "eval", "clauses_real.csv")]
FLAG_MIN = float(os.environ.get("HYBRID_FLAG_MIN", "0.60"))
SAFE_MIN = float(os.environ.get("HYBRID_SAFE_MIN", "0.70"))
MIN_ROWS = 20

_lock = threading.RLock()
_model = None
_attempted = False


def load_rows(paths=None):
    rows = []
    for path in paths or DATA_FILES:
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                rows.extend(r for r in csv.DictReader(f) if r.get("label") in LABELS and r.get("text"))
    return rows


def new_estimator():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, random_state=0)


def embed(embeddings, texts):
    return np.array(embeddings.embed_documents(list(texts)), dtype=np.float32)


def to_prob_dicts(classes, matrix):
    out = []
    for row in matrix:
        probs = {label: 0.0 for label in LABELS}
        for label, p in zip(classes, row, strict=True):
            probs[label] = float(p)
        out.append(probs)
    return out


class EmbeddingClassifier:
    def __init__(self, embeddings, rows):
        self.embeddings = embeddings
        self.estimator = new_estimator()
        self.estimator.fit(embed(embeddings, [r["text"] for r in rows]), [r["label"] for r in rows])
        self.size = len(rows)

    def predict_proba_many(self, texts):
        if not texts:
            return []
        matrix = self.estimator.predict_proba(embed(self.embeddings, texts))
        return to_prob_dicts(self.estimator.classes_, matrix)


def get_model():
    global _model, _attempted
    with _lock:
        if _model is None and not _attempted:
            _attempted = True
            try:
                from utils import rag_engine
                rows = load_rows()
                if len(rows) < MIN_ROWS or len({r["label"] for r in rows}) < 2:
                    print("ML classifier unavailable: not enough labelled clauses")
                else:
                    _model = EmbeddingClassifier(rag_engine.get_embeddings(), rows)
            except Exception as e:
                print(f"ML classifier unavailable: {type(e).__name__}")
        return _model


def set_model(model):
    global _model, _attempted
    with _lock:
        _model, _attempted = model, model is not None


def estimate(label, probability):
    return {"reason": "model estimate", "matched": f"{label}, {round(probability * 100)}% confidence"}


def ml_only(probs):
    label = max(LABELS, key=lambda lab: probs[lab])
    return {"label": label, "reasons": [] if label == "Safe" else [estimate(label, probs[label])]}


def combine(rule_result, probs, flag_min=None, safe_min=None):
    flag_min = FLAG_MIN if flag_min is None else flag_min
    safe_min = SAFE_MIN if safe_min is None else safe_min
    label = rule_result["label"]
    if label == "Safe":
        flagged = probs["Risky"] + probs["Suspicious"]
        if flagged >= flag_min:
            return {"label": "Risky", "reasons": [estimate("Risky", flagged)]}
    elif label == "Risky":
        only_weak = all(r["reason"] in WEAK_REASONS for r in rule_result["reasons"])
        if only_weak and probs["Safe"] >= safe_min:
            return {"label": "Safe", "reasons": []}
    return rule_result
