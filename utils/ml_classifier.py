import csv
import os
import threading
from collections import Counter

import numpy as np

from utils import legal_classifier
from utils.classifier import MITIGATOR_RULES, RISKY_RULES, SUSPICIOUS_RULES, WEAK_REASONS, rule_features

LABELS = ["Safe", "Risky", "Suspicious"]
KINDS = ("embed", "tfidf", "stacked")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILES = [os.path.join(BASE_DIR, "eval", "clauses.csv"), os.path.join(BASE_DIR, "eval", "clauses_real.csv")]
ALL_KINDS = KINDS + ("legalbert",)
GENERIC_THRESHOLDS = (0.60, 0.70)
KIND_THRESHOLDS = {"legalbert": (0.75, 0.85)}
RULE_NAMES = [(pattern, reason) for pattern, reason in SUSPICIOUS_RULES + RISKY_RULES]
MITIGATORS = len(MITIGATOR_RULES)
REGULARISATION = {"embed": 4.0, "tfidf": 10.0, "stacked": 5.0}
MIN_ROWS = 20

_lock = threading.RLock()
_models = {}
_tried = set()
_embedding_cache = {}
_warned = set()


def kind_from_env():
    kind = os.environ.get("ML_KIND", "auto").strip().lower()
    return kind if kind in ALL_KINDS or kind == "auto" else "auto"


def warn_once(message):
    if message not in _warned:
        _warned.add(message)
        print(f"WARNING: {message}")


def resolve_kind(kind=None):
    """Turn 'auto' or 'legalbert' into the model that can actually run, falling back to the embedding model."""
    kind = kind or kind_from_env()
    if kind in ("auto", "legalbert"):
        if legal_classifier.model_available():
            return "legalbert"
        warn_once(f"no trained Legal-BERT model in {legal_classifier.default_model_dir()}, using the embedding "
                  f"classifier instead. Train it with: {legal_classifier.TRAIN_COMMANDS}")
        return "embed"
    return kind


def thresholds(kind=None):
    """Hybrid thresholds (flag_min, safe_min). Defaults depend on the model, environment variables override them.

    Legal-BERT probabilities are sharper than logistic regression, so its defaults are higher. Tune them on held-out
    data with: python -m eval.run_eval --tune-hybrid
    """
    flag_min, safe_min = KIND_THRESHOLDS.get(kind, GENERIC_THRESHOLDS)
    return (float(os.environ.get("HYBRID_FLAG_MIN", flag_min)), float(os.environ.get("HYBRID_SAFE_MIN", safe_min)))


def load_rows(paths=None):
    rows = []
    for path in paths or DATA_FILES:
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                rows.extend(r for r in csv.DictReader(f) if r.get("label") in LABELS and r.get("text"))
    return rows


def new_estimator(kind="embed"):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=REGULARISATION[kind], class_weight="balanced", max_iter=3000, random_state=0)


def embed(embeddings, texts):
    texts = list(texts)
    missing = [t for t in dict.fromkeys(texts) if (id(embeddings), t) not in _embedding_cache]
    if missing:
        for text, vector in zip(missing, embeddings.embed_documents(missing), strict=True):
            _embedding_cache[(id(embeddings), text)] = vector
    return np.array([_embedding_cache[(id(embeddings), t)] for t in texts], dtype=np.float32)


def to_prob_dicts(classes, matrix):
    out = []
    for row in matrix:
        probs = {label: 0.0 for label in LABELS}
        for label, p in zip(classes, row, strict=True):
            probs[label] = float(p)
        out.append(probs)
    return out


class TextClassifier:
    def __init__(self, kind, rows, embeddings=None):
        if kind not in KINDS:
            raise ValueError(f"unknown classifier kind: {kind}")
        if kind == "embed" and embeddings is None:
            raise ValueError("the embed classifier needs an embedding model")
        self.kind = kind
        self.embeddings = embeddings
        self.vectorizer = None
        texts = [r["text"] for r in rows]
        self.estimator = new_estimator(kind)
        self.estimator.fit(self._fit_features(texts), [r["label"] for r in rows])
        self.size = len(rows)

    def _fit_features(self, texts):
        if self.kind == "embed":
            return embed(self.embeddings, texts)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2 if len(texts) >= 40 else 1,
                                          sublinear_tf=True, strip_accents="unicode")
        return self._add_rule_features(self.vectorizer.fit_transform(texts), texts)

    def _features(self, texts):
        if self.kind == "embed":
            return embed(self.embeddings, texts)
        return self._add_rule_features(self.vectorizer.transform(texts), texts)

    def _add_rule_features(self, matrix, texts):
        if self.kind != "stacked":
            return matrix
        from scipy.sparse import csr_matrix, hstack
        return hstack([matrix, csr_matrix(np.array([rule_features(t) for t in texts], dtype=np.float32))]).tocsr()

    def predict_proba_many(self, texts):
        texts = list(texts)
        if not texts:
            return []
        return to_prob_dicts(self.estimator.classes_, self.estimator.predict_proba(self._features(texts)))

    def _feature_names(self):
        names = list(self.vectorizer.get_feature_names_out())
        if self.kind == "stacked":
            names += [f"rule: {reason}" for _, reason in RULE_NAMES] + [f"softener {i + 1}" for i in range(MITIGATORS)]
            names += ["rules said Safe", "rules said Risky", "rules said Suspicious"]
        return names

    def explain_terms(self, texts, labels, top=3):
        texts = list(texts)
        if self.kind == "embed" or not texts:
            return [[] for _ in texts]
        classes = list(self.estimator.classes_)
        coef = self.estimator.coef_
        if len(classes) == 2:
            coef = np.vstack([-coef[0], coef[0]])
        names = self._feature_names()
        matrix = self._features(texts)
        out = []
        for row, label in zip(range(matrix.shape[0]), labels, strict=True):
            if label not in classes:
                out.append([])
                continue
            line = matrix.getrow(row)
            weights = coef[classes.index(label)]
            scored = sorted(((value * weights[col], col) for col, value in zip(line.indices, line.data, strict=True)),
                            reverse=True)
            out.append([names[col] for score, col in scored[:top] if score > 0])
        return out


def EmbeddingClassifier(embeddings, rows):
    return TextClassifier("embed", rows, embeddings)


def make_splits(labels, groups=None, folds=5, seed=0):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    index = np.arange(len(labels))
    if groups is not None and len(set(groups)) >= 2:
        return list(GroupKFold(n_splits=min(folds, len(set(groups)))).split(index, labels, groups))
    smallest = min(Counter(labels).values())
    splitter = StratifiedKFold(n_splits=max(2, min(folds, smallest)), shuffle=True, random_state=seed)
    return list(splitter.split(index, labels))


def cross_val_probs(kind, eval_rows, extra_rows=(), embeddings=None, groups=None, folds=5):
    labels = [r["label"] for r in eval_rows]
    out = [None] * len(eval_rows)
    for train_idx, test_idx in make_splits(labels, groups, folds):
        model = TextClassifier(kind, list(extra_rows) + [eval_rows[i] for i in train_idx], embeddings)
        probs = model.predict_proba_many([eval_rows[i]["text"] for i in test_idx])
        for i, p in zip(test_idx, probs, strict=True):
            out[i] = p
    return out


class LegalBertModel:
    """Adapter that gives the Legal-BERT classifier the same interface as TextClassifier."""

    kind = "legalbert"

    def __init__(self, classifier):
        self.classifier = classifier
        self.size = 0

    def predict_proba_many(self, texts):
        texts = list(texts)
        return to_prob_dicts(legal_classifier.LABELS, self.classifier.predict_proba(texts)) if texts else []

    def explain_terms(self, texts, labels, top=3):
        return [[] for _ in texts]


def get_model(kind=None):
    kind = resolve_kind(kind)
    with _lock:
        if kind not in _models and kind not in _tried:
            _tried.add(kind)
            try:
                rows = load_rows()
                if kind == "legalbert":
                    _models[kind] = LegalBertModel(legal_classifier.LegalBertRiskClassifier())
                elif len(rows) < MIN_ROWS or len({r["label"] for r in rows}) < 2:
                    print("ML classifier unavailable: not enough labelled clauses")
                else:
                    embeddings = None
                    if kind == "embed":
                        from utils import rag_engine
                        embeddings = rag_engine.get_embeddings()
                    _models[kind] = TextClassifier(kind, rows, embeddings)
            except Exception as e:
                print(f"ML classifier unavailable: {type(e).__name__}")
        return _models.get(kind)


def set_model(model, kind=None):
    with _lock:
        kind = kind or (model.kind if model is not None else kind_from_env())
        _tried.add(kind)
        if model is None:
            _models.pop(kind, None)
        else:
            _models[kind] = model


def reset():
    with _lock:
        _models.clear()
        _tried.clear()


def estimate(label, probability, kind=None):
    if kind == "legalbert":
        level = "high" if label == "Suspicious" else "elevated"
        return {"reason": f"model detected {level} risk",
                "matched": f"p={probability:.2f}, based on legal semantic features"}
    return {"reason": "model estimate", "matched": f"{label}, {round(probability * 100)}% confidence"}


def ml_only(probs, terms=(), kind=None):
    label = max(LABELS, key=lambda lab: probs[lab])
    if label == "Safe":
        return {"label": label, "reasons": []}
    reasons = [estimate(label, probs[label], kind)]
    if terms:
        reasons.append({"reason": "key words the model relied on", "matched": ", ".join(terms)})
    return {"label": label, "reasons": reasons}


def combine(rule_result, probs, flag_min=None, safe_min=None, kind=None):
    default_flag, default_safe = thresholds(kind)
    flag_min = default_flag if flag_min is None else flag_min
    safe_min = default_safe if safe_min is None else safe_min
    label = rule_result["label"]
    if label == "Safe":
        flagged = probs["Risky"] + probs["Suspicious"]
        if flagged >= flag_min:
            return {"label": "Risky", "reasons": [estimate("Risky", flagged, kind)]}
    elif label == "Risky":
        only_weak = all(r["reason"] in WEAK_REASONS for r in rule_result["reasons"])
        if only_weak and probs["Safe"] >= safe_min:
            return {"label": "Safe", "reasons": []}
    return rule_result
