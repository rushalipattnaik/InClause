import os

from utils.classifier import analyze_clause

MODES = ("rules", "ml", "hybrid")


def analysis_mode():
    mode = os.environ.get("CLASSIFIER", "rules").strip().lower()
    return mode if mode in MODES else "rules"


def analyze_all(clauses, mode=None):
    mode = mode or analysis_mode()
    rule_results = [analyze_clause(c) for c in clauses]
    if mode == "rules" or not clauses:
        return rule_results
    from utils import ml_classifier
    model = ml_classifier.get_model()
    if model is None:
        return rule_results
    probabilities = model.predict_proba_many(clauses)
    if mode == "ml":
        return [ml_classifier.ml_only(p) for p in probabilities]
    return [ml_classifier.combine(r, p) for r, p in zip(rule_results, probabilities, strict=True)]
