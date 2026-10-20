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
    try:
        probabilities = model.predict_proba_many(clauses)
    except Exception as e:
        ml_classifier.warn_once(f"the {model.kind} classifier failed ({type(e).__name__}), using the rules instead")
        return rule_results
    if mode == "ml":
        guesses = [max(ml_classifier.LABELS, key=lambda lab: p[lab]) for p in probabilities]
        terms = model.explain_terms(clauses, guesses)
        return [ml_classifier.ml_only(p, t, model.kind) for p, t in zip(probabilities, terms, strict=True)]
    return [ml_classifier.combine(r, p, kind=model.kind) for r, p in zip(rule_results, probabilities, strict=True)]
