import pytest

from utils import analysis, ml_classifier
from utils.classifier import analyze_clause

SAFE = ["This agreement commences on the effective date and continues for one year",
        "The headings are for convenience only and do not affect interpretation",
        "This agreement may be executed in counterparts each deemed an original",
        "Delivery shall be made within fifteen days of receipt of the advance",
        "Support is available by email on business days during office hours"] * 4
RISKY = ["A tariff surcharge is levied on each missed instalment",
         "An additional levy and surcharge applies to every delayed instalment",
         "The tariff includes a surcharge and a monthly levy on the balance",
         "A surcharge tariff is levied whenever the payment is delayed"] * 4
SUSP = ["The member forfeits everything and no refund shall ever be given",
        "All sums are forfeited and the member receives no refund at all",
        "Nothing is refunded and every sum paid is forfeited permanently"] * 4


def rows():
    return ([{"text": t, "label": "Safe"} for t in SAFE] + [{"text": t, "label": "Risky"} for t in RISKY]
            + [{"text": t, "label": "Suspicious"} for t in SUSP])


@pytest.fixture
def model(fake_embeddings):
    m = ml_classifier.EmbeddingClassifier(fake_embeddings, rows())
    ml_classifier.set_model(m)
    yield m
    ml_classifier.set_model(None)


def test_probabilities_are_valid_and_sensible(model):
    probs = model.predict_proba_many(["A tariff surcharge and levy apply to a missed instalment",
                                      "This agreement commences on the effective date for one year"])
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in probs)
    assert max(probs[0], key=probs[0].get) == "Risky"
    assert max(probs[1], key=probs[1].get) == "Safe"


def test_hybrid_rescues_a_clause_the_rules_miss(model):
    clause = "A tariff surcharge is levied on each missed instalment of the balance"
    assert analyze_clause(clause)["label"] == "Safe"
    result = analysis.analyze_all([clause], mode="hybrid")[0]
    assert result["label"] == "Risky" and result["reasons"][0]["reason"] == "model estimate"


def test_hybrid_never_overrides_a_suspicious_rule(model):
    clause = "The deposit is non-refundable and this commences on the effective date for one year"
    assert analysis.analyze_all([clause], mode="hybrid")[0]["label"] == "Suspicious"


def test_hybrid_demotes_a_weak_only_flag_when_the_model_is_confident():
    rule = {"label": "Risky", "reasons": [{"reason": "fees or charges", "matched": "fee"}]}
    assert ml_classifier.combine(rule, {"Safe": 0.9, "Risky": 0.07, "Suspicious": 0.03})["label"] == "Safe"
    strong = {"label": "Risky", "reasons": [{"reason": "penalty", "matched": "penalty"}]}
    assert ml_classifier.combine(strong, {"Safe": 0.95, "Risky": 0.03, "Suspicious": 0.02})["label"] == "Risky"


def test_ml_mode_uses_only_the_model(model):
    out = analysis.analyze_all(["This agreement commences on the effective date for one year"], mode="ml")[0]
    assert out == {"label": "Safe", "reasons": []}


def test_falls_back_to_rules_when_no_model_is_available(monkeypatch):
    monkeypatch.setattr(ml_classifier, "_model", None)
    monkeypatch.setattr(ml_classifier, "_attempted", True)
    clause = "The deposit is non-refundable under any circumstances at all."
    assert analysis.analyze_all([clause], mode="hybrid") == [analyze_clause(clause)]


def test_unknown_mode_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("CLASSIFIER", "nonsense")
    assert analysis.analysis_mode() == "rules"


def test_shipped_training_data_loads():
    data = ml_classifier.load_rows()
    assert len(data) >= 90 and {r["label"] for r in data} == {"Safe", "Risky", "Suspicious"}
