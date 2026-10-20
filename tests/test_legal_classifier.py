import json

import numpy as np
import pytest

from utils import analysis, legal_classifier, ml_classifier
from utils.legal_classifier import LegalBertRiskClassifier, ModelNotTrainedError


class StubClassifier(LegalBertRiskClassifier):
    """Skips loading any weights and returns fixed logits chosen from the text."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = []

    def _ensure_loaded(self):
        self._loaded = True

    def _forward(self, texts):
        self.batches.append(list(texts))
        rows = []
        for text in texts:
            if "forfeit" in text:
                rows.append([-1.0, 0.5, 4.0])
            elif "penalty" in text:
                rows.append([-0.5, 3.5, 0.0])
            else:
                rows.append([3.0, 0.0, -1.0])
        return np.array(rows)


@pytest.fixture(autouse=True)
def clean_state():
    ml_classifier.reset()
    ml_classifier._warned.clear()
    yield
    ml_classifier.reset()
    ml_classifier._warned.clear()


def test_probabilities_have_the_right_shape_and_sum_to_one():
    probs = StubClassifier(model_dir="unused").predict_proba(["a plain clause", "a penalty applies", "all fees forfeit"])
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_labels_map_to_the_right_columns():
    clf = StubClassifier(model_dir="unused")
    assert clf.predict(["a plain clause", "a penalty applies", "all fees forfeit"]) == ["Safe", "Risky", "Suspicious"]
    label, confidence = clf.predict_with_confidence(["all fees forfeit"])[0]
    assert label == "Suspicious" and 0.9 < confidence <= 1.0


def test_inference_is_batched():
    clf = StubClassifier(model_dir="unused", batch_size=2)
    clf.predict_proba([f"clause {i}" for i in range(5)])
    assert [len(b) for b in clf.batches] == [2, 2, 1]


def test_empty_input_gives_an_empty_result():
    assert StubClassifier(model_dir="unused").predict_proba([]).shape == (0, 3)


def test_missing_model_directory_gives_a_clear_error(tmp_path):
    clf = LegalBertRiskClassifier(model_dir=tmp_path / "nothing-here")
    with pytest.raises(ModelNotTrainedError, match="train_legal_bert"):
        clf.predict(["any clause"])


def test_column_order_follows_the_models_own_label_names():
    assert legal_classifier.column_order({0: "Suspicious", 1: "Safe", 2: "Risky"}) == [1, 2, 0]
    assert legal_classifier.column_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}) == [0, 1, 2]
    assert legal_classifier.column_order({"0": "Safe", "1": "Risky", "2": "Suspicious"}) == [0, 1, 2]


def test_softmax_is_stable_for_large_logits():
    out = legal_classifier.softmax(np.array([[1000.0, 0.0, -1000.0]]))
    assert np.allclose(out, [[1.0, 0.0, 0.0]])


def test_model_dir_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LEGAL_BERT_DIR", str(tmp_path / "m"))
    assert legal_classifier.default_model_dir() == tmp_path / "m"
    assert not legal_classifier.model_available()
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "config.json").write_text(json.dumps({}))
    assert legal_classifier.model_available()


def test_auto_falls_back_to_the_embedding_model_and_warns_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LEGAL_BERT_DIR", str(tmp_path / "missing"))
    monkeypatch.delenv("ML_KIND", raising=False)
    assert ml_classifier.resolve_kind() == "embed" and ml_classifier.resolve_kind("legalbert") == "embed"
    assert capsys.readouterr().out.count("no trained Legal-BERT model") == 1


def test_auto_uses_legal_bert_when_a_model_exists(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("LEGAL_BERT_DIR", str(tmp_path))
    monkeypatch.delenv("ML_KIND", raising=False)
    assert ml_classifier.resolve_kind() == "legalbert"
    assert isinstance(ml_classifier.get_model(), ml_classifier.LegalBertModel)


def test_default_thresholds_depend_on_the_model_and_the_environment_wins(monkeypatch):
    monkeypatch.delenv("HYBRID_FLAG_MIN", raising=False)
    monkeypatch.delenv("HYBRID_SAFE_MIN", raising=False)
    assert ml_classifier.thresholds("legalbert") == (0.75, 0.85)
    assert ml_classifier.thresholds("tfidf") == (0.60, 0.70)
    monkeypatch.setenv("HYBRID_FLAG_MIN", "0.9")
    assert ml_classifier.thresholds("legalbert") == (0.9, 0.85)


def test_legalbert_reasons_explain_the_flag():
    reason = ml_classifier.estimate("Suspicious", 0.82, "legalbert")
    assert reason["reason"] == "model detected high risk" and "p=0.82" in reason["matched"]
    assert ml_classifier.estimate("Risky", 0.6, "legalbert")["reason"] == "model detected elevated risk"


def test_hybrid_end_to_end_with_a_stub_model(monkeypatch):
    stub = StubClassifier(model_dir="unused")
    ml_classifier.set_model(ml_classifier.LegalBertModel(stub), kind="legalbert")
    monkeypatch.setenv("ML_KIND", "legalbert")
    monkeypatch.setattr(legal_classifier, "model_available", lambda *a, **k: True)
    clauses = ["A penalty applies to every delayed instalment of the balance.",
               "This agreement commences on the effective date for a term of one year."]
    out = analysis.analyze_all(clauses, mode="hybrid")
    assert out[0]["label"] == "Risky" and out[1]["label"] == "Safe"
    only_model = analysis.analyze_all(["Anything is forfeit when the member leaves early."], mode="ml")[0]
    assert only_model["label"] == "Suspicious" and only_model["reasons"][0]["reason"] == "model detected high risk"


def test_a_failing_model_falls_back_to_the_rules(monkeypatch, capsys):
    class Broken(StubClassifier):
        def _forward(self, texts):
            raise RuntimeError("weights are corrupt")
    ml_classifier.set_model(ml_classifier.LegalBertModel(Broken(model_dir="unused")), kind="legalbert")
    monkeypatch.setenv("ML_KIND", "legalbert")
    monkeypatch.setattr(legal_classifier, "model_available", lambda *a, **k: True)
    clause = "The deposit is non-refundable under any circumstances at all."
    assert analysis.analyze_all([clause], mode="hybrid")[0]["label"] == "Suspicious"
    assert "failed" in capsys.readouterr().out
