import csv
import json

import pytest

from eval import full_report, run_eval
from utils import ml_classifier


def probs_for(label, strength=0.9):
    rest = (1 - strength) / 2
    return {lab: (strength if lab == label else rest) for lab in ml_classifier.LABELS}


def test_tune_hybrid_picks_thresholds_that_improve_macro_f1():
    rows = [{"text": "The tenant must provide quarterly reports about activities.", "label": "Risky"},
            {"text": "This agreement commences on the effective date for one year.", "label": "Safe"}] * 6
    model_probs = [probs_for("Risky", 0.8), probs_for("Safe", 0.9)] * 6
    value, flag, safe = run_eval.tune_hybrid(rows, model_probs, kind="legalbert")
    assert value == pytest.approx(2 / 3) and flag == 0.9 and safe == 0.95


def test_tune_hybrid_prefers_the_more_conservative_threshold_on_a_tie():
    rows = [{"text": "This agreement commences on the effective date for one year.", "label": "Safe"}] * 4
    value, flag, safe = run_eval.tune_hybrid(rows, [probs_for("Safe")] * 4, kind="legalbert")
    assert value > 0 and (flag, safe) == (0.95, 0.95)


def test_training_overlap_counts_clauses_that_were_in_the_training_file(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"text": "Seen   clause\ttext", "label": "Safe", "source": "real"}) + "\n", encoding="utf-8")
    rows = [{"text": "Seen clause text"}, {"text": "A different clause"}]
    assert run_eval.training_overlap(rows, str(train)) == 1
    assert run_eval.training_overlap(rows, str(tmp_path / "missing.jsonl")) == 0


def test_legalbert_is_skipped_with_instructions_when_no_model_exists(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LEGAL_BERT_DIR", str(tmp_path / "none"))
    assert run_eval.legalbert_probabilities([{"text": "x"}]) is None
    assert "train_legal_bert" in capsys.readouterr().out


def make_real_file(path, count):
    with open(run_eval.CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:count]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "topic", "text", "assisted", "source"])
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({"id": i + 1, "split": "real", "label": r["label"], "topic": r["topic"], "text": r["text"],
                             "assisted": "no", "source": "rental" if i < count // 2 else "lease"})


@pytest.fixture
def real_file(tmp_path):
    path = tmp_path / "real.csv"
    make_real_file(path, 16)
    return path


def test_report_includes_legalbert_and_a_verdict_when_a_model_is_available(real_file, tmp_path, monkeypatch):
    def fake_legalbert(rows):
        return [probs_for(r["label"], 0.9) for r in rows]
    monkeypatch.setattr(run_eval, "legalbert_probabilities", fake_legalbert)
    md = full_report.run_report(real_path=str(real_file), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "ML (legalbert)" in md and "Hybrid (legalbert)" in md
    assert "## Verdict" in md and "ML beats rules by" in md
    assert "Each test fold is a whole contract" in md


def test_report_warns_when_legalbert_saw_the_clauses_it_is_tested_on(real_file, tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "legalbert_probabilities", lambda rows: [probs_for("Safe") for _ in rows])
    monkeypatch.setattr(run_eval, "training_overlap", lambda rows, *a, **k: 5)
    md = full_report.run_report(real_path=str(real_file), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "5 of your 16 clauses are in Legal-BERT's training data" in md


def test_report_without_a_trained_model_explains_how_to_get_one(real_file, tmp_path, monkeypatch):
    monkeypatch.setenv("LEGAL_BERT_DIR", str(tmp_path / "none"))
    md = full_report.run_report(real_path=str(real_file), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "No trained Legal-BERT model was found" in md and "ML (legalbert)" not in md
    assert "## Verdict" in md


def test_verdict_wording():
    rules = {"accuracy": 0.7, "macro_f1": 0.60, "flag": (0.8, 0.8, 0.8)}
    better = {"accuracy": 0.9, "macro_f1": 0.80, "flag": (0.9, 0.9, 0.9)}
    worse = {"accuracy": 0.5, "macro_f1": 0.40, "flag": (0.5, 0.5, 0.5)}
    lines = full_report.verdict_lines({"Rules": rules, "ML (legalbert)": better, "Hybrid (legalbert)": worse})
    assert lines[0] == "ML beats rules by 0.20 macro-F1 (ML (legalbert))."
    assert lines[1].startswith("Hybrid does NOT beat rules")
    lines = full_report.verdict_lines({"Rules": rules, "ML (tfidf)": worse})
    assert "ML does NOT beat rules" in lines[0] and "Keep `CLASSIFIER=rules`." in lines[0]
    assert lines[1] == "Hybrid was not evaluated."
