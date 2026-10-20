import json

import pytest

from scripts import train_legal_bert as tlb


def test_class_weights_are_inverse_frequency_in_label_order():
    weights = tlb.class_weights(["Safe"] * 6 + ["Risky"] * 3 + ["Suspicious"])
    assert weights == pytest.approx([10 / 18, 10 / 9, 10 / 3])
    assert tlb.class_weights(["Safe", "Safe"]) == pytest.approx([2 / 6, 0.0, 0.0])


def test_cuad_rows_count_for_less():
    rows = [{"source": "cuad:Some Contract"}, {"source": "real"}, {"source": "authored"}]
    assert tlb.sample_weights(rows, 0.4) == [0.4, 1.0, 1.0]


def test_metrics_report_is_json_serialisable_and_correct():
    y_true = ["Safe", "Safe", "Risky", "Suspicious"]
    y_pred = ["Safe", "Risky", "Risky", "Suspicious"]
    report = tlb.metrics_report(y_true, y_pred)
    json.dumps(report)
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["per_class"]["Risky"]["recall"] == 1.0 and report["per_class"]["Risky"]["precision"] == 0.5
    assert report["per_class"]["Safe"]["support"] == 2
    assert report["confusion"]["Safe"]["Risky"] == 1


def test_learning_rate_warms_up_then_decays_linearly():
    factors = [tlb.lr_factor(step, 10, 100) for step in (0, 5, 10, 55, 100)]
    assert factors == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])
    assert tlb.lr_factor(3, 0, 10) == pytest.approx(0.7)
    assert tlb.lr_factor(500, 10, 100) == 0.0


def test_training_sources_ignore_cuad():
    rows = [{"source": "cuad:A"}, {"source": "real"}, {"source": "real"}, {"source": "authored"}]
    assert tlb.training_sources(rows) == {"real": 2, "authored": 1}


def write_split(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def rows_for(labels):
    return [{"text": f"clause {i}", "label": lab, "source": "authored"} for i, lab in enumerate(labels)]


def test_dry_run_checks_data_without_needing_torch(tmp_path, capsys):
    write_split(tmp_path / "train.jsonl", rows_for(["Safe"] * 4 + ["Risky"] * 3 + ["Suspicious"] * 2))
    write_split(tmp_path / "val.jsonl", rows_for(["Safe", "Risky", "Suspicious"]))
    assert tlb.main(["--data-dir", str(tmp_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "train: 9 clauses" in out and "class weights" in out and "Dry run" in out


def test_missing_class_is_reported_before_any_training(tmp_path, capsys):
    write_split(tmp_path / "train.jsonl", rows_for(["Safe"] * 5 + ["Risky"] * 5))
    write_split(tmp_path / "val.jsonl", rows_for(["Safe", "Risky"]))
    assert tlb.main(["--data-dir", str(tmp_path)]) == 1
    assert "no examples of ['Suspicious']" in capsys.readouterr().out


def test_missing_data_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_training_data"):
        tlb.main(["--data-dir", str(tmp_path / "nope"), "--dry-run"])


def test_subsampling_is_deterministic(tmp_path, capsys):
    write_split(tmp_path / "train.jsonl", rows_for(["Safe"] * 30 + ["Risky"] * 30 + ["Suspicious"] * 30))
    write_split(tmp_path / "val.jsonl", rows_for(["Safe", "Risky", "Suspicious"]))
    tlb.main(["--data-dir", str(tmp_path), "--dry-run", "--max-train-samples", "45"])
    first = capsys.readouterr().out
    tlb.main(["--data-dir", str(tmp_path), "--dry-run", "--max-train-samples", "45"])
    assert first == capsys.readouterr().out and "train: 45 clauses" in first
