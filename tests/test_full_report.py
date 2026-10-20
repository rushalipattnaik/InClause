import csv

from eval import full_report, run_eval


def make_real_file(path, count, assisted=0):
    with open(run_eval.CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:count]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "topic", "text", "assisted"])
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({"id": i + 1, "split": "real", "label": r["label"], "topic": r["topic"],
                             "text": r["text"], "assisted": "yes" if i < assisted else "no"})


def test_full_report_with_enough_real_clauses(tmp_path):
    real = tmp_path / "real.csv"
    make_real_file(real, 14, assisted=3)
    out = tmp_path / "RESULTS.md"
    md = full_report.run_report(real_path=str(real), authored_path=run_eval.CSV_PATH, out_path=str(out), fake=True)
    assert out.read_text(encoding="utf-8") == md
    for needle in ("## In plain words", "matched your label on", "noticed", "3 of them accepted from an AI suggestion",
                   "### How to read the numbers", "## Classifier comparison", "## Retrieval", "Fake embeddings were used"):
        assert needle in md, needle
    assert "Keep `CLASSIFIER=rules`" in md or "CLASSIFIER=" in md
    assert "label at least 30" in md


def test_report_explains_what_to_do_when_there_is_too_little_real_data(tmp_path):
    real = tmp_path / "real.csv"
    make_real_file(real, 3)
    md = full_report.run_report(real_path=str(real), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "You have 3 labelled real clauses" in md and "python -m eval.label" in md
    assert "Authored set, sanity check only" in md


def test_report_works_when_the_real_file_does_not_exist(tmp_path):
    md = full_report.run_report(real_path=str(tmp_path / "missing.csv"), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "You have 0 labelled real clauses" in md


def test_resume_line_only_appears_with_enough_clauses_and_mentions_assisted_labels():
    scores = {"Rules": {"accuracy": 0.81, "macro_f1": 0.7, "flag": (0.9, 0.85, 0.87)}}
    assert full_report.resume_line(scores, 12, 0) is None
    line = full_report.resume_line(scores, 40, 10)
    assert "40 clauses" in line and "81%" in line and "85%" in line and "10 of the 40 labels" in line
    assert "plus or minus 12 points" in line


def test_recommendation_prefers_rules_unless_a_candidate_clearly_wins():
    rules = {"accuracy": 0.8, "macro_f1": 0.70, "flag": (0.9, 0.9, 0.9)}
    close = {"accuracy": 0.8, "macro_f1": 0.71, "flag": (0.9, 0.9, 0.9)}
    better = {"accuracy": 0.9, "macro_f1": 0.85, "flag": (0.9, 0.9, 0.9)}
    assert "Keep `CLASSIFIER=rules`" in full_report.recommendation({"Rules": rules, "Hybrid (tfidf)": close}, 40)
    assert "CLASSIFIER=hybrid" in full_report.recommendation({"Rules": rules, "Hybrid (tfidf)": better}, 40)
    assert "label at least 30" in full_report.recommendation({"Rules": rules}, 12)


def test_recommendation_names_the_model_kind_to_use():
    rules = {"accuracy": 0.7, "macro_f1": 0.60, "flag": (0.8, 0.8, 0.8)}
    winner = {"accuracy": 0.9, "macro_f1": 0.80, "flag": (0.9, 0.9, 0.9)}
    text = full_report.recommendation({"Rules": rules, "ML (stacked)": winner, "ML (embed)": rules}, 40)
    assert "CLASSIFIER=ml" in text and "ML_KIND=stacked" in text


def test_report_warns_when_sources_are_missing(tmp_path):
    real = tmp_path / "real.csv"
    make_real_file(real, 14)
    md = full_report.run_report(real_path=str(real), authored_path=run_eval.CSV_PATH,
                                out_path=str(tmp_path / "R.md"), fake=True)
    assert "Sources are not recorded" in md and "cross-validated (trained on the authored set plus your other clauses)" in md
    for kind in ("embed", "tfidf", "stacked"):
        assert f"ML ({kind})" in md and f"Hybrid ({kind})" in md
