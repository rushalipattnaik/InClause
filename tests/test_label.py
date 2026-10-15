import csv

from eval import label


def run(clauses, answers, path, topics=False):
    feed = iter(answers)
    printed = []
    saved = label.label_clauses(clauses, path=str(path), topics=topics,
                                input_fn=lambda _prompt: next(feed), output_fn=printed.append)
    return saved, printed


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_labels_are_saved_with_the_expected_schema(tmp_path):
    path = tmp_path / "real.csv"
    saved, _ = run(["Clause one text", "Clause two text", "Clause three text"], ["s", "k", "x"], path)
    got = rows(path)
    assert saved == 2
    assert [(r["id"], r["split"], r["label"], r["topic"], r["text"]) for r in got] == [
        ("1", "real", "Safe", "", "Clause one text"), ("2", "real", "Suspicious", "", "Clause three text")]


def test_quit_stops_and_resuming_skips_already_labelled_clauses(tmp_path):
    path = tmp_path / "real.csv"
    clauses = ["Alpha clause", "Beta clause", "Gamma clause"]
    run(clauses, ["r", "q"], path)
    saved, _ = run(clauses, ["s", "s"], path)
    got = rows(path)
    assert saved == 2 and [r["text"] for r in got] == clauses and [r["id"] for r in got] == ["1", "2", "3"]


def test_invalid_keys_are_ignored_until_a_valid_one(tmp_path):
    path = tmp_path / "real.csv"
    saved, _ = run(["Some clause"], ["z", "", "r"], path)
    assert saved == 1 and rows(path)[0]["label"] == "Risky"


def test_topic_prompt_accepts_ids_none_and_unknown(tmp_path):
    path = tmp_path / "real.csv"
    _, printed = run(["First", "Second", "Third"], ["r", "?", "bogus", "arbitration", "s", "n", "s", ""], path, topics=True)
    got = rows(path)
    assert [r["topic"] for r in got] == ["arbitration", "none", ""]
    assert any("arbitration" in line for line in printed) and any("not a valid topic" in line for line in printed)


def test_the_tools_prediction_is_never_shown(tmp_path):
    _, printed = run(["The deposit is non-refundable under any circumstances."], ["s"], tmp_path / "r.csv")
    assert not any(word in " ".join(printed).lower() for word in ("suspicious clause", "flagged", "why"))


def test_enter_accepts_the_suggestion_and_is_marked_assisted(tmp_path):
    path = tmp_path / "real.csv"
    feed = iter(["", "x", ""])
    label.label_clauses(["First clause text", "Second clause text", "Third clause text"], path=str(path),
                        suggest_fn=lambda c: "Risky", input_fn=lambda _p: next(feed), output_fn=lambda _m: None)
    got = rows(path)
    assert [(r["label"], r["assisted"]) for r in got] == [("Risky", "yes"), ("Suspicious", "no"), ("Risky", "yes")]


def test_suggestions_are_shown_but_enter_does_nothing_without_one(tmp_path):
    path = tmp_path / "real.csv"
    feed = iter(["", "", "s"])
    printed = []
    label.label_clauses(["Only clause here"], path=str(path), input_fn=lambda _p: next(feed), output_fn=printed.append)
    assert rows(path)[0]["assisted"] == "no" and not any("AI suggestion" in line for line in printed)


def test_ai_failure_falls_back_to_blind_labelling(tmp_path):
    path = tmp_path / "real.csv"

    def broken(_clause):
        raise ConnectionError("ollama is not running")
    feed = iter(["r", "s"])
    printed = []
    saved = label.label_clauses(["One clause", "Two clause"], path=str(path), suggest_fn=broken,
                                input_fn=lambda _p: next(feed), output_fn=printed.append)
    assert saved == 2 and sum("unavailable" in line for line in printed) == 1


def test_old_files_without_the_assisted_column_are_upgraded(tmp_path):
    path = tmp_path / "real.csv"
    path.write_text("id,split,label,topic,text\n1,real,Safe,,An older labelled clause\n", encoding="utf-8")
    run(["A new clause text"], ["r"], path)
    got = rows(path)
    assert [r["assisted"] for r in got] == ["no", "no"] and got[0]["text"] == "An older labelled clause"


def test_header_only_file_is_upgraded_too(tmp_path):
    path = tmp_path / "real.csv"
    path.write_text("id,split,label,topic,text\n", encoding="utf-8")
    run(["Brand new clause"], ["s"], path)
    assert rows(path)[0]["assisted"] == "no"
