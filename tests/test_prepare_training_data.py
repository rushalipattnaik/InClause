import json
import random

import pytest

from scripts import prepare_training_data as ptd

OFFICIAL_CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
    "Notice Period to Terminate Renewal", "Governing Law", "Most Favored Nation", "Non-Compete", "Exclusivity",
    "No-Solicit of Customers", "Competitive Restriction Exception", "No-Solicit of Employees", "Non-Disparagement",
    "Termination for Convenience", "Rofr/Rofo/Rofn", "Change of Control", "Anti-Assignment",
    "Revenue/Profit Sharing", "Price Restrictions", "Minimum Commitment", "Volume Restriction",
    "IP Ownership Assignment", "Joint IP Ownership", "License Grant", "Non-Transferable License",
    "Affiliate License-Licensor", "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable or Perpetual License", "Source Code Escrow", "Post-Termination Services", "Audit Rights",
    "Uncapped Liability", "Cap on Liability", "Liquidated Damages", "Warranty Duration", "Insurance",
    "Covenant Not to Sue", "Third Party Beneficiary",
]


def test_the_fixture_is_the_official_41_categories():
    assert len(OFFICIAL_CUAD_CATEGORIES) == 41 == len(set(OFFICIAL_CUAD_CATEGORIES))


def test_mapping_covers_every_official_category_and_uses_valid_labels():
    ptd.check_mapping(OFFICIAL_CUAD_CATEGORIES, official=OFFICIAL_CUAD_CATEGORIES)
    assert set(ptd.CUAD_CATEGORY_TO_LABEL.values()) <= set(ptd.LABELS)
    assert len(ptd.CUAD_CATEGORY_TO_LABEL) == 41


def test_an_unmapped_category_fails_loudly():
    with pytest.raises(ValueError, match="Brand New Category"):
        ptd.check_mapping(OFFICIAL_CUAD_CATEGORIES + ["Brand New Category"])


def test_a_missing_or_stale_mapping_entry_fails_loudly(monkeypatch):
    shrunk = {k: v for k, v in ptd.CUAD_CATEGORY_TO_LABEL.items() if k != "Insurance"}
    monkeypatch.setattr(ptd, "CUAD_CATEGORY_TO_LABEL", shrunk)
    with pytest.raises(ValueError, match="Insurance"):
        ptd.check_mapping(OFFICIAL_CUAD_CATEGORIES, official=OFFICIAL_CUAD_CATEGORIES)
    bad = dict(ptd.CUAD_CATEGORY_TO_LABEL, Insurance="Dangerous")
    monkeypatch.setattr(ptd, "CUAD_CATEGORY_TO_LABEL", bad)
    with pytest.raises(ValueError, match="invalid label"):
        ptd.check_mapping(OFFICIAL_CUAD_CATEGORIES)


def test_category_names_in_the_data_are_matched_case_insensitively():
    assert ptd.category_of("SOME CO-EX-10__Cap On Liability") == "Cap on Liability"
    assert ptd.category_of("A__B__Ip Ownership Assignment") == "IP Ownership Assignment"
    assert ptd.category_of("X__Something New") == "Something New"


def contract(title, annotated=True):
    law = f"This Agreement ({title}) shall be governed by the laws of the State of New York."
    licence = f"The Licensee under {title} receives an irrevocable and perpetual license to use the Software worldwide."
    assign = f"Neither party to {title} may assign this Agreement without the prior written consent of the other party."
    context = "\n\n".join([
        law, licence, assign,
        f"The parties to {title} will meet every quarter to review progress and share information about plans.",
        f"Each party to {title} shall bear its own costs in connection with the negotiation of this Agreement.",
        f"Notices under {title} shall be sent to the addresses written on the first page of this Agreement."])
    spans = {"Governing Law": law, "Irrevocable Or Perpetual License": licence, "Anti-Assignment": assign}
    qas = [{"id": f"{title}__{cat}", "question": "q", "is_impossible": not annotated,
            "answers": [{"text": text, "answer_start": 0}] if annotated else []} for cat, text in spans.items()]
    qas.append({"id": f"{title}__Insurance", "question": "q", "is_impossible": True, "answers": []})
    return {"title": title, "paragraphs": [{"context": context, "qas": qas}]}


def test_units_get_the_label_of_the_category_they_belong_to():
    rows = ptd.contract_units(contract("C1"), unannotated_per_contract=2, rng=random.Random(0))
    by_category = {r["category"]: r["label"] for r in rows}
    assert by_category["Governing Law"] == "Safe"
    assert by_category["Irrevocable or Perpetual License"] == "Suspicious"
    assert by_category["Anti-Assignment"] == "Risky"
    unannotated = [r for r in rows if r["category"] == "unannotated"]
    assert len(unannotated) == 2 and all(r["label"] == "Safe" for r in unannotated)
    assert all(r["source"] == "cuad:C1" for r in rows)


def test_a_contract_without_annotations_only_gives_capped_unannotated_safe_units():
    rows = ptd.contract_units(contract("C2", annotated=False), unannotated_per_contract=3, rng=random.Random(0))
    assert len(rows) == 3 and {r["label"] for r in rows} == {"Safe"}


def test_the_most_severe_overlapping_category_wins():
    units = ["The licence is irrevocable and perpetual and the licensee may use it worldwide."]
    spans = [("The licence is irrevocable and perpetual and the licensee may use it worldwide.", "Safe", "License Grant"),
             ("The licence is irrevocable and perpetual and the licensee may use it worldwide.", "Suspicious", "X")]
    assert ptd.label_units(units, spans) == [("Suspicious", "X")]


def test_duplicates_are_dropped_and_conflicting_labels_removed():
    rows = [{"text": "same text here", "label": "Safe"}, {"text": "same text here", "label": "Safe"},
            {"text": "conflict text", "label": "Safe"}, {"text": "conflict text", "label": "Risky"},
            {"text": "unique text", "label": "Risky"}]
    assert [r["text"] for r in ptd.drop_conflicts(rows)] == ["same text here", "unique text"]


def test_classes_are_capped_independently():
    rows = [{"text": f"s{i}", "label": "Safe"} for i in range(10)] + [{"text": "r", "label": "Risky"}]
    capped = ptd.cap_per_class(rows, 4, random.Random(0))
    assert sum(r["label"] == "Safe" for r in capped) == 4 and sum(r["label"] == "Risky" for r in capped) == 1


def many_rows():
    rows = []
    for i in range(30):
        for label in ptd.LABELS:
            rows.append({"text": f"{label} clause {i}", "label": label, "source": f"cuad:contract {i}"})
    return rows


def test_cuad_split_keeps_every_contract_in_one_place_and_is_deterministic():
    rows = many_rows()
    train, val, test = ptd.split_cuad_by_contract(rows)
    groups = [{r["source"] for r in part} for part in (train, val, test)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert len(train) + len(val) + len(test) == len(rows) and val and test
    assert ptd.split_cuad_by_contract(rows)[1] == val


def real_rows():
    return [{"text": f"real {i}", "label": "Safe", "source": "rental" if i < 3 else "lease"} for i in range(6)]


def test_your_real_clauses_are_the_held_out_test_set_by_default():
    train, val, test = ptd.assemble(many_rows(), [], real_rows(), set(), final=False)
    assert test == real_rows() and not any(r["source"] in ("rental", "lease") for r in train + val)


def test_train_on_real_moves_only_the_chosen_sources():
    train, _, test = ptd.assemble(many_rows(), [], real_rows(), {"rental"}, final=False)
    assert {r["source"] for r in test} == {"lease"}
    assert sum(r["source"] == "rental" for r in train) == 3


def test_final_mode_trains_on_everything(capsys):
    train, _, test = ptd.assemble(many_rows(), [], real_rows(), set(), final=True)
    assert sum(r["source"] in ("rental", "lease") for r in train) == 6
    assert "WARNING" not in capsys.readouterr().out and test


def test_without_real_clauses_the_test_set_falls_back_to_cuad_with_a_warning(capsys):
    _, _, test = ptd.assemble(many_rows(), [], [], set(), final=False)
    assert test and "WARNING" in capsys.readouterr().out and all(r["source"].startswith("cuad:") for r in test)


def test_authored_clauses_are_split_stratified_and_deterministic():
    rows = [{"text": f"{lab} {i}", "label": lab, "source": "authored"} for lab in ptd.LABELS for i in range(10)]
    train, val = ptd.split_authored(rows)
    assert len(train) == 24 and len(val) == 6 and {r["label"] for r in val} == set(ptd.LABELS)
    assert ptd.split_authored(rows)[1] == val


def test_main_writes_the_three_jsonl_files_from_a_cuad_file(tmp_path, capsys):
    data = {"version": "test", "data": [contract(f"Contract {i}") for i in range(14)]}
    cuad = tmp_path / "CUADv1.json"
    cuad.write_text(json.dumps(data), encoding="utf-8")
    assert ptd.main(["--cuad-json", str(cuad), "--out-dir", str(tmp_path / "out")]) == 0
    for name in ("train", "val", "test"):
        lines = (tmp_path / "out" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert lines and set(json.loads(lines[0])) >= {"text", "label", "source"}
    assert "train" in capsys.readouterr().out


def test_too_few_contracts_for_a_split_keeps_everything_in_train(capsys):
    rows = [{"text": f"clause {i}", "label": "Safe", "source": f"cuad:{i % 2}"} for i in range(8)]
    train, val, test = ptd.split_cuad_by_contract(rows)
    assert len(train) == 8 and not val and not test and "WARNING" in capsys.readouterr().out
