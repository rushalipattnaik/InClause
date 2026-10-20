"""Build train, validation and test JSONL files for the Legal-BERT risk classifier.

Sources: the official CUAD dataset (weak labels derived from its 41 clause categories), the author-labelled clauses
in eval/clauses.csv, and your own labelled clauses in eval/clauses_real.csv.

Run: python -m scripts.prepare_training_data
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.clause_splitter import split_into_clauses  # noqa: E402

CUAD_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/main/data.zip"
CUAD_JSON_NAME = "CUADv1.json"
LABELS = ("Safe", "Risky", "Suspicious")
SEVERITY = {"Safe": 0, "Risky": 1, "Suspicious": 2}
MIN_SPAN_CHARS = 25
MIN_PER_CLASS = 50
SEED = 42

CUAD_CATEGORY_TO_LABEL: dict[str, str] = {
    "Document Name": "Safe",
    "Parties": "Safe",
    "Agreement Date": "Safe",
    "Effective Date": "Safe",
    "Expiration Date": "Safe",
    "Renewal Term": "Risky",
    "Notice Period to Terminate Renewal": "Risky",
    "Governing Law": "Safe",
    "Most Favored Nation": "Risky",
    "Non-Compete": "Risky",
    "Exclusivity": "Risky",
    "No-Solicit of Customers": "Risky",
    "Competitive Restriction Exception": "Safe",
    "No-Solicit of Employees": "Risky",
    "Non-Disparagement": "Risky",
    "Termination for Convenience": "Risky",
    "Rofr/Rofo/Rofn": "Risky",
    "Change of Control": "Risky",
    "Anti-Assignment": "Risky",
    "Revenue/Profit Sharing": "Risky",
    "Price Restrictions": "Risky",
    "Minimum Commitment": "Risky",
    "Volume Restriction": "Risky",
    "IP Ownership Assignment": "Risky",
    "Joint IP Ownership": "Risky",
    "License Grant": "Safe",
    "Non-Transferable License": "Risky",
    "Affiliate License-Licensor": "Safe",
    "Affiliate License-Licensee": "Safe",
    "Unlimited/All-You-Can-Eat-License": "Risky",
    "Irrevocable or Perpetual License": "Suspicious",
    "Source Code Escrow": "Risky",
    "Post-Termination Services": "Risky",
    "Audit Rights": "Risky",
    "Uncapped Liability": "Suspicious",
    "Cap on Liability": "Risky",
    "Liquidated Damages": "Risky",
    "Warranty Duration": "Safe",
    "Insurance": "Risky",
    "Covenant Not to Sue": "Suspicious",
    "Third Party Beneficiary": "Safe",
}


def check_mapping(categories: list[str], official: list[str] | None = None) -> None:
    """Fail loudly if a category in the data has no label or a label is invalid.

    When the official category list is supplied, also fail if the mapping has an entry for a category that does not
    exist or if an official category is missing from the mapping.
    """
    problems = []
    unmapped = sorted(set(categories) - set(CUAD_CATEGORY_TO_LABEL))
    if unmapped:
        problems.append(f"categories with no mapping: {unmapped}")
    invalid = sorted(c for c, label in CUAD_CATEGORY_TO_LABEL.items() if label not in LABELS)
    if invalid:
        problems.append(f"categories mapped to an invalid label: {invalid}")
    if official is not None:
        stale = sorted(set(CUAD_CATEGORY_TO_LABEL) - set(official))
        missing = sorted(set(official) - set(CUAD_CATEGORY_TO_LABEL))
        if stale:
            problems.append(f"mapped names that are not official CUAD categories: {stale}")
        if missing:
            problems.append(f"official categories with no mapping: {missing}")
    if problems:
        raise ValueError("CUAD category mapping is out of date: " + "; ".join(problems))


_OFFICIAL_BY_LOWER = {name.lower(): name for name in CUAD_CATEGORY_TO_LABEL}


def category_of(qa_id: str) -> str:
    """CUAD question ids look like '<contract title>__<Category>', with capitalisation that differs from the
    official list (for example 'Cap On Liability'), so names are matched case-insensitively."""
    raw = qa_id.rsplit("__", 1)[1]
    return _OFFICIAL_BY_LOWER.get(raw.lower(), raw)


def normalise(text: str) -> str:
    return " ".join(text.split())


def download_cuad(dest_dir: Path) -> Path:
    """Download the official CUAD archive once and return the path of CUADv1.json."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / CUAD_JSON_NAME
    if target.exists():
        return target
    print(f"Downloading CUAD from {CUAD_URL} (about 18 MB) ...")
    with urllib.request.urlopen(CUAD_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target.write_bytes(archive.read(CUAD_JSON_NAME))
    return target


def load_cuad(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def contract_categories(contracts: list[dict]) -> list[str]:
    seen: set[str] = set()
    for contract in contracts:
        for paragraph in contract["paragraphs"]:
            seen.update(category_of(qa["id"]) for qa in paragraph["qas"])
    return sorted(seen)


def label_units(units: list[str], spans: list[tuple[str, str, str]]) -> list[tuple[str, str] | None]:
    """Label each unit with the most severe CUAD category whose answer span contains it, or None if unannotated."""
    out: list[tuple[str, str] | None] = []
    for unit in units:
        best: tuple[str, str] | None = None
        for span_text, label, category in spans:
            inside = unit in span_text
            covers = span_text in unit and len(span_text) >= 0.5 * len(unit)
            if (inside or covers) and (best is None or SEVERITY[label] > SEVERITY[best[0]]):
                best = (label, category)
        out.append(best)
    return out


def contract_units(contract: dict, unannotated_per_contract: int, rng: random.Random) -> list[dict]:
    """Turn one CUAD contract into labelled clause-sized units, using the same splitter the app uses."""
    title = contract["title"]
    rows: list[dict] = []
    for paragraph in contract["paragraphs"]:
        spans = []
        for qa in paragraph["qas"]:
            if qa["is_impossible"]:
                continue
            category = category_of(qa["id"])
            label = CUAD_CATEGORY_TO_LABEL[category]
            for answer in qa["answers"]:
                text = normalise(answer["text"])
                if len(text) >= MIN_SPAN_CHARS:
                    spans.append((text, label, category))
        units = [normalise(u) for u in split_into_clauses(paragraph["context"])]
        labelled = label_units(units, spans)
        unannotated = []
        for unit, hit in zip(units, labelled, strict=True):
            if hit is None:
                unannotated.append(unit)
            else:
                rows.append({"text": unit, "label": hit[0], "source": f"cuad:{title}", "category": hit[1]})
        rng.shuffle(unannotated)
        for unit in unannotated[:unannotated_per_contract]:
            rows.append({"text": unit, "label": "Safe", "source": f"cuad:{title}", "category": "unannotated"})
    return rows


def drop_conflicts(rows: list[dict]) -> list[dict]:
    """Remove duplicate texts, and drop any text that carries two different labels."""
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_text[row["text"]].add(row["label"])
    seen: set[str] = set()
    kept = []
    for row in rows:
        text = row["text"]
        if len(labels_by_text[text]) > 1 or text in seen:
            continue
        seen.add(text)
        kept.append(row)
    return kept


def cap_per_class(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    kept = []
    for label in LABELS:
        group = by_label[label]
        rng.shuffle(group)
        kept.extend(group[:limit])
    return kept


def split_cuad_by_contract(rows: list[dict], folds: int = 10, seed: int = SEED) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified, contract-grouped split so no contract appears in more than one of train, val and test."""
    from sklearn.model_selection import StratifiedGroupKFold

    labels = [r["label"] for r in rows]
    groups = [r["source"] for r in rows]
    folds = min(folds, len(set(groups)))
    if folds < 3:
        print("WARNING: fewer than 3 contracts, so there is no validation or test split from CUAD.")
        return list(rows), [], []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    train, val, test = [], [], []
    for fold, (_, index) in enumerate(splitter.split(rows, labels, groups)):
        target = val if fold == 0 else test if fold == 1 else train
        target.extend(rows[i] for i in index)
    return train, val, test


def read_label_csv(path: Path, default_source: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{"text": normalise(r["text"]), "label": r["label"],
                 "source": r.get("source") or default_source}
                for r in csv.DictReader(f) if r.get("label") in LABELS and r.get("text")]


def split_authored(rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    from sklearn.model_selection import train_test_split

    if len(rows) < 10:
        return rows, []
    train, val = train_test_split(rows, test_size=0.2, stratify=[r["label"] for r in rows], random_state=seed)
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict]) -> list[str]:
    counts = Counter(r["label"] for r in rows)
    print(f"{name:<6} {len(rows):>6} clauses: " + ", ".join(f"{lab} {counts.get(lab, 0)}" for lab in LABELS))
    return [lab for lab in LABELS if counts.get(lab, 0) < MIN_PER_CLASS]


def assemble(cuad_rows: list[dict], authored: list[dict], real: list[dict], train_on_real: set[str],
             final: bool, seed: int = SEED) -> tuple[list[dict], list[dict], list[dict]]:
    """Combine all sources into train, val and test. Your real clauses are held out as the test set by default."""
    train, val, cuad_test = split_cuad_by_contract(cuad_rows, seed=seed)
    authored_train, authored_val = split_authored(authored, seed)
    train += authored_train
    val += authored_val
    real_train = [r for r in real if final or r["source"] in train_on_real]
    real_test = [r for r in real if r not in real_train]
    train += real_train
    if real_test:
        test = real_test
    else:
        test = cuad_test
        if not final:
            print("WARNING: no held-out real clauses, so test.jsonl contains CUAD contracts. It measures agreement "
                  "with CUAD-derived labels, not with your own labels.")
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--cuad-json", help="path to CUADv1.json (downloaded automatically if omitted)")
    ap.add_argument("--unannotated-per-contract", type=int, default=6)
    ap.add_argument("--max-per-class", type=int, default=4000)
    ap.add_argument("--train-on-real", default="", help="comma-separated sources of your own clauses to train on")
    ap.add_argument("--final", action="store_true", help="train on all of your clauses (no honest test set left)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    cuad_path = Path(args.cuad_json) if args.cuad_json else download_cuad(out_dir / "raw")
    contracts = load_cuad(cuad_path)
    check_mapping(contract_categories(contracts))

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for contract in contracts:
        rows.extend(contract_units(contract, args.unannotated_per_contract, rng))
    rows = cap_per_class(drop_conflicts(rows), args.max_per_class, rng)

    authored = read_label_csv(ROOT / "eval" / "clauses.csv", "authored")
    real = read_label_csv(ROOT / "eval" / "clauses_real.csv", "real")
    train_on_real = {s.strip() for s in args.train_on_real.split(",") if s.strip()}
    train, val, test = assemble(rows, authored, real, train_on_real, args.final, args.seed)

    for name, split in (("train", train), ("val", val), ("test", test)):
        write_jsonl(out_dir / f"{name}.jsonl", split)
    small = {name: describe(name, split) for name, split in (("train", train), ("val", val), ("test", test))}
    for name, classes in small.items():
        if classes:
            print(f"WARNING: {name} has fewer than {MIN_PER_CLASS} examples of: {', '.join(classes)}")
    print(f"Wrote {out_dir}/train.jsonl, val.jsonl and test.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
