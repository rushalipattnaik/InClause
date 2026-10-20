import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import label  # noqa: E402


def parse_ids(spec):
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return ids


def set_source(path, ids, name):
    label.upgrade_file(path)
    rows = label.read_existing(path)
    changed = 0
    for row in rows:
        if str(row["id"]).isdigit() and int(row["id"]) in ids:
            row["source"] = name
            changed += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=label.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main():
    ap = argparse.ArgumentParser(description="Record which contract a group of labelled clauses came from.")
    ap.add_argument("--ids", required=True, help="clause ids, for example 1-45 or 1-10,50,60-62")
    ap.add_argument("--name", required=True, help="a short name for the contract, for example rental_agreement")
    args = ap.parse_args()
    changed = set_source(label.REAL_PATH, parse_ids(args.ids), args.name)
    print(f"Set the source of {changed} clauses to {args.name}")


if __name__ == "__main__":
    main()
