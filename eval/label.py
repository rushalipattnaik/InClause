import argparse
import csv
import json
import os
import random
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clause_splitter import split_into_clauses  # noqa: E402
from utils.extractor import extract_text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(HERE, "clauses_real.csv")
KB_PATH = os.path.join(os.path.dirname(HERE), "legal_data", "knowledge_base.json")
LABEL_KEYS = {"s": "Safe", "r": "Risky", "x": "Suspicious"}
FIELDS = ["id", "split", "label", "topic", "text", "assisted", "source"]


def read_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upgrade_file(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    if header == FIELDS:
        return
    rows = read_existing(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) or ("no" if key == "assisted" else "") for key in FIELDS})


def append_row(path, row):
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def ask_topic(topic_ids, input_fn, output_fn):
    while True:
        answer = input_fn("  topic id (Enter = unknown, n = none, ? = list): ").strip()
        if answer == "":
            return ""
        if answer.lower() == "n":
            return "none"
        if answer == "?":
            output_fn("  " + ", ".join(topic_ids))
        elif answer in topic_ids:
            return answer
        else:
            output_fn("  not a valid topic id, type ? to see the list")


def ask_label(suggestion, input_fn):
    accept = "[Enter] accept suggestion  " if suggestion else ""
    while True:
        key = input_fn(f"{accept}[s] safe  [r] risky  [x] suspicious  [k] skip  [q] quit > ").strip().lower()
        if key == "" and suggestion:
            return suggestion, "yes"
        if key in LABEL_KEYS:
            return LABEL_KEYS[key], "no"
        if key in ("k", "q"):
            return key, "no"


def label_clauses(clauses, path=REAL_PATH, topics=False, suggest_fn=None, source="", input_fn=input, output_fn=print):
    upgrade_file(path)
    existing = read_existing(path)
    done = {r["text"] for r in existing}
    next_id = max([int(r["id"]) for r in existing if str(r["id"]).isdigit()] + [0]) + 1
    topic_ids = [e["id"] for e in json.load(open(KB_PATH, encoding="utf-8"))] if topics else []
    todo = [c for c in clauses if c not in done]
    suggestions_on = suggest_fn is not None
    saved = 0
    for position, clause in enumerate(todo, 1):
        suggestion = None
        if suggestions_on:
            try:
                suggestion = suggest_fn(clause)
            except Exception:
                suggestions_on = False
                output_fn("AI suggestions are unavailable (is Ollama running?). Continuing without them.")
        output_fn(f"\n--- Clause {position} of {len(todo)} " + "-" * 30)
        output_fn(textwrap.fill(clause, width=100))
        if suggestion:
            output_fn(f"AI suggestion (decide for yourself): {suggestion}")
        choice, assisted = ask_label(suggestion, input_fn)
        if choice == "q":
            break
        if choice == "k":
            continue
        topic = ask_topic(topic_ids, input_fn, output_fn) if topics else ""
        append_row(path, {"id": next_id, "split": "real", "label": choice, "topic": topic,
                          "text": clause, "assisted": assisted, "source": source})
        next_id += 1
        saved += 1
    output_fn(f"\nSaved {saved} new labelled clauses to {path}")
    return saved


def main():
    ap = argparse.ArgumentParser(description="Label clauses from a real contract to build an honest test set.")
    ap.add_argument("document", help="PDF, DOCX or TXT file")
    ap.add_argument("--count", type=int, default=50, help="how many clauses to show (random sample)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topics", action="store_true", help="also record the knowledge-base topic for retrieval evaluation")
    ap.add_argument("--suggest", action="store_true",
                    help="show a local-AI suggestion you can accept with Enter (faster, but see docs/EVALUATION.md)")
    ap.add_argument("--model", help="Ollama model used for suggestions, for example mistral")
    args = ap.parse_args()

    text = extract_text(args.document)
    if text.startswith("Error:"):
        sys.exit(text)
    clauses = split_into_clauses(text)
    random.Random(args.seed).shuffle(clauses)

    suggest_fn = None
    if args.suggest:
        if args.model:
            os.environ["OLLAMA_MODEL"] = args.model
        from eval import llm_classifier
        suggest_fn = llm_classifier.classify_clause

    print("Label each clause from the point of view of the person signing.")
    print("Suspicious = removes a right, hides or uncaps a cost, or lets one side act alone.")
    print("Risky = a normal clause that still carries obligations, costs or penalties. Safe = neutral.")
    source = os.path.splitext(os.path.basename(args.document))[0]
    label_clauses(clauses[: args.count], topics=args.topics, suggest_fn=suggest_fn, source=source)


if __name__ == "__main__":
    main()
