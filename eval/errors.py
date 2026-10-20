import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval  # noqa: E402
from utils import classifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "ERRORS.md")
MAX_CHARS = 400

TITLES = {
    ("Safe", "Risky"): "False alarms: the tool said Risky, you said Safe",
    ("Safe", "Suspicious"): "False alarms: the tool said Suspicious, you said Safe",
    ("Risky", "Safe"): "Misses: the tool said Safe, you said Risky",
    ("Risky", "Suspicious"): "Over-rated: the tool said Suspicious, you said Risky",
    ("Suspicious", "Safe"): "Misses: the tool said Safe, you said Suspicious",
    ("Suspicious", "Risky"): "Under-rated: the tool said Risky, you said Suspicious",
}


def collect(rows):
    errors = []
    for row in rows:
        result = classifier.analyze_clause(row["text"])
        if result["label"] != row["label"]:
            errors.append({"id": row.get("id", ""), "text": row["text"], "true": row["label"],
                           "pred": result["label"], "reasons": result["reasons"]})
    return errors


def build_markdown(rows):
    errors = collect(rows)
    counts = Counter((e["true"], e["pred"]) for e in errors)
    out = ["# Where the rules disagree with you", "",
           f"{len(errors)} of {len(rows)} clauses disagree. This file contains contract text, so it is git-ignored.", "",
           "| You said | Tool said | Clauses |", "|---|---|---|"]
    for (true, pred), n in counts.most_common():
        out.append(f"| {true} | {pred} | {n} |")
    out.append("")
    for key, title in TITLES.items():
        group = [e for e in errors if (e["true"], e["pred"]) == key]
        if not group:
            continue
        out += [f"## {title} ({len(group)})", ""]
        for e in group:
            text = e["text"] if len(e["text"]) <= MAX_CHARS else e["text"][:MAX_CHARS] + "..."
            out.append(f"- **#{e['id']}** {text}")
            why = "; ".join(f"{r['reason']} (\"{r['matched']}\")" for r in e["reasons"]) or "no rule matched"
            out.append(f"  - Tool's reason: {why}")
        out.append("")
    return "\n".join(out), counts


def main():
    ap = argparse.ArgumentParser(description="List the clauses where the rule-based classifier disagrees with your labels.")
    ap.add_argument("--split", default="real", choices=["dev", "test", "real", "all"])
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    rows = run_eval.load_rows(args.split)
    if not rows:
        sys.exit("No labelled clauses found. Run python -m eval.label first.")
    markdown, counts = build_markdown(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"{sum(counts.values())} of {len(rows)} clauses disagree. Details written to {args.out}")
    for (true, pred), n in counts.most_common():
        print(f"  you said {true}, tool said {pred}: {n}")


if __name__ == "__main__":
    main()
