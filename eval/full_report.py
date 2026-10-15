import argparse
import contextlib
import datetime
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import run_eval  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "RESULTS.md")
ENOUGH = 30
MIN_GAIN = 0.03


def capture(fn, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue().strip()


def percent(value):
    return f"{round(value * 100)}%"


def plain_lines(scores):
    lines = []
    for name, s in scores.items():
        precision, recall, _ = s["flag"]
        lines.append(
            f"- **{name}** matched your label on {percent(s['accuracy'])} of clauses, noticed {percent(recall)} of the "
            f"clauses you marked risky or suspicious, and {percent(precision)} of the clauses it flagged were ones "
            f"you also marked risky or suspicious."
        )
    return lines


def recommendation(scores, count):
    rules = scores.get("Rules")
    hybrid = scores.get("Hybrid (rules + ML)")
    ml = scores.get("ML (embeddings + logistic regression)")
    if not rules:
        return "Keep `CLASSIFIER=rules`."
    best_name, best = "Rules", rules
    for name, candidate in (("Hybrid", hybrid), ("ML", ml)):
        if candidate and candidate["macro_f1"] > best["macro_f1"] + MIN_GAIN:
            best_name, best = name, candidate
    if best_name == "Rules":
        text = "Keep `CLASSIFIER=rules`: neither the ML model nor the hybrid beat it by a meaningful margin on your clauses."
    else:
        setting = "hybrid" if best_name == "Hybrid" else "ml"
        text = (f"{best_name} beat the rules by more than {MIN_GAIN:.2f} macro-F1 on your clauses. "
                f"You can set `CLASSIFIER={setting}` in `.env`.")
    if count < ENOUGH:
        text += f" With only {count} clauses this can change a lot, so label at least {ENOUGH} before deciding."
    return text


def resume_line(scores, count, assisted):
    if count < ENOUGH or "Rules" not in scores:
        return None
    s = scores["Rules"]
    precision, recall, _ = s["flag"]
    line = (f"Evaluated on {count} clauses I labelled from real contracts: the rule-based classifier matched my label on "
            f"{percent(s['accuracy'])} of clauses and noticed {percent(recall)} of the clauses I marked risky or suspicious.")
    if assisted:
        line += f" ({assisted} of the {count} labels were accepted from an AI suggestion, so say so if asked.)"
    return line


def run_report(real_path=run_eval.REAL_PATH, authored_path=run_eval.CSV_PATH, out_path=OUT_PATH,
               fake=False, llm=False, tune=True):
    authored = run_eval.read_csv(authored_path)
    real = run_eval.read_csv(real_path)
    assisted = sum(r.get("assisted") == "yes" for r in real)
    today = datetime.date.today().isoformat()
    out = ["# Results", "", f"Generated {today} by `python -m eval.full_report`.", ""]
    if fake:
        out += ["> **Fake embeddings were used. This only checks that the report runs; the numbers mean nothing.**", ""]

    embeddings, _ = capture(run_eval.setup_embeddings, fake)
    scores, ml_text = capture(run_eval.report_ml, embeddings, fake, authored_path, real_path)
    enough_real = len(real) >= run_eval.MIN_REAL_FOR_ML and scores

    out += ["## In plain words", ""]
    if enough_real:
        out += [f"Your test set: {len(real)} clauses you labelled from real contracts"
                f"{f', {assisted} of them accepted from an AI suggestion' if assisted else ''}.", ""]
        out += plain_lines(scores)
        if llm:
            from eval import llm_classifier
            y_true = [r["label"] for r in real]
            llm_scores = {"Local LLM (zero-shot)": run_eval.score(
                y_true, [llm_classifier.classify_clause(r["text"]) for r in real])}
            out += plain_lines(llm_scores)
            scores.update(llm_scores)
        out += ["", recommendation(scores, len(real)), ""]
        if assisted and assisted / len(real) > 0.5:
            out += ["> More than half of the labels were accepted AI suggestions. Be careful comparing against the "
                    "local LLM, because it may simply agree with itself.", ""]
        line = resume_line(scores, len(real), assisted)
        if line:
            out += ["Resume sentence you can use:", "", f"> {line}", ""]
    else:
        out += [f"You have {len(real)} labelled real clauses and the comparison needs at least "
                f"{run_eval.MIN_REAL_FOR_ML} (about {ENOUGH} is better). Run `python -m eval.label your_contract.pdf` "
                f"and then run this report again.", ""]

    out += ["### How to read the numbers", "",
            "- **Matched your label**: the share of clauses where the tool gave the same answer as you (accuracy).",
            "- **Noticed**: of the clauses you marked risky or suspicious, how many the tool flagged at all (recall).",
            "- **Flags you agreed with**: of the clauses the tool flagged, how many you also marked risky or suspicious "
            "(precision).",
            "- **Macro-F1**: one number from 0 to 1 that balances all three labels, higher is better.", ""]

    rule_rows = run_eval.rule_predictions(authored)
    _, sanity = capture(run_eval.report_classifier, "Authored set, rules only (sanity check)", authored, rule_rows, False)
    out += ["## Authored set, sanity check only", "",
            "These clauses and the rules were written by the same person, so this proves little.", "", sanity, ""]
    out += ["## Classifier comparison", "", ml_text, ""]
    _, retrieval = capture(run_eval.report_retrieval, fake, tune, (authored_path, real_path))
    out += ["## Retrieval", "", retrieval, ""]

    markdown = "\n".join(out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return markdown


def main():
    ap = argparse.ArgumentParser(description="Run every evaluation and write docs/RESULTS.md in plain language.")
    ap.add_argument("--llm", action="store_true", help="also score the local LLM as a classifier (Ollama must be running)")
    ap.add_argument("--no-tune", action="store_true", help="skip the retrieval settings recommendation")
    ap.add_argument("--fake-embeddings", action="store_true", help="plumbing check only")
    args = ap.parse_args()
    markdown = run_report(fake=args.fake_embeddings, llm=args.llm, tune=not args.no_tune)
    start = markdown.index("## In plain words")
    end = markdown.index("### How to read the numbers")
    print(markdown[start:end].strip())
    print(f"\nFull report written to {OUT_PATH}")


if __name__ == "__main__":
    main()
