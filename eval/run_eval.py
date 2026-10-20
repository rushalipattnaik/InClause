import argparse
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import legacy_classifier  # noqa: E402
from utils import classifier, ml_classifier  # noqa: E402

LABELS = ["Safe", "Risky", "Suspicious"]
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "clauses.csv")
REAL_PATH = os.path.join(HERE, "clauses_real.csv")
MIN_REAL_FOR_ML = 10
ROOT = os.path.dirname(HERE)
TRAIN_JSONL = os.path.join(ROOT, "data", "train.jsonl")
VAL_JSONL = os.path.join(ROOT, "data", "val.jsonl")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rows(split):
    if split == "real":
        return read_csv(REAL_PATH)
    rows = read_csv(CSV_PATH)
    return rows if split == "all" else [r for r in rows if r["split"] == split]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(y_true, y_pred):
    per_class, f1s = {}, []
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        per_class[lab] = prf(tp, fp, fn)
        f1s.append(per_class[lab][2])
    accuracy = sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)
    tp = sum(t != "Safe" and p != "Safe" for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == "Safe" and p != "Safe" for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t != "Safe" and p == "Safe" for t, p in zip(y_true, y_pred, strict=True))
    return {"per_class": per_class, "macro_f1": sum(f1s) / len(f1s), "accuracy": accuracy, "flag": prf(tp, fp, fn)}


def confusion(y_true, y_pred):
    return {(t, p): sum(a == t and b == p for a, b in zip(y_true, y_pred, strict=True)) for t in LABELS for p in LABELS}


def print_detail(name, y_true, y_pred, s):
    print(f"\n**{name}: per-class precision / recall / F1**\n")
    print("| Class | Precision | Recall | F1 |\n|---|---|---|---|")
    for lab in LABELS:
        p, r, f = s["per_class"][lab]
        print(f"| {lab} | {p:.2f} | {r:.2f} | {f:.2f} |")
    cm = confusion(y_true, y_pred)
    print("\nConfusion matrix (rows = true, columns = predicted):\n")
    print("| true \\ pred | " + " | ".join(LABELS) + " |\n|---|---|---|---|")
    for t in LABELS:
        print(f"| {t} | " + " | ".join(str(cm[(t, p)]) for p in LABELS) + " |")


def report_classifier(title, rows, predictions, detail=True, detail_for=None):
    y_true = [r["label"] for r in rows]
    print(f"\n### {title} ({len(rows)} clauses)\n")
    print("| System | Accuracy | Macro-F1 | Flag precision | Flag recall | Flag F1 |")
    print("|---|---|---|---|---|---|")
    scores = {}
    for name, y_pred in predictions.items():
        s = scores[name] = score(y_true, y_pred)
        fp_, fr_, ff_ = s["flag"]
        print(f"| {name} | {s['accuracy']:.2f} | {s['macro_f1']:.2f} | {fp_:.2f} | {fr_:.2f} | {ff_:.2f} |")
    print("\n'Flag' = Risky or Suspicious vs Safe (did the tool notice a clause that needs attention?).")
    if detail:
        for name, y_pred in predictions.items():
            if detail_for is None or name in detail_for:
                print_detail(name, y_true, y_pred, scores[name])
    return scores


def rule_predictions(rows):
    texts = [r["text"] for r in rows]
    return {"Original (substring rules)": [legacy_classifier.classify_clause(t) for t in texts],
            "Rules": [classifier.classify_clause(t) for t in texts]}


def setup_embeddings(fake):
    from utils import rag_engine
    if fake:
        from eval.fake_embeddings import HashingEmbeddings
        embeddings = HashingEmbeddings()
        rag_engine.set_embeddings(embeddings, index_dir=tempfile.mkdtemp(), tag="fake")
        print("\n(!) Using fake bag-of-words embeddings: this only checks the plumbing, the numbers mean nothing.")
        return embeddings
    return rag_engine.get_embeddings()


def hybrid_from(rows, probabilities, kind=None):
    rules = [classifier.analyze_clause(r["text"]) for r in rows]
    return [ml_classifier.combine(rule, p, kind=kind)["label"] for rule, p in zip(rules, probabilities, strict=True)]


def add_model_predictions(predictions, kind, rows, probabilities):
    predictions[f"ML ({kind})"] = [ml_classifier.ml_only(p, kind=kind)["label"] for p in probabilities]
    predictions[f"Hybrid ({kind})"] = hybrid_from(rows, probabilities, kind)


def legalbert_probabilities(rows):
    """Run the fine-tuned Legal-BERT on rows, or return None (with a note) if it is not trained or cannot run."""
    from utils import legal_classifier

    if not legal_classifier.model_available():
        print("\nNo trained Legal-BERT model was found, so it is not part of this comparison. "
              f"Train it with: {legal_classifier.TRAIN_COMMANDS}")
        return None
    try:
        model = legal_classifier.LegalBertRiskClassifier()
        return ml_classifier.to_prob_dicts(legal_classifier.LABELS, model.predict_proba([r["text"] for r in rows]))
    except Exception as e:
        print(f"\nLegal-BERT could not be run ({type(e).__name__}: {e}), so it is skipped.")
        return None


def training_overlap(rows, train_path=TRAIN_JSONL):
    """How many of these clauses appear in Legal-BERT's training file, which would inflate its score."""
    if not os.path.exists(train_path):
        return 0
    with open(train_path, encoding="utf-8") as f:
        seen = {" ".join(json.loads(line)["text"].split()) for line in f if line.strip()}
    return sum(" ".join(r["text"].split()) in seen for r in rows)


def tune_hybrid(rows, probabilities, kind=None):
    """Grid-search the hybrid thresholds for the best macro-F1. Returns (macro_f1, flag_min, safe_min)."""
    rules = [classifier.analyze_clause(r["text"]) for r in rows]
    y_true = [r["label"] for r in rows]
    grid = [round(0.50 + 0.05 * i, 2) for i in range(10)][::-1]
    best = (-1.0, None, None)
    for flag in grid:
        for safe in grid:
            predicted = [ml_classifier.combine(rule, p, flag_min=flag, safe_min=safe, kind=kind)["label"]
                         for rule, p in zip(rules, probabilities, strict=True)]
            value = score(y_true, predicted)["macro_f1"]
            if value > best[0] + 1e-9:
                best = (value, flag, safe)
    return best


def report_ml(embeddings, fake, authored_path=CSV_PATH, real_path=REAL_PATH):
    kinds = [k for k in ml_classifier.KINDS if k != "embed" or embeddings is not None]
    if embeddings is None:
        print("\nThe embedding model is not available, so only the word-based models are compared.")
    authored = read_csv(authored_path)
    real = read_csv(real_path)

    predictions = rule_predictions(authored)
    for kind in kinds:
        probs = ml_classifier.cross_val_probs(kind, authored, embeddings=embeddings)
        add_model_predictions(predictions, kind, authored, probs)
    report_classifier("Authored set, cross-validated (a model never scores a clause it trained on)",
                      authored, predictions, detail=False)

    if len(real) < MIN_REAL_FOR_ML:
        print(f"\nAdd at least {MIN_REAL_FOR_ML} of your own labelled clauses to eval/clauses_real.csv "
              f"(python -m eval.label your_contract.pdf) to see the honest comparison. You have {len(real)}.")
        return None

    legalbert = legalbert_probabilities(real)
    if legalbert is not None:
        print("\nLegal-BERT is scored as trained by scripts/train_legal_bert.py. It is not cross-validated, so the same "
              "predictions appear in both of the tables below.")
        leaked = training_overlap(real)
        if leaked:
            print(f"WARNING: {leaked} of your {len(real)} clauses are in Legal-BERT's training data, so its score is "
                  "inflated. Re-run scripts.prepare_training_data without --final or --train-on-real.")
    predictions = rule_predictions(real)
    for kind in kinds:
        model = ml_classifier.TextClassifier(kind, authored, embeddings)
        add_model_predictions(predictions, kind, real, model.predict_proba_many([r["text"] for r in real]))
    if legalbert is not None:
        add_model_predictions(predictions, "legalbert", real, legalbert)
    report_classifier("YOUR real clauses, models trained on the authored set only", real, predictions, detail=False)

    groups = [r.get("source") or None for r in real]
    strict = all(groups) and len(set(groups)) >= 2
    if strict:
        print(f"\nEach test fold is a whole contract the model never saw ({len(set(groups))} contracts).")
    else:
        print("\nSources are not recorded for your clauses, so clauses from the same contract can sit in both training "
              "and test folds and the model numbers may be optimistic. Run python -m eval.set_source to record them.")
    predictions = rule_predictions(real)
    for kind in kinds:
        probs = ml_classifier.cross_val_probs(kind, real, extra_rows=authored, embeddings=embeddings,
                                              groups=groups if strict else None)
        add_model_predictions(predictions, kind, real, probs)
    if legalbert is not None:
        add_model_predictions(predictions, "legalbert", real, legalbert)
    scores = report_classifier("YOUR real clauses, cross-validated (trained on the authored set plus your other clauses)",
                               real, predictions, detail=False)
    y_true = [r["label"] for r in real]
    shown = ["Rules"]
    for prefix in ("ML (", "Hybrid ("):
        names = [n for n in scores if n.startswith(prefix)]
        if names:
            shown.append(max(names, key=lambda n: scores[n]["macro_f1"]))
    for name in shown:
        print_detail(name, y_true, predictions[name], scores[name])
    return scores


def scored_hits(rows, k):
    from utils import rag_engine
    out = []
    for r in rows:
        hits = rag_engine.retrieve_reference(r["text"], k=k, min_similarity=-1.0, margin=1.0)
        out.append([(h["id"], h["score"]) for h in hits])
    return out


def apply_filter(hits, threshold, margin):
    best = max((s for _, s in hits), default=0.0)
    return [i for i, s in hits if s >= threshold and s >= best - margin]


def retrieval_rows(paths=(CSV_PATH, REAL_PATH)):
    rows = [r for path in paths for r in read_csv(path)]
    return [r for r in rows if r.get("topic")]


def evaluate_setting(rows, hits, threshold, margin):
    with_topic = [(r, h) for r, h in zip(rows, hits, strict=True) if r["topic"] != "none"]
    none_rows = [(r, h) for r, h in zip(rows, hits, strict=True) if r["topic"] == "none"]
    found = shown = correct = 0
    for r, h in with_topic:
        got = apply_filter(h, threshold, margin)
        found += r["topic"] in got
        shown += len(got)
        correct += got.count(r["topic"])
    empty = sum(not apply_filter(h, threshold, margin) for _, h in none_rows)
    return (found / len(with_topic) if with_topic else 0.0,
            correct / shown if shown else 0.0,
            empty / len(none_rows) if none_rows else None)


def harmonic_mean(values):
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def best_setting(rows, top_k_hits):
    best = (-1.0, 0.0, 1.0)
    for thr in [round(0.20 + 0.05 * i, 2) for i in range(11)]:
        for margin in (0.05, 0.10, 0.15, 0.20, 0.30, 1.0):
            found, prec, empty = evaluate_setting(rows, top_k_hits, thr, margin)
            value = harmonic_mean([found, prec] + ([empty] if empty is not None else []))
            if value > best[0]:
                best = (value, thr, margin)
    return best


def report_retrieval(fake, tune, paths=(CSV_PATH, REAL_PATH)):
    from utils import rag_engine
    rows = retrieval_rows(paths)
    if not rows:
        print("\nNo rows with a knowledge-base topic to evaluate retrieval on.")
        return
    k = rag_engine.TOP_K
    hits = scored_hits(rows, max(k, 3))
    with_topic = [(r, h) for r, h in zip(rows, hits, strict=True) if r["topic"] != "none"]
    hit1 = sum(r["topic"] in [i for i, _ in h[:1]] for r, h in with_topic) / len(with_topic)
    hit3 = sum(r["topic"] in [i for i, _ in h[:3]] for r, h in with_topic) / len(with_topic)
    none_count = sum(r["topic"] == "none" for r in rows)
    print(f"\n### Retrieval on {len(with_topic)} clauses with a matching knowledge-base entry\n")
    print(f"- hit@1 (right entry ranked first): {hit1:.2f}")
    print(f"- hit@3 (right entry in top 3): {hit3:.2f}\n")
    print(f"### Threshold sweep (k = {k}, margin = {rag_engine.MARGIN}); {none_count} clauses have NO relevant entry\n")
    print("| Min similarity | Right entry found | Shown notes that were the expected one | Nothing shown for unrelated clauses |")
    print("|---|---|---|---|")
    top_k_hits = [h[:k] for h in hits]
    for thr in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
        found, prec, empty = evaluate_setting(rows, top_k_hits, thr, rag_engine.MARGIN)
        print(f"| {thr:.1f} | {found:.2f} | {prec:.2f} | {'n/a' if empty is None else f'{empty:.2f}'} |")
    print("\n'Expected one' is strict: another note may also be relevant, so precision is a lower bound.")
    if tune:
        value, thr, margin = best_setting(rows, top_k_hits)
        print(f"\n### Recommended settings (balance of found / precision / empty = {value:.2f})\n")
        print(f"Put these lines in your .env:\n\nRAG_MIN_SIMILARITY={thr}\nRAG_MARGIN={margin}\n")
        print("The balance is a harmonic mean, so a setting that hides almost every note cannot win.")
        print("Tuned on the clauses and topic labels in eval/*.csv. Add your own labelled clauses and run again.")
    if fake:
        print("\n(!) Fake embeddings: these settings are meaningless.")


def run_tune_hybrid(val_path=VAL_JSONL):
    rows = read_jsonl_rows(val_path)
    if not rows:
        sys.exit(f"No validation data in {val_path}. Run: python -m scripts.prepare_training_data")
    probabilities = legalbert_probabilities(rows)
    if probabilities is None:
        sys.exit("A trained Legal-BERT model is needed to tune the hybrid thresholds.")
    defaults = ml_classifier.thresholds("legalbert")
    default_f1 = score([r["label"] for r in rows], hybrid_from(rows, probabilities, "legalbert"))["macro_f1"]
    value, flag, safe = tune_hybrid(rows, probabilities, "legalbert")
    print(f"\nValidation clauses: {len(rows)} (CUAD-derived labels plus the authored set, none of your real clauses)")
    print(f"Hybrid macro-F1 with the defaults {defaults}: {default_f1:.3f}")
    print(f"Best on the validation split: {value:.3f} with flag_min={flag}, safe_min={safe}\n")
    print(f"Put these lines in your .env:\n\nHYBRID_FLAG_MIN={flag}\nHYBRID_SAFE_MIN={safe}\n")
    print("These were tuned on validation data. Measure the final result on your own held-out clauses, not on these.")


def read_jsonl_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "real", "all"],
                    help="dev/test/all: eval/clauses.csv; real: your own clauses in eval/clauses_real.csv")
    ap.add_argument("--llm", action="store_true", help="also score the local LLM as a classifier (Ollama must be running)")
    ap.add_argument("--ml", action="store_true", help="compare rules, the embedding classifier and the hybrid")
    ap.add_argument("--retrieval", action="store_true", help="evaluate retrieval and sweep the similarity threshold")
    ap.add_argument("--tune", action="store_true", help="with --retrieval: recommend RAG_MIN_SIMILARITY and RAG_MARGIN")
    ap.add_argument("--tune-hybrid", action="store_true",
                    help="recommend HYBRID_FLAG_MIN and HYBRID_SAFE_MIN for Legal-BERT using data/val.jsonl")
    ap.add_argument("--fake-embeddings", action="store_true", help="plumbing check only; numbers are not meaningful")
    args = ap.parse_args()

    if args.tune_hybrid:
        return run_tune_hybrid()
    if args.ml or args.retrieval or args.tune:
        embeddings = setup_embeddings(args.fake_embeddings)
    if args.ml:
        report_ml(embeddings, args.fake_embeddings)
        if not (args.llm or args.retrieval or args.tune):
            return
    if args.retrieval or args.tune:
        report_retrieval(args.fake_embeddings, args.tune)
        if not args.llm:
            return

    rows = load_rows(args.split)
    if not rows:
        sys.exit("No rows in this split. For --split real, add labelled clauses to eval/clauses_real.csv.")
    print(f"Split: {args.split} ({len(rows)} clauses)")
    predictions = rule_predictions(rows)
    if args.llm:
        from eval import llm_classifier
        predictions["Local LLM (zero-shot)"] = [llm_classifier.classify_clause(r["text"]) for r in rows]
    report_classifier("Classifier", rows, predictions)


if __name__ == "__main__":
    main()
