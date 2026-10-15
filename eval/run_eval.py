import argparse
import csv
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


def report_classifier(title, rows, predictions, detail=True):
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
            print(f"\n**{name}: per-class precision / recall / F1**\n")
            print("| Class | Precision | Recall | F1 |\n|---|---|---|---|")
            for lab in LABELS:
                p, r, f = scores[name]["per_class"][lab]
                print(f"| {lab} | {p:.2f} | {r:.2f} | {f:.2f} |")
            cm = confusion(y_true, y_pred)
            print("\nConfusion matrix (rows = true, columns = predicted):\n")
            print("| true \\ pred | " + " | ".join(LABELS) + " |\n|---|---|---|---|")
            for t in LABELS:
                print(f"| {t} | " + " | ".join(str(cm[(t, p)]) for p in LABELS) + " |")
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


def hybrid_from(rows, probabilities):
    rules = [classifier.analyze_clause(r["text"]) for r in rows]
    return [ml_classifier.combine(rule, p)["label"] for rule, p in zip(rules, probabilities, strict=True)]


def report_ml(embeddings, fake, authored_path=CSV_PATH, real_path=REAL_PATH):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    authored = read_csv(authored_path)
    X = ml_classifier.embed(embeddings, [r["text"] for r in authored])
    y = [r["label"] for r in authored]
    smallest = min(y.count(lab) for lab in set(y))
    folds = max(2, min(5, smallest))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    matrix = cross_val_predict(ml_classifier.new_estimator(), X, y, cv=cv, method="predict_proba")
    probs = ml_classifier.to_prob_dicts(sorted(set(y)), matrix)
    predictions = rule_predictions(authored)
    predictions["ML (embeddings + logistic regression)"] = [ml_classifier.ml_only(p)["label"] for p in probs]
    predictions["Hybrid (rules + ML)"] = hybrid_from(authored, probs)
    report_classifier(f"Authored set, {folds}-fold cross-validation (the ML model never scores a clause it trained on)",
                      authored, predictions, detail=False)

    real = read_csv(real_path)
    if len(real) < MIN_REAL_FOR_ML:
        print(f"\nAdd at least {MIN_REAL_FOR_ML} of your own labelled clauses to eval/clauses_real.csv "
              f"(python -m eval.label your_contract.pdf) to see the honest comparison. You have {len(real)}.")
        return
    model = ml_classifier.EmbeddingClassifier(embeddings, authored)
    probs = model.predict_proba_many([r["text"] for r in real])
    predictions = rule_predictions(real)
    predictions["ML (embeddings + logistic regression)"] = [ml_classifier.ml_only(p)["label"] for p in probs]
    predictions["Hybrid (rules + ML)"] = hybrid_from(real, probs)
    return report_classifier("YOUR real clauses (model trained on the authored set only)", real, predictions)


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
        best = None
        for thr in [round(0.20 + 0.05 * i, 2) for i in range(11)]:
            for margin in (0.05, 0.10, 0.15, 0.20, 0.30, 1.0):
                found, prec, empty = evaluate_setting(rows, top_k_hits, thr, margin)
                parts = [found, prec] + ([empty] if empty is not None else [])
                value = sum(parts) / len(parts)
                if best is None or value > best[0]:
                    best = (value, thr, margin, found, prec, empty)
        value, thr, margin, found, prec, empty = best
        print(f"\n### Recommended settings (best average of found / precision / empty = {value:.2f})\n")
        print(f"Put these lines in your .env:\n\nRAG_MIN_SIMILARITY={thr}\nRAG_MARGIN={margin}\n")
        print("Tuned on the clauses and topic labels in eval/*.csv. Add your own labelled clauses and run again.")
    if fake:
        print("\n(!) Fake embeddings: these settings are meaningless.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "real", "all"],
                    help="dev/test/all: eval/clauses.csv; real: your own clauses in eval/clauses_real.csv")
    ap.add_argument("--llm", action="store_true", help="also score the local LLM as a classifier (Ollama must be running)")
    ap.add_argument("--ml", action="store_true", help="compare rules, the embedding classifier and the hybrid")
    ap.add_argument("--retrieval", action="store_true", help="evaluate retrieval and sweep the similarity threshold")
    ap.add_argument("--tune", action="store_true", help="with --retrieval: recommend RAG_MIN_SIMILARITY and RAG_MARGIN")
    ap.add_argument("--fake-embeddings", action="store_true", help="plumbing check only; numbers are not meaningful")
    args = ap.parse_args()

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
