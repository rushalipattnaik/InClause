"""Fine-tune Legal-BERT as a Safe / Risky / Suspicious clause classifier.

Reads data/train.jsonl and data/val.jsonl (made by scripts/prepare_training_data.py), trains with class-weighted loss,
keeps the epoch with the best validation macro-F1, and saves it to models/legal-bert-risk.

Run: python -m scripts.train_legal_bert            (add --dry-run to only check the data and the plan)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import confusion, score  # noqa: E402
from utils.legal_classifier import CONFIG_NAME, DEFAULT_MAX_LENGTH, LABELS, default_model_dir  # noqa: E402

BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
REPORT_PATH = ROOT / "eval" / "train_report.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m scripts.prepare_training_data")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_weights(labels: list[str]) -> list[float]:
    """Inverse-frequency weights in LABELS order: n / (k * count). Classes that never occur get weight 0."""
    counts = Counter(labels)
    total, k = len(labels), len(LABELS)
    return [total / (k * counts[label]) if counts[label] else 0.0 for label in LABELS]


def sample_weights(rows: list[dict], cuad_weight: float) -> list[float]:
    """CUAD rows carry noisy, category-derived labels, so they count for less than clauses you labelled yourself."""
    return [cuad_weight if str(r.get("source", "")).startswith("cuad:") else 1.0 for r in rows]


def metrics_report(y_true: list[str], y_pred: list[str]) -> dict:
    s = score(y_true, y_pred)
    matrix = confusion(y_true, y_pred)
    return {
        "accuracy": s["accuracy"],
        "macro_f1": s["macro_f1"],
        "per_class": {lab: {"precision": p, "recall": r, "f1": f, "support": y_true.count(lab)}
                      for lab, (p, r, f) in s["per_class"].items()},
        "confusion": {t: {p: matrix[(t, p)] for p in LABELS} for t in LABELS},
    }


def training_sources(rows: list[dict]) -> dict[str, int]:
    """Counts of non-CUAD sources, saved with the model so the evaluation can detect leakage."""
    return dict(Counter(r["source"] for r in rows if not str(r["source"]).startswith("cuad:")))


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear warm-up to 1.0, then linear decay to 0.0 at the last step."""
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def plan_text(train_rows: list[dict], val_rows: list[dict], args: argparse.Namespace) -> str:
    counts = Counter(r["label"] for r in train_rows)
    weights = class_weights([r["label"] for r in train_rows])
    steps = math.ceil(len(train_rows) / args.batch_size) * args.epochs
    lines = [f"train: {len(train_rows)} clauses ({', '.join(f'{lab} {counts[lab]}' for lab in LABELS)})",
             f"val:   {len(val_rows)} clauses",
             "class weights: " + ", ".join(f"{lab} {w:.2f}" for lab, w in zip(LABELS, weights, strict=True)),
             f"{args.epochs} epochs x batch {args.batch_size} = {steps} optimiser steps, max length {args.max_length}",
             f"CUAD rows count {args.cuad_weight:g}x, your own rows 1x"]
    return "\n".join(lines)


def predict_labels(model, tokenizer, rows: list[dict], device: str, max_length: int, batch_size: int) -> list[str]:
    import torch

    model.eval()
    out: list[str] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            texts = [r["text"] for r in rows[start:start + batch_size]]
            encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                                return_tensors="pt").to(device)
            out.extend(LABELS[i] for i in model(**encoded).logits.argmax(dim=-1).tolist())
    return out


def train(args: argparse.Namespace, train_rows: list[dict], val_rows: list[dict]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    label_ids = {label: i for i, label in enumerate(LABELS)}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=label_ids).to(device)

    weights = torch.tensor(class_weights([r["label"] for r in train_rows]), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    per_row_weight = sample_weights(train_rows, args.cuad_weight)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))

    output_dir = Path(args.output_dir)
    best_f1, best_epoch, history, step, started = -1.0, 0, [], 0, time.time()
    y_val = [r["label"] for r in val_rows]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            chosen = order[start:start + args.batch_size]
            encoded = tokenizer([train_rows[i]["text"] for i in chosen], truncation=True, max_length=args.max_length,
                                padding=True, return_tensors="pt").to(device)
            labels = torch.tensor([label_ids[train_rows[i]["label"]] for i in chosen], device=device)
            row_weight = torch.tensor([per_row_weight[i] for i in chosen], dtype=torch.float, device=device)
            per_row_loss = loss_fn(model(**encoded).logits, labels) * row_weight
            loss = per_row_loss.sum() / (weights[labels] * row_weight).sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.detach().item()
            step += 1
            if step % 20 == 0 or step == total_steps:
                elapsed = time.time() - started
                print(f"epoch {epoch}/{args.epochs} step {step}/{total_steps} loss {running / (start // args.batch_size + 1):.4f}"
                      f" elapsed {elapsed / 60:.1f} min, about {elapsed / step * (total_steps - step) / 60:.1f} min left")
        report = metrics_report(y_val, predict_labels(model, tokenizer, val_rows, device, args.max_length, args.batch_size))
        history.append({"epoch": epoch, "val_macro_f1": report["macro_f1"], "val_accuracy": report["accuracy"]})
        print(f"epoch {epoch}: validation macro-F1 {report['macro_f1']:.3f}, accuracy {report['accuracy']:.3f}")
        if report["macro_f1"] > best_f1:
            best_f1, best_epoch, best_report = report["macro_f1"], epoch, report
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            (output_dir / CONFIG_NAME).write_text(json.dumps({
                "labels": LABELS, "max_length": args.max_length, "base_model": args.base_model, "epoch": epoch,
                "val_macro_f1": report["macro_f1"], "train_sources": training_sources(train_rows)}, indent=2),
                encoding="utf-8")
    return {"best_epoch": best_epoch, "val": best_report, "history": history}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--output-dir", default=str(default_model_dir()))
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--cuad-weight", type=float, default=0.5)
    ap.add_argument("--max-train-samples", type=int, default=0, help="train on a random subset (faster on a CPU)")
    ap.add_argument("--device", default="", help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="check the data and print the plan without training")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    if args.max_train_samples and len(train_rows) > args.max_train_samples:
        train_rows = random.Random(args.seed).sample(train_rows, args.max_train_samples)
    print(plan_text(train_rows, val_rows, args))
    missing = [lab for lab in LABELS if not any(r["label"] == lab for r in train_rows)]
    if missing:
        print(f"ERROR: the training data has no examples of {missing}")
        return 1
    if args.dry_run:
        print("Dry run: nothing was trained.")
        return 0

    result = train(args, train_rows, val_rows)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({**result, "settings": {k: v for k, v in vars(args).items()}}, indent=2),
                           encoding="utf-8")
    print(f"Best epoch {result['best_epoch']}, validation macro-F1 {result['val']['macro_f1']:.3f}")
    print(f"Model saved to {args.output_dir}; validation report written to {report_path}")
    print("Next: python -m eval.full_report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
