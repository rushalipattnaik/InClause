# Results

Generated 2026-10-15 by `python -m eval.full_report`.

## In plain words

Your test set: 85 clauses you labelled from real contracts.

- **Original (substring rules)** matched your label on 59% of clauses, noticed 30% of the clauses you marked risky or suspicious, and 81% of the clauses it flagged were ones you also marked risky or suspicious.
- **Rules** matched your label on 74% of clauses, noticed 79% of the clauses you marked risky or suspicious, and 81% of the clauses it flagged were ones you also marked risky or suspicious.
- **ML (embeddings + logistic regression)** matched your label on 54% of clauses, noticed 67% of the clauses you marked risky or suspicious, and 72% of the clauses it flagged were ones you also marked risky or suspicious.
- **Hybrid (rules + ML)** matched your label on 68% of clauses, noticed 86% of the clauses you marked risky or suspicious, and 70% of the clauses it flagged were ones you also marked risky or suspicious.

Keep `CLASSIFIER=rules`: neither the ML model nor the hybrid beat it by a meaningful margin on your clauses.

Resume sentence you can use:

> Evaluated on 85 clauses I labelled from real contracts: the rule-based classifier matched my label on 74% of clauses and noticed 79% of the clauses I marked risky or suspicious.

### How to read the numbers

- **Matched your label**: the share of clauses where the tool gave the same answer as you (accuracy).
- **Noticed**: of the clauses you marked risky or suspicious, how many the tool flagged at all (recall).
- **Flags you agreed with**: of the clauses the tool flagged, how many you also marked risky or suspicious (precision).
- **Macro-F1**: one number from 0 to 1 that balances all three labels, higher is better.

## Authored set, sanity check only

These clauses and the rules were written by the same person, so this proves little.

### Authored set, rules only (sanity check) (91 clauses)

| System | Accuracy | Macro-F1 | Flag precision | Flag recall | Flag F1 |
|---|---|---|---|---|---|
| Original (substring rules) | 0.49 | 0.43 | 1.00 | 0.31 | 0.47 |
| Rules | 0.92 | 0.92 | 0.95 | 0.97 | 0.96 |

'Flag' = Risky or Suspicious vs Safe (did the tool notice a clause that needs attention?).

## Classifier comparison

### Authored set, 5-fold cross-validation (the ML model never scores a clause it trained on) (91 clauses)

| System | Accuracy | Macro-F1 | Flag precision | Flag recall | Flag F1 |
|---|---|---|---|---|---|
| Original (substring rules) | 0.49 | 0.43 | 1.00 | 0.31 | 0.47 |
| Rules | 0.92 | 0.92 | 0.95 | 0.97 | 0.96 |
| ML (embeddings + logistic regression) | 0.55 | 0.54 | 0.80 | 0.80 | 0.80 |
| Hybrid (rules + ML) | 0.82 | 0.82 | 0.82 | 0.98 | 0.89 |

'Flag' = Risky or Suspicious vs Safe (did the tool notice a clause that needs attention?).

### YOUR real clauses (model trained on the authored set only) (85 clauses)

| System | Accuracy | Macro-F1 | Flag precision | Flag recall | Flag F1 |
|---|---|---|---|---|---|
| Original (substring rules) | 0.59 | 0.51 | 0.81 | 0.30 | 0.44 |
| Rules | 0.74 | 0.74 | 0.81 | 0.79 | 0.80 |
| ML (embeddings + logistic regression) | 0.54 | 0.47 | 0.72 | 0.67 | 0.70 |
| Hybrid (rules + ML) | 0.68 | 0.70 | 0.70 | 0.86 | 0.77 |

'Flag' = Risky or Suspicious vs Safe (did the tool notice a clause that needs attention?).

**Original (substring rules): per-class precision / recall / F1**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.57 | 0.93 | 0.70 |
| Risky | 0.50 | 0.19 | 0.27 |
| Suspicious | 1.00 | 0.38 | 0.55 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | Safe | Risky | Suspicious |
|---|---|---|---|
| Safe | 39 | 3 | 0 |
| Risky | 22 | 5 | 0 |
| Suspicious | 8 | 2 | 6 |

**Rules: per-class precision / recall / F1**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.79 | 0.81 | 0.80 |
| Risky | 0.60 | 0.67 | 0.63 |
| Suspicious | 0.92 | 0.69 | 0.79 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | Safe | Risky | Suspicious |
|---|---|---|---|
| Safe | 34 | 8 | 0 |
| Risky | 8 | 18 | 1 |
| Suspicious | 1 | 4 | 11 |

**ML (embeddings + logistic regression): per-class precision / recall / F1**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.69 | 0.74 | 0.71 |
| Risky | 0.44 | 0.15 | 0.22 |
| Suspicious | 0.35 | 0.69 | 0.47 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | Safe | Risky | Suspicious |
|---|---|---|---|
| Safe | 31 | 2 | 9 |
| Risky | 12 | 4 | 11 |
| Suspicious | 2 | 3 | 11 |

**Hybrid (rules + ML): per-class precision / recall / F1**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Safe | 0.81 | 0.62 | 0.70 |
| Risky | 0.51 | 0.78 | 0.62 |
| Suspicious | 0.92 | 0.69 | 0.79 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | Safe | Risky | Suspicious |
|---|---|---|---|
| Safe | 26 | 16 | 0 |
| Risky | 5 | 21 | 1 |
| Suspicious | 1 | 4 | 11 |

## Retrieval

### Retrieval on 76 clauses with a matching knowledge-base entry

- hit@1 (right entry ranked first): 0.72
- hit@3 (right entry in top 3): 0.87

### Threshold sweep (k = 2, margin = 0.15); 15 clauses have NO relevant entry

| Min similarity | Right entry found | Shown notes that were the expected one | Nothing shown for unrelated clauses |
|---|---|---|---|
| 0.0 | 0.82 | 0.46 | 0.00 |
| 0.2 | 0.82 | 0.46 | 0.07 |
| 0.3 | 0.82 | 0.47 | 0.07 |
| 0.4 | 0.71 | 0.51 | 0.27 |
| 0.5 | 0.51 | 0.72 | 0.67 |
| 0.6 | 0.18 | 0.82 | 1.00 |

'Expected one' is strict: another note may also be relevant, so precision is a lower bound.

### Recommended settings (best average of found / precision / empty = 0.71)

Put these lines in your .env:

RAG_MIN_SIMILARITY=0.65
RAG_MARGIN=0.05

Tuned on the clauses and topic labels in eval/*.csv. Add your own labelled clauses and run again.
