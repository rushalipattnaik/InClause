# Evaluation

## Quick start

1. `python -m eval.label my_contract.pdf --count 50`. You see one clause at a time and press `s` (safe), `r` (risky)
   or `x` (suspicious). It takes about 15 minutes for 50 clauses. Use a few different contracts if you can.
2. `python -m eval.full_report`. It runs every comparison and writes `docs/RESULTS.md`, with the numbers explained in
   plain words, a recommendation on which classifier to use, and (once you have 30 or more clauses) a sentence you can
   use on a resume.

Why can't the tool just report an accuracy by itself? Accuracy means "how often the answer matches the correct answer".
A person has to decide the correct answer, because if the tool marked its own work the score would be meaningless.

### Faster, with a caveat
`python -m eval.label my_contract.pdf --suggest --model mistral` shows a local-AI suggestion for each clause, and Enter
accepts it. This is quicker, but people tend to accept suggestions without really deciding, which makes the score look
better than it is. Every accepted suggestion is recorded in the `assisted` column, and the report states how many there
were and warns when more than half were accepted. For a number you can defend, label at least the first 20 blind.

## What is in `eval/`

| File | Purpose |
|---|---|
| `clauses.csv` | 91 clauses (leases, employment, terms of service, loans, NDAs, sales) labelled Safe, Risky or Suspicious, with the knowledge-base note that should be retrieved. Written by the project author together with the rules |
| `clauses_real.csv` | Clauses from real contracts, labelled by you with `python -m eval.label`. Empty until you add some |
| `label.py` | Blind labelling tool: shows a clause, never the tool's prediction |
| `run_eval.py` | Metrics, comparison of classifiers, retrieval sweep and tuning |
| `legacy_classifier.py` | The original substring rules, kept as the baseline |
| `llm_classifier.py` | The local LLM asked to assign the same labels (needs Ollama) |

Labelling rubric, from the point of view of the person signing: **Suspicious** removes a right, hides or uncaps a cost,
or lets one side act alone. **Risky** is a normal clause that still carries obligations, costs or penalties.
**Safe** is neutral or informational.

## Other commands

```
python -m eval.full_report --llm               # also score a zero-shot local LLM (Ollama must be running)
python -m eval.run_eval --split all            # rules vs the original baseline on the authored set
python -m eval.run_eval --ml                   # classifier comparison only
python -m eval.run_eval --retrieval --tune     # retrieval quality, and recommended RAG_MIN_SIMILARITY and RAG_MARGIN
```
Adding `--fake-embeddings` runs any of these with a bag-of-words stand-in so you can check the plumbing without
downloading a model. Its numbers are meaningless.

## Protocols

1. **Authored set, 5-fold cross-validation.** The ML model never scores a clause it trained on. The rules were written
   with these clauses in view, so their score here is optimistic.
2. **Your real clauses.** The ML model trains on the authored set only and is scored on `clauses_real.csv`. This is the
   comparison to trust, and the only one worth quoting.

## Sanity check on the authored set

`python -m eval.run_eval --split all`

| System | Accuracy | Macro-F1 | Flag precision | Flag recall |
|---|---|---|---|---|
| Original substring rules | 0.49 | 0.43 | 1.00 | 0.31 |
| Rules | 0.92 | 0.92 | 0.95 | 0.97 |

"Flag" means Risky or Suspicious versus Safe. These figures only show that the rules cover the vocabulary in this set.
They are not an accuracy claim, because the same person wrote the clauses and the rules.

## Results on real contracts

`python -m eval.full_report` writes them to `docs/RESULTS.md`. Commit that file once it contains numbers from your own
labelled clauses, and state how many clauses you labelled and how many labels were AI-suggested.

## Regression tests from a real document

Clauses from a real rental agreement that were once flagged wrongly (an owner paying a fee, "successors and assigns"
in a party definition, a refundable deposit, handing back a flat subject to normal wear and tear) are locked in as unit
tests in `tests/test_classifier.py` and `tests/test_real_lease.py`. They are regression tests, not an evaluation: the
rules were adjusted after seeing them.

## Known limitations

- Rules and the ML model judge words and embeddings, not legal meaning.
- Flags assume the reader is the weaker party. There is no switch for the other side.
- A small local model can explain a clause wrongly. The prompt forbids invented citations and a guard removes sentences
  that cite a law absent from the clause and notes, but plain factual mistakes are not caught.
- The legal notes are short summaries and every one starts as "not yet reviewed". Check them against the primary
  sources with [KNOWLEDGE_BASE_REVIEW.md](KNOWLEDGE_BASE_REVIEW.md) before relying on them.
- Scanned PDFs have no text layer and are not supported.
- `langchain-community` is being retired upstream. FAISS still works, but moving to a standalone integration is future work.
