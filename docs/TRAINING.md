# Training the ML classifier

This guide covers the Legal-BERT path: where the training data comes from, how it is split, how to train, and how to
judge the result. Read the section on label quality before you trust the model.

## Commands

```
python -m scripts.prepare_training_data
python -m scripts.train_legal_bert
python -m eval.full_report
python -m eval.run_eval --tune-hybrid
```

1. `prepare_training_data` downloads the official CUAD archive (about 18 MB) from the CUAD GitHub repository,
   converts it to clause-sized units with the same splitter the app uses, adds your labelled clauses from `eval/`, and
   writes `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl` (fields `text`, `label`, `source`, and `category`
   for CUAD rows). It prints the class distribution and warns if a class has fewer than 50 examples.
2. `train_legal_bert` fine-tunes `nlpaueb/legal-bert-base-uncased` and saves the best epoch to `models/legal-bert-risk`,
   plus `eval/train_report.json` with per-class F1 on the validation split. Add `--dry-run` first to check the data.
3. `eval.full_report` scores the rules, the ML models and the hybrids on **your** held-out clauses and writes
   `docs/RESULTS.md`, including a verdict.
4. `eval.run_eval --tune-hybrid` recommends `HYBRID_FLAG_MIN` and `HYBRID_SAFE_MIN` for Legal-BERT using the
   validation split.

## Where the labels come from, and why to be careful

CUAD marks, in 510 commercial contracts, the spans that belong to 41 clause categories. It does not say whether a clause
is bad for the person signing. The mapping below turns a category into Safe, Risky or Suspicious using the same rubric
as the rest of this project, so every label is a weak, category-level guess:

| Label | CUAD categories |
|---|---|
| Safe | Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Governing Law, Competitive Restriction Exception, License Grant, Affiliate License-Licensor, Affiliate License-Licensee, Warranty Duration, Third Party Beneficiary |
| Risky | Renewal Term, Notice Period to Terminate Renewal, Most Favored Nation, Non-Compete, Exclusivity, No-Solicit of Customers, No-Solicit of Employees, Non-Disparagement, Termination for Convenience, Rofr/Rofo/Rofn, Change of Control, Anti-Assignment, Revenue/Profit Sharing, Price Restrictions, Minimum Commitment, Volume Restriction, IP Ownership Assignment, Joint IP Ownership, Non-Transferable License, Unlimited/All-You-Can-Eat-License, Source Code Escrow, Post-Termination Services, Audit Rights, Cap on Liability, Liquidated Damages, Insurance |
| Suspicious | Irrevocable or Perpetual License, Uncapped Liability, Covenant Not to Sue |

Units that no CUAD annotation covers are sampled as "Safe" (at most 6 per contract). That is a convenience, not a fact.

Evidence of the noise, measured on the 874-unit CUAD test sample from the first preparation run:

- The hand-written rules agree with the CUAD-derived labels on only 52% of units.
- 76 of the 223 "unannotated, so Safe" units trip a rule, for example a clause letting a supplier sell
  products to anyone "in its sole discretion".
- A governing-law clause is always labelled Safe whoever it favours, and a span can be a fragment that is not a risk
  on its own.

Because of this, CUAD rows count half as much as your own clauses in the loss (`--cuad-weight 0.5`), and the model is
never scored on CUAD labels when you have held-out clauses of your own.

## How the data is split

- CUAD units are split **by contract**, stratified by label (`StratifiedGroupKFold`, seed 42), so no contract is in two
  splits. About 80% goes to train, 10% to validation and 10% to a CUAD test set that is only used if you have no clauses
  of your own.
- `eval/clauses.csv` (the author-labelled set) is split 80/20 stratified into train and validation.
- **Your own clauses in `eval/clauses_real.csv` are the test set by default.** The model never trains on them, which
  is what makes the comparison with the rules honest. `--train-on-real rental_agreement` moves one contract into
  training, and `--final` trains on all of them for the model you deploy (there is then no honest test left, so measure
  first, then re-train with `--final`).

This deliberately differs from a plain random stratified split: a random split would mix clauses from the same contract
across train and test and would test agreement with CUAD instead of agreement with you.

## Training time

I could not run the training code where this was written, so these are rough guesses, not measurements.

- On a laptop CPU, a full run (about 6,900 clauses, 4 epochs, length 256) may take many hours. For a CPU, use a subset
  and shorter inputs: `python -m scripts.train_legal_bert --max-train-samples 2500 --max-length 128 --epochs 3`.
- On a free GPU it should take minutes to a quarter of an hour. Google Colab and Kaggle offer free GPUs: copy the
  project there, run the first two commands, then download the `models/legal-bert-risk` folder (about 440 MB) into the
  same place in your project.
- Use `--dry-run` first. It prints the number of optimiser steps without importing PyTorch.

## Judging the result

Run `python -m eval.full_report`. If the model beats the rules on your clauses by a clear margin the report says so and
tells you what to put in `.env`; if not, it says to keep `CLASSIFIER=rules`. Either outcome is a legitimate finding.
The report also warns if any of your clauses were in the model's training data.

If the model does not beat the rules, the likely reasons are the noisy labels above and the small number of your own
clauses. More labelled clauses from different contract types is the most reliable fix. A second training stage that
fine-tunes only on your clauses, starting from this model, would be a sensible next step.

## Assumptions to review

- The category mapping above. These are judgement calls, for example Renewal Term as Risky, Audit Rights as Risky (the
  direction of the audit is unknown), Exclusivity and Most Favored Nation as Risky and not Suspicious, and only Uncapped
  Liability, Irrevocable or Perpetual License and Covenant Not to Sue as Suspicious.
- The category names come verbatim from `category_descriptions.csv` in the official CUAD repository. The question ids in
  `CUADv1.json` capitalise some names differently (for example "Cap On Liability"), so they are matched ignoring case.
- Check the CUAD licence on the Atticus Project site before sharing anything derived from it. `data/` and `models/` are
  git-ignored so they are not published with your repository.
