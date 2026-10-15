# Design decisions

## Local first
The document and everything derived from it stay on the user's machine. That rules out hosted LLM APIs and a public
server, and it shapes the rest of the design: a local Ollama model, local embeddings, results held only in memory,
and a page that makes no external requests. The cost is speed and model quality, which the interface hides by
showing flags immediately and filling in explanations as they finish.

## Instant flags, background explanations
A 3B model on a laptop CPU needs seconds per clause, too long for one web request. Upload therefore extracts, splits
and classifies synchronously (pure Python, milliseconds) and hands the slow work to one background worker with a
priority queue (Suspicious, then Risky, then Safe). The browser polls a versioned snapshot, so unchanged polls are
cheap. There is deliberately one Gunicorn worker: jobs live in that process's memory.

## Retrieval
Notes and clauses are embedded with MiniLM into unit vectors and searched with FAISS inner product, so a score is a
cosine similarity. Showing nothing beats showing something irrelevant, so a note must reach a minimum similarity and
also come within a margin of the best match. Both numbers are tunable with `python -m eval.run_eval --retrieval --tune`.
The index is cached on disk together with a hash of the whole knowledge base, so editing any note rebuilds it.

## Three classifiers, one decision rule
- **Rules** are instant and every flag shows the matching words. Weak triggers such as "fee" or "termination" are
  cleared when the same clause puts the cost on the owner or company, shares it, or grants the signer a right.
- **ML** is logistic regression on the same embeddings used for retrieval, trained from the labelled CSV files at
  first use. It costs no extra model and can learn from clauses the rules never saw.
- **Hybrid** keeps every Suspicious rule decision, lets the model rescue clauses the rules passed (capped at Risky),
  and lets it clear weak rule flags only when it is confident.
The default stays on rules until a measurement on real clauses shows otherwise. A weak model can make the hybrid worse
than the rules alone, which is why the comparison exists.

## Guarding the model's output
The prompt tells the model to stay faithful to the clause and not to invent citations. After generation, a preamble
stripper removes openings like "Here is an explanation:", and a citation guard drops any sentence that cites a section
or Act that appears in neither the clause nor the retrieved notes.

## Text cleanup
PDF extraction can produce ligatures ("ﬁ") and words glued across line wraps. Text is normalised, then a conservative
repair splits only unknown lowercase words into known words and two stacked function words ("bythe"). Proper nouns,
capitalised words and standard legal vocabulary are left alone.

## Privacy model, precisely
Nothing leaves the machine, uploads are read from a temporary file that is removed immediately, results exist only in
memory and expire, responses are marked no-store, and document text is inserted into the page as text, never HTML.
This does not protect against malware on the same computer or against someone with access to its memory.

## What would come next
Cross-clause reasoning, OCR for scanned files, a perspective switch for the other side of the contract, and a
reviewed, cited knowledge base maintained by a lawyer.
