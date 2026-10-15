import os

LABELS = ["Safe", "Risky", "Suspicious"]
PROMPT = """You label clauses from contracts for a non-lawyer reader.
Suspicious = removes a right, hides or uncaps a cost, or lets one side act alone without recourse.
Risky = a normal clause that still carries obligations, costs or penalties worth attention.
Safe = neutral or purely informational.
Answer with exactly one word: Safe, Risky or Suspicious.

Clause: {clause}
Label:"""

_llm = None


def classify_clause(clause):
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
                          base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                          temperature=0, num_predict=5)
    answer = _llm.invoke(PROMPT.format(clause=clause[:800])).content.strip().lower()
    for label in LABELS:
        if label.lower() in answer:
            return label
    return "Safe"
