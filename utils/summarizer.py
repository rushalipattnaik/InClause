import os
import re

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

SYSTEM = (
    "You help people understand contracts. Explain the clause in simple, everyday words in 2-3 short "
    "sentences, from the point of view of the person signing (tenant, employee, customer or borrower). "
    "Stay faithful to the clause: never add conditions, exceptions, reasons or examples that are not in it. "
    "Start directly with the explanation, with no introduction. Use the reference notes only if they are "
    "relevant, and only mention a law or section number if it appears in the notes; never invent citations. "
    "You are not a lawyer and must not give legal advice."
)
HUMAN = "Reference notes:\n{context}\n\nClause:\n{clause}\n\nWhat does this clause mean for the person signing?"

_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)|this is|below is)\b[^:\n]{0,100}:\s*|^\s*(?:explanation|summary)\s*:\s*",
    re.IGNORECASE,
)

_SECTION = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)", re.IGNORECASE)
_ACT = re.compile(r"\b(?:[A-Z][A-Za-z&']+\s+){1,6}Act\b")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
NO_RELIABLE_EXPLANATION = "No reliable explanation could be generated for this clause."

_chain = None


def format_context(references):
    if not references:
        return "No relevant reference notes found."
    lines = []
    for r in references:
        if isinstance(r, dict):
            lines.append(f"- {r['title']} ({r['source']}): {r['text']}")
        else:
            lines.append(f"- {r}")
    return "\n".join(lines)


def clean_output(text):
    text = _PREAMBLE.sub("", text.strip())
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return re.sub(r"[ \t]+", " ", text).strip()


def strip_unsupported_citations(text, allowed_text):
    allowed_numbers = {m.group(1).lower() for m in _SECTION.finditer(allowed_text)}
    allowed_lower = allowed_text.lower()
    kept = []
    for sentence in _SENTENCES.split(text):
        wrong_section = any(m.group(1).lower() not in allowed_numbers for m in _SECTION.finditer(sentence))
        wrong_act = any(m.group(0).lower() not in allowed_lower for m in _ACT.finditer(sentence))
        if not (wrong_section or wrong_act):
            kept.append(sentence)
    return " ".join(kept).strip() or NO_RELIABLE_EXPLANATION


def _get_chain():
    global _chain
    if _chain is None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
        llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0.2, num_predict=220)
        _chain = prompt | llm | StrOutputParser()
    return _chain


def summarize_clause(clause, references=None):
    try:
        context = format_context(references)
        answer = clean_output(_get_chain().invoke({"clause": clause[:800], "context": context}))
        return strip_unsupported_citations(answer, f"{clause} {context}")
    except Exception as e:
        print(f"Ollama error: {type(e).__name__}")
        raise RuntimeError(
            f"Local AI is offline. Make sure Ollama is running and '{MODEL}' is pulled (ollama pull {MODEL})."
        ) from e
