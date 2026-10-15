import re

MIN_CLAUSE_CHARS = 30
LONG_PARAGRAPH_CHARS = 280
MAX_CLAUSE_CHARS = 1500

_MARKER = re.compile(
    r"^(?:\d+(?:\.\d+)+(?:\([a-z]\))?\s"
    r"|\d+[.)]\s"
    r"|\(\s*[a-zA-Z]{1,4}\s*\)\s"
    r"|(?:clause|section|article)\s+\d+[.:)]?\s)",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d{1,3})$", re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)[A-Z][A-Za-z ,&/'-]{2,50}$")

_PLACEHOLDER = re.compile(
    r"\(\s*[^)]{0,40}?\b(?:name|address|signature|number|complete|date|city|state|amount|witness)\b[^)]*\)|_{3,}",
    re.IGNORECASE,
)
_SIGNING_BLOCK = re.compile(r"^(?:\d+[.)]\s*)?in witness whereof\b|signature of the\b", re.IGNORECASE)

_ABBREVIATIONS = ["Rs.", "No.", "Nos.", "Inc.", "Ltd.", "Pvt.", "Co.", "Corp.", "e.g.", "i.e.",
                  "etc.", "Mr.", "Mrs.", "Ms.", "Dr.", "Sec.", "Art.", "approx.", "vs.", "viz."]
_DOT = "\u2024"
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d])")


def _is_heading(line):
    if _NUMBERED_HEADING.match(line) and not line.endswith((".", ";", ",")):
        return True
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and line.isupper() and len(line) < 80 and not line.endswith((".", ";"))


def _paragraphs(text):
    paragraphs, current = [], []

    def flush():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if _PAGE_NUMBER.match(line):
            continue
        if _is_heading(line):
            flush()
            continue
        if _MARKER.match(line):
            flush()
        current.append(line)
    flush()
    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs]


def _split_sentences(paragraph):
    protected = paragraph
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _DOT))
    parts = _SENTENCE_END.split(protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def _merge_short(pieces):
    merged = []
    for piece in pieces:
        if merged and len(piece) < 40:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _is_template_noise(clause):
    return bool(_SIGNING_BLOCK.search(clause)) or len(_PLACEHOLDER.findall(clause)) >= 3


def split_into_clauses(text):
    clauses = []
    for paragraph in _paragraphs(text):
        if len(paragraph) > LONG_PARAGRAPH_CHARS:
            pieces = _merge_short(_split_sentences(paragraph))
        else:
            pieces = [paragraph]
        clauses.extend(pieces)
    return [c[:MAX_CLAUSE_CHARS] for c in clauses if len(c) >= MIN_CLAUSE_CHARS and not _is_template_noise(c)]
