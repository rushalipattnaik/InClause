import re
import unicodedata

PROTECTED = {
    "hereinafter", "hereinbefore", "hereinabove", "herein", "hereof", "hereto", "hereby", "hereunder",
    "hereafter", "thereof", "thereto", "thereby", "therein", "thereafter", "thereunder", "thereupon",
    "whereas", "whereof", "whereby", "wherein", "aforesaid", "aforementioned", "notwithstanding",
    "forthwith", "henceforth", "heretofore", "inasmuch", "insofar", "otherwise", "whatsoever", "whomsoever",
}
_FUNCTION_WORDS = ("the", "of", "to", "and", "or", "by", "in", "on", "for", "with", "any", "all", "his", "her", "its")
_GLUED_CAMEL = re.compile(r"^(" + "|".join(_FUNCTION_WORDS) + r")([A-Z][a-z]+)$")
_PREFIX_WORDS = ("the", "of", "to", "and", "or", "by", "in", "on", "for", "with", "any", "all", "at", "as", "if", "is", "be")
_SUFFIX_WORDS = ("the", "of", "to", "and", "or", "by", "in", "on", "for", "with", "shall", "will", "may", "any", "all")
_FIRST_WORDS = ("by", "to", "of", "in", "on", "for", "and", "with", "at", "as", "or", "from")
_SECOND_WORDS = ("the", "any", "all", "his", "her", "its", "such", "this", "that", "each")
_FUNCTION_PAIRS = {a + b: len(a) for a in _FIRST_WORDS for b in _SECOND_WORDS}
_TOKEN = re.compile(r"[A-Za-z]+")
_SOFT_HYPHEN = re.compile("[\u00ad\u200b\u200c\u200d\ufeff]")

_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            import wordninja
            _model = wordninja.DEFAULT_LANGUAGE_MODEL
        except Exception:
            _model = False
    return _model


def _known(model, word):
    return word.lower() in model._wordcost


def _merge_protected(parts):
    merged, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and (parts[i] + parts[i + 1]).lower() in PROTECTED:
            merged.append(parts[i] + parts[i + 1])
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged


def _split_function_word(token, model):
    lower = token.lower()
    for word in _PREFIX_WORDS:
        rest = lower[len(word):]
        if lower.startswith(word) and len(rest) >= 3 and _known(model, rest):
            return f"{token[:len(word)]} {token[len(word):]}"
    for word in _SUFFIX_WORDS:
        head = lower[:-len(word)]
        if lower.endswith(word) and len(head) >= 3 and _known(model, head):
            return f"{token[:-len(word)]} {token[-len(word):]}"
    return None


def _repair_token(token, model):
    camel = _GLUED_CAMEL.match(token)
    if camel:
        return f"{camel.group(1)} {camel.group(2)}"
    cut = _FUNCTION_PAIRS.get(token.lower())
    if cut:
        return f"{token[:cut]} {token[cut:]}"
    if len(token) < 4 or token.lower() in PROTECTED or _known(model, token) or token.isupper():
        return token
    glued = _split_function_word(token, model)
    if glued:
        return glued
    if len(token) < 9 or not token.islower():
        return token
    parts = _merge_protected(model.split(token))
    if len(parts) < 2 or len(parts) > 3:
        return token
    if not all(p.lower() in PROTECTED or (_known(model, p) and (len(p) > 1 or p in "ai")) for p in parts):
        return token
    return " ".join(parts)


def repair_glued_words(text):
    model = _load_model()
    if not model:
        return text
    return _TOKEN.sub(lambda m: _repair_token(m.group(0), model), text)


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = text.replace("\u00a0", " ")
    return repair_glued_words(text)
