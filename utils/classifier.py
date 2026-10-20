import re

SUSPICIOUS_RULES = [
    (r"non-?\s?refundable|no refund|not be entitled to any refund|without (?:any )?refund", "no refund"),
    (r"\bforfeit(?:s|ed|ure)?\b", "forfeiture"),
    (r"\bwaive[sd]?\b|\bwaiver\b", "waives a right"),
    (r"sole (?:and absolute )?(?:discretion|right)|(?:at|in) (?:its|their|our|his|her) (?:sole |absolute )?discretion", "decided at one party's discretion"),
    (r"without (?:any )?(?:prior |written )*(?:notice|intimation)", "no notice required"),
    (r"\bfor any reason\b|\bany reason whatsoever\b", "any reason allowed"),
    (r"\bunlimited\b|without (?:any )?limit|however arising", "uncapped liability or cost"),
    (r"\birrevocabl[ey]\b|\bperpetual\b", "irrevocable or perpetual grant"),
    (r"\bunconditionally\b", "unconditional commitment"),
    (r"automatically renew|auto-?renew|renew automatically|renews automatically", "renews automatically"),
    (r"continued use .{0,60}(?:accept|agree|constitutes)", "changes accepted by continued use"),
    (r"under (?:any|no) circumstances|(?:any|every) (?:reason|purpose|kind) whatsoever", "no exceptions"),
    (r"(?:final|binding) and (?:final|binding)|decision shall be final", "one-sided final decision"),
    (r"(?:may not|shall not|not be entitled to|cannot) (?:sue|bring|initiate)[^.]{0,40}(?:court|proceeding)", "gives up the right to sue"),
    (r"whether or not (?:during|in the course of) working hours", "claims work done outside work"),
    (r"full indemnity basis", "full legal costs"),
    (r"\bwithhold (?:any )?payment\b", "payment can be withheld"),
    (r"appointed by (?:the )?(?:client|company|landlord|lender|seller|licensor|employer)", "one side picks the arbitrator"),
    (r"class action", "gives up class action rights"),
]

RISKY_RULES = [
    (r"\bpenalt(?:y|ies)\b|liquidated damages|penal interest", "penalty"),
    (r"\blate (?:fee|payment|charge)s?\b|\boverdue\b|\bper annum\b|interest at|\binterest rate\b", "late fee or interest"),
    (r"\bterminat(?:e|es|ed|ion|ing)\b", "termination terms"),
    (r"\bliabilit(?:y|ies)\b|\bliable\b", "liability"),
    (r"\bindemnif(?:y|ies|ication)\b|hold (?:us |it |them )?harmless", "indemnity"),
    (r"\barbitrat\w*", "arbitration"),
    (r"non-?compet\w*|not (?:work|engage|be engaged|open)\b[^.]{0,60}\b(?:competitor|similar business|any business)", "non-compete"),
    (r"\bconfidential\w*", "confidentiality"),
    (r"\bbreach(?:es|ed)?\b|\bdefault\b", "breach or default"),
    (r"governed by|exclusive jurisdiction|courts? (?:at|of|in)\b", "governing law and courts"),
    (r"notice period", "notice period"),
    (r"(?:prior |written )+consent|consent of the", "needs consent"),
    (r"\bprepayment\b|\bforeclosure\b|\bprocessing fees?\b", "loan charges"),
    (r"\bas is\b|without warrant(?:y|ies)|disclaim\w*", "no warranty"),
    (r"personal data|share your|promotional|marketing purposes", "data sharing or marketing"),
    (r"\bfees?\b|\bcharges?\b", "fees or charges"),
    (r"\bassign(?!s\b)\w*\b|\bassigns\b(?<!and assigns)|\bsublet\b", "assignment or subletting"),
    (r"\bexclusive(?:ly)?\b", "exclusivity"),
    (r"\bincrease[sd]?\b|\bescalat\w+", "price or rent increase"),
    (r"\bcompensation\b|\bdamages\b", "compensation"),
    (r"intellectual property|\binventions?\b|\blicen[cs]e\b|reverse engineer", "intellectual property"),
    (r"\bdeposit\b", "deposit"),
]

WEAK_REASONS = {"fees or charges", "termination terms", "deposit", "compensation"}

_STRONG_PARTY = r"(?:owner|landlord|lessor|company|employer|seller|lender|licensor|provider|society|management)"
_SIGNER = r"(?:tenant|employee|borrower|buyer|customer|licensee|lessee|member|consumer)"
MITIGATOR_RULES = [
    rf"(?:paid|borne|payable|met|reimbursed|refunded|returned)\s+by\s+(?:the\s+)?{_STRONG_PARTY}",
    r"(?:borne|shared|split|paid)\s+(?:equally|jointly)|equally\s+(?:borne|shared|split|paid)|each party (?:shall )?(?:bear|pay)",
    rf"{_STRONG_PARTY}\s+(?:shall|will|agrees to|undertakes to)\s+(?:pay|bear|refund|return|reimburse|keep|provide|maintain)",
    rf"{_SIGNER}\s+(?:shall be|will be|is|are)\s+entitled to|entitled to (?:remove|receive|claim|refund|retain)",
    r"free and harmless",
    r"\brefundable\b|\brefunded\b",
    r"(?:subject to )?(?:natural |normal |fair )?wear (?:and|&) tear",
]

_SUSPICIOUS = [(re.compile(p, re.IGNORECASE), r) for p, r in SUSPICIOUS_RULES]
_RISKY = [(re.compile(p, re.IGNORECASE), r) for p, r in RISKY_RULES]
_MITIGATORS = [re.compile(p, re.IGNORECASE) for p in MITIGATOR_RULES]


def _matches(rules, text):
    found = []
    for pattern, reason in rules:
        m = pattern.search(text)
        if m:
            found.append({"reason": reason, "matched": m.group(0).strip()})
    return found


def analyze_clause(clause):
    suspicious = _matches(_SUSPICIOUS, clause)
    if suspicious:
        return {"label": "Suspicious", "reasons": suspicious}
    risky = _matches(_RISKY, clause)
    if risky:
        only_weak = all(r["reason"] in WEAK_REASONS for r in risky)
        if only_weak and any(m.search(clause) for m in _MITIGATORS):
            return {"label": "Safe", "reasons": []}
        return {"label": "Risky", "reasons": risky}
    return {"label": "Safe", "reasons": []}


def classify_clause(clause):
    return analyze_clause(clause)["label"]


def rule_features(clause):
    hits = [int(bool(pattern.search(clause))) for pattern, _ in _SUSPICIOUS + _RISKY]
    hits += [int(bool(m.search(clause))) for m in _MITIGATORS]
    label = analyze_clause(clause)["label"]
    return hits + [int(label == name) for name in ("Safe", "Risky", "Suspicious")]
