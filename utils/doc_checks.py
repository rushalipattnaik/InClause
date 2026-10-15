import re

TYPE_PATTERNS = {
    "rental agreement": r"\b(?:tenant|landlord|lessee|lessor|rent|premises|tenancy|demised)\b",
    "employment contract": r"\b(?:employee|employer|employment|salary|probation|designation)\b",
    "loan agreement": r"\b(?:borrower|lender|loan|emi|repayment|principal)\b",
    "terms of service": r"terms of (?:service|use)|\bsubscription\b|your account|\byou agree\b|\bthe service\b",
}
MIN_EVIDENCE = 5

CHECKS = {
    "rental agreement": [
        ("Notice period for ending the agreement", [r"notice"]),
        ("Security deposit and when it is returned", [r"deposit", r"refund|return"]),
        ("Rent increases", [r"increase|escalat|enhance|revis"]),
        ("Lock-in period", [r"lock-?\s?in"]),
        ("Repairs and maintenance", [r"repair|maintenance"]),
        ("Dispute resolution or court jurisdiction", [r"arbitrat|jurisdiction|\bcourts?\b"]),
    ],
    "employment contract": [
        ("Notice period", [r"notice"]),
        ("Termination", [r"terminat"]),
        ("Salary or compensation", [r"salary|compensation|\bctc\b|remuneration"]),
        ("Leave", [r"\bleave\b"]),
        ("Confidentiality", [r"confidential"]),
    ],
    "loan agreement": [
        ("Interest rate", [r"interest"]),
        ("Repayment schedule", [r"instal+ments?|\bemi\b|repay"]),
        ("Prepayment or foreclosure", [r"prepay|foreclos|pre-?closure"]),
        ("What happens on default", [r"default|penal"]),
        ("Security or guarantee", [r"secur|collateral|mortgage|guarant"]),
    ],
    "terms of service": [
        ("Refunds or cancellation", [r"refund|cancel"]),
        ("Privacy and personal data", [r"privacy|personal data|\bdata\b"]),
        ("Termination or suspension", [r"terminat|suspend"]),
        ("Limit on liability", [r"liabil"]),
        ("Changes to the terms", [r"change|modify|amend"]),
        ("Governing law", [r"governing law|jurisdiction|laws of"]),
    ],
}


def detect_type(text):
    counts = {name: len(re.findall(pattern, text, re.IGNORECASE)) for name, pattern in TYPE_PATTERNS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] >= MIN_EVIDENCE else None


def check_document(text):
    doc_type = detect_type(text)
    if not doc_type:
        return None
    missing = [label for label, patterns in CHECKS[doc_type]
               if not all(re.search(p, text, re.IGNORECASE) for p in patterns)]
    return {"type": doc_type, "missing": missing}
