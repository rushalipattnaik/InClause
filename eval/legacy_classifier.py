def classify_clause(clause):
    clause_lower = clause.lower()
    suspicious = ["hidden", "unlimited", "waive", "forfeit", "non-refundable"]
    risky = ["penalty", "terminate", "liability", "eviction", "lawsuit", "fee"]
    for word in suspicious:
        if word in clause_lower:
            return "Suspicious"
    for word in risky:
        if word in clause_lower:
            return "Risky"
    return "Safe"
