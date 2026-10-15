from utils.classifier import analyze_clause, classify_clause


def test_suspicious_clause_has_reason():
    r = analyze_clause("The security deposit is non-refundable and may be kept by the Landlord.")
    assert r["label"] == "Suspicious"
    assert r["reasons"][0]["matched"].lower() == "non-refundable"


def test_risky_clause():
    assert classify_clause("Either party may terminate this agreement by giving thirty days notice.") == "Risky"


def test_safe_clause():
    r = analyze_clause("This agreement commences on the Effective Date and continues for one year.")
    assert r == {"label": "Safe", "reasons": []}


def test_word_boundaries_avoid_false_positives():
    assert classify_clause("We value your feedback and our coffee machine is in the lobby area.") == "Safe"


def test_suspicious_beats_risky():
    assert classify_clause("Late fees apply and the deposit is forfeited.") == "Suspicious"


def test_ordinary_sublet_ban_is_risky_not_suspicious():
    clause = ("That the Tenant shall not sublet, assign or part with the demised premises in whole or "
              "part thereof to any person in any circumstances whatsoever and the same shall be used "
              "for bonafide residential purposes only.")
    assert classify_clause(clause) == "Risky"


def test_whatsoever_after_refund_reason_is_still_suspicious():
    assert classify_clause("No refund of advance rent shall be made for any reason whatsoever.") == "Suspicious"


REAL_LEASE = [
    ("On termination of the tenancy or earlier, the Tenant will be entitled to remove such equipment and "
     "restore the changes made, if any, to the original state.", "Safe"),
    ("Further, any other payment in the nature of subscription or periodical fee to the welfare "
     "association shall be paid by the Owner.", "Safe"),
    ("That the Rent Agreement will be registered in front of the Registrar and the charges towards stamp "
     "duty, court fee and lawyer fees will be equally borne by the Owner and Tenant.", "Safe"),
    ("That this Rent Agreement can be terminated before the expiry of this tenancy period by serving one "
     "month prior notice in writing by either party.", "Risky"),
    ("That in case the Premises are not vacated by the Tenant at the termination of the Rent period, the "
     "Tenant will pay damages calculated at two times the rent.", "Risky"),
    ("That no structural additions or alterations shall be made by the Tenant in the premises without the "
     "prior written consent of the Owner.", "Risky"),
    ("That the Owner shall hold the right to visit the demised premises for inspection, not exceeding once "
     "in a month.", "Safe"),
]


def test_regression_clauses_from_a_real_rental_agreement():
    for text, expected in REAL_LEASE:
        assert classify_clause(text) == expected, text


def test_burden_on_the_signer_is_not_demoted():
    assert classify_clause("The Tenant shall pay all charges and fees levied by the society.") == "Risky"


def test_successors_and_assigns_boilerplate_is_not_flagged():
    clause = ("hereinafter called the OWNER, which expression shall include their heirs, legal "
              "representatives, successors and assigns, of the ONE PART")
    assert classify_clause(clause) == "Safe"


def test_real_assignment_restriction_is_still_flagged():
    assert classify_clause("The Tenant shall not assign this lease to any other person.") == "Risky"
