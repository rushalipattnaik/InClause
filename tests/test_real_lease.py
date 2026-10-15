import pytest

from utils.classifier import classify_clause
from utils.clause_splitter import split_into_clauses
from utils.text_cleanup import clean_text

SAMPLE_RENT_AGREEMENT = [
    ("hereinafter called the \u201cTENANT\u201d, which expression shall include its legal representatives,successors "
     "and assigns) of the OTHER PART, having a permanent address at Complete permanent Address", "Safe"),
    ("That the Tenant shall pay to the Owner a monthly maintenance charge of Rs.(Amount in Numbers) towards the "
     "maintenance of Generator & Elevator, Salaries towards guards, Chargesfor Electricity Maintenance for "
     "Common Areasand towards maintaining the lawn.", "Risky"),
    ("The Tenant will pay to the Owner an interest-free refundable security deposit of Rs.(Amountin Numbers) "
     "vide cheque no (Cheque Number) dated (date of the cheque) at the time of signing the Rent Agreement.", "Safe"),
    ("8. That the Tenant shall not sublet, assign or part with the demised premises in whole or part thereof "
     "to any person in any circumstances whatsoever and the same shall be used for the bona fide residential "
     "purposes only.", "Risky"),
    ("That no structural additions or alterations shall be made by the Tenant in the premises without the prior "
     "written consent of the Owner but the Tenant can install air-conditioners in thespace provided.", "Risky"),
    ("Further, any other payment in the nature of subscription or periodical fee to the welfare association "
     "shall be paid bythe Owner.", "Safe"),
    ("15. That this Rent Agreement can be terminated before the expiry of this tenancy period by serving One month "
     "prior notice in writing by either party.", "Risky"),
    ("That it shall be the responsibility of the Tenant to hand over the vacant and peaceful possession of the "
     "demised premises on expiry of the Rent period, or on its early termination, as stated hereinabove in the "
     "same condition subject to natural wear and tear.", "Safe"),
    ("That in case, where the Premises are not vacated by the Tenant, at the termination of the Rentperiod, the "
     "Tenant will pay damages calculated at two times the rent for any period of occupation.", "Risky"),
    ("On termination of the tenancy or earlier, the Tenant will be entitled toremove such equipment and restore "
     "the changes made, if any, to the original state.", "Safe"),
    ("21. That the Rent Agreement will be registered in front of the Registrar and the charges towards stamp duty, "
     "court fee & lawyer/coordinator will be equally borne by the Owner and Tenant.", "Safe"),
    ("In case the Owner fails to refund the security deposit to the Tenant on early termination, the Tenant is "
     "entitled to hold possession of the Rented premises, without payment of rent and/or any other charges "
     "whatsoever, till such time the Owner refunds the security deposit.", "Safe"),
]


@pytest.mark.parametrize("text, expected", SAMPLE_RENT_AGREEMENT)
def test_label_after_cleanup(text, expected):
    cleaned = clean_text(text)
    assert classify_clause(cleaned) == expected, cleaned


def test_cleanup_produces_readable_text():
    cleaned = clean_text(SAMPLE_RENT_AGREEMENT[5][0])
    assert "paid by the Owner" in cleaned
    assert "Charges for Electricity" in clean_text(SAMPLE_RENT_AGREEMENT[1][0])


def test_pipeline_on_a_wrapped_numbered_document():
    raw = ("1. Rent\n7. That the Tenant shall pay a monthly rent on or before the 7th day of each\nmonth without fail "
           "and the same shall be paid bythe Tenant.\n\n8. That the Tenant shall not sublet or assign the premises "
           "to any person.\n")
    clauses = split_into_clauses(clean_text(raw))
    assert len(clauses) == 2 and "by the Tenant" in clauses[0]
