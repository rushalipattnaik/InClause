from utils.doc_checks import check_document, detect_type

RENTAL = ("The Tenant shall pay rent to the Landlord for the premises. The tenancy may be ended by one month "
          "notice. The security deposit shall be refunded. Repairs are the Tenant's duty. Disputes go to the "
          "civil courts. The Tenant shall keep the premises clean.")


def test_rental_agreement_is_detected_and_gaps_are_listed():
    result = check_document(RENTAL)
    assert result["type"] == "rental agreement"
    assert "Lock-in period" in result["missing"] and "Rent increases" in result["missing"]
    assert "Notice period for ending the agreement" not in result["missing"]
    assert "Security deposit and when it is returned" not in result["missing"]


def test_a_complete_document_has_nothing_missing():
    text = RENTAL + " There is a lock-in of six months. Rent will increase yearly."
    assert check_document(text)["missing"] == []


def test_unrecognised_documents_return_none():
    assert check_document("This is a short note about lunch plans for next week.") is None
    assert detect_type("") is None


def test_loan_and_employment_and_terms_are_recognised():
    assert detect_type("The Borrower shall repay the loan to the Lender. The loan principal and EMI are due.") == "loan agreement"
    assert detect_type("The Employee and the Employer agree on employment, salary and a probation period.") == "employment contract"
    assert detect_type("By using the service you agree to these terms of service. Your account is yours. You agree to pay for the service. The service may change.") == "terms of service"
