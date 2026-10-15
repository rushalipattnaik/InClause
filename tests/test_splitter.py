from utils.clause_splitter import split_into_clauses


def test_wrapped_lines_are_joined():
    text = "The Tenant shall pay rent of Rs. 18,000 on or before\nthe 5th day of each month without fail."
    assert split_into_clauses(text) == [
        "The Tenant shall pay rent of Rs. 18,000 on or before the 5th day of each month without fail."
    ]


def test_numbered_clauses_split_and_headings_dropped():
    text = "1. Rent\n1.1 The rent is payable monthly in advance.\n1.2 The deposit is held by the Landlord.\n(a) The Tenant shall not sublet the premises to anyone."
    out = split_into_clauses(text)
    assert len(out) == 3
    assert out[0].startswith("1.1") and out[1].startswith("1.2") and out[2].startswith("(a)")
    assert not any(c.strip() == "1. Rent" for c in out)


def test_page_numbers_and_all_caps_headings_removed():
    text = "TERMINATION\nPage 3 of 10\n7\nEither party may terminate by thirty days notice in writing."
    assert split_into_clauses(text) == ["Either party may terminate by thirty days notice in writing."]


def test_abbreviations_do_not_split_sentences():
    text = ("The Tenant shall pay Rs. 5,000 to Acme Pvt. Ltd. every month. "
            "The Landlord may revise this amount only with written consent of the Tenant. "
            "Any revision must be recorded in writing and signed by both parties.")
    out = split_into_clauses(text)
    assert any("Rs. 5,000 to Acme Pvt. Ltd. every month" in c for c in out)


def test_long_paragraph_is_split_into_sentences():
    sentences = [f"Clause number {i} says that the party must comply with obligation {i} fully." for i in range(8)]
    out = split_into_clauses(" ".join(sentences))
    assert len(out) > 1


def test_short_fragments_dropped():
    assert split_into_clauses("Signed.\n\nOK") == []


def test_numbered_line_starting_with_amount_is_not_a_new_clause():
    text = "The late fee is charged for every week beyond the due date, up to a maximum of\n50 percent of the monthly rent in total."
    assert len(split_into_clauses(text)) == 1


def test_signature_blocks_and_template_placeholders_are_dropped():
    text = (
        "IN WITNESS WHEREOF BOTH PARTIES AGREES AND SIGNS THIS AGREEMENT ON THIS DAY AND YEAR\n\n"
        "1. (Name of the Witness) (Signature of the Witness) (Address of the Witness)\n\n"
        "2. (Name of the Witness) (Signature of the Witness) (Address of the Witness) (Signature of the Owner)\n\n"
        "The (Complete Address) of the Property, consisting (Number of Bedrooms) bedrooms and inventory of (Number of Fans).\n\n"
        "The Tenant shall pay the rent on or before the fifth day of every month without delay."
    )
    assert split_into_clauses(text) == ["The Tenant shall pay the rent on or before the fifth day of every month without delay."]


def test_one_blank_in_a_real_clause_is_kept():
    text = "The monthly rent shall be Rs. ______ payable in advance on the first day of each month."
    assert len(split_into_clauses(text)) == 1


def test_party_block_with_fill_in_fields_is_dropped():
    text = ("(Father's Name of the Owner), (Permanent Address of the Owner) (hereinafter called the OWNER) "
            "of the ONE PART AND in favour of (Name of the Tenant), S/o or D/o someone.")
    assert split_into_clauses(text) == []
