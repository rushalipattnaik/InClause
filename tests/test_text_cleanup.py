from utils.text_cleanup import clean_text


def test_ligatures_are_normalised():
    assert clean_text("for the bona\ufb01de use of \ufb01ttings") == "for the bonafide use of fittings"


def test_glued_words_from_line_wraps_are_repaired():
    out = clean_text("in whole or partthereof and at his own expense is ownexpense and orgovernment rules")
    assert "part thereof" in out and "own expense" in out and "or government" in out


def test_function_word_glued_to_capitalised_word():
    assert clean_text("shall be paid by theOwner") == "shall be paid by the Owner"


def test_legal_vocabulary_and_names_are_left_alone():
    text = ("The Landlord hereinafter called the Lessor, notwithstanding the aforesaid, hereby agrees. "
            "Indemnification and subletting apply to Hyderabad and Pattnaik.")
    assert clean_text(text) == text


def test_ordinary_text_is_unchanged():
    text = "The Tenant shall pay the monthly rent on or before the fifth day of each calendar month."
    assert clean_text(text) == text


def test_short_glued_function_words_from_a_real_pdf_are_repaired():
    glued = ("paid bythe Owner. Chargesfor Electricity Maintenance for Common Areasand towards the lawn. "
             "Rs.(Amountin Numbers). entitled toremove such equipment in thespace provided. "
             "the Tenantshall pay. executed byand between")
    out = clean_text(glued)
    for expected in ("by the Owner", "Charges for Electricity", "Areas and towards", "Amount in Numbers",
                     "to remove such", "the space provided", "Tenant shall pay", "by and between"):
        assert expected in out, expected


def test_names_and_places_are_not_split():
    names = "Hyderabad Pattnaik Maharashtra Chandigarh Bengaluru Visakhapatnam Rushali Bhopal Andrew Anderson Toronto Intern"
    assert clean_text(names) == names


def test_two_function_words_glued_together():
    out = clean_text("held bythe Owner and given tothe Tenant, the rent ofthe month, kept inthe premises")
    assert out == "held by the Owner and given to the Tenant, the rent of the month, kept in the premises"


def test_real_words_that_start_with_function_words_survive():
    text = "Information, inform, together, another, onward, intent, entitled, tomorrow and bythe-way"
    assert clean_text(text).startswith("Information, inform, together, another, onward, intent, entitled, tomorrow and")
