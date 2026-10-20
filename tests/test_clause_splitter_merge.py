from utils.clause_splitter import merge_continuations, split_into_clauses


def test_lowercase_fragment_is_merged_into_the_previous_clause():
    out = merge_continuations(["The Tenant shall pay the rent on the first day of every", "month without any delay."])
    assert out == ["The Tenant shall pay the rent on the first day of every month without any delay."]


def test_unfinished_clause_followed_by_a_non_clause_start_is_merged():
    assert merge_continuations(["The Tenant shall pay the deposit of", "- Rs. 5,000 held by the Owner."]) == [
        "The Tenant shall pay the deposit of - Rs. 5,000 held by the Owner."]


def test_a_new_capitalised_clause_is_not_merged_even_without_closing_punctuation():
    clauses = ["The Tenant shall keep the premises clean", "The Owner shall repair the structure"]
    assert merge_continuations(clauses) == clauses


def test_a_lowercase_start_merges_even_after_closing_punctuation():
    for end in (".", "!", "?", ";", ":", ")"):
        out = merge_continuations([f"The first clause ends here{end}", "and this one starts in lowercase"])
        assert len(out) == 1


def test_a_capitalised_start_after_closing_punctuation_stays_a_separate_clause():
    for end in (".", "!", "?", ";", ":", ")"):
        clauses = [f"The first clause ends here{end}", "The second clause starts a new sentence."]
        assert merge_continuations(clauses) == clauses


def test_numbered_items_are_never_merged():
    clauses = ["The Tenant shall pay rent monthly and on time", "(a) the rent is due on the first of the month"]
    assert merge_continuations(clauses) == clauses
    numbered = ["1.1 The deposit is held by the Owner", "1.2 the deposit is refunded on exit"]
    assert merge_continuations(numbered) == numbered


def test_a_sentence_broken_by_a_page_break_is_repaired_end_to_end():
    text = ("The Tenant shall pay the monthly rent on or before the fifth day of each\n\n"
            "Page 2 of 5\n\n"
            "month and any amount that is overdue will attract a late fee as stated below.\n\n"
            "The Owner shall repair all structural defects in the premises at his own cost.")
    out = split_into_clauses(text)
    assert len(out) == 2
    assert out[0].startswith("The Tenant shall pay") and "each month and any amount" in out[0]


def test_the_first_fragment_and_an_empty_list_are_safe():
    assert merge_continuations([]) == []
    assert merge_continuations(["only one clause here"]) == ["only one clause here"]
