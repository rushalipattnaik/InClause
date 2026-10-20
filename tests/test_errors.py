from eval import errors


def rows():
    return [
        {"id": "1", "label": "Safe", "text": "The deposit is non-refundable under any circumstances."},
        {"id": "2", "label": "Risky", "text": "A late fee of Rs. 100 shall be charged on any payment after the due date."},
        {"id": "3", "label": "Safe", "text": "This agreement commences on the effective date and continues for one year."},
        {"id": "4", "label": "Risky", "text": "The tenant must provide quarterly reports about the activities carried on in the premises."},
    ]


def test_only_disagreements_are_listed_with_the_tools_reason():
    markdown, counts = errors.build_markdown(rows())
    assert counts[("Safe", "Suspicious")] == 1 and counts[("Risky", "Safe")] == 1
    assert "2 of 4 clauses disagree" in markdown
    assert "#1" in markdown and "non-refundable" in markdown and "#4" in markdown
    assert "#2" not in markdown and "#3" not in markdown
    assert "no rule matched" in markdown


def test_long_clauses_are_truncated():
    long_text = "The tenant shall pay a late fee " + "very " * 200
    markdown, _ = errors.build_markdown([{"id": "9", "label": "Safe", "text": long_text}])
    assert "..." in markdown and len(markdown) < 1500


def test_perfect_agreement_produces_an_empty_report():
    markdown, counts = errors.build_markdown([rows()[2]])
    assert "0 of 1 clauses disagree" in markdown and not counts
