import pytest

from utils import summarizer
from utils.summarizer import clean_output, format_context


@pytest.mark.parametrize("raw, expected", [
    ("Here's an explanation of the clause in plain English: This clause says you cannot sublet.",
     "This clause says you cannot sublet."),
    ("Sure! Here is a simple explanation:\nYou must pay rent monthly.", "You must pay rent monthly."),
    ("Explanation: The deposit is not returned.", "The deposit is not returned."),
    ('"You may leave with one month notice."', "You may leave with one month notice."),
    ("You must pay rent by the fifth.", "You must pay rent by the fifth."),
])
def test_clean_output_removes_preambles(raw, expected):
    assert clean_output(raw) == expected


def test_context_includes_title_source_and_text():
    ctx = format_context([{"title": "Deposits", "source": "General", "text": "Held as protection."}])
    assert "Deposits (General): Held as protection." in ctx
    assert format_context([]) == "No relevant reference notes found."


def test_prompt_never_asks_for_plain_english():
    assert "plain english" not in (summarizer.SYSTEM + summarizer.HUMAN).lower()


def test_offline_model_raises_a_helpful_error(monkeypatch):
    class Broken:
        def invoke(self, _):
            raise ConnectionError("refused")
    monkeypatch.setattr(summarizer, "_chain", Broken())
    with pytest.raises(RuntimeError, match="ollama pull"):
        summarizer.summarize_clause("Some clause text here.", [])


def test_invented_section_numbers_are_removed():
    notes = "Liquidated damages (s.74 of the Indian Contract Act, 1872): reasonable compensation only."
    text = "You may owe a penalty. Under Section 74 you only pay reasonable compensation. Section 99 also applies."
    out = summarizer.strip_unsupported_citations(text, notes)
    assert "Section 74" in out and "Section 99" not in out and out.startswith("You may owe a penalty.")


def test_invented_acts_are_removed_but_cited_ones_kept():
    notes = "Under the Transfer of Property Act, 1882 (s.106) a month-to-month lease needs fifteen days notice."
    text = "The Transfer of Property Act sets notice rules. The Companies Act also controls this."
    out = summarizer.strip_unsupported_citations(text, notes)
    assert "Transfer of Property Act" in out and "Companies Act" not in out


def test_a_clause_that_mentions_a_section_may_be_referred_to():
    out = summarizer.strip_unsupported_citations("Section 5 of the agreement fixes the rent.", "As per Section 5 rent is due.")
    assert out == "Section 5 of the agreement fixes the rent."


def test_fallback_when_nothing_is_left():
    assert summarizer.strip_unsupported_citations("Section 12 applies.", "no notes") == summarizer.NO_RELIABLE_EXPLANATION


def test_rupee_amounts_are_not_mistaken_for_sections():
    text = "You pay Rs.500 every month and the deposit is Rs.5000."
    assert summarizer.strip_unsupported_citations(text, "") == text
