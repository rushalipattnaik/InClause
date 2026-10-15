import json

import pytest

from utils import rag_engine

ENTRIES = [
    {"id": "deposit", "title": "Security deposits", "source": "test",
     "text": "A security deposit protects the landlord against unpaid rent and damage to the premises."},
    {"id": "arbitration", "title": "Arbitration", "source": "test",
     "text": "Arbitration sends disputes to a private arbitrator instead of a court."},
]


@pytest.fixture
def kb(tmp_path, monkeypatch, fake_embeddings):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(ENTRIES))
    monkeypatch.setattr(rag_engine, "KB_PATH", str(path))
    rag_engine.set_embeddings(fake_embeddings, index_dir=str(tmp_path / "idx"), tag="fake")
    yield path
    rag_engine.set_embeddings(None, tag=rag_engine.EMBED_MODEL)


def test_relevant_query_returns_right_entry_with_metadata(kb):
    hits = rag_engine.retrieve_reference("The landlord keeps the security deposit for unpaid rent", min_similarity=0.2)
    assert hits and hits[0]["id"] == "deposit"
    assert hits[0]["source"] == "test" and 0 < hits[0]["score"] <= 1.0


def test_irrelevant_query_returns_nothing(kb):
    assert rag_engine.retrieve_reference("The moon orbits quickly around saturn tonight", min_similarity=0.3) == []


def test_threshold_controls_results(kb):
    q = "arbitrator for disputes"
    assert len(rag_engine.retrieve_reference(q, k=2, min_similarity=0.0, margin=1.0)) == 2
    assert len(rag_engine.retrieve_reference(q, k=2, min_similarity=0.99)) == 0


def test_index_rebuilds_when_corpus_changes(kb, tmp_path):
    assert rag_engine.retrieve_reference("force majeure floods", min_similarity=0.2) == []
    new = ENTRIES + [{"id": "fm", "title": "Force majeure", "source": "test",
                      "text": "Force majeure excuses performance during floods, war or strikes."}]
    kb.write_text(json.dumps(new))
    rag_engine.set_embeddings(rag_engine._embeddings, index_dir=str(tmp_path / "idx"), tag="fake")
    hits = rag_engine.retrieve_reference("force majeure floods", min_similarity=0.2)
    assert hits and hits[0]["id"] == "fm"


def test_blank_query(kb):
    assert rag_engine.retrieve_reference("   ") == []


def test_margin_drops_a_note_far_below_the_best_match(kb):
    q = "The landlord keeps the security deposit for unpaid rent and arbitrator disputes"
    wide = rag_engine.retrieve_reference(q, k=2, min_similarity=0.0, margin=1.0)
    narrow = rag_engine.retrieve_reference(q, k=2, min_similarity=0.0, margin=0.01)
    assert len(wide) == 2 and len(narrow) == 1 and narrow[0]["id"] == wide[0]["id"]


def test_reviewed_flag_is_returned_and_changes_rebuild_the_index(kb, tmp_path):
    first = rag_engine.retrieve_reference("security deposit unpaid rent landlord", min_similarity=0.2)
    assert first[0]["reviewed"] is False
    flagged = [dict(e, reviewed=True) if e["id"] == "deposit" else e for e in ENTRIES]
    kb.write_text(json.dumps(flagged))
    rag_engine.set_embeddings(rag_engine._embeddings, index_dir=str(tmp_path / "idx"), tag="fake")
    again = rag_engine.retrieve_reference("security deposit unpaid rent landlord", min_similarity=0.2)
    assert again[0]["reviewed"] is True


def test_get_embeddings_returns_the_configured_model(kb, fake_embeddings):
    assert rag_engine.get_embeddings() is fake_embeddings


def test_shipped_knowledge_base_is_well_formed():
    entries = rag_engine.load_entries(rag_engine.__dict__["BASE_DIR"] + "/legal_data/knowledge_base.json")
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)) and len(ids) >= 30
    for e in entries:
        assert {"id", "title", "text", "source", "reviewed"} <= set(e)
        assert isinstance(e["reviewed"], bool) and len(e["text"]) > 60
