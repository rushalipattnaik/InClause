import hashlib
import json
import os
import threading

from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.environ.get("KB_PATH", os.path.join(BASE_DIR, "legal_data", "knowledge_base.json"))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(BASE_DIR, "legal_data", "faiss_index"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.40"))
MARGIN = float(os.environ.get("RAG_MARGIN", "0.15"))
TOP_K = int(os.environ.get("RAG_TOP_K", "2"))

_lock = threading.RLock()
_embeddings = None
_embeddings_tag = EMBED_MODEL
_index_dir = INDEX_DIR
_db = None
_failed = False


def load_entries(path=None):
    with open(path or KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _entry_text(entry):
    return f"{entry['title']}. {entry['text']}"


def _corpus_hash(entries, tag):
    payload = json.dumps([tag, entries], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_or_load(embeddings, entries, index_dir, tag):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    # Scores equal cosine similarity only because the embedding model returns unit-length vectors.
    settings = {"distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT}
    hash_file = os.path.join(index_dir, "corpus.sha256")
    current = _corpus_hash(entries, tag)

    if os.path.exists(hash_file) and open(hash_file).read().strip() == current:
        try:
            return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True, **settings)
        except Exception as e:
            print(f"Index cache unreadable, rebuilding: {e}")

    docs = [
        Document(page_content=_entry_text(e), metadata={"id": e["id"], "title": e["title"],
                                                        "source": e["source"], "body": e["text"],
                                                        "reviewed": bool(e.get("reviewed", False))})
        for e in entries
    ]
    db = FAISS.from_documents(docs, embeddings, **settings)
    os.makedirs(index_dir, exist_ok=True)
    db.save_local(index_dir)
    with open(hash_file, "w") as f:
        f.write(current)
    return db


def get_embeddings():
    global _embeddings
    with _lock:
        if _embeddings is None:
            _embeddings = _default_embeddings()
        return _embeddings


def set_embeddings(embeddings, index_dir=None, tag="custom"):
    global _embeddings, _embeddings_tag, _index_dir, _db, _failed
    with _lock:
        _embeddings, _embeddings_tag, _db, _failed = embeddings, tag, None, False
        if index_dir:
            _index_dir = index_dir


def _get_db():
    global _embeddings, _db, _failed
    if _db is not None or _failed:
        return _db
    with _lock:
        if _db is None and not _failed:
            try:
                if _embeddings is None:
                    _embeddings = _default_embeddings()
                _db = _build_or_load(_embeddings, load_entries(), _index_dir, _embeddings_tag)
            except Exception as e:
                print(f"RAG engine unavailable: {e}")
                _failed = True
    return _db


def retrieve_reference(query, k=None, min_similarity=None, margin=None):
    db = _get_db()
    if db is None or not query.strip():
        return []
    k = TOP_K if k is None else k
    threshold = MIN_SIMILARITY if min_similarity is None else min_similarity
    window = MARGIN if margin is None else margin
    try:
        hits = db.similarity_search_with_score(query[:600], k=k)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
    results = []
    best = max((score for _, score in hits), default=0.0)
    for doc, score in hits:
        if score >= threshold and score >= best - window:
            results.append({
                "id": doc.metadata["id"],
                "title": doc.metadata["title"],
                "source": doc.metadata["source"],
                "text": doc.metadata["body"],
                "reviewed": doc.metadata.get("reviewed", False),
                "score": round(float(score), 3),
            })
    return results


def warm_up():
    if _get_db() is None:
        raise RuntimeError("Could not build the FAISS index")
