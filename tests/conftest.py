import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.fake_embeddings import HashingEmbeddings  # noqa: E402


@pytest.fixture
def fake_embeddings():
    return HashingEmbeddings()
