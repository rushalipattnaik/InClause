import re
import zlib

import numpy as np
from langchain_core.embeddings import Embeddings

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by",
         "with", "as", "that", "this", "it", "at", "from", "which", "can", "will", "if", "not", "no"}


class HashingEmbeddings(Embeddings):
    def __init__(self, dim=512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            if w not in _STOP and len(w) > 2:
                v[zlib.crc32(w.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)
