"""Lightweight ANN store for embeddings.

Uses FAISS if available, numpy brute-force otherwise.
For fewer than ~10k vectors, numpy is faster than FAISS overhead.
"""
import os
from typing import List, Optional, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[int] = []
        self._faiss = None
        if os.environ.get("USE_FAISS", "auto").lower() in ("1", "true", "yes", "on", "auto"):
            try:
                import faiss  # noqa
                self._faiss = faiss
            except Exception:
                self._faiss = None

    def build(self, vectors: List[List[float]], ids: Optional[List[int]] = None) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        # Normalize for cosine via inner product.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        self._vectors = arr
        self._ids = ids if ids is not None else list(range(len(vectors)))

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        if self._vectors is None or len(self._vectors) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        n = np.linalg.norm(q) or 1.0
        q = q / n
        scores = self._vectors @ q
        idx = np.argsort(-scores)[:top_k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in idx]