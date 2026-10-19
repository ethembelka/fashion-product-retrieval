"""Exact cosine search with FAISS IndexFlatIP over L2-normalised embeddings."""

from __future__ import annotations

import faiss
import numpy as np


class FlatIPIndex:
    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)

    def add(self, emb: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices), each (n_queries, k), best first."""
        k = min(k, self.index.ntotal)
        return self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)

    @property
    def size_mb(self) -> float:
        return self.index.ntotal * self.index.d * 4 / 1e6
