"""A tiny in-memory vector index.

The corpus here is small (hundreds of chunks), so an exact cosine scan over a
normalized matrix is both fastest and simplest -- no approximate-nearest-neighbor
library needed. The interface (`build`, `search`, `save`, `load`) is the seam
where you would swap in FAISS or pgvector once the corpus grows to millions of
chunks; nothing else in the codebase would change.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .transcripts import Chunk


class VectorIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder_name: str) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings count mismatch")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder_name = embedder_name

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        matrix = embedder.embed_documents([c.text for c in chunks])
        return cls(chunks, matrix, embedder.name)

    def score_all(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query against every chunk.

        Vectors are unit-normalized, so the matrix-vector product is cosine.
        """
        return self.embeddings @ query_vec.astype(np.float32)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, cosine_score), highest first.

        argpartition finds the top-k in O(n) before we sort just those k.
        """
        scores = self.score_all(query_vec)
        k = min(k, len(self.chunks))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        (path / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], indent=2)
        )
        (path / "meta.json").write_text(
            json.dumps({"embedder_name": self.embedder_name, "dim": self.dim})
        )

    @classmethod
    def load(cls, path: Path | str) -> "VectorIndex":
        path = Path(path)
        embeddings = np.load(path / "embeddings.npy")
        chunks = [Chunk(**c) for c in json.loads((path / "chunks.json").read_text())]
        meta = json.loads((path / "meta.json").read_text())
        return cls(chunks, embeddings, meta["embedder_name"])
