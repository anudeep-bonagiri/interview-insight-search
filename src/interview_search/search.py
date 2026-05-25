"""High-level search: turn a natural-language question into cited passages.

`SearchEngine` ties an embedder, a dense vector index, and a BM25 lexical index
together. It supports three retrieval modes:

    dense    semantic similarity only (good at paraphrase)
    lexical  BM25 only (good at exact terms: product names, error strings)
    hybrid   both, fused with Reciprocal Rank Fusion (the default)

Results carry everything needed to show a passage and trace it back to the
exact moment in the exact interview it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .embeddings import Embedder, get_embedder
from .index import VectorIndex
from .lexical import BM25Index
from .transcripts import Chunk, build_chunks

MODES = ("dense", "lexical", "hybrid")
_RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass
class SearchResult:
    rank: int
    score: float
    chunk: Chunk

    @property
    def citation(self) -> str:
        return self.chunk.citation()


def _reciprocal_rank_fusion(score_arrays: list[np.ndarray]) -> np.ndarray:
    """Fuse several score arrays by the rank each assigns, not the raw scores.

    RRF avoids having to normalize cosine against BM25: each ranker contributes
    1 / (k + rank) for the position it gives a document.
    """
    n = len(score_arrays[0])
    fused = np.zeros(n, dtype=np.float32)
    for scores in score_arrays:
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        fused += 1.0 / (_RRF_K + ranks + 1)
    return fused


class SearchEngine:
    def __init__(self, index: VectorIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.index.chunks)
        return self._bm25

    @classmethod
    def build(
        cls,
        data_dir: Path | str | None = None,
        embedder_name: str | None = None,
        target_words: int = config.TARGET_WORDS,
        overlap_turns: int = config.OVERLAP_TURNS,
    ) -> "SearchEngine":
        embedder = get_embedder(embedder_name)
        chunks = build_chunks(data_dir, target_words, overlap_turns)
        index = VectorIndex.build(chunks, embedder)
        return cls(index, embedder)

    @classmethod
    def load(cls, index_dir: Path | str | None = None) -> "SearchEngine":
        index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
        index = VectorIndex.load(index_dir)
        embedder = get_embedder(index.embedder_name)
        return cls(index, embedder)

    def _scores(self, query: str, mode: str) -> np.ndarray:
        if mode == "dense":
            return self.index.score_all(self.embedder.embed_query(query))
        if mode == "lexical":
            return self.bm25.score_all(query)
        if mode == "hybrid":
            dense = self.index.score_all(self.embedder.embed_query(query))
            lexical = self.bm25.score_all(query)
            return _reciprocal_rank_fusion([dense, lexical])
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    def search(self, query: str, k: int = 5, mode: str | None = None) -> list[SearchResult]:
        mode = mode or config.DEFAULT_MODE
        scores = self._scores(query, mode)
        k = min(k, len(self.index.chunks))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            SearchResult(rank=i + 1, score=float(scores[idx]), chunk=self.index.chunks[idx])
            for i, idx in enumerate(top)
        ]
