"""Okapi BM25 lexical retrieval over chunk text.

Dense embeddings are strong on paraphrase but can miss exact tokens that carry
meaning in this domain: product names, error strings, "QuickBooks", "per seat".
BM25 is the standard lexical counterweight. It is cheap to build from the chunk
text already in the index, so the hybrid retriever can fuse the two rankings
without storing anything extra on disk.

Implemented in plain numpy. The corpus is small, so an exact pass over every
document per query is the simplest correct option.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .transcripts import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._doc_tokens = [_tokenize(c.text) for c in chunks]
        self._doc_freqs = [Counter(toks) for toks in self._doc_tokens]
        self._doc_len = np.array([len(toks) for toks in self._doc_tokens], dtype=np.float32)
        self._avgdl = float(self._doc_len.mean()) if len(chunks) else 0.0

        n = len(chunks)
        df: Counter = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        # Okapi BM25 idf with the +1 smoothing that keeps it non-negative.
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self._avgdl == 0.0:
            return scores
        terms = _tokenize(query)
        for i, freqs in enumerate(self._doc_freqs):
            denom_len = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
            s = 0.0
            for term in terms:
                tf = freqs.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * (tf * (self.k1 + 1)) / (tf + denom_len)
            scores[i] = s
        return scores
