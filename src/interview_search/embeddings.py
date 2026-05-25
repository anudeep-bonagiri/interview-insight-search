"""Pluggable text embedders.

Two backends, one interface:

* ``FastEmbedEmbedder`` -- real semantic embeddings via fastembed (ONNX). Default
  model is BAAI/bge-small-en-v1.5: 384-dim, strong retrieval quality, ~130 MB,
  no torch and no API key. The model downloads once on first use.
* ``HashingEmbedder`` -- a deterministic, dependency-free feature-hashing
  embedder. It captures lexical overlap only, but it needs no network and gives
  byte-identical vectors every run, so tests and offline environments use it.

Both return L2-normalized vectors, so a dot product equals cosine similarity.
Document and query embedding are separate methods because instruction-tuned
models (like bge) prepend a query prefix that improves retrieval.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (lexical, no dependencies)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """Real semantic embeddings via fastembed (ONNX, no torch, no API key)."""

    def __init__(self, model_name: str = config.FASTEMBED_MODEL) -> None:
        from fastembed import TextEmbedding  # imported lazily; heavy-ish

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe dimensionality once so callers can rely on `dim`.
        self.dim = int(next(iter(self._model.embed(["dimension probe"]))).shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        embed = getattr(self._model, "passage_embed", self._model.embed)
        matrix = np.vstack(list(embed(texts))).astype(np.float32)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        embed = getattr(self._model, "query_embed", self._model.embed)
        vec = np.asarray(next(iter(embed([text]))), dtype=np.float32)
        return _l2_normalize(vec[None, :])[0]


def get_embedder(name: str | None = None) -> Embedder:
    """Resolve an embedder by short name or fastembed model id.

    "fastembed"  -> default fastembed model
    "hashing"    -> deterministic fallback
    any other id -> treated as a fastembed model name (e.g. "BAAI/bge-base-en-v1.5")
    """
    name = name or config.DEFAULT_EMBEDDER
    # "hashing" or a saved "hashing-<dim>" identifier.
    if name == "hashing" or name.startswith("hashing-"):
        _, _, suffix = name.partition("-")
        return HashingEmbedder(dim=int(suffix)) if suffix.isdigit() else HashingEmbedder()
    if name == "fastembed":
        return FastEmbedEmbedder()
    return FastEmbedEmbedder(model_name=name)
