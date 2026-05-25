"""Search plumbing: deterministic, ranked, round-trips through save/load.

These use the dependency-free HashingEmbedder so they run offline and produce
identical results every time.
"""

import numpy as np

from interview_search.embeddings import HashingEmbedder, get_embedder
from interview_search.search import SearchEngine


def _engine():
    return SearchEngine.build(embedder_name="hashing")


def test_results_ranked_descending():
    engine = _engine()
    results = engine.search("receipt scanning on my phone", k=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_lexical_query_finds_expected_interview():
    # The hashing embedder is lexical, so a query with distinctive terms should
    # surface the interview that actually discusses them.
    engine = _engine()
    results = engine.search("QuickBooks sync batch failed silently", k=3)
    assert any(r.chunk.interview_id == "int-004" for r in results)


def test_embeddings_are_unit_normalized():
    emb = HashingEmbedder()
    vecs = emb.embed_documents(["hello world", "another document here"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_index_save_load_roundtrip(tmp_path):
    engine = _engine()
    engine.index.save(tmp_path / "idx")
    loaded = SearchEngine.load(tmp_path / "idx")
    assert len(loaded.index.chunks) == len(engine.index.chunks)
    a = engine.search("pricing per seat", k=3)
    b = loaded.search("pricing per seat", k=3)
    assert [r.chunk.chunk_id for r in a] == [r.chunk.chunk_id for r in b]


def test_get_embedder_factory():
    assert get_embedder("hashing").name.startswith("hashing")
