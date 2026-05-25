"""Lexical (BM25) and hybrid (RRF) retrieval modes.

Uses the deterministic hashing embedder so dense and hybrid runs are repeatable
offline.
"""

import numpy as np

from interview_search.lexical import BM25Index
from interview_search.search import MODES, SearchEngine, _reciprocal_rank_fusion
from interview_search.transcripts import build_chunks


def _engine():
    return SearchEngine.build(embedder_name="hashing")


def test_bm25_scores_distinctive_terms_highest():
    chunks = build_chunks()
    bm25 = BM25Index(chunks)
    scores = bm25.score_all("QuickBooks sync batch failed silently")
    best = chunks[int(np.argmax(scores))]
    assert best.interview_id == "int-004"
    assert scores.max() > 0


def test_lexical_mode_finds_exact_terms():
    engine = _engine()
    results = engine.search("QuickBooks sync", k=3, mode="lexical")
    assert any(r.chunk.interview_id == "int-004" for r in results)


def test_all_modes_return_ranked_results():
    engine = _engine()
    for mode in MODES:
        results = engine.search("pricing per seat cost", k=5, mode=mode)
        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_rrf_rewards_agreement():
    # Two rankers that agree should rank doc 0 first.
    a = np.array([0.9, 0.5, 0.1])
    b = np.array([0.8, 0.4, 0.2])
    fused = _reciprocal_rank_fusion([a, b])
    assert int(np.argmax(fused)) == 0
    assert fused.shape == (3,)


def test_hybrid_recovers_lexical_hit_for_product_term():
    # A product/term-heavy query should surface the integration interview in
    # hybrid mode, blending semantic and lexical signal.
    engine = _engine()
    results = engine.search("Xero connector reliability dashboard", k=5, mode="hybrid")
    assert any(r.chunk.interview_id == "int-004" for r in results)
