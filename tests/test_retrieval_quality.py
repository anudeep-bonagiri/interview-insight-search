"""Quality regression guard on the real (fastembed) embedder.

This test checks that retrieval works and keeps working. It asserts a quality
floor on the labeled query set using real semantic embeddings. It is skipped
(not failed) when the model cannot be loaded, for example in an offline CI
environment, so the rest of the suite stays runnable everywhere.
"""

import pytest

from interview_search.evaluate import evaluate, load_queries
from interview_search.search import SearchEngine


@pytest.fixture(scope="module")
def fastembed_engine():
    pytest.importorskip("fastembed")
    try:
        return SearchEngine.build(embedder_name="fastembed")
    except Exception as e:  # model download/runtime unavailable offline
        pytest.skip(f"fastembed model unavailable: {e}")


def test_semantic_retrieval_quality_floor(fastembed_engine):
    queries = load_queries()
    report = evaluate(fastembed_engine, queries, k_values=(1, 3, 5))
    agg = report.aggregate

    # Floors are deliberately below observed performance so the guard catches
    # real regressions without being flaky. See README for current numbers.
    assert agg["hit"][5] >= 0.8, f"hit@5={agg['hit'][5]:.2f}"
    assert agg["mrr"] >= 0.5, f"mrr={agg['mrr']:.2f}"
    assert agg["recall"][5] >= 0.4, f"recall@5={agg['recall'][5]:.2f}"
