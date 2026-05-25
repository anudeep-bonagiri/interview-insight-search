"""Evaluation harness: metric math is correct and the report is sane.

Metric arithmetic is verified on a hand-built scenario. The harness is then run
end-to-end on the real data with the deterministic hashing embedder to confirm
it produces well-formed, monotonic metrics.
"""

import math

from interview_search.evaluate import (
    Query,
    chunk_is_relevant,
    evaluate,
    load_queries,
    _dcg,
)
from interview_search.search import SearchEngine
from interview_search.transcripts import Chunk


def _chunk(interview_id, start_turn, end_turn):
    return Chunk(
        chunk_id=f"{interview_id}::c{start_turn:03d}",
        interview_id=interview_id,
        interview_title="t",
        participant_name="p",
        participant_role="r",
        date="2026-01-01",
        speakers=["p"],
        start_ts="00:00:00",
        end_ts="00:00:10",
        start_turn=start_turn,
        end_turn=end_turn,
        text="x",
    )


def test_chunk_is_relevant_overlap():
    gold = {"int-001": {3, 4, 5}}
    assert chunk_is_relevant(_chunk("int-001", 2, 3), gold)      # overlaps turn 3
    assert not chunk_is_relevant(_chunk("int-001", 6, 8), gold)  # no overlap
    assert not chunk_is_relevant(_chunk("int-002", 3, 5), gold)  # wrong interview


def test_dcg_matches_definition():
    # Relevances [1, 0, 1] -> 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5
    assert math.isclose(_dcg([1, 0, 1]), 1.0 + 0.5)


def test_loaded_queries_have_gold():
    queries = load_queries()
    assert len(queries) >= 5
    for q in queries:
        assert q.gold
        assert all(isinstance(turns, set) for turns in q.gold.values())


def test_evaluate_metrics_are_well_formed_and_monotonic():
    engine = SearchEngine.build(embedder_name="hashing")
    queries = load_queries()
    report = evaluate(engine, queries, k_values=(1, 3, 5, 10))
    agg = report.aggregate

    # Every metric is a valid fraction.
    for k in (1, 3, 5, 10):
        assert 0.0 <= agg["hit"][k] <= 1.0
        assert 0.0 <= agg["recall"][k] <= 1.0
        assert 0.0 <= agg["ndcg"][k] <= 1.0
    assert 0.0 <= agg["mrr"] <= 1.0

    # Hit-rate and recall are non-decreasing in k.
    assert agg["hit"][1] <= agg["hit"][5] <= agg["hit"][10]
    assert agg["recall"][1] <= agg["recall"][5] <= agg["recall"][10]

    # Even the lexical fallback should retrieve something relevant overall.
    assert agg["hit"][10] > 0.0


def test_perfect_ranking_scores_one():
    # A tiny synthetic engine where the top result is always gold.
    engine = SearchEngine.build(embedder_name="hashing")
    # Construct a query whose gold is exactly the top retrieved chunk.
    top = engine.search("receipt scanning", k=1)[0].chunk
    q = Query(id="synthetic", text="receipt scanning", gold={top.interview_id: set(top.covered_turns())})
    report = evaluate(engine, [q], k_values=(1,))
    assert report.aggregate["hit"][1] == 1.0
    assert report.aggregate["mrr"] == 1.0
