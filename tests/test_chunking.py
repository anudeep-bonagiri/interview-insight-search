"""Chunking must cover every turn and produce traceable, well-formed chunks."""

from interview_search import config
from interview_search.transcripts import build_chunks, chunk_interview, load_interviews


def test_chunks_cover_every_turn():
    for interview in load_interviews():
        chunks = chunk_interview(interview)
        covered = set()
        for c in chunks:
            covered.update(c.covered_turns())
        assert covered == set(range(len(interview.turns))), interview.id


def test_chunk_ids_unique_and_formatted():
    chunks = build_chunks()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all("::c" in cid for cid in ids)


def test_chunk_turn_ranges_are_contiguous_and_ordered():
    for interview in load_interviews():
        for c in chunk_interview(interview):
            assert c.start_turn <= c.end_turn
            assert c.start_turn >= 0
            assert c.end_turn < len(interview.turns)


def test_citation_includes_source_metadata():
    chunk = build_chunks()[0]
    citation = chunk.citation()
    assert chunk.interview_id in citation
    assert chunk.participant_name in citation
    assert chunk.start_ts in citation


def test_target_words_is_respected_loosely():
    # Each chunk (except the last per interview) should reach the target.
    for interview in load_interviews():
        chunks = chunk_interview(interview, target_words=config.TARGET_WORDS)
        for c in chunks[:-1]:
            words = sum(len(line.split()) for line in c.text.splitlines())
            assert words >= config.TARGET_WORDS // 2
