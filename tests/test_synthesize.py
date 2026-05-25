"""Synthesis context and the no-key fallback render real citations.

Guards against passing an unrendered citation (a bound method) into the prompt
or the fallback output.
"""

from interview_search.search import SearchEngine
from interview_search.synthesize import _format_context, synthesize


def _results():
    engine = SearchEngine.build(embedder_name="hashing")
    return engine.search("pricing per seat", k=3, mode="hybrid")


def test_context_contains_rendered_citation():
    ctx = _format_context(_results())
    assert "int-" in ctx
    assert "bound method" not in ctx


def test_fallback_without_key_returns_passages_with_citations(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = synthesize("why do customers churn", _results())
    assert result.used_llm is False
    assert "bound method" not in result.answer
    assert "int-" in result.answer
    assert result.note and "ANTHROPIC_API_KEY" in result.note


def test_empty_results_is_handled():
    result = synthesize("anything", [])
    assert result.used_llm is False
    assert result.answer
