"""Grounded insight synthesis over retrieved interview passages.

This is the "share insights" layer: given a research question and the passages
retrieval found, Claude writes a synthesized answer that cites its evidence by
number and refuses to go beyond what the passages support. The aim is an answer
a researcher could paste into a report and trust, because every claim traces
back to a specific moment in a specific interview.

Design notes:
* Uses the official Anthropic SDK and defaults to claude-opus-4-7.
* The grounding rules live in a stable system prompt with a cache breakpoint,
  so repeated questions reuse the cached prefix. The retrieved passages (which
  vary per question) go in the user turn, after the cached prefix.
* Degrades gracefully: with no API key, it returns the retrieved passages and a
  note, so the demo is still useful offline. The retrieval + eval layers never
  depend on this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import config
from .search import SearchResult

SYSTEM_PROMPT = """You are a customer-research analyst. You answer questions \
using ONLY the interview excerpts provided in the user message. Each excerpt is \
numbered and labeled with its source (interview id, timestamp range, and \
participant).

Rules:
- Ground every claim in the excerpts. After each claim, cite the supporting \
excerpt(s) inline like [1] or [2][4].
- Do not use outside knowledge or invent details. If the excerpts do not \
contain enough information to answer, say so plainly and state what is missing.
- Prefer specifics (what participants actually said) over generic summary.
- Be concise: a short synthesis a researcher could drop into a report, not an \
essay. Note when participants disagree.
- End with a one-line "Confidence:" note reflecting how well the excerpts \
support the answer."""


@dataclass
class SynthesisResult:
    question: str
    answer: str
    model: str | None
    used_llm: bool
    sources: list[SearchResult]
    note: str | None = None


def _format_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.rank}] {r.citation}\n"
            f"interview: {r.chunk.interview_title}\n"
            f"{r.chunk.text}"
        )
    return "\n\n".join(blocks)


def _fallback(question: str, results: list[SearchResult], note: str) -> SynthesisResult:
    """No API key (or SDK unavailable): return the retrieved evidence verbatim."""
    lines = [
        "Synthesis is disabled (no Anthropic API key), so here are the most "
        "relevant passages retrieved for your question:\n",
    ]
    for r in results:
        snippet = r.chunk.text.replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        lines.append(f"[{r.rank}] {r.citation}\n    {snippet}")
    return SynthesisResult(
        question=question,
        answer="\n".join(lines),
        model=None,
        used_llm=False,
        sources=results,
        note=note,
    )


def synthesize(
    question: str,
    results: list[SearchResult],
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1500,
) -> SynthesisResult:
    """Answer `question` grounded in `results`, citing sources by rank number."""
    if not results:
        return SynthesisResult(
            question=question,
            answer="No relevant passages were retrieved, so there is nothing to "
            "synthesize. Try rephrasing the question.",
            model=None,
            used_llm=False,
            sources=results,
        )

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback(
            question,
            results,
            note="Set ANTHROPIC_API_KEY to enable Claude-grounded synthesis.",
        )

    try:
        import anthropic
    except ImportError:
        return _fallback(question, results, note="The `anthropic` package is not installed.")

    model = model or config.SYNTHESIS_MODEL
    context = _format_context(results)
    user_content = (
        f"Research question: {question}\n\n"
        f"Interview excerpts:\n\n{context}\n\n"
        "Answer the research question using only these excerpts, with inline "
        "citations like [1]."
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Stable prefix, cached so repeated questions do not re-pay for it.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as e:
        return _fallback(question, results, note=f"Claude request failed: {e}")

    answer = "".join(block.text for block in response.content if block.type == "text").strip()
    return SynthesisResult(
        question=question,
        answer=answer,
        model=model,
        used_llm=True,
        sources=results,
    )
