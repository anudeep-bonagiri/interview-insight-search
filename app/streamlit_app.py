"""Streamlit UI for interview-insight-search.

    streamlit run app/streamlit_app.py

Visual semantic search over the interview transcripts, with grounded
citations and an optional Claude-synthesized answer. The search engine is built
once and cached for the session.
"""

from __future__ import annotations

import os

import streamlit as st

from interview_search.search import SearchEngine
from interview_search.synthesize import synthesize

st.set_page_config(page_title="Interview Insight Search", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Building the search index (downloads a small model on first run)...")
def get_engine(embedder_name: str) -> SearchEngine:
    return SearchEngine.build(embedder_name=embedder_name)


def main() -> None:
    st.title("Interview Insight Search")
    st.caption(
        "Semantic search over customer-research interview transcripts, with "
        "citations back to the exact moment in each interview."
    )

    with st.sidebar:
        st.header("Settings")
        embedder = st.selectbox(
            "Embedder",
            ["fastembed", "hashing"],
            help="fastembed = real semantic embeddings (BGE). hashing = lexical fallback, no download.",
        )
        mode = st.selectbox(
            "Retrieval mode",
            ["hybrid", "dense", "lexical"],
            help="hybrid fuses semantic and BM25. dense is semantic only. lexical is BM25 only.",
        )
        k = st.slider("Passages to retrieve", 3, 12, 6)
        synth = st.toggle("Synthesize an answer with Claude", value=False)
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if synth and not has_key:
            st.warning("Set ANTHROPIC_API_KEY to enable synthesis. Showing passages only.")
        st.markdown("---")
        st.markdown("**Try:**")
        for example in [
            "Why are customers thinking about cancelling?",
            "Problems with mobile receipt scanning",
            "What do customers love most?",
            "Issues with the QuickBooks sync",
        ]:
            st.markdown(f"- {example}")

    engine = get_engine(embedder)

    query = st.text_input(
        "Ask a research question or search the interviews",
        placeholder="e.g. What makes onboarding difficult?",
    )
    if not query:
        return

    results = engine.search(query, k=k, mode=mode)

    if synth:
        with st.spinner("Synthesizing a grounded answer..."):
            result = synthesize(query, results)
        st.subheader("Synthesized answer")
        st.markdown(result.answer)
        if result.model:
            st.caption(f"model: {result.model}")
        if result.note:
            st.info(result.note)
        st.markdown("---")

    st.subheader(f"Top {len(results)} passages")
    for r in results:
        with st.container(border=True):
            cols = st.columns([0.8, 0.2])
            cols[0].markdown(f"**{r.chunk.interview_title}**  \n`{r.chunk.citation}`")
            cols[1].metric("similarity", f"{r.score:.3f}")
            st.markdown(r.chunk.text.replace("\n", "  \n"))


if __name__ == "__main__":
    main()
