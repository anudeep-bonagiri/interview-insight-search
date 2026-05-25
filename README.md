# Interview Insight Search

[![CI](https://github.com/anudeep-bonagiri/interview-insight-search/actions/workflows/ci.yml/badge.svg)](https://github.com/anudeep-bonagiri/interview-insight-search/actions/workflows/ci.yml)

Semantic search and grounded insight synthesis over customer-research interview
transcripts. Ask a question in plain English, get back the passages that answer
it, cited down to the interview and timestamp. Optionally, get a Claude-written
synthesis that stays within what the transcripts support.

This is a small, runnable demo of the problems a customer-research platform
deals with: searching across many hours of interviews, tracing insights back to
their source, and measuring whether retrieval is any good.

```
$ iis search "why are customers thinking about cancelling" -k 2

Top 2 passages for: "why are customers thinking about cancelling"  (mode: hybrid)

  1. score=0.033  [int-002 00:00:06-00:00:51, Sofia, Co-founder]
     Pricing concerns and considering a switch
     ... the price keeps creeping. We started at the base plan and now we're
     paying per seat for people who barely log in ...

  2. score=0.032  [int-002 00:02:34-00:03:24, Sofia, Co-founder]
     Pricing concerns and considering a switch
     Sofia: Cost is number one. Number two is that the QuickBooks sync has been
     flaky for us, so we're already half living in two tools ...
```

Neither passage contains the word "cancelling". Retrieval matched on meaning.

## What it solves

A research platform lives or dies on retrieval and trust:

- **Find the right moment across many interviews.** A researcher does not
  remember which of 200 interviews mentioned a pricing objection, and keyword
  search misses it when the participant said "the math doesn't work" instead of
  "too expensive."
- **Keep every insight traceable.** An insight you cannot trace to a real quote
  is a liability. Each result carries `[interview-id, timestamp range,
  participant]`, and the synthesis cites those numbers inline.
- **Know whether it works.** "It feels good" is not a quality bar. This ships
  with a labeled evaluation set, three retrieval modes you can compare, and a
  test that fails CI when retrieval quality drops.

## Quickstart

```bash
git clone https://github.com/anudeep-bonagiri/interview-insight-search
cd interview-insight-search
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: numpy, pyyaml, fastembed (no torch, no API key)

iis build-index             # embeds the transcripts (downloads a ~130MB model once)
iis search "problems with mobile receipt scanning"
iis eval --compare          # retrieval quality across all three modes
```

Optional extras:

```bash
pip install -e ".[synthesis]"   # Claude-grounded answers (needs ANTHROPIC_API_KEY)
pip install -e ".[ui]"          # Streamlit web UI
export ANTHROPIC_API_KEY=sk-ant-...
iis ask "what do customers love most?"
streamlit run app/streamlit_app.py
```

Everything except `ask` runs with no API key. If `ANTHROPIC_API_KEY` is unset,
`ask` falls back to returning the top passages.

## Retrieval modes

The same query can be served three ways, selectable with `--mode`:

- `dense`: semantic similarity from embeddings. Strong on paraphrase.
- `lexical`: Okapi BM25. Strong on exact terms like product names and error
  strings.
- `hybrid` (default): both, combined with Reciprocal Rank Fusion. It keeps the
  semantic wins and recovers the exact-term hits that dense alone can miss.

## How it works

```
transcripts (JSON, speaker turns + timestamps)
   |  chunk by turns (~90 words, 1-turn overlap, ranges tracked)
   v
chunks --embed--> dense vector index  --\
   |                                      >-- Reciprocal Rank Fusion --> ranked
   |              BM25 lexical index   --/                                passages
   |                                                                        |
   |   query ----------------------------------------------------------------+
   v                                                                        |
ranked passages with citations --> Claude synthesis (grounded, optional)    |
   |                                                                        |
   +--> evaluation harness (recall@k, MRR, nDCG vs. labeled queries) <------+
```

- **Chunking** (`transcripts.py`) groups consecutive turns to a target word
  count with a one-turn overlap, so a participant's answer is never severed from
  the question that prompted it. Each chunk records the turn range it covers,
  which is how the evaluator maps relevance labels onto retrieved chunks.
- **Embeddings** (`embeddings.py`) sit behind one interface. The default is
  `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim, ONNX): real semantic
  quality with no torch and no API key. A dependency-free `HashingEmbedder` is
  the deterministic, offline fallback used in tests and CI.
- **Retrieval** (`search.py`, `index.py`, `lexical.py`) runs an exact cosine
  scan for dense, Okapi BM25 for lexical, and fuses their rankings for hybrid.
  The corpus is small, so exact scans are the simplest correct choice. The
  `build/search/save/load` interface is the seam where FAISS or pgvector slots
  in at scale.
- **Synthesis** (`synthesize.py`) uses the Anthropic SDK. The grounding rules
  live in a cached system prompt; retrieved passages go in the user turn. The
  model cites by number and states when the evidence is insufficient instead of
  filling the gap.

## Evaluation

Retrieval is only trustworthy if it is measured. `data/eval/queries.yaml` labels
10 research questions against the specific interview turns that answer them.
Labels are at the turn level, so they stay valid regardless of chunking. A chunk
counts as relevant if it overlaps any labeled turn.

`iis eval --compare` on the default embedder (`BAAI/bge-small-en-v1.5`,
21 chunks):

| mode    | hit@1 | hit@3 | hit@5 | recall@5 | nDCG@10 | MRR  |
|---------|-------|-------|-------|----------|---------|------|
| dense   | 0.80  | 1.00  | 1.00  | 0.87     | 0.89    | 0.90 |
| lexical | 0.80  | 0.90  | 1.00  | 0.74     | 0.80    | 0.87 |
| hybrid  | 0.90  | 1.00  | 1.00  | 0.83     | 0.90    | 0.95 |

Hybrid is the default because it improves the first-result metrics that matter
most when a researcher wants the answer up top: MRR rises from 0.90 to 0.95,
hit@1 from 0.80 to 0.90, and nDCG from 0.89 to 0.90. Dense keeps a small edge in
deep recall (recall@10 0.95 vs 0.93), which is the kind of tradeoff the harness
exists to surface. `tests/test_retrieval_quality.py` asserts a floor on these
numbers so a future change that quietly degrades retrieval fails CI.

## Design decisions

- **Local-first, zero-key core.** Retrieval and eval run with no API key and no
  GPU. A reviewer can clone the repo and see real numbers in two commands. The
  LLM layer is additive, not required.
- **fastembed over a torch stack.** Same retrieval quality, a much lighter
  install, and a faster cold start. The clone-and-run experience is what gets
  judged here, so I optimized for it.
- **Eval before polish.** I wrote the labeled set and the metrics early, so every
  later change (hybrid included) was a measured decision rather than a guess.
- **Citations are built in, not bolted on.** A `Chunk` knows its source and
  timestamp range, a `SearchResult` exposes a citation, and synthesis cites by
  rank.
- **Grounded, not confident.** The synthesis prompt forbids outside knowledge
  and requires an evidence-confidence line, because a fluent answer with no basis
  is the worst output a research tool can give.
- **A swap seam for scale.** Nothing assumes the in-memory index. Moving to FAISS
  or pgvector means reimplementing one class.

## Project layout

```
data/interviews/        6 customer-research interviews (speaker turns + timestamps)
data/eval/queries.yaml  labeled relevance judgments for evaluation
src/interview_search/
  transcripts.py        load + chunk transcripts
  embeddings.py         pluggable embedders (fastembed default, hashing fallback)
  index.py              dense cosine index (save/load)
  lexical.py            Okapi BM25 index
  search.py             dense / lexical / hybrid retrieval, with citations
  synthesize.py         Claude-grounded answer synthesis
  evaluate.py           recall@k / MRR / nDCG over labeled queries
  cli.py                the `iis` command
app/streamlit_app.py    web UI
tests/                  chunking, search, modes, eval-math, real-embedder quality guard
.github/workflows/      CI: install and run the test suite on every push
```

## The data

The transcripts are synthetic interviews about a fictional small-business
expense app ("Maple"), hand-written to contain realistic, recurring research
themes: onboarding friction, pricing, integration reliability, mobile OCR
accuracy, support, reporting, and churn risk. Synthetic data keeps the demo
self-contained and free of privacy concerns, and lets me hand-label a precise
evaluation set. The pipeline is content-agnostic: drop real transcripts (with
`id`, `turns`, `speaker`, `ts`, `text`) into `data/interviews/` and rebuild.

## Deploy the web UI

The repo is ready for Streamlit Community Cloud (free). `requirements.txt`
installs the package and the UI, and `.streamlit/config.toml` sets the theme.

1. Push the repo to GitHub (done).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. New app, then point it at this repo:
   - Repository: `anudeep-bonagiri/interview-insight-search`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

   Or use the prefilled link:
   https://share.streamlit.io/deploy?repository=anudeep-bonagiri/interview-insight-search&branch=main&mainModule=app/streamlit_app.py
4. (Optional, for the synthesis tab) In the app's Settings, Secrets, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   Search and eval work without it.

First load builds the index and downloads the embedding model once, so it takes
a moment; after that it is cached.

## What's next

- **Cross-encoder reranking** on top of hybrid, with the eval harness deciding
  whether it earns its latency.
- **Real ASR transcripts** from Whisper or Deepgram, where timestamps and
  speaker diarization come for free.
- **Answer-faithfulness evals**: score the synthesis for citation accuracy and
  unsupported claims, not just retrieval.
- **A persistent index** (pgvector) for millions of chunks.

Built by Anudeep Bonagiri. The synthesis layer uses Claude via the official
Anthropic SDK.
