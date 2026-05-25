"""Command-line interface for interview-insight-search.

    iis build-index                 # embed transcripts into a saved index
    iis search "onboarding pain"    # semantic search with cited passages
    iis ask "why do users churn?"   # Claude-synthesized, grounded answer
    iis eval                        # retrieval quality on the labeled set

Run `iis <command> -h` for per-command options. Everything works without an
index on disk; search/ask/eval build one in memory from the transcripts when
none is found.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import config
from .evaluate import evaluate, load_queries
from .search import MODES, SearchEngine
from .synthesize import synthesize


def _get_engine(args) -> SearchEngine:
    """Load a saved index, or build one in memory from the transcripts."""
    index_dir = Path(getattr(args, "index", None) or config.INDEX_DIR)
    rebuild = getattr(args, "rebuild", False)
    embedder = getattr(args, "embedder", None)
    if index_dir.exists() and not rebuild:
        engine = SearchEngine.load(index_dir)
        if embedder and embedder != engine.embedder.name and embedder != "fastembed":
            print(
                f"note: saved index uses '{engine.embedder.name}'; "
                f"--embedder {embedder} ignored. Use build-index to change it.",
                file=sys.stderr,
            )
        return engine
    print(
        f"No saved index at {index_dir}; building one in memory "
        f"(embedder: {embedder or config.DEFAULT_EMBEDDER}).",
        file=sys.stderr,
    )
    return SearchEngine.build(embedder_name=embedder)


def _print_results(results, show_text=True) -> None:
    for r in results:
        print(f"  {r.rank}. score={r.score:.3f}  {r.citation}")
        print(f"     {r.chunk.interview_title}")
        if show_text:
            snippet = " ".join(r.chunk.text.split())
            for line in textwrap.wrap(snippet, width=88)[:4]:
                print(f"     {line}")
        print()


def cmd_build_index(args) -> int:
    engine = SearchEngine.build(
        embedder_name=args.embedder,
        target_words=args.target_words,
        overlap_turns=args.overlap,
    )
    out = Path(args.out or config.INDEX_DIR)
    engine.index.save(out)
    print(
        f"Indexed {len(engine.index.chunks)} chunks "
        f"({engine.index.dim}-dim, embedder: {engine.embedder.name})\n"
        f"Saved to {out}"
    )
    return 0


def cmd_search(args) -> int:
    engine = _get_engine(args)
    results = engine.search(args.query, k=args.k, mode=args.mode)
    mode = args.mode or config.DEFAULT_MODE
    print(f'\nTop {len(results)} passages for: "{args.query}"  (mode: {mode})\n')
    _print_results(results)
    return 0


def cmd_ask(args) -> int:
    engine = _get_engine(args)
    results = engine.search(args.question, k=args.k, mode=args.mode)
    result = synthesize(args.question, results, model=args.model)
    print(f'\nQuestion: {args.question}\n')
    print(result.answer)
    if result.used_llm:
        print(f"\n(model: {result.model})")
        print("\nSources:")
        _print_results(results, show_text=False)
    if result.note:
        print(f"\nnote: {result.note}")
    return 0


def cmd_eval(args) -> int:
    engine = _get_engine(args)
    queries = load_queries()
    k_values = tuple(int(k) for k in args.k.split(","))
    header = (f"embedder: {engine.embedder.name}, "
              f"{len(queries)} queries, {len(engine.index.chunks)} chunks")

    if args.compare:
        print(f"\nRetrieval comparison across modes  ({header})\n")
        cols = " ".join(f"hit@{k:<3}" for k in k_values)
        cols += " ".join(f"rec@{k:<3}" for k in k_values)
        line = f"{'mode':<9} {cols} {'nDCG@'+str(max(k_values)):<8} {'mrr':<6}"
        print(line)
        print("-" * len(line))
        for mode in MODES:
            agg = evaluate(engine, queries, k_values=k_values, mode=mode).aggregate
            row = f"{mode:<9} "
            row += " ".join(f"{agg['hit'][k]:<6.2f}" for k in k_values)
            row += " ".join(f"{agg['recall'][k]:<6.2f}" for k in k_values)
            row += f" {agg['ndcg'][max(k_values)]:<8.2f} {agg['mrr']:<6.2f}"
            print(row)
        return 0

    mode = args.mode or config.DEFAULT_MODE
    report = evaluate(engine, queries, k_values=k_values, mode=mode)
    print(f"\nRetrieval evaluation  (mode: {mode}, {header})\n")
    print(report.format_table())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iis",
        description="Semantic search and grounded insight synthesis over "
        "research interview transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="embed transcripts into a saved index")
    p_build.add_argument("--embedder", default=None, help="fastembed (default) | hashing | <model id>")
    p_build.add_argument("--out", default=None, help=f"output dir (default: {config.INDEX_DIR})")
    p_build.add_argument("--target-words", type=int, default=config.TARGET_WORDS)
    p_build.add_argument("--overlap", type=int, default=config.OVERLAP_TURNS)
    p_build.set_defaults(func=cmd_build_index)

    p_search = sub.add_parser("search", help="semantic search with cited passages")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5, help="number of passages (default 5)")
    p_search.add_argument("--mode", choices=MODES, default=None,
                          help=f"retrieval mode (default: {config.DEFAULT_MODE})")
    p_search.add_argument("--index", default=None)
    p_search.add_argument("--embedder", default=None)
    p_search.add_argument("--rebuild", action="store_true", help="build in memory, ignore saved index")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Claude-synthesized, grounded answer with citations")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=6, help="passages to ground on (default 6)")
    p_ask.add_argument("--mode", choices=MODES, default=None,
                       help=f"retrieval mode (default: {config.DEFAULT_MODE})")
    p_ask.add_argument("--model", default=None, help=f"Claude model (default: {config.SYNTHESIS_MODEL})")
    p_ask.add_argument("--index", default=None)
    p_ask.add_argument("--embedder", default=None)
    p_ask.add_argument("--rebuild", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="retrieval quality on the labeled query set")
    p_eval.add_argument("-k", default="1,3,5,10", help="comma-separated cutoffs (default 1,3,5,10)")
    p_eval.add_argument("--mode", choices=MODES, default=None,
                        help=f"retrieval mode (default: {config.DEFAULT_MODE})")
    p_eval.add_argument("--compare", action="store_true", help="compare all retrieval modes")
    p_eval.add_argument("--index", default=None)
    p_eval.add_argument("--embedder", default=None)
    p_eval.add_argument("--rebuild", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
