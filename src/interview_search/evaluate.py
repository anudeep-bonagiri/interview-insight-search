"""Retrieval evaluation against the labeled query set.

Relevance labels live at the (interview, turn) level (see data/eval/queries.yaml).
A retrieved chunk is relevant to a query if the chunk's covered turn range
intersects any labeled turn for that query, in the same interview. This keeps
the labels independent of chunking parameters: re-chunk however you like and the
gold standard still applies.

Metrics reported (averaged over queries):
  hit@k     -- fraction of queries with >=1 relevant chunk in the top k
  recall@k  -- relevant chunks retrieved in top k / all relevant chunks in index
  ndcg@k    -- normalized discounted cumulative gain, binary relevance
  mrr       -- mean reciprocal rank of the first relevant chunk
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config
from .search import SearchEngine
from .transcripts import Chunk


@dataclass
class Query:
    id: str
    text: str
    # interview_id -> set of relevant turn indices
    gold: dict[str, set[int]]


def load_queries(path: Path | str | None = None) -> list[Query]:
    path = Path(path) if path else config.EVAL_PATH
    raw = yaml.safe_load(Path(path).read_text())
    queries: list[Query] = []
    for item in raw["queries"]:
        gold: dict[str, set[int]] = {}
        for rel in item["relevant"]:
            gold.setdefault(rel["interview"], set()).update(rel["turns"])
        queries.append(Query(id=item["id"], text=item["query"], gold=gold))
    return queries


def chunk_is_relevant(chunk: Chunk, gold: dict[str, set[int]]) -> bool:
    relevant_turns = gold.get(chunk.interview_id)
    if not relevant_turns:
        return False
    return any(t in relevant_turns for t in chunk.covered_turns())


def _count_relevant_in_index(engine: SearchEngine, gold: dict[str, set[int]]) -> int:
    return sum(1 for c in engine.index.chunks if chunk_is_relevant(c, gold))


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


@dataclass
class EvalReport:
    k_values: list[int]
    per_query: list[dict]
    aggregate: dict

    def format_table(self) -> str:
        lines = []
        header = f"{'query':<6} " + " ".join(f"hit@{k:<3}" for k in self.k_values)
        header += " ".join(f"rec@{k:<3}" for k in self.k_values) + f" {'mrr':<6}"
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.per_query:
            cells = f"{row['id']:<6} "
            cells += " ".join(f"{row['hit'][k]:<6.0f}" for k in self.k_values)
            cells += " ".join(f"{row['recall'][k]:<6.2f}" for k in self.k_values)
            cells += f" {row['rr']:<6.2f}"
            lines.append(cells)
        lines.append("-" * len(header))
        agg = self.aggregate
        cells = f"{'MEAN':<6} "
        cells += " ".join(f"{agg['hit'][k]:<6.2f}" for k in self.k_values)
        cells += " ".join(f"{agg['recall'][k]:<6.2f}" for k in self.k_values)
        cells += f" {agg['mrr']:<6.2f}"
        lines.append(cells)
        ndcg = self.aggregate["ndcg"]
        lines.append("")
        lines.append("nDCG: " + "  ".join(f"@{k}={ndcg[k]:.2f}" for k in self.k_values))
        return "\n".join(lines)


def evaluate(
    engine: SearchEngine,
    queries: list[Query],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    mode: str | None = None,
) -> EvalReport:
    max_k = max(k_values)
    per_query: list[dict] = []

    for q in queries:
        results = engine.search(q.text, k=max_k, mode=mode)
        rels = [1 if chunk_is_relevant(r.chunk, q.gold) else 0 for r in results]
        total_relevant = _count_relevant_in_index(engine, q.gold)

        hit = {k: float(any(rels[:k])) for k in k_values}
        recall = {
            k: (sum(rels[:k]) / total_relevant if total_relevant else 0.0)
            for k in k_values
        }
        ideal = [1] * total_relevant
        ndcg = {
            k: (_dcg(rels[:k]) / _dcg(ideal[:k]) if total_relevant else 0.0)
            for k in k_values
        }
        rr = 0.0
        for rank, rel in enumerate(rels, start=1):
            if rel:
                rr = 1.0 / rank
                break
        per_query.append(
            {"id": q.id, "hit": hit, "recall": recall, "ndcg": ndcg, "rr": rr}
        )

    n = len(queries)
    aggregate = {
        "hit": {k: sum(r["hit"][k] for r in per_query) / n for k in k_values},
        "recall": {k: sum(r["recall"][k] for r in per_query) / n for k in k_values},
        "ndcg": {k: sum(r["ndcg"][k] for r in per_query) / n for k in k_values},
        "mrr": sum(r["rr"] for r in per_query) / n,
    }
    return EvalReport(list(k_values), per_query, aggregate)
