# eval/run_eval.py
"""
Reproducible retrieval-quality eval for search_skills.

Usage: python -m eval.run_eval [--top-n 5]

Computes Precision@N and MRR against eval/queries.yaml (hand-picked queries
with known-relevant skill IDs) using the same code path search_skills the
MCP tool uses. Prints a summary and writes a timestamped JSON report to
eval/reports/. This is the yardstick for any change touching embeddings,
similarity edges, or graph structure - run it before and after.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from neo4j import AsyncGraphDatabase

from core.neo4j_client import Neo4jClient
from server.graph.traversal import get_layer1
from server.search.vector_search import search_skills

ROOT = Path(__file__).resolve().parent
QUERIES_PATH = ROOT / "queries.yaml"
REPORTS_DIR = ROOT / "reports"

# How many top vector hits feed the 1-hop graph expansion in --mode=graph.
GRAPH_EXPANSION_SEED = 3


def _load_queries() -> list[dict]:
    return yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))


def _score(case: dict, retrieved: list[str]) -> dict:
    """Recall-based scoring: does the retrieved set contain the expected id(s)?
    Precision@N is misleading here because most queries have 1-3 expected_ids,
    not N - see eval/reports for the discussion.
    """
    expected = set(case["expected_ids"])
    hits = [rid for rid in retrieved if rid in expected]
    reciprocal_rank = next(
        (1.0 / (i + 1) for i, rid in enumerate(retrieved) if rid in expected), 0.0
    )
    return {
        "query": case["query"],
        "expected_ids": sorted(expected),
        "retrieved_ids": retrieved,
        "hits": hits,
        "recall": round(len(set(hits)) / len(expected), 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


async def _eval_one_vector(driver, case: dict, top_n: int) -> dict:
    candidates = await search_skills(driver, case["query"], top_n=top_n)
    return _score(case, [c.id for c in candidates])


async def _eval_one_graph(driver, client: Neo4jClient, case: dict, top_n: int) -> dict:
    """Vector top-3 seeds, expanded with their 1-hop neighbors (any edge type)."""
    seeds = await search_skills(driver, case["query"], top_n=GRAPH_EXPANSION_SEED)
    retrieved = [c.id for c in seeds]
    for seed in seeds:
        neighbors = await get_layer1(client, seed.id, direction="outbound")
        for n in neighbors:
            if n.id not in retrieved:
                retrieved.append(n.id)
    return _score(case, retrieved[:top_n] if len(retrieved) > top_n else retrieved)


async def run(top_n: int, mode: str) -> dict:
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)
    try:
        cases = _load_queries()
        if mode == "graph":
            results = [await _eval_one_graph(driver, client, case, top_n) for case in cases]
        else:
            results = [await _eval_one_vector(driver, case, top_n) for case in cases]
    finally:
        await driver.close()

    n = len(results)
    mean_recall = sum(r["recall"] for r in results) / n
    mean_rr = sum(r["reciprocal_rank"] for r in results) / n
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "top_n": top_n,
        "num_queries": n,
        "mean_recall": round(mean_recall, 4),
        "mrr": round(mean_rr, 4),
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the search_skills retrieval-quality eval.")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--mode", choices=["vector", "graph"], default="vector",
                         help="vector = plain top-N search; graph = top-3 seeds + their 1-hop neighbors")
    args = parser.parse_args()

    report = asyncio.run(run(args.top_n, args.mode))

    for r in report["results"]:
        status = "OK  " if r["hits"] else "MISS"
        print(f"{status} recall={r['recall']:.2f}  {r['query']!r}")
    print()
    print(f"Mode: {report['mode']}")
    print(f"Queries: {report['num_queries']}")
    print(f"Mean recall: {report['mean_recall']}")
    print(f"MRR: {report['mrr']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"{report['timestamp'].replace(':', '-')}-{report['mode']}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
