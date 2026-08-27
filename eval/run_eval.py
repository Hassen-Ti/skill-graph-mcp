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

from server.search.vector_search import search_skills

ROOT = Path(__file__).resolve().parent
QUERIES_PATH = ROOT / "queries.yaml"
REPORTS_DIR = ROOT / "reports"


def _load_queries() -> list[dict]:
    return yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))


async def _eval_one(driver, case: dict, top_n: int) -> dict:
    expected = set(case["expected_ids"])
    candidates = await search_skills(driver, case["query"], top_n=top_n)
    retrieved = [c.id for c in candidates]
    hits = [rid for rid in retrieved if rid in expected]
    precision = len(hits) / top_n
    reciprocal_rank = next(
        (1.0 / (i + 1) for i, rid in enumerate(retrieved) if rid in expected), 0.0
    )
    return {
        "query": case["query"],
        "expected_ids": sorted(expected),
        "retrieved_ids": retrieved,
        "hits": hits,
        f"precision_at_{top_n}": round(precision, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


async def run(top_n: int) -> dict:
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        cases = _load_queries()
        results = [await _eval_one(driver, case, top_n) for case in cases]
    finally:
        await driver.close()

    n = len(results)
    mean_precision = sum(r[f"precision_at_{top_n}"] for r in results) / n
    mean_rr = sum(r["reciprocal_rank"] for r in results) / n
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_n": top_n,
        "num_queries": n,
        f"mean_precision_at_{top_n}": round(mean_precision, 4),
        "mrr": round(mean_rr, 4),
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the search_skills retrieval-quality eval.")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    report = asyncio.run(run(args.top_n))

    for r in report["results"]:
        status = "OK  " if r["hits"] else "MISS"
        print(f"{status} P@{args.top_n}={r[f'precision_at_{args.top_n}']:.2f}  {r['query']!r}")
    print()
    print(f"Queries: {report['num_queries']}")
    print(f"Mean Precision@{args.top_n}: {report[f'mean_precision_at_{args.top_n}']}")
    print(f"MRR: {report['mrr']}")

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"{report['timestamp'].replace(':', '-')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
