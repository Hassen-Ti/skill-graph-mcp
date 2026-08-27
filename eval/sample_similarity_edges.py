# eval/sample_similarity_edges.py
"""
Print a random sample of auto-generated COLLABORATES_WITH edges (source =
'similarity') with both skills' names/descriptions, for manual relevance
spot-checking. Usage: python -m eval.sample_similarity_edges [--n 40]
"""
from __future__ import annotations

import argparse
import asyncio
import os

from neo4j import AsyncGraphDatabase

_QUERY = """
MATCH (a:Skill)-[r:COLLABORATES_WITH {source: 'similarity'}]->(b:Skill)
WITH a, r, b, rand() AS rnd
ORDER BY rnd
LIMIT $n
RETURN a.id AS a_id, a.description AS a_desc,
       b.id AS b_id, b.description AS b_desc,
       r.score AS score
"""


async def run(n: int) -> None:
    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        async with driver.session() as session:
            result = await session.run(_QUERY, {"n": n})
            async for record in result:
                print(f"[{record['score']:.3f}] {record['a_id']}  <->  {record['b_id']}")
                print(f"   A: {record['a_desc'][:100]}")
                print(f"   B: {record['b_desc'][:100]}")
                print()
    finally:
        await driver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    asyncio.run(run(args.n))


if __name__ == "__main__":
    main()
