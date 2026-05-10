#!/usr/bin/env python3
"""
Push V3 embeddings (payload.instructions) to Neo4j.

Loads pre-generated embeddings from the H005 analytics cache
and updates the Neo4j vector index in batches.

Usage:
    python scripts/push_embeddings_v3.py [--dry-run]
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

HERE      = Path(__file__).resolve().parent.parent
V3_CACHE  = (
    HERE.parent / "skill-graph-viz" / "analytics" / "hypotheses"
    / "H005_search_quality" / "embeddings_v3_payload.json"
)

NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "skillgraph"
BATCH_SIZE = 50


def push(embeddings: dict[str, list[float]], dry_run: bool) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    ids   = list(embeddings.keys())
    total = len(ids)
    updated = 0

    with driver.session() as session:
        for i in range(0, total, BATCH_SIZE):
            batch_ids = ids[i:i + BATCH_SIZE]
            batch     = [{"id": sid, "emb": embeddings[sid]} for sid in batch_ids]

            if not dry_run:
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (s:Skill {id: row.id})
                    SET s.embedding = row.emb
                    """,
                    batch=batch,
                )
            updated += len(batch_ids)
            print(f"  {'[DRY-RUN] ' if dry_run else ''}Updated {updated}/{total}")

    driver.close()


def verify(sample_ids: list[str]) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        for sid in sample_ids:
            row = session.run(
                "MATCH (s:Skill {id: $id}) RETURN size(s.embedding) AS dim",
                id=sid,
            ).single()
            dim = row["dim"] if row else "NOT FOUND"
            print(f"  {sid}: embedding dim = {dim}")
    driver.close()


def main(dry_run: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""

    print(f"Loading V3 embeddings from cache...")
    if not V3_CACHE.exists():
        print(f"ERROR: cache not found at {V3_CACHE}")
        print("Run analytics/hypotheses/H005_search_quality/experiment.py first.")
        sys.exit(1)

    with open(V3_CACHE, encoding="utf-8") as f:
        embeddings = json.load(f)

    print(f"  {len(embeddings)} skills loaded")
    sample = list(embeddings.keys())[:3]
    print(f"  Embedding dim: {len(embeddings[sample[0]])}D")
    print(f"  Sample IDs: {sample}")

    print(f"\n{prefix}Pushing to Neo4j ({NEO4J_URI})...")
    push(embeddings, dry_run)

    if not dry_run:
        print("\nVerifying sample skills in Neo4j...")
        verify(sample)

    print(f"\n{prefix}Done. {len(embeddings)} skills updated with V3 embeddings.")
    if not dry_run:
        print("Neo4j vector index will use payload.instructions embeddings for search_skills.")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
