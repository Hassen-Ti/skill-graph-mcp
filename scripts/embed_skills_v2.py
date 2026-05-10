#!/usr/bin/env python3
"""
V2 Embedding pipeline — richer text, analytics output.

Changes vs embed_skills.py (v1):
  - Embeds name + description + capabilities + requirements (not description only)
  - Writes results to viz/analytics/ (no YAML modification, no index_metadata.json update)
  - Optional --push-neo4j flag to update Neo4j with the new embeddings
  - Designed for comparison and analysis before committing to V2

Usage:
    python scripts/embed_skills_v2.py [--dry-run] [--push-neo4j]
"""

import json
import math
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

HERE        = Path(__file__).resolve().parent.parent
STAGING_DIR = HERE / "staging" / "skills"
ANALYTICS   = HERE.parent / "skill-graph-viz" / "analytics"
ANALYTICS.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100
SIM_THRESHOLD = 0.55   # same as v1 for comparison


# ── Text builder ──────────────────────────────────────────────

def build_rich_text(skill: dict) -> str:
    parts = [skill.get("name", skill["id"]) + "."]
    if skill.get("description"):
        parts.append(skill["description"])
    caps = skill.get("capabilities", [])
    if caps:
        parts.append(" ".join(caps))
    reqs = skill.get("requirements", [])
    if reqs:
        parts.append(" ".join(reqs))
    return " ".join(parts)


# ── Load skills ───────────────────────────────────────────────

def load_staging_skills() -> list[dict]:
    skills = []
    for f in sorted(STAGING_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        data["_file"] = f
        skills.append(data)
    return skills


# ── Embeddings ────────────────────────────────────────────────

def generate_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)
    return all_embeddings


# ── Cosine similarity (numpy) ─────────────────────────────────

def cosine_sim_matrix(embeddings: list[list[float]]) -> np.ndarray:
    mat   = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat   = mat / np.maximum(norms, 1e-10)
    return mat @ mat.T


# ── Similarity stats ──────────────────────────────────────────

def compute_similarity_stats(sim_matrix: np.ndarray, ids: list[str]) -> dict:
    n = len(ids)
    # Upper triangle only, no diagonal
    upper = sim_matrix[np.triu_indices(n, k=1)]

    # Degree per node at threshold
    above = (sim_matrix >= SIM_THRESHOLD).astype(int)
    np.fill_diagonal(above, 0)
    degrees = above.sum(axis=1).tolist()

    # Histogram buckets [0, 0.1, 0.2, ..., 1.0]
    hist, _ = np.histogram(upper, bins=10, range=(0.0, 1.0))

    return {
        "n_skills": n,
        "threshold": SIM_THRESHOLD,
        "similarity_distribution": {
            "min":    float(upper.min()),
            "max":    float(upper.max()),
            "mean":   float(upper.mean()),
            "median": float(np.median(upper)),
            "p25":    float(np.percentile(upper, 25)),
            "p75":    float(np.percentile(upper, 75)),
        },
        "histogram_0_to_1": hist.tolist(),
        "degree_at_threshold": {
            "mean":     float(np.mean(degrees)),
            "median":   float(np.median(degrees)),
            "isolated": int(sum(1 for d in degrees if d == 0)),
            "max":      int(max(degrees)),
        },
        "pairs_above_threshold": int((above.sum()) // 2),
    }


# ── Neo4j push ────────────────────────────────────────────────

def push_to_neo4j(skills: list[dict], embeddings: list[list[float]]) -> None:
    from neo4j import GraphDatabase
    NEO4J_URI  = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "skillgraph")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        for skill, emb in zip(skills, embeddings):
            session.run(
                "MATCH (s:Skill {id: $id}) SET s.embedding = $emb",
                id=skill["id"], emb=emb,
            )
    driver.close()
    print(f"  Pushed {len(skills)} embeddings to Neo4j")


# ── Main ──────────────────────────────────────────────────────

def main(dry_run: bool = False, push_neo4j: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""

    print(f"{prefix}Loading staging skills...")
    skills = load_staging_skills()
    print(f"  {len(skills)} skills found")

    texts = [build_rich_text(s) for s in skills]
    ids   = [s["id"] for s in skills]

    # Sample the first 3 to confirm format
    print("\nSample rich texts:")
    for t in texts[:3]:
        print(f"  {t[:120]}...")

    if dry_run:
        print(f"\n[DRY-RUN] Would embed {len(skills)} skills. Exiting.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(f"\nGenerating V2 embeddings ({EMBED_MODEL}, rich text)...")
    embeddings = generate_embeddings(texts, client)

    # ── Save embeddings ──────────────────────────────────────
    emb_path = ANALYTICS / "embeddings_v2.json"
    payload   = {sid: emb for sid, emb in zip(ids, embeddings)}
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"\nSaved embeddings -> {emb_path}")

    # ── Similarity stats ─────────────────────────────────────
    print("Computing similarity matrix...")
    sim_mat = cosine_sim_matrix(embeddings)
    stats   = compute_similarity_stats(sim_mat, ids)

    stats_path = ANALYTICS / "similarity_stats_v2.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved similarity stats -> {stats_path}")

    # ── Print summary ────────────────────────────────────────
    d = stats["similarity_distribution"]
    deg = stats["degree_at_threshold"]
    print(f"""
V2 Embedding stats:
  Skills embedded : {stats['n_skills']}
  Similarity range: {d['min']:.3f} – {d['max']:.3f}
  Mean / Median   : {d['mean']:.3f} / {d['median']:.3f}
  P25 / P75       : {d['p25']:.3f} / {d['p75']:.3f}
  Pairs ≥ {SIM_THRESHOLD}    : {stats['pairs_above_threshold']}
  Avg degree      : {deg['mean']:.1f}
  Isolated nodes  : {deg['isolated']}
""")

    # ── Optional Neo4j push ──────────────────────────────────
    if push_neo4j:
        print("Pushing embeddings to Neo4j...")
        push_to_neo4j(skills, embeddings)


if __name__ == "__main__":
    dry_run    = "--dry-run" in sys.argv
    push_neo4j = "--push-neo4j" in sys.argv
    main(dry_run=dry_run, push_neo4j=push_neo4j)
