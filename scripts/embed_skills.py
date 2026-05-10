#!/usr/bin/env python3
"""
Sprint 2 — Embedding pipeline + semantic edge injection.

Steps:
  1. Generate OpenAI embeddings for all staging/skills/*.yaml descriptions
  2. Compute cosine similarity (scipy sparse matrix)
  3. For each skill: top-5 neighbors above threshold → inject as collaborates_with edges
  4. Update YAML files in-place with new edges
  5. Update index_metadata.json (SHA256 drift detection)

Usage:
    python scripts/embed_skills.py [--dry-run]
"""

import json
import math
import hashlib
import sys
import time
import yaml
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent
STAGING_DIR = HERE / "staging" / "skills"
INDEX_METADATA = HERE / "index_metadata.json"

SIMILARITY_THRESHOLD = 0.55
TOP_K = 8
BATCH_SIZE = 100
EMBED_MODEL = "text-embedding-3-small"


def load_staging_skills() -> list[dict]:
    """Load all YAML files from staging/skills/."""
    skills = []
    for f in sorted(STAGING_DIR.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        data["_file"] = f
        skills.append(data)
    return skills


def load_index_metadata() -> dict:
    if INDEX_METADATA.exists():
        with open(INDEX_METADATA) as f:
            return json.load(f)
    return {}


def description_hash(description: str) -> str:
    return hashlib.sha256(description.encode()).hexdigest()


def needs_embedding(skill: dict, metadata: dict) -> bool:
    sid = skill["id"]
    desc = skill.get("description", "")
    current_hash = description_hash(desc)
    return metadata.get(sid, {}).get("description_hash") != current_hash


def generate_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Call OpenAI embeddings API in batches."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)  # avoid rate limit
    return all_embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_existing_edges(skill: dict) -> set[str]:
    """Return set of target IDs already in this skill's edges."""
    return {e["to"] for e in skill.get("edges", [])}


def inject_semantic_edges(skills: list[dict], embeddings: list[list[float]], dry_run: bool) -> dict:
    """
    For each skill, find top-K neighbors above threshold.
    Inject collaborates_with edges into YAML files.
    Returns stats.
    """
    valid_ids = {s["id"] for s in skills}
    stats = {"edges_added": 0, "skills_updated": 0}

    # Build similarity for each skill (O(n²) but n=718 is fine)
    print("Computing cosine similarities...")
    new_edges: dict[str, list[tuple[float, str]]] = defaultdict(list)

    n = len(skills)
    for i in range(n):
        sid = skills[i]["id"]
        existing = build_existing_edges(skills[i])

        candidates = []
        for j in range(n):
            if i == j:
                continue
            tid = skills[j]["id"]
            if tid in existing:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= SIMILARITY_THRESHOLD:
                candidates.append((sim, tid))

        # Keep top-K
        candidates.sort(reverse=True)
        new_edges[sid] = candidates[:TOP_K]

    # Inject into YAML
    for skill in skills:
        sid = skill["id"]
        to_add = new_edges.get(sid, [])
        if not to_add:
            continue

        edges = skill.get("edges", [])
        for _, tid in to_add:
            edges.append({"to": tid, "type": "collaborates_with"})
            stats["edges_added"] += 1

        skill["edges"] = edges
        stats["skills_updated"] += 1

        if not dry_run:
            f = skill["_file"]
            out = {k: v for k, v in skill.items() if not k.startswith("_")}
            with open(f, "w", encoding="utf-8") as fh:
                yaml.dump(out, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return stats


def main(dry_run: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"{prefix}Loading staging skills...")
    skills = load_staging_skills()
    print(f"  {len(skills)} skills found")

    metadata = load_index_metadata()

    # Identify which skills need (re-)embedding
    to_embed = [s for s in skills if needs_embedding(s, metadata)]
    already_ok = len(skills) - len(to_embed)
    print(f"  {already_ok} already embedded (hash match), {len(to_embed)} need embedding")

    if to_embed:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        texts = [s.get("description", s["id"]) for s in to_embed]
        print(f"{prefix}Generating embeddings for {len(to_embed)} skills...")
        new_embeddings = generate_embeddings(texts, client)

        # Build full embeddings list (reuse existing where possible)
        # For similarity we need ALL embeddings — embed everything if cache missing
        if already_ok > 0:
            print("  Re-embedding all (first run or partial cache) to build full matrix...")
            all_texts = [s.get("description", s["id"]) for s in skills]
            all_embeddings = generate_embeddings(all_texts, client)
        else:
            all_embeddings = new_embeddings

        # Update metadata
        new_meta = dict(metadata)
        for i, skill in enumerate(skills):
            new_meta[skill["id"]] = {
                "description_hash": description_hash(skill.get("description", "")),
            }

        if not dry_run:
            with open(INDEX_METADATA, "w") as f:
                json.dump(new_meta, f, indent=2)
            print(f"  Updated {INDEX_METADATA}")
    else:
        print("All embeddings up to date — loading cached descriptions for similarity...")
        # We still need embeddings to compute similarity — regenerate
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        all_texts = [s.get("description", s["id"]) for s in skills]
        print(f"{prefix}Generating embeddings for similarity computation...")
        all_embeddings = generate_embeddings(all_texts, client)

    print(f"{prefix}Injecting semantic edges (threshold={SIMILARITY_THRESHOLD}, top_k={TOP_K})...")
    stats = inject_semantic_edges(skills, all_embeddings, dry_run)

    # Report degree distribution
    degree: dict[str, int] = defaultdict(int)
    for skill in skills:
        for e in skill.get("edges", []):
            degree[skill["id"]] += 1
            degree[e["to"]] += 1

    isolated = sum(1 for s in skills if degree[s["id"]] == 0)
    avg_degree = sum(degree.values()) / len(skills) if skills else 0

    print(f"\n{prefix}Results:")
    print(f"  Semantic edges added: {stats['edges_added']}")
    print(f"  Skills updated: {stats['skills_updated']}")
    print(f"  Isolated nodes (degree=0): {isolated}/{len(skills)}")
    print(f"  Avg degree: {avg_degree:.1f}")

    if isolated > 0:
        print(f"  WARNING: {isolated} isolated nodes — consider lowering threshold")
    if avg_degree < 5:
        print(f"  WARNING: avg degree {avg_degree:.1f} < 5 — consider lowering threshold")
    else:
        print(f"\nSprint 2 criterion met: avg degree >= 5, {isolated} isolated nodes.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
