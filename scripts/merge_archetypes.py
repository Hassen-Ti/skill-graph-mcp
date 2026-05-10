#!/usr/bin/env python3
"""
Sprint 3 — Archetype merger.

Connects the 14 existing hand-crafted skills (archetypes) to the 718 staging skills
via embedding similarity. Each archetype gets 'enables' edges to its top-N most
relevant staging skills.

Steps:
  1. Embed archetype descriptions
  2. Embed all staging skill descriptions (reuse from embed_skills if possible)
  3. For each archetype: find top-N staging skills with similarity > threshold
  4. Patch archetype YAML files with new 'enables' edges
  5. Optionally: add 'collaborates_with' back-edges in staging YAMLs

Usage:
    python scripts/merge_archetypes.py [--dry-run]
"""

import json
import sys
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import os
import math
from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent
ARCHETYPES_DIR = HERE / "skills"
STAGING_DIR = HERE / "staging" / "skills"

ARCHETYPE_SIMILARITY_THRESHOLD = 0.45
TOP_N_PER_ARCHETYPE = 15
BATCH_SIZE = 100
EMBED_MODEL = "text-embedding-3-small"


def load_yaml_skills(directory: Path) -> list[dict]:
    skills = []
    for f in sorted(directory.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        data["_file"] = f
        skills.append(data)
    return skills


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embs.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.3)
    return all_embs


def existing_edge_targets(skill: dict) -> set[str]:
    return {e["to"] for e in skill.get("edges", [])}


def add_edge(skill: dict, target_id: str, edge_type: str) -> bool:
    """Add edge if not already present. Returns True if added."""
    existing = existing_edge_targets(skill)
    if target_id in existing:
        return False
    if "edges" not in skill:
        skill["edges"] = []
    skill["edges"].append({"to": target_id, "type": edge_type})
    return True


def write_yaml(skill: dict) -> None:
    f = skill["_file"]
    out = {k: v for k, v in skill.items() if not k.startswith("_")}
    with open(f, "w", encoding="utf-8") as fh:
        yaml.dump(out, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)


def main(dry_run: bool = False) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""

    print(f"{prefix}Loading archetypes and staging skills...")
    archetypes = load_yaml_skills(ARCHETYPES_DIR)
    staging = load_yaml_skills(STAGING_DIR)

    # Clusters don't need outbound semantic edges — they already have 'enables'
    archetype_roles = [a for a in archetypes if a.get("type") != "cluster"]
    staging_ids = {s["id"] for s in staging}

    print(f"  {len(archetypes)} archetypes ({len(archetype_roles)} non-cluster)")
    print(f"  {len(staging)} staging skills")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"{prefix}Embedding archetype descriptions...")
    arch_texts = [a.get("description", a["id"]) for a in archetype_roles]
    arch_embeddings = embed_texts(arch_texts, client)

    print(f"{prefix}Embedding staging descriptions...")
    staging_texts = [s.get("description", s["id"]) for s in staging]
    staging_embeddings = embed_texts(staging_texts, client)

    staging_map = {s["id"]: (i, s) for i, s in enumerate(staging)}

    stats = {"archetype_edges_added": 0, "back_edges_added": 0}

    print(f"\n{prefix}Computing archetype -> staging similarities...")
    for ai, archetype in enumerate(archetype_roles):
        arch_id = archetype["id"]
        arch_emb = arch_embeddings[ai]

        candidates = []
        for si, staging_skill in enumerate(staging):
            sid = staging_skill["id"]
            sim = cosine_similarity(arch_emb, staging_embeddings[si])
            if sim >= ARCHETYPE_SIMILARITY_THRESHOLD:
                candidates.append((sim, sid))

        candidates.sort(reverse=True)
        top_candidates = candidates[:TOP_N_PER_ARCHETYPE]

        print(f"  {arch_id}: {len(top_candidates)} matches")
        for sim, sid in top_candidates[:5]:
            print(f"    -> {sid} ({sim:.3f})")

        if not dry_run:
            for sim, sid in top_candidates:
                if add_edge(archetype, sid, "enables"):
                    stats["archetype_edges_added"] += 1

                # Back-edge: staging skill collaborates_with archetype
                _, staging_skill = staging_map[sid]
                if add_edge(staging_skill, arch_id, "collaborates_with"):
                    stats["back_edges_added"] += 1
                    write_yaml(staging_skill)

            write_yaml(archetype)

    print(f"\n{prefix}Results:")
    print(f"  Archetype 'enables' edges added: {stats['archetype_edges_added']}")
    print(f"  Back 'collaborates_with' edges added: {stats['back_edges_added']}")

    if not dry_run:
        # Recount isolated staging nodes
        degree = {}
        for s in staging:
            # reload to get fresh state
            with open(s["_file"], encoding="utf-8") as fh:
                fresh = yaml.safe_load(fh)
            degree[fresh["id"]] = len(fresh.get("edges", []))

        isolated = sum(1 for v in degree.values() if v == 0)
        print(f"  Remaining staging nodes with 0 edges: {isolated}")

    print(f"\nSprint 3 criterion: each archetype has >= {TOP_N_PER_ARCHETYPE} enables edges.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
