# registry/similarity.py
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def link_similar_skills(
    client: Any, top_k: int = 5, threshold: float = 0.75, dry_run: bool = False,
) -> dict[str, int]:
    """Generate COLLABORATES_WITH edges between skills whose embeddings are
    close by cosine similarity (via the existing vector index's KNN query).

    Safe to rerun: previously auto-generated edges (tagged source='similarity')
    are cleared and recomputed from scratch each time. Never touches manually
    curated edges — a pair already connected by any edge (manual or a prior
    similarity pass) is always skipped.
    """
    if not dry_run:
        await client.delete_similarity_edges()
    embeddings = await client.get_all_skill_embeddings()
    stats = {
        "nodes_processed": 0,
        "created": 0,
        "skipped_existing": 0,
        "skipped_below_threshold": 0,
    }
    pending_pairs: set[frozenset[str]] = set()
    for skill_id in sorted(embeddings):
        neighbors = await client.vector_knn(embeddings[skill_id], top_k + 1)
        stats["nodes_processed"] += 1
        candidates = [n for n in neighbors if n["id"] != skill_id][:top_k]
        for n in candidates:
            if n["semantic_score"] < threshold:
                stats["skipped_below_threshold"] += 1
                continue
            if dry_run:
                pair = frozenset((skill_id, n["id"]))
                if pair in pending_pairs or await client.manual_edge_exists(skill_id, n["id"]):
                    stats["skipped_existing"] += 1
                else:
                    stats["created"] += 1
                    pending_pairs.add(pair)
                continue
            if await client.similarity_edge_exists_or_create(skill_id, n["id"], n["semantic_score"]):
                stats["created"] += 1
            else:
                stats["skipped_existing"] += 1
    if not dry_run:
        await client.recompute_hub_scores()
    logger.info(
        "Similarity linking: %d nodes, %d edges created, %d skipped (already connected), "
        "%d skipped (below threshold).",
        stats["nodes_processed"], stats["created"], stats["skipped_existing"], stats["skipped_below_threshold"],
    )
    return stats
