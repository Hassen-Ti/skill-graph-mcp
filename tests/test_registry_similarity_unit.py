# tests/test_registry_similarity_unit.py
"""
Unit tests for registry/similarity.py's link_similar_skills orchestration.
The Neo4jClient is fully mocked (AsyncMock) — no live Neo4j required.

Covers the Codex-flagged dry-run bug fixed this session: a symmetric pair
(A sees B in its top-k, and B sees A in its) must be counted as ONE would-be
edge in a dry run, matching what a real run's WHERE-NOT-EXISTS guard does.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from registry.similarity import link_similar_skills


@pytest.mark.asyncio
async def test_real_run_clears_previous_similarity_edges_first():
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={})
    client.recompute_hub_scores = AsyncMock()

    await link_similar_skills(client, dry_run=False)

    client.delete_similarity_edges.assert_awaited_once()
    client.recompute_hub_scores.assert_awaited_once()


@pytest.mark.asyncio
async def test_dry_run_never_deletes_or_recomputes():
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={})
    client.recompute_hub_scores = AsyncMock()

    await link_similar_skills(client, dry_run=True)

    client.delete_similarity_edges.assert_not_awaited()
    client.recompute_hub_scores.assert_not_awaited()


@pytest.mark.asyncio
async def test_excludes_self_and_respects_top_k():
    """vector_knn is asked for top_k+1 (self always ranks first); self must not appear
    in the final candidate list, and the list is capped at top_k."""
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.recompute_hub_scores = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={"a": [0.1]})
    client.vector_knn = AsyncMock(return_value=[
        {"id": "a", "semantic_score": 1.0},  # self
        {"id": "b", "semantic_score": 0.9},
        {"id": "c", "semantic_score": 0.85},
        {"id": "d", "semantic_score": 0.8},  # beyond top_k=2, must be dropped
    ])
    client.manual_edge_exists = AsyncMock(return_value=False)
    client.similarity_edge_exists_or_create = AsyncMock(return_value=True)

    stats = await link_similar_skills(client, top_k=2, threshold=0.5, dry_run=False)

    client.vector_knn.assert_awaited_once_with([0.1], 3)  # top_k + 1
    assert stats["created"] == 2  # b and c only, d dropped by the top_k cap


@pytest.mark.asyncio
async def test_below_threshold_candidates_are_skipped():
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.recompute_hub_scores = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={"a": [0.1]})
    client.vector_knn = AsyncMock(return_value=[
        {"id": "a", "semantic_score": 1.0},
        {"id": "b", "semantic_score": 0.5},  # below threshold
    ])
    client.manual_edge_exists = AsyncMock(return_value=False)
    client.similarity_edge_exists_or_create = AsyncMock(return_value=True)

    stats = await link_similar_skills(client, top_k=5, threshold=0.75, dry_run=False)

    assert stats["created"] == 0
    assert stats["skipped_below_threshold"] == 1


@pytest.mark.asyncio
async def test_real_run_uses_manual_edge_exists_result_for_stats():
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.recompute_hub_scores = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={"a": [0.1]})
    client.vector_knn = AsyncMock(return_value=[
        {"id": "a", "semantic_score": 1.0},
        {"id": "b", "semantic_score": 0.9},
    ])
    client.similarity_edge_exists_or_create = AsyncMock(return_value=False)  # already connected

    stats = await link_similar_skills(client, top_k=5, threshold=0.75, dry_run=False)

    assert stats["created"] == 0
    assert stats["skipped_existing"] == 1


@pytest.mark.asyncio
async def test_dry_run_counts_symmetric_pair_only_once():
    """The exact bug Codex flagged: A's top-k includes B and B's top-k includes A.
    A real run creates ONE edge (second direction sees the first and skips).
    Dry-run must predict the same count, not double it, via the in-memory
    pending_pairs set."""
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.recompute_hub_scores = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={"a": [0.1], "b": [0.2]})
    client.manual_edge_exists = AsyncMock(return_value=False)  # no pre-existing curated edge

    async def fake_knn(embedding, k):
        if embedding == [0.1]:  # querying for "a"
            return [{"id": "a", "semantic_score": 1.0}, {"id": "b", "semantic_score": 0.9}]
        return [{"id": "b", "semantic_score": 1.0}, {"id": "a", "semantic_score": 0.9}]

    client.vector_knn = AsyncMock(side_effect=fake_knn)

    stats = await link_similar_skills(client, top_k=5, threshold=0.75, dry_run=True)

    assert stats["created"] == 1
    assert stats["skipped_existing"] == 1  # the second direction hits the pending_pairs guard


@pytest.mark.asyncio
async def test_dry_run_predicts_zero_for_already_curated_pair():
    """A pair that already has a real (non-similarity) edge must never be predicted
    as 'created' by dry-run — manual_edge_exists is the source of truth here."""
    client = MagicMock()
    client.delete_similarity_edges = AsyncMock()
    client.recompute_hub_scores = AsyncMock()
    client.get_all_skill_embeddings = AsyncMock(return_value={"a": [0.1]})
    client.vector_knn = AsyncMock(return_value=[
        {"id": "a", "semantic_score": 1.0},
        {"id": "b", "semantic_score": 0.9},
    ])
    client.manual_edge_exists = AsyncMock(return_value=True)

    stats = await link_similar_skills(client, top_k=5, threshold=0.75, dry_run=True)

    assert stats["created"] == 0
    assert stats["skipped_existing"] == 1
