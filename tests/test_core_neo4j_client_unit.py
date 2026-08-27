# tests/test_core_neo4j_client_unit.py
"""
Unit tests for core/neo4j_client.py's Cypher-building logic. Neo4jClient._run
is mocked directly (rather than the driver/session chain) so these tests
check what query/params get built without needing a live database.

Covers the two Codex-flagged correctness bugs fixed this session:
- upsert_edge must strip stale similarity tags on a promoted (curated) edge.
- manual_edge_exists must ignore similarity-tagged edges (what a real `link`
  run would see after clearing them), so dry-run predictions match reality.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.neo4j_client import Neo4jClient


@pytest.fixture
def client():
    return Neo4jClient(driver=MagicMock())


class TestUpsertEdge:
    @pytest.mark.asyncio
    async def test_upsert_edge_rejects_unknown_type(self, client):
        with pytest.raises(ValueError, match="Invalid edge_type"):
            await client.upsert_edge("a", "b", "INVENTED_RELATION")

    @pytest.mark.asyncio
    async def test_upsert_edge_strips_similarity_tag_on_write(self, client, mocker):
        """The promotion bug: a curated edge write must clear any prior source/score
        properties, or a later delete_similarity_edges() would wrongly remove it."""
        mock_run = mocker.patch.object(client, "_run", new=AsyncMock(return_value=[]))

        await client.upsert_edge("a", "b", "collaborates_with")

        query = mock_run.call_args.args[0]
        assert "REMOVE r.source, r.score" in query
        assert "MERGE (a)-[r:COLLABORATES_WITH]->(b)" in query

    @pytest.mark.asyncio
    async def test_upsert_edge_uppercases_type_and_passes_params(self, client, mocker):
        mock_run = mocker.patch.object(client, "_run", new=AsyncMock(return_value=[]))

        await client.upsert_edge("skill-a", "skill-b", "requires")

        query, params = mock_run.call_args.args[0], mock_run.call_args.args[1]
        assert "REQUIRES" in query
        assert params == {"from_id": "skill-a", "to_id": "skill-b"}


class TestManualEdgeExists:
    @pytest.mark.asyncio
    async def test_queries_excluding_similarity_source(self, client, mocker):
        mock_run = mocker.patch.object(
            client, "_run", new=AsyncMock(return_value=[{"exists": True}]),
        )

        result = await client.manual_edge_exists("a", "b")

        assert result is True
        query = mock_run.call_args.args[0]
        assert "r.source IS NULL OR r.source <> 'similarity'" in query

    @pytest.mark.asyncio
    async def test_returns_false_when_no_records(self, client, mocker):
        mocker.patch.object(client, "_run", new=AsyncMock(return_value=[{"exists": False}]))
        assert await client.manual_edge_exists("a", "b") is False


class TestSimilarityEdgeExistsOrCreate:
    @pytest.mark.asyncio
    async def test_creates_when_no_existing_edge(self, client, mocker):
        mocker.patch.object(client, "_run", new=AsyncMock(return_value=[{"r": {}}]))
        created = await client.similarity_edge_exists_or_create("a", "b", 0.81)
        assert created is True

    @pytest.mark.asyncio
    async def test_skips_when_any_edge_already_exists(self, client, mocker):
        """WHERE NOT EXISTS guards the CREATE — an existing edge of ANY kind means zero rows back."""
        mocker.patch.object(client, "_run", new=AsyncMock(return_value=[]))
        created = await client.similarity_edge_exists_or_create("a", "b", 0.81)
        assert created is False

    @pytest.mark.asyncio
    async def test_passes_score_as_parameter(self, client, mocker):
        mock_run = mocker.patch.object(client, "_run", new=AsyncMock(return_value=[{"r": {}}]))
        await client.similarity_edge_exists_or_create("a", "b", 0.8123)
        assert mock_run.call_args.args[1]["score"] == 0.8123


class TestDeleteSimilarityEdges:
    @pytest.mark.asyncio
    async def test_only_targets_similarity_tagged_edges(self, client, mocker):
        mock_run = mocker.patch.object(client, "_run", new=AsyncMock(return_value=[]))
        await client.delete_similarity_edges()
        query = mock_run.call_args.args[0]
        assert "source: 'similarity'" in query
        assert "COLLABORATES_WITH" in query


class TestGetAllSkillEmbeddings:
    @pytest.mark.asyncio
    async def test_builds_id_to_embedding_dict(self, client, mocker):
        mocker.patch.object(
            client, "_run",
            new=AsyncMock(return_value=[
                {"id": "a", "embedding": [0.1, 0.2]},
                {"id": "b", "embedding": [0.3, 0.4]},
            ]),
        )
        result = await client.get_all_skill_embeddings()
        assert result == {"a": [0.1, 0.2], "b": [0.3, 0.4]}
