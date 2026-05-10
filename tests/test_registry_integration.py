# tests/test_registry_integration.py
"""
Integration test suite for registry/loader.py + registry/embedder.py.

Prerequisites:
    - Neo4j running on bolt://localhost:7687 (docker-compose up -d)
    - NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars (or defaults)
    - pip install -e ".[dev]"

Each test creates its own isolated YAML/directory fixtures, cleans up after itself
via DETACH DELETE, and makes no assumptions about pre-existing graph state.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def neo4j_client():
    """Provide a connected Neo4jClient and clean up after the test."""
    from neo4j import AsyncGraphDatabase
    from server.graph.neo4j_client import Neo4jClient

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "skillgraph")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)
    yield client
    await driver.close()


@pytest_asyncio.fixture(autouse=True)
async def cleanup_neo4j(neo4j_client):
    """Delete only Skill nodes created during the test; preserve pre-existing ones."""
    records = await neo4j_client._run("MATCH (s:Skill) RETURN s.id AS id")
    pre_existing = {r["id"] for r in records}

    yield

    records = await neo4j_client._run("MATCH (s:Skill) RETURN s.id AS id")
    new_ids = [r["id"] for r in records if r["id"] not in pre_existing]
    if new_ids:
        await neo4j_client._run(
            "MATCH (s:Skill) WHERE s.id IN $ids DETACH DELETE s",
            {"ids": new_ids},
        )


@pytest.fixture
def schema_path(tmp_path) -> Path:
    """Write the canonical skills/schema.json to a temp dir."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["id", "name", "type", "author", "version", "description"],
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
            "name": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": ["role", "tool", "domain", "responsibility", "cluster"],
            },
            "author": {"type": "string"},
            "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
            "description": {"type": "string", "minLength": 1, "maxLength": 500},
            "priority": {"type": "integer", "minimum": 1, "maximum": 3},
            "extends": {"type": "string"},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["to", "type"],
                    "additionalProperties": False,
                    "properties": {
                        "to": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "requires",
                                "enables",
                                "collaborates_with",
                                "uses",
                                "part_of",
                                "extends",
                            ],
                        },
                    },
                },
            },
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "instructions": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "exclude_tools": {"type": "array", "items": {"type": "string"}},
                    "knowledge": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(schema))
    return p


def _write_yaml(directory: Path, filename: str, content: str) -> Path:
    p = directory / filename
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Mock embedder module (avoids calling OpenAI during integration tests)
# ---------------------------------------------------------------------------


class _MockEmbedder:
    """Minimal embedder stub — records which skills were updated, no real API calls."""

    def __init__(self):
        self.updated: list[str] = []

    async def update_embeddings(self, client, registry):
        self.updated.extend(registry.keys())


# ---------------------------------------------------------------------------
# Test: load a single YAML file
# ---------------------------------------------------------------------------


class TestLoadSingleSkillYaml:
    async def test_load_single_skill_yaml(self, tmp_path, schema_path, neo4j_client):
        """Parse + write one YAML; verify the node exists in Neo4j with correct fields."""
        from registry.loader import load_skill_file
        from registry.loader import _write_skill_node

        yaml_path = _write_yaml(
            tmp_path,
            "my_skill.yaml",
            """
            id: my_skill
            name: My Skill
            type: role
            author: tester
            version: 1.0.0
            description: Integration test skill.
            priority: 1
            edges: []
            payload:
              instructions: "Do X."
              tools: [bash]
              knowledge: []
            """,
        )

        skill = load_skill_file(yaml_path, schema_path)
        await _write_skill_node(neo4j_client, skill)

        records = await neo4j_client._run(
            "MATCH (s:Skill {id: 'my_skill'}) RETURN s"
        )
        assert len(records) == 1
        node = records[0]["s"]
        assert node["name"] == "My Skill"
        assert node["version"] == "1.0.0"
        assert node["type"] == "role"
        assert node["author"] == "tester"


# ---------------------------------------------------------------------------
# Test: dry-run — no nodes created
# ---------------------------------------------------------------------------


class TestDryRun:
    async def test_load_skills_directory_dry_run(
        self, tmp_path, schema_path, neo4j_client
    ):
        """dry_run=True must validate without writing any nodes to Neo4j."""
        from registry.loader import load_skills_directory

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        for i in range(1, 3):
            _write_yaml(
                skills_dir,
                f"skill_{i:02d}.yaml",
                f"""
                id: skill_{i:02d}
                name: Skill {i}
                type: domain
                author: tester
                version: 1.0.0
                description: Skill number {i} for dry-run test.
                edges: []
                """,
            )

        # Capture pre-existing node count — dry_run must not add any new nodes.
        before = await neo4j_client._run("MATCH (s:Skill) RETURN count(s) AS n")
        pre_count = before[0]["n"]

        embedder = _MockEmbedder()
        await load_skills_directory(
            skills_dir=skills_dir,
            schema_path=schema_path,
            client=neo4j_client,
            embedder_module=embedder,
            dry_run=True,
        )

        # No new nodes should have been created.
        records = await neo4j_client._run("MATCH (s:Skill) RETURN count(s) AS n")
        assert records[0]["n"] == pre_count

        # Embedder should not have been called.
        assert embedder.updated == []


# ---------------------------------------------------------------------------
# Test: real load — nodes written to Neo4j
# ---------------------------------------------------------------------------


class TestLoadWritesNodes:
    async def test_load_skills_directory_writes_nodes(
        self, tmp_path, schema_path, neo4j_client
    ):
        """load_skills_directory must persist all loaded skills as Skill nodes."""
        from registry.loader import load_skills_directory

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        for i in range(1, 3):
            _write_yaml(
                skills_dir,
                f"skill_{i:02d}.yaml",
                f"""
                id: skill_{i:02d}
                name: Skill {i}
                type: domain
                author: tester
                version: 1.0.0
                description: Skill number {i} for write test.
                edges: []
                """,
            )

        embedder = _MockEmbedder()
        await load_skills_directory(
            skills_dir=skills_dir,
            schema_path=schema_path,
            client=neo4j_client,
            embedder_module=embedder,
            dry_run=False,
        )

        records = await neo4j_client._run("MATCH (s:Skill) RETURN s.id AS id ORDER BY s.id")
        ids = [r["id"] for r in records]
        assert "skill_01" in ids
        assert "skill_02" in ids


# ---------------------------------------------------------------------------
# Test: orphan edge detection
# ---------------------------------------------------------------------------


class TestOrphanEdgeDetection:
    async def test_orphan_edge_detection(
        self, tmp_path, schema_path, neo4j_client
    ):
        """An edge referencing an unknown skill_id must raise KeyError before any write."""
        from registry.loader import load_skills_directory

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        _write_yaml(
            skills_dir,
            "skill_a.yaml",
            """
            id: skill_a
            name: Skill A
            type: role
            author: tester
            version: 1.0.0
            description: Has an edge to a non-existent skill.
            edges:
              - to: does_not_exist
                type: requires
            """,
        )

        # Orphan detection fires before any writes; graph count must stay the same.
        before = await neo4j_client._run("MATCH (s:Skill) RETURN count(s) AS n")
        pre_count = before[0]["n"]

        embedder = _MockEmbedder()
        with pytest.raises(KeyError, match="does_not_exist"):
            await load_skills_directory(
                skills_dir=skills_dir,
                schema_path=schema_path,
                client=neo4j_client,
                embedder_module=embedder,
                dry_run=False,
            )

        # Verify no nodes were committed.
        records = await neo4j_client._run("MATCH (s:Skill) RETURN count(s) AS n")
        assert records[0]["n"] == pre_count


# ---------------------------------------------------------------------------
# Test: cycle detection
# ---------------------------------------------------------------------------


class TestCycleDetection:
    async def test_cycle_detection_after_load(
        self, tmp_path, schema_path, neo4j_client
    ):
        """A A->B->A requires cycle must be detected and raise RuntimeError after load."""
        from registry.loader import load_skills_directory

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        _write_yaml(
            skills_dir,
            "skill_a.yaml",
            """
            id: skill_a
            name: Skill A
            type: role
            author: tester
            version: 1.0.0
            description: Skill A in a cycle.
            edges:
              - to: skill_b
                type: requires
            """,
        )
        _write_yaml(
            skills_dir,
            "skill_b.yaml",
            """
            id: skill_b
            name: Skill B
            type: role
            author: tester
            version: 1.0.0
            description: Skill B in a cycle.
            edges:
              - to: skill_a
                type: requires
            """,
        )

        embedder = _MockEmbedder()
        with pytest.raises(RuntimeError, match="[Cc]ycle"):
            await load_skills_directory(
                skills_dir=skills_dir,
                schema_path=schema_path,
                client=neo4j_client,
                embedder_module=embedder,
                dry_run=False,
            )


# ---------------------------------------------------------------------------
# Test: hub_score recomputation
# ---------------------------------------------------------------------------


class TestHubScoreRecomputation:
    async def test_hub_score_recomputed(
        self, tmp_path, schema_path, neo4j_client
    ):
        """After loading 3 skills with differing connectivity,
        hub_score = 1.0 for the most-connected node."""
        from registry.loader import load_skills_directory

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # hub_skill has 2 outgoing edges -> degree 2 (highest).
        # leaf_a and leaf_b have 0 outgoing edges -> degree 0 (lowest).
        # hub_score for hub_skill should be 1.0 after normalization.
        _write_yaml(
            skills_dir,
            "hub_skill.yaml",
            """
            id: hub_skill
            name: Hub Skill
            type: cluster
            author: tester
            version: 1.0.0
            description: Central hub node with 2 outgoing edges.
            edges:
              - to: leaf_a
                type: enables
              - to: leaf_b
                type: enables
            """,
        )
        _write_yaml(
            skills_dir,
            "leaf_a.yaml",
            """
            id: leaf_a
            name: Leaf A
            type: domain
            author: tester
            version: 1.0.0
            description: First leaf node with no outgoing edges.
            edges: []
            """,
        )
        _write_yaml(
            skills_dir,
            "leaf_b.yaml",
            """
            id: leaf_b
            name: Leaf B
            type: domain
            author: tester
            version: 1.0.0
            description: Second leaf node with no outgoing edges.
            edges: []
            """,
        )

        embedder = _MockEmbedder()
        await load_skills_directory(
            skills_dir=skills_dir,
            schema_path=schema_path,
            client=neo4j_client,
            embedder_module=embedder,
            dry_run=False,
        )

        records = await neo4j_client._run(
            "MATCH (s:Skill {id: 'hub_skill'}) RETURN s.hub_score AS hs"
        )
        assert len(records) == 1
        hub_score = records[0]["hs"]
        # hub_skill has 2 outbound edges (highest among test nodes), but other
        # production nodes may have higher degree — so hub_score may be < 1.0.
        # Invariant: hub_skill must beat the leaves (0 outbound).
        assert hub_score > 0.0, f"hub_skill.hub_score should be > 0, got {hub_score}"

        # leaf nodes have 0 outgoing edges → lower hub_score than hub_skill
        leaf_records = await neo4j_client._run(
            "MATCH (s:Skill) WHERE s.id IN ['leaf_a', 'leaf_b'] RETURN s.hub_score AS hs"
        )
        for rec in leaf_records:
            assert rec["hs"] < hub_score
