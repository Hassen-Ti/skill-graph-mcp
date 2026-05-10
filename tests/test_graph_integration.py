"""
Integration tests for Neo4jClient and traversal.py.

Requires a running Neo4j instance (see docker-compose.yml).
Each test creates its own data and cleans up with DETACH DELETE afterwards.
No test depends on data created by another test.
"""
from __future__ import annotations

import pytest
from neo4j import AsyncDriver

from server.graph.neo4j_client import Neo4jClient
from server.graph.traversal import (
    build_skill_context_object,
    get_layer1,
    resolve_extends_chain,
)
from server.models.skill_node import SkillContextObject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(neo4j_driver: AsyncDriver) -> Neo4jClient:
    """Provide a Neo4jClient connected to the test Neo4j instance."""
    c = Neo4jClient(neo4j_driver)
    await c.setup_schema()
    return c


# ---------------------------------------------------------------------------
# Test 1: write and read a Skill node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_write_and_read_skill(client: Neo4jClient):
    """Upsert a Skill node, then read it back — all fields must match."""
    skill_data = {
        "id": "integ_test_skill_01",
        "name": "Integration Test Skill",
        "description": "A skill created during integration testing.",
        "type": "role",
        "hub_score": 0.0,
        "degree": 0,
        "context_cost": 300,
        "payload": {
            "instructions": "You are an integration test agent.",
            "tools": ["bash", "python"],
            "knowledge": ["test_kb"],
            "exclude_tools": [],
        },
    }

    try:
        await client.upsert_skill_node(skill_data)
        result = await client.get_skill_node("integ_test_skill_01")

        assert result is not None, "get_skill_node returned None after upsert"
        assert result["id"] == "integ_test_skill_01"
        assert result["name"] == "Integration Test Skill"
        assert result["description"] == "A skill created during integration testing."
        assert result["payload"]["instructions"] == "You are an integration test agent."
        assert "bash" in result["payload"]["tools"]
        assert "python" in result["payload"]["tools"]
        assert "test_kb" in result["payload"]["knowledge"]

    finally:
        await client.delete_skill_node("integ_test_skill_01")
        cleaned = await client.get_skill_node("integ_test_skill_01")
        assert cleaned is None, "Cleanup failed: node still present after DETACH DELETE"


# ---------------------------------------------------------------------------
# Test 2: write and read an edge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_neo4j_write_and_read_edge(client: Neo4jClient):
    """Upsert two Skill nodes and a REQUIRES edge, then verify get_outbound_neighbors."""
    src_id = "integ_edge_src_01"
    dst_id = "integ_edge_dst_01"

    try:
        await client.upsert_skill_node({"id": src_id, "name": "Source", "description": "src"})
        await client.upsert_skill_node({"id": dst_id, "name": "Destination", "description": "dst"})
        await client.upsert_edge(src_id, dst_id, "REQUIRES")

        neighbors = await client.get_outbound_neighbors(src_id)

        ids = [n["id"] for n in neighbors]
        assert dst_id in ids, f"Expected {dst_id} in outbound neighbors of {src_id}, got {ids}"
        edge_types = [n["edge_type"] for n in neighbors if n["id"] == dst_id]
        assert "REQUIRES" in edge_types

    finally:
        await client.delete_skill_node(src_id)
        await client.delete_skill_node(dst_id)


# ---------------------------------------------------------------------------
# Test 3: hub_score recomputation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hub_score_recomputed(client: Neo4jClient):
    """After upserting skills with different out-degrees, recompute_hub_scores must
    assign hub_score = degree / max_degree.

    hub_test_a: 2 outbound edges → degree=2 → hub_score=1.0
    hub_test_b: 1 outbound edge  → degree=1 → hub_score=0.5
    """
    ids = ["hub_test_a", "hub_test_b", "hub_test_c", "hub_test_d"]

    try:
        for skill_id in ids:
            await client.upsert_skill_node(
                {"id": skill_id, "name": skill_id, "description": skill_id}
            )

        await client.upsert_edge("hub_test_a", "hub_test_c", "REQUIRES")
        await client.upsert_edge("hub_test_a", "hub_test_d", "ENABLES")
        await client.upsert_edge("hub_test_b", "hub_test_c", "REQUIRES")

        await client.recompute_hub_scores()

        node_a = await client.get_skill_node("hub_test_a")
        node_b = await client.get_skill_node("hub_test_b")

        assert node_a is not None
        assert node_b is not None

        # hub_test_a has degree=2, hub_test_b has degree=1.
        # hub_score = degree / max_degree_across_all_nodes.
        # The absolute values depend on what else is in the graph,
        # but the 2:1 ratio is invariant.
        assert node_a["hub_score"] > node_b["hub_score"], (
            f"Expected hub_test_a.hub_score > hub_test_b.hub_score, "
            f"got {node_a['hub_score']} vs {node_b['hub_score']}"
        )
        ratio = node_a["hub_score"] / node_b["hub_score"]
        assert abs(ratio - 2.0) < 0.01, (
            f"Expected hub_test_a/hub_test_b score ratio ≈ 2.0, got {ratio}"
        )

    finally:
        for skill_id in ids:
            await client.delete_skill_node(skill_id)


# ---------------------------------------------------------------------------
# Test 4: cycle detection — no cycles in a DAG
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cycle_detection_no_cycles(client: Neo4jClient):
    """A simple DAG (A → B → C) must not trigger cycle detection."""
    ids = ["cycle_test_a", "cycle_test_b", "cycle_test_c"]

    try:
        for skill_id in ids:
            await client.upsert_skill_node(
                {"id": skill_id, "name": skill_id, "description": skill_id}
            )
        await client.upsert_edge("cycle_test_a", "cycle_test_b", "REQUIRES")
        await client.upsert_edge("cycle_test_b", "cycle_test_c", "REQUIRES")

        cycles = await client.detect_cycles()

        cycle_set = set(cycles)
        for skill_id in ids:
            assert skill_id not in cycle_set, (
                f"DAG node {skill_id!r} incorrectly flagged as part of a cycle"
            )

    finally:
        for skill_id in ids:
            await client.delete_skill_node(skill_id)


# ---------------------------------------------------------------------------
# Test 5: build_skill_context_object integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_skill_context_object_integration(client: Neo4jClient):
    """Load a complete skill graph fragment and verify that build_skill_context_object
    returns a valid SkillContextObject with populated metadata, payload, and layer1."""
    root_id = "ctx_root_01"
    neighbor_a_id = "ctx_neighbor_a_01"
    neighbor_b_id = "ctx_neighbor_b_01"

    try:
        await client.upsert_skill_node({
            "id": root_id,
            "name": "Context Root",
            "description": "Root skill for context object integration test.",
            "type": "role",
            "hub_score": 0.8,
            "degree": 2,
            "context_cost": 500,
            "payload": {
                "instructions": "You are the root agent.",
                "tools": ["bash", "git"],
                "knowledge": ["root_kb"],
                "exclude_tools": [],
            },
        })
        await client.upsert_skill_node({
            "id": neighbor_a_id,
            "name": "Neighbor A",
            "description": "First neighbor.",
            "type": "role",
            "hub_score": 0.7,
            "degree": 0,
            "context_cost": 200,
            "payload": {
                "instructions": "Neighbor A instructions.",
                "tools": ["python"],
                "knowledge": [],
                "exclude_tools": [],
            },
        })
        await client.upsert_skill_node({
            "id": neighbor_b_id,
            "name": "Neighbor B",
            "description": "Second neighbor.",
            "type": "role",
            "hub_score": 0.4,
            "degree": 0,
            "context_cost": 150,
            "payload": {
                "instructions": "Neighbor B instructions.",
                "tools": ["curl"],
                "knowledge": [],
                "exclude_tools": [],
            },
        })
        await client.upsert_edge(root_id, neighbor_a_id, "REQUIRES")
        await client.upsert_edge(root_id, neighbor_b_id, "ENABLES")

        ctx = await build_skill_context_object(client, root_id, depth="shallow")

        assert isinstance(ctx, SkillContextObject)

        # Node metadata (field is 'metadata' in our implementation)
        assert ctx.metadata.id == root_id
        assert ctx.metadata.name == "Context Root"
        assert ctx.metadata.hub_score == pytest.approx(0.8, abs=0.01)

        # Payload
        assert ctx.payload is not None
        assert ctx.payload.instructions == "You are the root agent."
        assert "bash" in ctx.payload.tools
        assert "root_kb" in ctx.payload.knowledge

        # Layer 1: both neighbors must be present
        l1_ids = {n.id for n in ctx.layer_1}
        assert neighbor_a_id in l1_ids
        assert neighbor_b_id in l1_ids

        # Layer 2: not requested (depth='shallow') → must be empty
        assert ctx.layer_2 == []

    finally:
        await client.delete_skill_node(root_id)
        await client.delete_skill_node(neighbor_a_id)
        await client.delete_skill_node(neighbor_b_id)
