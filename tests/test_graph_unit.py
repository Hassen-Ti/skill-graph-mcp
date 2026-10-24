"""
Unit tests for server/graph/traversal.py
All Neo4jClient calls are mocked with AsyncMock — no Neo4j instance required.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from server.graph.traversal import (
    get_layer1,
    get_layer2,
    resolve_extends_chain,
    build_skill_context_object,
    MAX_EXTEND_DEPTH,
    HUB_SCORE_THRESHOLD,
    L2_CAP,
)
from server.models.skill_node import NeighborMetadata, SkillNodeMetadata, SkillPayload, SkillContextObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_neighbor(skill_id: str, hub_score: float = 0.5, edge_type: str = "REQUIRES") -> dict:
    return {
        "id": skill_id,
        "name": skill_id.replace("_", " ").title(),
        "description": f"Description of {skill_id}",
        "edge_type": edge_type,
        "hub_score": hub_score,
        "context_cost": 100,
    }


def make_skill_node_dict(
    skill_id: str,
    hub_score: float = 0.5,
    instructions: str = "original instructions",
    tools: list[str] | None = None,
    knowledge: list[str] | None = None,
    extends: str | None = None,
) -> dict:
    node = {
        "id": skill_id,
        "name": skill_id.replace("_", " ").title(),
        "description": f"Description of {skill_id}",
        "type": "role",
        "hub_score": hub_score,
        "degree": 2,
        "context_cost": 200,
        "payload": {
            "instructions": instructions,
            "tools": tools or ["bash"],
            "knowledge": knowledge or [],
            "exclude_tools": [],
        },
    }
    if extends:
        node["extends"] = extends
    return node


# ---------------------------------------------------------------------------
# Task 2.1 — Test 1: get_layer1_outbound
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_layer1_outbound():
    """get_layer1 with direction='outbound' returns 2 NeighborMetadata built from
    the 2 dicts returned by client.get_outbound_neighbors."""
    client = MagicMock()
    client.get_outbound_neighbors = AsyncMock(
        return_value=[
            make_neighbor("security", hub_score=0.8),
            make_neighbor("api_design", hub_score=0.4),
        ]
    )

    result = await get_layer1(client, "backend_dev", direction="outbound")

    client.get_outbound_neighbors.assert_awaited_once_with("backend_dev", edge_type=None)
    assert len(result) == 2
    assert all(isinstance(n, NeighborMetadata) for n in result)
    ids = {n.id for n in result}
    assert ids == {"security", "api_design"}


# ---------------------------------------------------------------------------
# Task 2.1 — Test 2: get_layer1_inbound
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_layer1_inbound():
    """get_layer1 with direction='inbound' calls get_inbound_neighbors."""
    client = MagicMock()
    client.get_inbound_neighbors = AsyncMock(
        return_value=[make_neighbor("parent_skill", hub_score=0.9)]
    )

    result = await get_layer1(client, "child_skill", direction="inbound")

    client.get_inbound_neighbors.assert_awaited_once_with("child_skill", edge_type=None)
    assert len(result) == 1
    assert result[0].id == "parent_skill"


# ---------------------------------------------------------------------------
# Task 2.1 — Test 3: get_layer2_filters_hub_score
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_layer2_filters_hub_score():
    """get_layer2 keeps neighbors with hub_score >= threshold and discards those below."""
    client = MagicMock()

    # layer_1 has 2 neighbors; each has outbound neighbors
    layer1 = [
        NeighborMetadata(id="node_a", name="Node A", description="A", edge_type="REQUIRES", hub_score=0.7, context_cost=100),
        NeighborMetadata(id="node_b", name="Node B", description="B", edge_type="ENABLES", hub_score=0.5, context_cost=100),
    ]

    # node_a's neighbors: one above threshold, one below
    # node_b's neighbors: one above threshold
    async def mock_outbound(skill_id: str, edge_type=None):
        if skill_id == "node_a":
            return [
                make_neighbor("high_hub", hub_score=0.8),   # kept
                make_neighbor("low_hub", hub_score=0.3),    # filtered out
            ]
        if skill_id == "node_b":
            return [make_neighbor("another_high", hub_score=0.75)]  # kept
        return []

    client.get_outbound_neighbors = AsyncMock(side_effect=mock_outbound)

    result = await get_layer2(client, layer1, hub_threshold=0.6, cap=20)

    result_ids = {n.id for n in result}
    assert "high_hub" in result_ids
    assert "another_high" in result_ids
    assert "low_hub" not in result_ids
    # layer1 nodes themselves should not appear in layer2
    assert "node_a" not in result_ids
    assert "node_b" not in result_ids


# ---------------------------------------------------------------------------
# Task 2.1 — Test 4: get_layer2_hard_cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_layer2_hard_cap():
    """get_layer2 returns at most `cap` neighbors even when candidates exceed it."""
    client = MagicMock()

    # 5 layer1 nodes, each with 10 outbound neighbors all above threshold → 50 candidates
    layer1 = [
        NeighborMetadata(id=f"l1_{i}", name=f"L1 {i}", description="", edge_type="REQUIRES", hub_score=0.9, context_cost=100)
        for i in range(5)
    ]

    async def mock_outbound(skill_id: str, edge_type=None):
        base = int(skill_id.split("_")[1]) * 10
        return [make_neighbor(f"candidate_{base + j}", hub_score=0.9) for j in range(10)]

    client.get_outbound_neighbors = AsyncMock(side_effect=mock_outbound)

    result = await get_layer2(client, layer1, hub_threshold=0.6, cap=20)

    assert len(result) <= 20


# ---------------------------------------------------------------------------
# Task 2.1 — Test 5: resolve_extends_no_parent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_extends_no_parent():
    """resolve_extends_chain on a skill with no EXTENDS edge returns original instructions."""
    client = MagicMock()

    skill_dict = make_skill_node_dict(
        "standalone_skill",
        instructions="Do the thing.",
        tools=["bash", "python"],
        knowledge=["ref_a"],
    )
    client.get_skill_node = AsyncMock(return_value=skill_dict)
    client.get_outbound_neighbors = AsyncMock(return_value=[])  # no extends

    result = await resolve_extends_chain(client, "standalone_skill")

    assert result["instructions"] == "Do the thing."
    assert result["tools"] == {"bash", "python"}
    assert result["knowledge"] == {"ref_a"}


# ---------------------------------------------------------------------------
# Task 2.1 — Test 6: resolve_extends_one_parent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_extends_one_parent():
    """resolve_extends_chain: child instructions replace parent; tools and knowledge are
    the union of child and parent."""
    client = MagicMock()

    parent_dict = make_skill_node_dict(
        "parent_skill",
        instructions="Parent instructions.",
        tools=["bash", "git"],
        knowledge=["parent_kb"],
    )
    child_dict = make_skill_node_dict(
        "child_skill",
        instructions="Child instructions.",  # must replace parent
        tools=["python"],                     # union with parent: bash + git + python
        knowledge=["child_kb"],               # union with parent: parent_kb + child_kb
    )

    async def mock_get_skill_node(skill_id: str):
        if skill_id == "child_skill":
            return child_dict
        if skill_id == "parent_skill":
            return parent_dict
        return None

    async def mock_outbound(skill_id: str, edge_type=None):
        if skill_id == "child_skill" and edge_type == "EXTENDS":
            return [{"id": "parent_skill", "name": "Parent Skill", "description": "", "edge_type": "EXTENDS", "hub_score": 0.7, "context_cost": 200}]
        return []

    client.get_skill_node = AsyncMock(side_effect=mock_get_skill_node)
    client.get_outbound_neighbors = AsyncMock(side_effect=mock_outbound)

    result = await resolve_extends_chain(client, "child_skill")

    # Instructions = total substitution — child wins
    assert result["instructions"] == "Child instructions."
    # Tools = union of child + parent
    assert result["tools"] == {"bash", "git", "python"}
    # Knowledge = union of child + parent
    assert result["knowledge"] == {"parent_kb", "child_kb"}


# ---------------------------------------------------------------------------
# Task 2.1 — Test 7: resolve_extends_depth_exceeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_extends_depth_exceeded():
    """resolve_extends_chain raises ValueError when depth exceeds MAX_EXTEND_DEPTH."""
    client = MagicMock()

    # We call resolve_extends_chain with depth already at MAX_EXTEND_DEPTH
    with pytest.raises(ValueError, match="extends chain exceeds max depth"):
        await resolve_extends_chain(client, "some_skill", depth=MAX_EXTEND_DEPTH)
