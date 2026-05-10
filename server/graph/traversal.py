"""
Graph traversal logic for the Skill Graph project.

Builds on Neo4jClient to implement:
- 2-layer BFS (L1 full, L2 filtered by hub_score and capped)
- extends chain resolution (eager, single-parent, max depth 3)
- SkillContextObject builder
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from server.models.skill_node import (
    NeighborMetadata,
    SkillContextObject,
    SkillNodeMetadata,
    SkillPayload,
)

if TYPE_CHECKING:
    from server.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

HUB_SCORE_THRESHOLD = 0.6
L2_CAP = 20
MAX_EXTEND_DEPTH = 3


# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------

async def get_layer1(
    client: "Neo4jClient",
    skill_id: str,
    direction: str = "outbound",
) -> list[NeighborMetadata]:
    """Return immediate neighbors of skill_id.

    direction: 'outbound' | 'inbound'
    Returns a list of NeighborMetadata (distance=1).
    """
    if direction == "outbound":
        raw = await client.get_outbound_neighbors(skill_id, edge_type=None)
    elif direction == "inbound":
        raw = await client.get_inbound_neighbors(skill_id, edge_type=None)
    else:
        raise ValueError(f"Invalid direction: {direction!r}. Must be 'outbound' or 'inbound'.")

    return [
        NeighborMetadata(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            edge_type=r["edge_type"],
            hub_score=r.get("hub_score", 0.0),
            context_cost=r.get("context_cost", 0),
            distance=1,
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# Layer 2
# ---------------------------------------------------------------------------

async def get_layer2(
    client: "Neo4jClient",
    layer1: list[NeighborMetadata],
    hub_threshold: float = HUB_SCORE_THRESHOLD,
    cap: int = L2_CAP,
) -> list[NeighborMetadata]:
    """Return depth-2 neighbors of all nodes in layer1.

    Filters:
    - hub_score >= hub_threshold
    - nodes already present in layer1 are excluded (de-duplication)
    - hard cap of `cap` nodes

    Returns a list of NeighborMetadata (distance=2).
    """
    layer1_ids = {n.id for n in layer1}
    seen_ids: set[str] = set(layer1_ids)
    results: list[NeighborMetadata] = []

    for l1_node in layer1:
        if len(results) >= cap:
            break
        raw_neighbors = await client.get_outbound_neighbors(l1_node.id, edge_type=None)
        for r in raw_neighbors:
            if len(results) >= cap:
                break
            nid = r["id"]
            hub = r.get("hub_score", 0.0)
            if nid in seen_ids:
                continue
            if hub < hub_threshold:
                continue
            seen_ids.add(nid)
            results.append(
                NeighborMetadata(
                    id=nid,
                    name=r["name"],
                    description=r["description"],
                    edge_type=r["edge_type"],
                    hub_score=hub,
                    context_cost=r.get("context_cost", 0),
                    distance=2,
                )
            )

    return results


# ---------------------------------------------------------------------------
# extends resolution
# ---------------------------------------------------------------------------

async def resolve_extends_chain(
    client: "Neo4jClient",
    skill_id: str,
    depth: int = 0,
) -> dict:
    """Resolve the extends inheritance chain for skill_id.

    Returns a dict:
      {
        "instructions": str,         # child instructions replace parent (total substitution)
        "tools": set[str],           # union of entire chain minus exclude_tools
        "knowledge": set[str],       # union of entire chain
      }

    Raises ValueError if depth >= MAX_EXTEND_DEPTH.
    """
    if depth >= MAX_EXTEND_DEPTH:
        raise ValueError(
            f"extends chain exceeds max depth {MAX_EXTEND_DEPTH} "
            f"(skill_id={skill_id!r}, depth={depth})"
        )

    node = await client.get_skill_node(skill_id)
    if node is None:
        raise KeyError(f"Skill not found: {skill_id!r}")

    payload = node.get("payload") or {}
    own_instructions: str = payload.get("instructions", "")
    own_tools: set[str] = set(payload.get("tools", []))
    own_knowledge: set[str] = set(payload.get("knowledge", []))
    exclude_tools: set[str] = set(payload.get("exclude_tools", []))

    # Check for an EXTENDS outbound edge
    extends_neighbors = await client.get_outbound_neighbors(skill_id, edge_type="EXTENDS")

    if not extends_neighbors:
        return {
            "instructions": own_instructions,
            "tools": own_tools - exclude_tools,
            "knowledge": own_knowledge,
        }

    # Single parent (v1 — multiple inheritance deferred to v2)
    parent_id = extends_neighbors[0]["id"]
    parent_resolved = await resolve_extends_chain(client, parent_id, depth=depth + 1)

    # Instruction substitution: child wins entirely
    merged_instructions = own_instructions

    # Tools: union of child + parent, then remove exclude_tools from the full set.
    # Parentheses are explicit to avoid the - operator being applied before |
    # (both have the same precedence and are evaluated left-to-right by default).
    merged_tools = (own_tools | parent_resolved["tools"]) - exclude_tools

    # Knowledge: strict union
    merged_knowledge = own_knowledge | parent_resolved["knowledge"]

    return {
        "instructions": merged_instructions,
        "tools": merged_tools,
        "knowledge": merged_knowledge,
    }


# ---------------------------------------------------------------------------
# SkillContextObject builder
# ---------------------------------------------------------------------------

async def build_skill_context_object(
    client: "Neo4jClient",
    skill_id: str,
    depth: str = "shallow",
) -> SkillContextObject:
    """Build the full SkillContextObject for a skill.

    depth='shallow': returns node metadata + resolved payload + layer1 only.
    depth='deep':    also includes layer2.
    """
    node_dict = await client.get_skill_node(skill_id)
    if node_dict is None:
        raise KeyError(f"Skill not found: {skill_id!r}")

    # Build SkillNodeMetadata (fields with defaults handle partial Neo4j data)
    node_meta = SkillNodeMetadata(
        id=node_dict["id"],
        name=node_dict.get("name", ""),
        description=node_dict.get("description", ""),
        type=node_dict.get("type", "role"),
        hub_score=node_dict.get("hub_score", 0.0),
        degree=node_dict.get("degree", 0),
        context_cost=node_dict.get("context_cost", 0),
    )

    # Resolve extends chain to get merged payload
    skill_payload: SkillPayload | None = None
    try:
        resolved = await resolve_extends_chain(client, skill_id)
        skill_payload = SkillPayload(
            instructions=resolved["instructions"],
            tools=sorted(resolved["tools"]),
            knowledge=sorted(resolved["knowledge"]),
        )
    except (KeyError, ValueError) as exc:
        logger.warning("Could not resolve extends chain for %r: %s", skill_id, exc)
        raw_payload = node_dict.get("payload") or {}
        if raw_payload:
            skill_payload = SkillPayload(
                instructions=raw_payload.get("instructions", ""),
                tools=raw_payload.get("tools", []),
                knowledge=raw_payload.get("knowledge", []),
            )

    # Layer 1
    layer1 = await get_layer1(client, skill_id, direction="outbound")

    # Layer 2 (only for deep depth)
    layer2: list[NeighborMetadata] = []
    if depth == "deep":
        layer2 = await get_layer2(client, layer1)

    return SkillContextObject(
        metadata=node_meta,
        payload=skill_payload,
        layer_1=layer1,
        layer_2=layer2,
    )
