# server/graph/traversal.py
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from server.models.skill_node import NeighborMetadata, SkillContextObject, SkillNodeMetadata, SkillPayload

if TYPE_CHECKING:
    from core.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

HUB_SCORE_THRESHOLD = 0.6
L2_CAP = 20
MAX_EXTEND_DEPTH = 3


async def get_layer1(client, skill_id: str, direction: str = "outbound") -> list[NeighborMetadata]:
    if direction == "outbound":
        raw = await client.get_outbound_neighbors(skill_id, edge_type=None)
    elif direction == "inbound":
        raw = await client.get_inbound_neighbors(skill_id, edge_type=None)
    else:
        raise ValueError(f"Invalid direction: {direction!r}")
    return [
        NeighborMetadata(
            id=r["id"], name=r["name"], description=r["description"],
            edge_type=r["edge_type"], hub_score=r.get("hub_score", 0.0),
            context_cost=r.get("context_cost", 0), distance=1,
        )
        for r in raw
    ]


async def get_layer2(
    client, layer1: list[NeighborMetadata],
    hub_threshold: float = HUB_SCORE_THRESHOLD, cap: int = L2_CAP,
) -> list[NeighborMetadata]:
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
            if nid in seen_ids or hub < hub_threshold:
                continue
            seen_ids.add(nid)
            results.append(NeighborMetadata(
                id=nid, name=r["name"], description=r["description"],
                edge_type=r["edge_type"], hub_score=hub,
                context_cost=r.get("context_cost", 0), distance=2,
            ))
    return results


async def resolve_extends_chain(client, skill_id: str, depth: int = 0) -> dict:
    if depth >= MAX_EXTEND_DEPTH:
        raise ValueError(f"extends chain exceeds max depth {MAX_EXTEND_DEPTH}")
    node = await client.get_skill_node(skill_id)
    if node is None:
        raise KeyError(f"Skill not found: {skill_id!r}")
    payload = node.get("payload") or {}
    own_instructions: str = payload.get("instructions", "")
    own_tools: set[str] = set(payload.get("tools", []))
    own_knowledge: set[str] = set(payload.get("knowledge", []))
    exclude_tools: set[str] = set(payload.get("exclude_tools", []))
    extends_neighbors = await client.get_outbound_neighbors(skill_id, edge_type="EXTENDS")
    if not extends_neighbors:
        return {"instructions": own_instructions, "tools": own_tools - exclude_tools, "knowledge": own_knowledge}
    parent_id = extends_neighbors[0]["id"]
    parent_resolved = await resolve_extends_chain(client, parent_id, depth=depth + 1)
    # Explicit parentheses to avoid operator precedence bug (- has same precedence as |)
    merged_tools = (own_tools | parent_resolved["tools"]) - exclude_tools
    return {
        "instructions": own_instructions,
        "tools": merged_tools,
        "knowledge": own_knowledge | parent_resolved["knowledge"],
    }


async def build_skill_context_object(client, skill_id: str, depth: str = "shallow") -> SkillContextObject:
    node_dict = await client.get_skill_node(skill_id)
    if node_dict is None:
        raise KeyError(f"Skill not found: {skill_id!r}")
    node_meta = SkillNodeMetadata(
        id=node_dict["id"], name=node_dict.get("name", ""),
        description=node_dict.get("description", ""), type=node_dict.get("type", "role"),
        hub_score=node_dict.get("hub_score", 0.0), degree=node_dict.get("degree", 0),
        context_cost=node_dict.get("context_cost", 0),
    )
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
    layer1 = await get_layer1(client, skill_id, direction="outbound")
    layer2 = await get_layer2(client, layer1) if depth == "deep" else []
    return SkillContextObject(metadata=node_meta, payload=skill_payload, layer_1=layer1, layer_2=layer2)
