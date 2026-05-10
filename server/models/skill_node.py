# server/models/skill_node.py
"""
Core Pydantic v2 data models for Skill Graph nodes and derived structures.

Models:
  SkillNodeMetadata   — full metadata for a single skill node
  SkillPayload        — actionable content attached to a node (instructions, tools, knowledge)
  NeighborMetadata    — lightweight view of an adjacent node used in context objects
  SkillContextObject  — assembled context passed to the LLM for a selected skill
  SkillCandidate      — ranked search result returned by skill discovery tools
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The five allowed node categories in the skill graph.
NodeType = Literal["role", "tool", "domain", "responsibility", "cluster"]


class SkillNodeMetadata(BaseModel):
    """Full metadata record for a skill node as stored in Neo4j."""

    id: str
    name: str
    description: str
    author: str = ""
    version: str = ""
    type: NodeType = "role"
    priority: int = Field(default=1, ge=1, le=3)
    prerequisites: list[str] = []
    hub_score: float = Field(default=0.0, ge=0.0, le=1.0)
    degree: int = 0
    context_cost: int = 0


class SkillPayload(BaseModel):
    """Actionable content attached to a skill node."""

    instructions: str = ""
    tools: list[str] = []
    knowledge: list[str] = []
    exclude_tools: list[str] = []


class NeighborMetadata(BaseModel):
    """Lightweight representation of a neighbor node used in SkillContextObject."""

    id: str
    name: str
    description: str
    edge_type: str
    hub_score: float
    context_cost: int
    distance: int = 1  # 1 = direct neighbor, 2 = two hops away


class SkillContextObject(BaseModel):
    """
    Full context object assembled for a selected skill.
    Passed to the LLM as structured context.
    """

    metadata: SkillNodeMetadata
    layer_1: list[NeighborMetadata]
    layer_2: list[NeighborMetadata]
    payload: SkillPayload | None = None


class SkillCandidate(BaseModel):
    """
    Ranked candidate returned by skill search / discovery.
    Combines semantic similarity with graph centrality (hub_score).
    """

    id: str
    name: str
    semantic_score: float
    hub_score: float
