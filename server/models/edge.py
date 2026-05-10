# server/models/edge.py
"""
Edge type definitions for the Skill Graph.

EdgeType      — the semantic relationship between two SkillNodes.
EdgeDirection — the traversal direction used when querying neighbors.
"""

from enum import Enum


class EdgeType(str, Enum):
    """Typed relationships between skill nodes in the graph."""

    REQUIRES = "requires"
    ENABLES = "enables"
    COLLABORATES_WITH = "collaborates_with"
    USES = "uses"
    PART_OF = "part_of"
    EXTENDS = "extends"


class EdgeDirection(str, Enum):
    """Traversal direction for neighbor queries."""

    OUTBOUND = "outbound"
    INBOUND = "inbound"
    BOTH = "both"
