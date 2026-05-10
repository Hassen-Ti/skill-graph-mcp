# tests/test_models.py
"""
TDD test suite for server/models/edge.py and server/models/skill_node.py.
Written BEFORE implementation. All tests must fail with ImportError first,
then pass green after implementation.
"""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestEdgeTypeEnum:
    """Task: test_edge_type_enum"""

    def test_edge_type_enum(self):
        from server.models.edge import EdgeType

        assert EdgeType.REQUIRES == "requires"
        assert EdgeType.ENABLES == "enables"
        assert EdgeType.COLLABORATES_WITH == "collaborates_with"
        assert EdgeType.USES == "uses"
        assert EdgeType.PART_OF == "part_of"
        assert EdgeType.EXTENDS == "extends"
        assert len(EdgeType) == 6


class TestEdgeDirectionEnum:
    """Task: test_edge_direction_enum"""

    def test_edge_direction_enum(self):
        from server.models.edge import EdgeDirection

        assert EdgeDirection.OUTBOUND == "outbound"
        assert EdgeDirection.INBOUND == "inbound"
        assert EdgeDirection.BOTH == "both"
        assert len(EdgeDirection) == 3


# ---------------------------------------------------------------------------
# SkillNodeMetadata tests
# ---------------------------------------------------------------------------

class TestSkillNodeMetadata:
    """Tasks: test_skill_node_metadata_required_fields, test_skill_node_metadata_defaults"""

    def _valid_payload(self) -> dict:
        return {
            "id": "python-basics",
            "name": "Python Basics",
            "description": "Foundational Python knowledge",
            "author": "alice",
            "version": "1.0.0",
            "type": "domain",
            "priority": 2,
            "prerequisites": [],
            "hub_score": 0.75,
            "degree": 5,
            "context_cost": 300,
        }

    def test_skill_node_metadata_required_fields(self):
        from server.models.skill_node import SkillNodeMetadata

        node = SkillNodeMetadata(**self._valid_payload())

        assert node.id == "python-basics"
        assert node.name == "Python Basics"
        assert node.description == "Foundational Python knowledge"
        assert node.author == "alice"
        assert node.version == "1.0.0"
        assert node.type == "domain"
        assert node.priority == 2
        assert node.prerequisites == []
        assert node.hub_score == 0.75
        assert node.degree == 5
        assert node.context_cost == 300

    def test_skill_node_metadata_defaults(self):
        """prerequisites has no default — it must be supplied; verify empty list accepted."""
        from server.models.skill_node import SkillNodeMetadata

        payload = self._valid_payload()
        payload["prerequisites"] = []
        node = SkillNodeMetadata(**payload)

        assert node.prerequisites == []

    def test_skill_node_metadata_missing_required_field_raises(self):
        """Omitting a required field must raise ValidationError."""
        from server.models.skill_node import SkillNodeMetadata

        payload = self._valid_payload()
        del payload["name"]

        with pytest.raises(ValidationError):
            SkillNodeMetadata(**payload)

    def test_skill_node_metadata_invalid_type_raises(self):
        """NodeType must be one of the five literals."""
        from server.models.skill_node import SkillNodeMetadata

        payload = self._valid_payload()
        payload["type"] = "invalid_type"

        with pytest.raises(ValidationError):
            SkillNodeMetadata(**payload)


# ---------------------------------------------------------------------------
# priority bounds
# ---------------------------------------------------------------------------

class TestSkillNodeMetadataPriorityBounds:
    """Task: test_skill_node_metadata_priority_bounds"""

    def _base(self) -> dict:
        return {
            "id": "x",
            "name": "X",
            "description": "d",
            "author": "a",
            "version": "1.0",
            "type": "tool",
            "prerequisites": [],
            "hub_score": 0.5,
            "degree": 1,
            "context_cost": 100,
        }

    def test_priority_1_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "priority": 1})
        assert node.priority == 1

    def test_priority_2_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "priority": 2})
        assert node.priority == 2

    def test_priority_3_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "priority": 3})
        assert node.priority == 3

    def test_priority_0_raises(self):
        from server.models.skill_node import SkillNodeMetadata
        with pytest.raises(ValidationError):
            SkillNodeMetadata(**{**self._base(), "priority": 0})

    def test_priority_4_raises(self):
        from server.models.skill_node import SkillNodeMetadata
        with pytest.raises(ValidationError):
            SkillNodeMetadata(**{**self._base(), "priority": 4})

    def test_priority_negative_raises(self):
        from server.models.skill_node import SkillNodeMetadata
        with pytest.raises(ValidationError):
            SkillNodeMetadata(**{**self._base(), "priority": -1})


# ---------------------------------------------------------------------------
# hub_score bounds
# ---------------------------------------------------------------------------

class TestHubScoreBounds:
    """Task: test_hub_score_bounds"""

    def _base(self) -> dict:
        return {
            "id": "x",
            "name": "X",
            "description": "d",
            "author": "a",
            "version": "1.0",
            "type": "role",
            "priority": 1,
            "prerequisites": [],
            "degree": 1,
            "context_cost": 50,
        }

    def test_hub_score_zero_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "hub_score": 0.0})
        assert node.hub_score == 0.0

    def test_hub_score_one_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "hub_score": 1.0})
        assert node.hub_score == 1.0

    def test_hub_score_midrange_valid(self):
        from server.models.skill_node import SkillNodeMetadata
        node = SkillNodeMetadata(**{**self._base(), "hub_score": 0.42})
        assert node.hub_score == pytest.approx(0.42)

    def test_hub_score_negative_raises(self):
        from server.models.skill_node import SkillNodeMetadata
        with pytest.raises(ValidationError):
            SkillNodeMetadata(**{**self._base(), "hub_score": -0.01})

    def test_hub_score_above_one_raises(self):
        from server.models.skill_node import SkillNodeMetadata
        with pytest.raises(ValidationError):
            SkillNodeMetadata(**{**self._base(), "hub_score": 1.01})


# ---------------------------------------------------------------------------
# SkillPayload defaults
# ---------------------------------------------------------------------------

class TestSkillPayloadDefaults:
    """Task: test_skill_payload_defaults"""

    def test_skill_payload_all_defaults(self):
        from server.models.skill_node import SkillPayload

        payload = SkillPayload()
        assert payload.instructions == ""
        assert payload.tools == []
        assert payload.knowledge == []

    def test_skill_payload_with_values(self):
        from server.models.skill_node import SkillPayload

        payload = SkillPayload(
            instructions="Do X then Y",
            tools=["bash", "python"],
            knowledge=["PEP8", "typing"],
        )
        assert payload.instructions == "Do X then Y"
        assert payload.tools == ["bash", "python"]
        assert payload.knowledge == ["PEP8", "typing"]


# ---------------------------------------------------------------------------
# NeighborMetadata distance default
# ---------------------------------------------------------------------------

class TestNeighborMetadataDistanceDefault:
    """Task: test_neighbor_metadata_distance_default"""

    def _valid_neighbor(self) -> dict:
        return {
            "id": "neighbor-1",
            "name": "Neighbor One",
            "description": "A neighboring skill",
            "edge_type": "requires",
            "hub_score": 0.6,
            "context_cost": 200,
        }

    def test_distance_defaults_to_1(self):
        from server.models.skill_node import NeighborMetadata

        neighbor = NeighborMetadata(**self._valid_neighbor())
        assert neighbor.distance == 1

    def test_distance_can_be_set_to_2(self):
        from server.models.skill_node import NeighborMetadata

        neighbor = NeighborMetadata(**{**self._valid_neighbor(), "distance": 2})
        assert neighbor.distance == 2

    def test_neighbor_required_fields(self):
        from server.models.skill_node import NeighborMetadata

        neighbor = NeighborMetadata(**self._valid_neighbor())
        assert neighbor.id == "neighbor-1"
        assert neighbor.name == "Neighbor One"
        assert neighbor.edge_type == "requires"
        assert neighbor.hub_score == 0.6
        assert neighbor.context_cost == 200


# ---------------------------------------------------------------------------
# SkillContextObject structure
# ---------------------------------------------------------------------------

class TestSkillContextObjectStructure:
    """Task: test_skill_context_object_structure"""

    def _make_metadata(self):
        from server.models.skill_node import SkillNodeMetadata
        return SkillNodeMetadata(
            id="root-skill",
            name="Root Skill",
            description="The root node",
            author="bob",
            version="2.0.0",
            type="cluster",
            priority=1,
            prerequisites=["prereq-1"],
            hub_score=0.9,
            degree=10,
            context_cost=500,
        )

    def _make_neighbor(self, distance: int = 1):
        from server.models.skill_node import NeighborMetadata
        return NeighborMetadata(
            id=f"n-{distance}",
            name=f"Neighbor {distance}",
            description="A neighbor",
            edge_type="enables",
            hub_score=0.4,
            context_cost=150,
            distance=distance,
        )

    def test_skill_context_object_structure(self):
        from server.models.skill_node import SkillContextObject, SkillPayload

        ctx = SkillContextObject(
            metadata=self._make_metadata(),
            layer_1=[self._make_neighbor(1)],
            layer_2=[self._make_neighbor(2)],
            payload=SkillPayload(instructions="Step by step", tools=["git"]),
        )

        assert ctx.metadata.id == "root-skill"
        assert len(ctx.layer_1) == 1
        assert ctx.layer_1[0].distance == 1
        assert len(ctx.layer_2) == 1
        assert ctx.layer_2[0].distance == 2
        assert ctx.payload.instructions == "Step by step"
        assert ctx.payload.tools == ["git"]
        assert ctx.payload.knowledge == []

    def test_skill_context_object_empty_layers_valid(self):
        from server.models.skill_node import SkillContextObject, SkillPayload

        ctx = SkillContextObject(
            metadata=self._make_metadata(),
            layer_1=[],
            layer_2=[],
            payload=SkillPayload(),
        )
        assert ctx.layer_1 == []
        assert ctx.layer_2 == []


# ---------------------------------------------------------------------------
# SkillCandidate
# ---------------------------------------------------------------------------

class TestSkillCandidate:
    """Task: test_skill_candidate"""

    def test_skill_candidate_fields(self):
        from server.models.skill_node import SkillCandidate

        candidate = SkillCandidate(
            id="ml-pipeline",
            name="ML Pipeline",
            semantic_score=0.87,
            hub_score=0.55,
        )

        assert candidate.id == "ml-pipeline"
        assert candidate.name == "ML Pipeline"
        assert candidate.semantic_score == pytest.approx(0.87)
        assert candidate.hub_score == pytest.approx(0.55)

    def test_skill_candidate_missing_field_raises(self):
        from server.models.skill_node import SkillCandidate

        with pytest.raises(ValidationError):
            SkillCandidate(id="x", name="X", semantic_score=0.5)
            # hub_score is missing
