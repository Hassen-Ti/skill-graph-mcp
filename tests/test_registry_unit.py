# tests/test_registry_unit.py
"""
TDD unit test suite for registry/loader.py — pure Python, no Neo4j.
Written BEFORE implementation. All tests must fail with ImportError first,
then pass green after Task 4.4.
"""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema(tmp_path: Path) -> Path:
    """Write the canonical skills/schema.json into a temp directory and return its path."""
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
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    return schema_path


def _valid_skill() -> dict:
    return {
        "id": "test_skill",
        "name": "Test Skill",
        "type": "role",
        "author": "tester",
        "version": "1.0.0",
        "description": "A valid skill for testing purposes.",
        "priority": 2,
        "edges": [],
        "payload": {
            "instructions": "Do X then Y.",
            "tools": ["bash"],
            "knowledge": ["python_basics"],
        },
    }


# ---------------------------------------------------------------------------
# validate_yaml
# ---------------------------------------------------------------------------


class TestValidateYaml:
    """Tests for registry.loader.validate_yaml."""

    def test_validate_yaml_valid(self, tmp_path):
        """A fully valid skill dict should return an empty errors list."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        errors = validate_yaml(_valid_skill(), schema_path)
        assert errors == []

    def test_validate_yaml_missing_required_id(self, tmp_path):
        """Omitting the required 'id' field must produce an error mentioning 'id'."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        del skill["id"]

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("id" in e for e in errors)

    def test_validate_yaml_invalid_id_format(self, tmp_path):
        """An id with spaces and uppercase must violate the ^[a-z0-9_]+$ pattern."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["id"] = "My Skill!"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("id" in e for e in errors)

    def test_validate_yaml_bad_version(self, tmp_path):
        """version='1.0' (missing patch segment) must violate the semver pattern."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["version"] = "1.0"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("version" in e for e in errors)

    def test_validate_yaml_unknown_type(self, tmp_path):
        """type='wizard' is not in the enum; must produce a validation error."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["type"] = "wizard"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0


# ---------------------------------------------------------------------------
# resolve_extends_chain_loader
# ---------------------------------------------------------------------------


class TestResolveExtendsChain:
    """Tests for registry.loader.resolve_extends_chain_loader."""

    def _make_registry(self, skills: list[dict]) -> dict:
        """Build a {skill_id: skill_dict} registry from a list of skill dicts."""
        return {s["id"]: s for s in skills}

    def test_resolve_extends_no_parent(self):
        """A skill with no 'extends' key should return its own payload as-is."""
        from registry.loader import resolve_extends_chain_loader

        skill = {
            "id": "child",
            "payload": {
                "instructions": "Child instructions.",
                "tools": ["bash"],
                "knowledge": ["doc_a"],
            },
        }
        registry = self._make_registry([skill])

        result = resolve_extends_chain_loader("child", registry)

        assert result["instructions"] == "Child instructions."
        assert "bash" in result["tools"]
        assert "doc_a" in result["knowledge"]

    def test_resolve_extends_one_parent(self):
        """
        instructions: child substitutes parent totally.
        tools: union(child.tools, parent.tools).
        knowledge: union(child.knowledge, parent.knowledge).
        """
        from registry.loader import resolve_extends_chain_loader

        parent = {
            "id": "parent",
            "payload": {
                "instructions": "Parent instructions.",
                "tools": ["git"],
                "knowledge": ["doc_parent"],
            },
        }
        child = {
            "id": "child",
            "extends": "parent",
            "payload": {
                "instructions": "Child instructions.",
                "tools": ["bash"],
                "knowledge": ["doc_child"],
            },
        }
        registry = self._make_registry([parent, child])

        result = resolve_extends_chain_loader("child", registry)

        # instructions: total substitution — child wins
        assert result["instructions"] == "Child instructions."
        # tools: union
        assert "bash" in result["tools"]
        assert "git" in result["tools"]
        # knowledge: union
        assert "doc_child" in result["knowledge"]
        assert "doc_parent" in result["knowledge"]

    def test_resolve_extends_exclude_tools(self):
        """exclude_tools on the child must remove the specified tool from the union."""
        from registry.loader import resolve_extends_chain_loader

        parent = {
            "id": "parent",
            "payload": {
                "instructions": "Parent instructions.",
                "tools": ["bash", "git"],
                "knowledge": [],
            },
        }
        child = {
            "id": "child",
            "extends": "parent",
            "payload": {
                "instructions": "Child instructions.",
                "tools": ["python"],
                "exclude_tools": ["bash"],
                "knowledge": [],
            },
        }
        registry = self._make_registry([parent, child])

        result = resolve_extends_chain_loader("child", registry)

        assert "bash" not in result["tools"]
        assert "git" in result["tools"]
        assert "python" in result["tools"]

    def test_resolve_extends_depth_exceeded(self):
        """A chain of 5 levels (depth > 3) must raise ValueError mentioning 'max depth'."""
        from registry.loader import resolve_extends_chain_loader

        # Build chain: a -> b -> c -> d -> e (4 extends hops = depth 4 > MAX_EXTEND_DEPTH=3)
        skills = []
        for i, (sid, parent_id) in enumerate(
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", None)]
        ):
            s = {
                "id": sid,
                "payload": {
                    "instructions": f"Instructions for {sid}",
                    "tools": [],
                    "knowledge": [],
                },
            }
            if parent_id:
                s["extends"] = parent_id
            skills.append(s)

        registry = self._make_registry(skills)

        with pytest.raises(ValueError, match="max depth"):
            resolve_extends_chain_loader("a", registry)

    def test_resolve_extends_missing_parent(self):
        """extends pointing to a non-existent skill_id must raise KeyError."""
        from registry.loader import resolve_extends_chain_loader

        child = {
            "id": "child",
            "extends": "nonexistent_parent",
            "payload": {
                "instructions": "Child instructions.",
                "tools": [],
                "knowledge": [],
            },
        }
        registry = self._make_registry([child])

        with pytest.raises(KeyError):
            resolve_extends_chain_loader("child", registry)
