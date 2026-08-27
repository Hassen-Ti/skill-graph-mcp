# tests/test_registry_loader_unit.py
"""
Unit tests for registry/loader.py — pure Python, no Neo4j.

Adapted from the version recovered off the `master` branch (commit 94b6815):
uses the real skills/schema.json instead of a hand-duplicated copy, so these
tests can't silently drift from the actual schema.
"""

import shutil
from pathlib import Path

import pytest

REAL_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "skills" / "schema.json"


def _make_schema(tmp_path: Path) -> Path:
    """Copy the real skills/schema.json into a temp directory and return its path."""
    schema_path = tmp_path / "schema.json"
    shutil.copy(REAL_SCHEMA_PATH, schema_path)
    return schema_path


def _valid_skill() -> dict:
    return {
        "id": "test-skill",
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
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        errors = validate_yaml(_valid_skill(), schema_path)
        assert errors == []

    def test_validate_yaml_missing_required_id(self, tmp_path):
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        del skill["id"]

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("id" in e for e in errors)

    def test_validate_yaml_invalid_id_format(self, tmp_path):
        """An id with spaces and uppercase violates ^[a-z0-9_-]+$."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["id"] = "My Skill!"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("id" in e for e in errors)

    def test_validate_yaml_id_allows_hyphens(self, tmp_path):
        """The current schema allows hyphens in id (unlike the historical underscore-only pattern)."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["id"] = "test-skill-with-hyphens"

        errors = validate_yaml(skill, schema_path)

        assert errors == []

    def test_validate_yaml_bad_version(self, tmp_path):
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["version"] = "1.0"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0
        assert any("version" in e for e in errors)

    def test_validate_yaml_unknown_type(self, tmp_path):
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["type"] = "wizard"

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0

    def test_validate_yaml_unknown_edge_type_rejected(self, tmp_path):
        """The edges[].type enum must reject anything outside the 6 known relationship types."""
        from registry.loader import validate_yaml

        schema_path = _make_schema(tmp_path)
        skill = _valid_skill()
        skill["edges"] = [{"to": "other-skill", "type": "invented_relation"}]

        errors = validate_yaml(skill, schema_path)

        assert len(errors) > 0


# ---------------------------------------------------------------------------
# resolve_extends_chain_loader
# ---------------------------------------------------------------------------


class TestResolveExtendsChain:
    """Tests for registry.loader.resolve_extends_chain_loader."""

    def _make_registry(self, skills: list[dict]) -> dict:
        return {s["id"]: s for s in skills}

    def test_resolve_extends_no_parent(self):
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

        assert result["instructions"] == "Child instructions."
        assert "bash" in result["tools"]
        assert "git" in result["tools"]
        assert "doc_child" in result["knowledge"]
        assert "doc_parent" in result["knowledge"]

    def test_resolve_extends_exclude_tools(self):
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
        from registry.loader import resolve_extends_chain_loader

        skills = []
        for sid, parent_id in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", None)]:
            s = {
                "id": sid,
                "payload": {"instructions": f"Instructions for {sid}", "tools": [], "knowledge": []},
            }
            if parent_id:
                s["extends"] = parent_id
            skills.append(s)

        registry = self._make_registry(skills)

        with pytest.raises(ValueError, match="max depth"):
            resolve_extends_chain_loader("a", registry)

    def test_resolve_extends_missing_parent(self):
        from registry.loader import resolve_extends_chain_loader

        child = {
            "id": "child",
            "extends": "nonexistent_parent",
            "payload": {"instructions": "Child instructions.", "tools": [], "knowledge": []},
        }
        registry = self._make_registry([child])

        with pytest.raises(KeyError):
            resolve_extends_chain_loader("child", registry)


# ---------------------------------------------------------------------------
# _detect_orphan_edges — covers the exact failure mode pipeline/enrich_skills.py's
# _prune_dangling_edges (Codex-flagged bug, fixed this session) exists to prevent.
# ---------------------------------------------------------------------------


class TestDetectOrphanEdges:
    def test_detect_orphan_edges_raises_on_unknown_target(self):
        from registry.loader import _detect_orphan_edges

        registry = {
            "a": {"id": "a", "edges": [{"to": "missing-skill", "type": "collaborates_with"}]},
        }

        with pytest.raises(KeyError):
            _detect_orphan_edges(registry)

    def test_detect_orphan_edges_passes_when_all_targets_known(self):
        from registry.loader import _detect_orphan_edges

        registry = {
            "a": {"id": "a", "edges": [{"to": "b", "type": "collaborates_with"}]},
            "b": {"id": "b", "edges": []},
        }

        _detect_orphan_edges(registry)  # must not raise
