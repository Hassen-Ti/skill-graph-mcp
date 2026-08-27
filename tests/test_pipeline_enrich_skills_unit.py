# tests/test_pipeline_enrich_skills_unit.py
"""
Unit tests for pipeline/enrich_skills.py — pure filesystem/YAML logic, no
Neo4j or OpenAI. STAGING_DIR/SKILLS_LIB are monkeypatched to tmp_path
fixtures so nothing here touches the real staging/skills/ or the raw
source library.

Covers the Codex-flagged bug fixed this session: _prune_dangling_edges must
strip edges pointing at a skill about to be deleted from every surviving
YAML, or a subsequent registry.cli load rejects the orphan reference.
"""
from pathlib import Path

import pytest
import yaml

import pipeline.enrich_skills as m


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")


class TestParseSkillMd:
    def test_parses_frontmatter_and_body(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(
            "---\nname: My Skill\ndescription: Does a thing.\n---\nBody content here.",
            encoding="utf-8",
        )

        fm, body = m.parse_skill_md(skill_md)

        assert fm == {"name": "My Skill", "description": "Does a thing."}
        assert body == "Body content here."

    def test_no_frontmatter_returns_full_text_as_body(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("Just a body, no frontmatter.", encoding="utf-8")

        fm, body = m.parse_skill_md(skill_md)

        assert fm == {}
        assert body == "Just a body, no frontmatter."


class TestTruncate:
    def test_no_op_under_limit(self):
        assert m._truncate("short", 500) == "short"

    def test_truncates_and_appends_ellipsis(self):
        result = m._truncate("x" * 600, 500)
        assert len(result) == 503
        assert result.endswith("...")


class TestBuildInstructions:
    def test_inlines_nested_sub_skills(self, tmp_path):
        skill_dir = tmp_path / "parent"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Parent body.", encoding="utf-8")
        sub_dir = skill_dir / "sub-one"
        sub_dir.mkdir()
        (sub_dir / "SKILL.md").write_text(
            "---\nname: Sub One\n---\nSub body.", encoding="utf-8",
        )

        instructions = m.build_instructions(skill_dir)

        assert "Parent body." in instructions
        assert "## Sub One" in instructions
        assert "Sub body." in instructions


class TestPruneDanglingEdges:
    def test_strips_edge_to_removed_id_keeps_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "STAGING_DIR", tmp_path)
        _write_yaml(tmp_path / "a.yaml", {
            "id": "a", "edges": [
                {"to": "b", "type": "collaborates_with"},
                {"to": "doomed", "type": "collaborates_with"},
            ],
            "payload": {"instructions": "instructions for a"},
        })
        _write_yaml(tmp_path / "doomed.yaml", {"id": "doomed", "payload": {"instructions": "x"}})

        m._prune_dangling_edges({"doomed"}, dry_run=False)

        result = yaml.safe_load((tmp_path / "a.yaml").read_text(encoding="utf-8"))
        assert result["edges"] == [{"to": "b", "type": "collaborates_with"}]

    def test_drops_edges_key_entirely_when_all_edges_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "STAGING_DIR", tmp_path)
        _write_yaml(tmp_path / "a.yaml", {
            "id": "a", "edges": [{"to": "doomed", "type": "collaborates_with"}],
            "payload": {"instructions": "instructions for a"},
        })

        m._prune_dangling_edges({"doomed"}, dry_run=False)

        result = yaml.safe_load((tmp_path / "a.yaml").read_text(encoding="utf-8"))
        assert "edges" not in result

    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "STAGING_DIR", tmp_path)
        original = {
            "id": "a", "edges": [{"to": "doomed", "type": "collaborates_with"}],
            "payload": {"instructions": "instructions for a"},
        }
        _write_yaml(tmp_path / "a.yaml", original)

        m._prune_dangling_edges({"doomed"}, dry_run=True)

        result = yaml.safe_load((tmp_path / "a.yaml").read_text(encoding="utf-8"))
        assert result == original

    def test_skill_being_removed_itself_is_not_rewritten(self, tmp_path, monkeypatch):
        """The doomed file itself is skipped by _prune_dangling_edges (clean_staging
        deletes it separately) — pruning must not touch it."""
        monkeypatch.setattr(m, "STAGING_DIR", tmp_path)
        _write_yaml(tmp_path / "doomed.yaml", {
            "id": "doomed", "edges": [{"to": "doomed", "type": "collaborates_with"}],
        })

        m._prune_dangling_edges({"doomed"}, dry_run=False)  # must not raise or rewrite

        result = yaml.safe_load((tmp_path / "doomed.yaml").read_text(encoding="utf-8"))
        assert result["edges"] == [{"to": "doomed", "type": "collaborates_with"}]

    def test_leaves_unaffected_yamls_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "STAGING_DIR", tmp_path)
        _write_yaml(tmp_path / "unrelated.yaml", {
            "id": "unrelated", "edges": [{"to": "still-here", "type": "collaborates_with"}],
        })

        m._prune_dangling_edges({"doomed"}, dry_run=False)

        result = yaml.safe_load((tmp_path / "unrelated.yaml").read_text(encoding="utf-8"))
        assert result["edges"] == [{"to": "still-here", "type": "collaborates_with"}]


class TestCleanStagingIntegration:
    def test_clean_staging_prunes_before_deleting(self, tmp_path, monkeypatch):
        """End-to-end: a skill removed from the source lib is deleted from staging,
        AND any other skill's edge pointing at it is pruned first — never left dangling."""
        staging = tmp_path / "staging"
        staging.mkdir()
        lib = tmp_path / "lib"
        lib.mkdir()
        (lib / "a").mkdir()
        (lib / "a" / "SKILL.md").write_text("---\nname: A\n---\nBody A.", encoding="utf-8")
        # "doomed" exists in staging but no longer in the source lib

        monkeypatch.setattr(m, "STAGING_DIR", staging)
        monkeypatch.setattr(m, "SKILLS_LIB", lib)

        _write_yaml(staging / "a.yaml", {
            "id": "a", "edges": [{"to": "doomed", "type": "collaborates_with"}],
            "payload": {"instructions": "x"},
        })
        _write_yaml(staging / "doomed.yaml", {"id": "doomed", "payload": {"instructions": "x"}})

        m.clean_staging(dry_run=False)

        assert not (staging / "doomed.yaml").exists()
        result = yaml.safe_load((staging / "a.yaml").read_text(encoding="utf-8"))
        assert "edges" not in result
