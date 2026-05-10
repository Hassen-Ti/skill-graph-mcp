"""
Sprint 1 — Unit tests for convert_skills.py
No external dependencies, no Neo4j, no OpenAI.
"""

import json
import pytest
import jsonschema
from pathlib import Path

SKILL_GRAPH = Path(__file__).resolve().parent.parent
STAGING_DIR = SKILL_GRAPH / "staging" / "skills"
SCHEMA_FILE = SKILL_GRAPH / "skills" / "schema.json"
REPORT_FILE = SKILL_GRAPH / "staging" / "conversion_report.json"

import sys
sys.path.insert(0, str(SKILL_GRAPH / "scripts"))
from convert_skills import (
    classify_type,
    strip_frontmatter,
    parse_mcp_tools,
    extract_related_skills,
    truncate,
)


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def staging_yamls():
    import yaml
    files = list(STAGING_DIR.glob("*.yaml"))
    result = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            result.append((f.name, yaml.safe_load(fh)))
    return result


# ── classify_type ────────────────────────────────────────────────

class TestClassifyType:
    def test_tool_suffix_pro(self):
        assert classify_type("bash-pro", "You are...") == "tool"

    def test_tool_suffix_automation(self):
        assert classify_type("github-automation", "Automate...") == "tool"

    def test_role_from_description(self):
        assert classify_type("senior-engineer", "You are a senior engineer...") == "role"

    def test_domain_default(self):
        assert classify_type("kubernetes-networking", "Deep dive into K8s networking") == "domain"

    def test_tool_docker(self):
        assert classify_type("docker-compose", "...") == "tool"


# ── strip_frontmatter ─────────────────────────────────────────────

class TestStripFrontmatter:
    def test_strips_yaml_block(self):
        text = "---\nname: foo\n---\n# Body here\nSome content."
        result = strip_frontmatter(text)
        assert result.startswith("# Body here")
        assert "name: foo" not in result

    def test_no_frontmatter(self):
        text = "# Plain content\nNo frontmatter."
        result = strip_frontmatter(text)
        assert result == text

    def test_empty_body(self):
        text = "---\nname: foo\n---\n"
        assert strip_frontmatter(text) == ""


# ── parse_mcp_tools ───────────────────────────────────────────────

class TestParseMcpTools:
    def test_extracts_mcp_list(self):
        text = "---\nname: foo\nrequires:\n  mcp:\n    - rube\n    - zapier\n---\nBody"
        tools = parse_mcp_tools(text)
        assert tools == ["rube", "zapier"]

    def test_no_requires(self):
        text = "---\nname: foo\n---\nBody"
        assert parse_mcp_tools(text) == []

    def test_requires_no_mcp(self):
        text = "---\nname: foo\nrequires:\n  something: else\n---"
        assert parse_mcp_tools(text) == []


# ── extract_related_skills ────────────────────────────────────────

class TestExtractRelatedSkills:
    VALID_IDS = {"react", "nextjs", "typescript", "tailwind", "unknown-skill"}

    def test_backtick_in_related_section(self):
        text = "## Related Skills\nWorks well with `react`, `nextjs`.\n## Other"
        result = extract_related_skills(text, self.VALID_IDS)
        assert "react" in result
        assert "nextjs" in result

    def test_inline_works_well_with(self):
        text = "Some text. Works well with: `typescript`, `tailwind`."
        result = extract_related_skills(text, self.VALID_IDS)
        assert "typescript" in result

    def test_invalid_ids_filtered(self):
        text = "## Related Skills\n`nonexistent-skill-xyz`"
        result = extract_related_skills(text, self.VALID_IDS)
        assert "nonexistent-skill-xyz" not in result

    def test_no_duplicates(self):
        text = "## Related Skills\n`react` and `react` again"
        result = extract_related_skills(text, self.VALID_IDS)
        assert result.count("react") == 1


# ── truncate ──────────────────────────────────────────────────────

class TestTruncate:
    def test_short_string_unchanged(self):
        s = "short"
        assert truncate(s, 500) == s

    def test_long_string_truncated(self):
        s = "x" * 600
        result = truncate(s, 500)
        assert len(result) == 500

    def test_exact_length_unchanged(self):
        s = "x" * 500
        assert truncate(s, 500) == s


# ── Staging output validation ─────────────────────────────────────

class TestStagingOutput:
    def test_staging_dir_exists(self):
        assert STAGING_DIR.exists(), "staging/skills/ directory not found — run convert_skills.py first"

    def test_file_count(self, staging_yamls):
        assert len(staging_yamls) >= 718, f"Expected ≥718 files, got {len(staging_yamls)}"

    def test_all_files_valid_schema(self, staging_yamls, schema):
        failures = []
        for fname, data in staging_yamls:
            try:
                jsonschema.validate(data, schema)
            except jsonschema.ValidationError as e:
                failures.append(f"{fname}: {e.message}")
        assert not failures, f"Schema failures:\n" + "\n".join(failures[:10])

    def test_cluster_nodes_present(self, staging_yamls):
        cluster_ids = {d["id"] for _, d in staging_yamls if d.get("type") == "cluster"}
        expected = {"core-dev-cluster", "security-core-cluster", "data-core-cluster",
                    "ops-core-cluster", "k8s-core-cluster"}
        assert expected.issubset(cluster_ids), f"Missing clusters: {expected - cluster_ids}"

    def test_all_skills_have_instructions(self, staging_yamls):
        missing = [f for f, d in staging_yamls
                   if d.get("type") != "cluster" and
                   not d.get("payload", {}).get("instructions")]
        pct = len(missing) / len(staging_yamls) * 100
        assert pct <= 5, f"{pct:.1f}% of skills missing instructions (threshold: 5%)"

    def test_no_schema_failures_in_report(self):
        assert REPORT_FILE.exists(), "conversion_report.json not found"
        with open(REPORT_FILE) as f:
            report = json.load(f)
        assert report["stats"]["failed"] == 0, f"Schema failures: {report['stats']['failed']}"

    def test_hyphenated_ids_accepted(self, staging_yamls, schema):
        hyphenated = [(f, d) for f, d in staging_yamls if "-" in d.get("id", "")]
        assert len(hyphenated) > 100, "Expected many hyphenated IDs"
        for fname, data in hyphenated[:10]:
            jsonschema.validate(data, schema)  # should not raise

    def test_valid_types_only(self, staging_yamls):
        valid_types = {"role", "tool", "domain", "responsibility", "cluster"}
        invalid = [(f, d["type"]) for f, d in staging_yamls if d.get("type") not in valid_types]
        assert not invalid, f"Invalid types: {invalid[:5]}"
