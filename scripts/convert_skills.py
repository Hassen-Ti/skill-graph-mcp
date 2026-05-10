#!/usr/bin/env python3
"""
Sprint 1 — SKILL.md → YAML converter.

Reads 713 skills from sklills_lib, converts each to the topic_2 YAML format,
outputs into staging/skills/. Also creates 5 bundle cluster nodes.

Usage:
    python scripts/convert_skills.py [--dry-run]
"""

import json
import re
import sys
import yaml
import jsonschema
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent.parent           # skill-graph/
SKILLS_LIB = Path(r"C:\Users\tilio\Desktop\Projets Dev AI\sklills_lib")
SKILLS_INDEX = SKILLS_LIB / "skills_index.json"
BUNDLES_FILE = SKILLS_LIB / "data" / "bundles.json"
SCHEMA_FILE = HERE / "skills" / "schema.json"
STAGING_DIR = HERE / "staging" / "skills"

TOOL_PATTERNS = [
    "-pro", "-automation", "bash", "git", "docker",
    "kubectl", "terraform", "ansible", "helm", "aws-cli",
    "gcloud", "azure-cli", "npm", "pip", "poetry",
]

BUNDLE_CLUSTER_DEFS = {
    "core-dev": {
        "id": "core-dev-cluster",
        "name": "Core Development",
        "description": "Entry cluster for core development: languages, frameworks, backend, APIs.",
        "archetype_link": "software_engineer",
    },
    "security-core": {
        "id": "security-core-cluster",
        "name": "Security",
        "description": "Entry cluster for security engineering, compliance, and vulnerability research.",
        "archetype_link": "security",
    },
    "data-core": {
        "id": "data-core-cluster",
        "name": "Data & ML",
        "description": "Entry cluster for data engineering, analytics, and machine learning.",
        "archetype_link": "data_engineer",
    },
    "ops-core": {
        "id": "ops-core-cluster",
        "name": "Operations & DevOps",
        "description": "Entry cluster for DevOps, SRE, observability, and delivery pipelines.",
        "archetype_link": "devops_engineer",
    },
    "k8s-core": {
        "id": "k8s-core-cluster",
        "name": "Kubernetes & Cloud Native",
        "description": "Entry cluster for Kubernetes, service mesh, and cloud-native infrastructure.",
        "archetype_link": "devops_engineer",
    },
}


def load_schema() -> dict:
    with open(SCHEMA_FILE) as f:
        return json.load(f)


def load_index() -> list[dict]:
    with open(SKILLS_INDEX, encoding="utf-8") as f:
        return json.load(f)


def load_bundles() -> dict:
    with open(BUNDLES_FILE, encoding="utf-8") as f:
        return json.load(f)["bundles"]


def read_skill_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter block, return body only."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def parse_frontmatter_author(text: str) -> str | None:
    """Extract author field from YAML frontmatter if present."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    fm_block = text[3:end]
    m = re.search(r"^author:\s*(.+)$", fm_block, re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_mcp_tools(text: str) -> list[str]:
    """Extract requires.mcp list from frontmatter."""
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    if end == -1:
        return []
    fm_block = text[3:end]

    # Look for 'requires:' block then '  mcp:' list
    in_requires = False
    in_mcp = False
    tools = []
    for line in fm_block.splitlines():
        if re.match(r"^requires\s*:", line):
            in_requires = True
            continue
        if in_requires and re.match(r"\s+mcp\s*:", line):
            in_mcp = True
            continue
        if in_mcp:
            m = re.match(r"\s+-\s+(.+)", line)
            if m:
                tools.append(m.group(1).strip())
            elif not line.startswith(" "):
                break
    return tools


def extract_related_skills(text: str, valid_ids: set[str]) -> list[str]:
    """Extract skill IDs mentioned in Works well with / Related Skills section."""
    related = []
    lines = text.splitlines()
    in_related = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^##\s+(related skills?|works well with)", stripped, re.IGNORECASE):
            in_related = True
            continue
        if in_related and re.match(r"^##\s+", stripped):
            break
        if in_related:
            found = re.findall(r"`([a-z][a-z0-9\-]+)`", stripped)
            related.extend(found)

    # Also scan anywhere for "works well with: `x`, `y`"
    for m in re.finditer(r"works well with[:\s]+(.+)", text, re.IGNORECASE):
        found = re.findall(r"`([a-z][a-z0-9\-]+)`", m.group(1))
        related.extend(found)

    return [r for r in dict.fromkeys(related) if r in valid_ids]


def classify_type(skill_id: str, description: str) -> str:
    if any(p in skill_id for p in TOOL_PATTERNS):
        return "tool"
    desc_lower = description.lower()
    if desc_lower.startswith("you are a") or desc_lower.startswith("you're a"):
        return "role"
    return "domain"


def truncate(s: str, max_len: int = 500) -> str:
    return s[:max_len].rstrip() if len(s) > max_len else s


def convert_skill(entry: dict, valid_ids: set[str], schema: dict) -> tuple[dict, list[str]]:
    """Convert one skills_index entry to a YAML-ready dict. Returns (yaml_dict, warnings)."""
    warnings = []
    sid = entry["id"]
    skill_path = SKILLS_LIB / entry["path"] / "SKILL.md"
    raw = read_skill_md(skill_path) if skill_path.exists() else ""

    description = truncate(entry.get("description", "") or sid)
    if not description:
        description = sid
        warnings.append(f"{sid}: empty description, using ID as fallback")

    author = parse_frontmatter_author(raw) or "sklills_lib"
    mcp_tools = parse_mcp_tools(raw)
    body = strip_frontmatter(raw)
    related = extract_related_skills(raw, valid_ids)
    skill_type = classify_type(sid, description)

    priority_map = {"cluster": 1, "role": 2, "domain": 2, "tool": 3, "responsibility": 2}
    priority = priority_map.get(skill_type, 2)

    yaml_dict: dict = {
        "id": sid,
        "name": entry.get("name", sid),
        "type": skill_type,
        "author": author,
        "version": "1.0.0",
        "description": description,
        "priority": priority,
    }

    edges = [{"to": ref, "type": "collaborates_with"} for ref in related]
    if edges:
        yaml_dict["edges"] = edges

    payload: dict = {}
    if body.strip():
        payload["instructions"] = body.strip()
    if mcp_tools:
        payload["tools"] = mcp_tools
    if payload:
        yaml_dict["payload"] = payload

    if not body.strip():
        warnings.append(f"{sid}: no instructions body extracted")

    try:
        jsonschema.validate(yaml_dict, schema)
    except jsonschema.ValidationError as e:
        warnings.append(f"{sid}: schema validation FAILED — {e.message}")

    return yaml_dict, warnings


def make_cluster_yaml(bundle_name: str, bundle_skills: list[str], valid_ids: set[str], schema: dict) -> dict:
    defn = BUNDLE_CLUSTER_DEFS[bundle_name]
    edges = []
    if defn["archetype_link"]:
        edges.append({"to": defn["archetype_link"], "type": "collaborates_with"})
    # Connect to first 10 skills in bundle as entry points
    for s in bundle_skills[:10]:
        if s in valid_ids:
            edges.append({"to": s, "type": "enables"})

    cluster = {
        "id": defn["id"],
        "name": defn["name"],
        "type": "cluster",
        "author": "sklills_lib",
        "version": "1.0.0",
        "description": defn["description"],
        "priority": 1,
        "edges": edges,
    }
    jsonschema.validate(cluster, schema)
    return cluster


def write_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def main(dry_run: bool = False) -> None:
    print(f"{'[DRY-RUN] ' if dry_run else ''}Loading index + schema...")
    schema = load_schema()
    index = load_index()
    bundles = load_bundles()
    valid_ids = {e["id"] for e in index}

    # Build skill→bundle map
    skill_bundle: dict[str, str] = {}
    for bname, bdata in bundles.items():
        for s in bdata["skills"]:
            if s not in skill_bundle:
                skill_bundle[s] = bname

    stats = {"converted": 0, "failed": 0, "no_instructions": 0, "edges": 0}
    all_warnings: list[str] = []

    print(f"Converting {len(index)} skills...")
    for i, entry in enumerate(index):
        yaml_dict, warnings = convert_skill(entry, valid_ids, schema)
        all_warnings.extend(warnings)

        schema_failed = any("schema validation FAILED" in w for w in warnings)
        if schema_failed:
            stats["failed"] += 1
            print(f"  FAIL {entry['id']}: {[w for w in warnings if 'FAILED' in w]}")
            continue

        if any("no instructions" in w for w in warnings):
            stats["no_instructions"] += 1

        stats["converted"] += 1
        stats["edges"] += len(yaml_dict.get("edges", []))

        if not dry_run:
            out_path = STAGING_DIR / f"{entry['id']}.yaml"
            write_yaml(yaml_dict, out_path)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(index)}")

    # Create 5 cluster nodes
    print("Creating bundle cluster nodes...")
    for bname, bdata in bundles.items():
        if bname not in BUNDLE_CLUSTER_DEFS:
            continue
        cluster = make_cluster_yaml(bname, bdata["skills"], valid_ids, schema)
        stats["converted"] += 1
        if not dry_run:
            out_path = STAGING_DIR / f"{cluster['id']}.yaml"
            write_yaml(cluster, out_path)
        print(f"  cluster: {cluster['id']} ({len(cluster['edges'])} edges)")

    # Report
    report = {
        "converted": stats["converted"],
        "failed": stats["failed"],
        "no_instructions": stats["no_instructions"],
        "explicit_edges": stats["edges"],
        "warnings_count": len(all_warnings),
    }

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Results:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if not dry_run:
        report_path = HERE / "staging" / "conversion_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"stats": report, "warnings": all_warnings[:100]}, f, indent=2)
        print(f"\nStaging dir: {STAGING_DIR}")
        print(f"Files written: {stats['converted']}")

    no_instr_pct = stats["no_instructions"] / max(stats["converted"], 1) * 100
    print(f"\nSkills without instructions: {stats['no_instructions']} ({no_instr_pct:.1f}%)")
    print(f"Schema failures: {stats['failed']}")

    if stats["failed"] > 0:
        print("\nWARNING: schema failures detected — check conversion_report.json")
        sys.exit(1)
    else:
        print("\nSprint 1 criterion met: 0 schema failures.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
