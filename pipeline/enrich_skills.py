"""
pipeline/enrich_skills.py

Converts raw SKILL.md files from the skills library into enriched YAML files
for the skill-graph Neo4j ingestion pipeline.

Mapping:
    folder name                          -> id
    SKILL.md frontmatter.name            -> name
    SKILL.md frontmatter.description     -> description (<=500 chars)
    SKILL.md body + nested sub-skills    -> payload.instructions

Nested sub-skills:
    If a skill folder contains sub-folders with their own SKILL.md, their
    content is appended inline to the parent instructions. The sub-skills are
    NOT created as separate YAMLs.

    Example: game-development/ has 10 sub-folders -> one rich YAML.

Behaviour:
    - Existing YAML in staging/skills/ -> updates description + payload.instructions.
      All other fields (edges, type, priority, author) are preserved.
    - No existing YAML -> creates a new minimal one (type: domain, priority: 2).
    - --clean flag removes staging YAMLs that have no source in the lib
      (orphans) and sub-skill YAMLs that are now inlined in their parent.

Usage:
    python -m pipeline.enrich_skills [--dry-run] [--skill <id>] [--clean]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]          # prod/skill-graph/
SKILLS_LIB = ROOT.parents[1] / "data" / "sklills_lib" / "skills"
STAGING_DIR = ROOT / "staging" / "skills"

DESCRIPTION_MAX = 497  # schema max=500; 3 chars reserved for "..."


# ---------------------------------------------------------------------------
# SKILL.md parsing
# ---------------------------------------------------------------------------

def parse_skill_md(path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body_str) from a SKILL.md file."""
    text = path.read_text(encoding="utf-8").strip()
    fm: dict = {}
    body: str = text

    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            try:
                fm = yaml.safe_load(text[3:end].strip()) or {}
            except yaml.YAMLError:
                fm = {}
            body = text[end + 3:].strip()

    return fm, body


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def build_instructions(skill_dir: Path) -> str:
    """
    Build the full instructions text for a skill:
    parent body + all nested sub-skill bodies (appended with separator).
    """
    _, body = parse_skill_md(skill_dir / "SKILL.md")

    sub_skill_dirs = sorted(
        d for d in skill_dir.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    )

    for sub_dir in sub_skill_dirs:
        sub_fm, sub_body = parse_skill_md(sub_dir / "SKILL.md")
        label = sub_fm.get("name") or sub_dir.name
        body += f"\n\n---\n\n## {label}\n\n{sub_body}"

    return body


# ---------------------------------------------------------------------------
# YAML serialisation
# ---------------------------------------------------------------------------

class _LiteralStr(str):
    """Forces YAML literal block scalar (|) style."""


def _literal_representer(dumper: yaml.Dumper, data: _LiteralStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(_LiteralStr, _literal_representer)


def _dump_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _nested_sub_skill_ids(lib: Path) -> set[str]:
    """Return the set of skill IDs that are nested inside another skill."""
    nested: set[str] = set()
    for skill_dir in lib.iterdir():
        if not skill_dir.is_dir():
            continue
        for sub in skill_dir.iterdir():
            if sub.is_dir() and (sub / "SKILL.md").exists():
                nested.add(sub.name)
    return nested


def enrich_skill(skill_id: str, dry_run: bool = False) -> str:
    skill_dir = SKILLS_LIB / skill_id
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        return f"SKIP  {skill_id} -- no SKILL.md"

    fm, _ = parse_skill_md(skill_md)
    description_raw = fm.get("description", "")
    if not description_raw:
        return f"SKIP  {skill_id} -- no description in frontmatter"

    description = _truncate(
        str(description_raw).replace("\n", " ").strip(),
        DESCRIPTION_MAX,
    )
    instructions = _LiteralStr(build_instructions(skill_dir))

    yaml_path = STAGING_DIR / f"{skill_id}.yaml"

    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        existing["description"] = description
        existing.setdefault("payload", {})["instructions"] = instructions
        if not dry_run:
            yaml_path.write_text(_dump_yaml(existing), encoding="utf-8")
        return f"UPDATE {skill_id}"

    new_skill = {
        "id": skill_id,
        "name": fm.get("name", skill_id),
        "type": "domain",
        "author": "sklills_lib",
        "version": "1.0.0",
        "description": description,
        "priority": 2,
        "payload": {"instructions": instructions},
    }
    if not dry_run:
        yaml_path.write_text(_dump_yaml(new_skill), encoding="utf-8")
    return f"CREATE {skill_id}"


def clean_staging(dry_run: bool = False) -> None:
    """Remove staging YAMLs that have no source in the lib (orphans + inlined sub-skills)."""
    lib_root_ids = {
        d.name for d in SKILLS_LIB.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    }
    nested_ids = _nested_sub_skill_ids(SKILLS_LIB)
    to_remove = nested_ids | (
        {f.stem for f in STAGING_DIR.glob("*.yaml")} - lib_root_ids
    )

    for skill_id in sorted(to_remove):
        yaml_path = STAGING_DIR / f"{skill_id}.yaml"
        if yaml_path.exists():
            print(f"  DELETE {skill_id}.yaml")
            if not dry_run:
                yaml_path.unlink()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(dry_run: bool, single: str | None, clean: bool) -> None:
    if not SKILLS_LIB.exists():
        print(f"ERROR: lib not found at {SKILLS_LIB}", file=sys.stderr)
        sys.exit(1)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    nested_ids = _nested_sub_skill_ids(SKILLS_LIB)

    if single:
        print(enrich_skill(single, dry_run=dry_run))
        return

    skill_dirs = sorted(
        d for d in SKILLS_LIB.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in nested_ids
    )

    counts: dict[str, int] = {}
    for skill_dir in skill_dirs:
        status = enrich_skill(skill_dir.name, dry_run=dry_run)
        key = status.split()[0]
        counts[key] = counts.get(key, 0) + 1
        if key != "UPDATE":
            print(status)

    tag = "[DRY RUN] " if dry_run else ""
    print(
        f"\n{tag}Enrichment -- "
        f"{counts.get('UPDATE', 0)} updated, "
        f"{counts.get('CREATE', 0)} created, "
        f"{counts.get('SKIP', 0)} skipped."
    )

    if clean:
        print(f"\n{tag}Cleaning orphan/inlined YAMLs from staging:")
        clean_staging(dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich staging YAMLs from raw SKILL.md files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skill", metavar="ID")
    parser.add_argument("--clean", action="store_true", help="Delete orphan and inlined sub-skill YAMLs from staging.")
    args = parser.parse_args()
    run(dry_run=args.dry_run, single=args.skill, clean=args.clean)


if __name__ == "__main__":
    main()
