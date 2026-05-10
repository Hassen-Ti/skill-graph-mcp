# registry/loader.py
"""
Registry loader for Skill Graph.

Responsibilities:
- Validate YAML skill files against the JSON Schema.
- Resolve 'extends' chains with substitution/union semantics.
- Load individual YAML files.
- Atomically load a full directory of YAMLs into Neo4j.
- Detect orphan edges and cycles.
- Recompute hub_score after every successful load.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)

# Maximum allowed depth for the 'extends' resolution chain.
MAX_EXTEND_DEPTH: int = 3


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_yaml(skill_dict: dict, schema_path: Path) -> list[str]:
    """Validate a parsed skill dict against the JSON Schema at schema_path.

    Returns:
        An empty list if the dict is valid.
        A list of human-readable error strings if validation fails.
    """
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(skill_dict), key=lambda e: list(e.path))

    return [_format_error(e) for e in errors]


def _format_error(error: jsonschema.ValidationError) -> str:
    """Convert a jsonschema ValidationError into a short human-readable string."""
    path = ".".join(str(p) for p in error.absolute_path)
    field = path if path else error.validator_value
    # Include the failing field name in the message so tests can assert on it.
    if error.absolute_path:
        field_name = str(list(error.absolute_path)[-1])
        return f"{field_name}: {error.message}"
    # Top-level errors (e.g. missing required field)
    msg = error.message
    # jsonschema reports missing required as "'id' is a required property" — keep as-is
    return msg


# ---------------------------------------------------------------------------
# Extends resolution
# ---------------------------------------------------------------------------


def resolve_extends_chain_loader(
    skill_id: str, registry: dict[str, dict], depth: int = 0
) -> dict[str, Any]:
    """Resolve the extends inheritance chain for a skill in the in-memory registry.

    Semantics:
        instructions : total substitution — child always wins.
        tools        : union(child.tools, parent.tools) MINUS child.exclude_tools.
        knowledge    : union(child.knowledge, parent.knowledge).

    Args:
        skill_id:  The ID of the skill to resolve.
        registry:  A flat {skill_id: skill_dict} mapping of all loaded skills.
        depth:     Current recursion depth (0 at root call).

    Returns:
        {"instructions": str, "tools": set[str], "knowledge": set[str]}

    Raises:
        ValueError: If depth > MAX_EXTEND_DEPTH.
        KeyError:   If a referenced parent skill_id is absent from registry.
    """
    if depth > MAX_EXTEND_DEPTH:
        raise ValueError(
            f"extends chain exceeds max depth {MAX_EXTEND_DEPTH} at '{skill_id}'"
        )

    if skill_id not in registry:
        raise KeyError(f"extends references unknown skill '{skill_id}'")

    skill = registry[skill_id]
    payload = skill.get("payload") or {}

    own_instructions: str = payload.get("instructions", "")
    own_tools: set[str] = set(payload.get("tools") or [])
    own_exclude: set[str] = set(payload.get("exclude_tools") or [])
    own_knowledge: set[str] = set(payload.get("knowledge") or [])

    parent_id: str | None = skill.get("extends")

    if parent_id is None:
        # No parent — return own payload directly.
        return {
            "instructions": own_instructions,
            "tools": own_tools - own_exclude,
            "knowledge": own_knowledge,
        }

    # Recurse into the parent chain.
    parent_resolved = resolve_extends_chain_loader(parent_id, registry, depth + 1)

    # instructions: total substitution — child overrides parent.
    # Fallback to parent if child has no instructions.
    resolved_instructions = own_instructions if own_instructions else parent_resolved["instructions"]

    # tools: union then subtract exclude_tools.
    resolved_tools = (own_tools | parent_resolved["tools"]) - own_exclude

    # knowledge: union.
    resolved_knowledge = own_knowledge | parent_resolved["knowledge"]

    return {
        "instructions": resolved_instructions,
        "tools": resolved_tools,
        "knowledge": resolved_knowledge,
    }


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_skill_file(yaml_path: Path, schema_path: Path) -> dict:
    """Parse a YAML skill file and validate it against the JSON Schema.

    Args:
        yaml_path:   Path to the .yaml/.yml file.
        schema_path: Path to the JSON Schema file.

    Returns:
        The parsed skill dict if valid.

    Raises:
        ValueError: If the file fails schema validation.
        yaml.YAMLError: If the file is not valid YAML.
        FileNotFoundError: If yaml_path does not exist.
    """
    with yaml_path.open("r", encoding="utf-8") as fh:
        skill_dict = yaml.safe_load(fh)

    if not isinstance(skill_dict, dict):
        raise ValueError(f"{yaml_path}: YAML root must be a mapping, got {type(skill_dict)}")

    errors = validate_yaml(skill_dict, schema_path)
    if errors:
        joined = "; ".join(errors)
        raise ValueError(f"{yaml_path}: schema validation failed — {joined}")

    return skill_dict


# ---------------------------------------------------------------------------
# Directory loading (atomic)
# ---------------------------------------------------------------------------


async def load_skills_directory(
    skills_dir: Path,
    schema_path: Path,
    client,  # Neo4jClient from Plan 02
    embedder_module,  # registry.embedder module (injected to avoid circular import)
    dry_run: bool = False,
) -> None:
    """Load all YAML skill files in skills_dir into Neo4j atomically.

    Process:
        1. Parse and validate every YAML file. Abort on first schema error.
        2. Build an in-memory registry {skill_id: skill_dict}.
        3. Detect orphan edges (edge.to references an unknown skill_id).
        4. If dry_run=True, print validation summary and return without writing.
        5. Write all skill nodes to Neo4j (MERGE on id).
        6. Write all edges.
        7. Recompute hub_score for all nodes.
        8. Run cycle detection on the full graph. Raise RuntimeError if cycles found.
        9. Trigger embedding updates via embedder_module.update_embeddings.

    Args:
        skills_dir:      Directory containing .yaml/.yml skill files.
        schema_path:     Path to the JSON Schema file.
        client:          Injected Neo4jClient instance.
        embedder_module: Module exposing update_embeddings(client, registry) coroutine.
        dry_run:         If True, validate only — no writes to Neo4j.

    Raises:
        ValueError:    If any YAML fails schema validation or has orphan edges.
        RuntimeError:  If cycles are detected in the graph after load.
    """
    yaml_files = sorted(
        [p for p in skills_dir.iterdir() if p.suffix in {".yaml", ".yml"}]
    )

    if not yaml_files:
        logger.warning("load_skills_directory: no YAML files found in %s", skills_dir)
        return

    # --- Phase 1: Parse + validate all files ---
    registry: dict[str, dict] = {}
    for yaml_path in yaml_files:
        skill = load_skill_file(yaml_path, schema_path)
        skill_id = skill["id"]
        if skill_id in registry:
            raise ValueError(
                f"Duplicate skill id '{skill_id}' found in {yaml_path}"
            )
        registry[skill_id] = skill

    # --- Phase 2: Orphan edge detection ---
    _detect_orphan_edges(registry)

    if dry_run:
        print(f"Validated {len(registry)} skills OK.")
        return

    # --- Phase 3: Write nodes ---
    for skill_id, skill in registry.items():
        await _write_skill_node(client, skill)

    # --- Phase 4: Write edges ---
    for skill_id, skill in registry.items():
        for edge in skill.get("edges") or []:
            await _write_skill_edge(client, skill_id, edge["to"], edge["type"])

    # --- Phase 4b: Create EXTENDS edges for skills with 'extends' field ---
    for skill_id, skill in registry.items():
        if parent_id := skill.get("extends"):
            await _write_skill_edge(client, skill_id, parent_id, "EXTENDS")

    # --- Phase 5: Recompute hub_score ---
    await _recompute_hub_scores(client)

    # --- Phase 6: Cycle detection ---
    cycles = await _detect_cycles(client)
    if cycles:
        raise RuntimeError(
            f"Cycle detected in skill graph: {cycles[0]}"
        )

    # --- Phase 7: Update embeddings ---
    await embedder_module.update_embeddings(client, registry)

    logger.info("Loaded %d skills into Neo4j.", len(registry))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_orphan_edges(registry: dict[str, dict]) -> None:
    """Raise KeyError if any edge references a skill_id not in registry."""
    known_ids = set(registry.keys())
    for skill_id, skill in registry.items():
        for edge in skill.get("edges") or []:
            target = edge["to"]
            if target not in known_ids:
                raise KeyError(
                    f"Skill '{skill_id}' has edge to unknown skill '{target}'"
                )


async def _write_skill_node(client, skill: dict) -> None:
    """Merge a single skill node into Neo4j using Neo4jClient.upsert_skill_node."""
    import tiktoken
    payload = skill.get("payload") or {}
    enc = tiktoken.get_encoding("cl100k_base")
    instructions_text = payload.get("instructions", "")
    context_cost = len(enc.encode(instructions_text))

    skill_data = {
        "id":           skill["id"],
        "name":         skill["name"],
        "description":  skill["description"],
        "type":         skill["type"],
        "author":       skill.get("author", ""),
        "version":      skill.get("version", ""),
        "priority":     skill.get("priority", 2),
        "context_cost": context_cost,
        "payload": {
            "instructions":  instructions_text,
            "tools":         payload.get("tools") or [],
            "knowledge":     payload.get("knowledge") or [],
            "exclude_tools": payload.get("exclude_tools") or [],
        },
    }
    await client.upsert_skill_node(skill_data)


async def _write_skill_edge(client, from_id: str, to_id: str, edge_type: str) -> None:
    """Merge a typed directed edge between two skill nodes using Neo4jClient.upsert_edge."""
    et = edge_type.upper().replace("-", "_")
    await client.upsert_edge(from_id, to_id, et)


async def _recompute_hub_scores(client) -> None:
    """Recompute hub_score for all Skill nodes via Neo4jClient."""
    await client.recompute_hub_scores()


async def _detect_cycles(client) -> list[str]:
    """Return node IDs involved in cycles via Neo4jClient."""
    return await client.detect_cycles()
