# registry/embedder.py
"""
Embedding drift detection and update for Skill Graph.

Strategy:
    Each skill's full content (description + instructions) is hashed with SHA-256.
    The hash is persisted in index_metadata.json alongside the skill_id.
    On each update_embeddings call, only skills whose current content hash
    differs from the stored hash (or are absent) trigger a new embedding call.
    This makes incremental loads cheap: unchanged skills are skipped entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location of the drift-detection metadata file.
# Override via the INDEX_METADATA_PATH environment variable.
_DEFAULT_METADATA_PATH = Path("index_metadata.json")


def _metadata_path() -> Path:
    """Return the active path for index_metadata.json (env-overrideable)."""
    return Path(os.getenv("INDEX_METADATA_PATH", str(_DEFAULT_METADATA_PATH)))


def _hash_content(content: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_embed_text(skill: dict) -> str:
    """Combine description + payload instructions into a single embedding text."""
    description = skill.get("description", "")
    instructions = (skill.get("payload") or {}).get("instructions", "")
    if instructions:
        return f"{description}\n\n{instructions}"
    return description


def load_index_metadata() -> dict[str, str]:
    """Load {skill_id: description_hash} from index_metadata.json.

    Returns an empty dict if the file does not exist or is malformed.
    """
    path = _metadata_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            logger.warning("index_metadata.json has unexpected format; resetting.")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read index_metadata.json: %s", exc)
        return {}


def save_index_metadata(metadata: dict[str, str]) -> None:
    """Persist {skill_id: description_hash} to index_metadata.json.

    Writes atomically: first to a .tmp file, then renames to the final path
    to prevent corruption on interrupted writes.
    """
    path = _metadata_path()
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


async def update_embeddings(client: Any, registry: dict[str, dict]) -> None:
    """Generate or update embeddings for skills whose content has changed.

    Drift detection:
        A skill triggers an embedding update when its current content's
        SHA-256 hash differs from the value stored in index_metadata.json,
        OR when the embedding is absent from Neo4j (e.g. after a graph clear).

    Args:
        client:   Neo4jClient instance (must expose ._run() and ._driver).
        registry: {skill_id: skill_dict} in-memory registry built by the loader.
    """
    from server.search.vector_search import update_skill_embedding

    records = await client._run(
        "MATCH (s:Skill) WHERE s.embedding IS NOT NULL RETURN s.id AS id"
    )
    embedded_in_graph: set[str] = {r["id"] for r in records}

    stored_metadata = load_index_metadata()
    updated_metadata = dict(stored_metadata)
    updated_count = 0

    for skill_id, skill in registry.items():
        embed_text = _build_embed_text(skill)
        current_hash = _hash_content(embed_text)
        stored_hash = stored_metadata.get(skill_id)

        hash_unchanged = current_hash == stored_hash
        embedding_present = skill_id in embedded_in_graph

        if hash_unchanged and embedding_present:
            logger.debug("Skipping '%s' — content unchanged and embedding present.", skill_id)
            continue

        logger.info("Updating embedding for '%s'.", skill_id)
        await update_skill_embedding(client._driver, skill_id, embed_text)
        updated_metadata[skill_id] = current_hash
        updated_count += 1

    save_index_metadata(updated_metadata)
    logger.info(
        "Embedding update complete: %d updated, %d skipped.",
        updated_count,
        len(registry) - updated_count,
    )
