# registry/embedder.py
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
_DEFAULT_METADATA_PATH = Path("index_metadata.json")


def _metadata_path() -> Path:
    return Path(os.getenv("INDEX_METADATA_PATH", str(_DEFAULT_METADATA_PATH)))


def _build_embed_text(skill: dict) -> str:
    """Combine description + payload instructions into a single embedding text."""
    description = skill.get("description", "")
    instructions = (skill.get("payload") or {}).get("instructions", "")
    if instructions:
        return f"{description}\n\n{instructions}"
    return description


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_index_metadata() -> dict[str, str]:
    path = _metadata_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_index_metadata(metadata: dict[str, str]) -> None:
    path = _metadata_path()
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


async def update_embeddings(client: Any, registry: dict[str, dict]) -> None:
    from core.embeddings import embed_text as compute_embedding
    try:
        records = await client._run("MATCH (s:Skill) WHERE s.embedding IS NOT NULL RETURN s.id AS id")
        embedded_in_graph: set[str] = {r["id"] for r in records}
    except Exception:
        embedded_in_graph = set()
    stored_metadata = load_index_metadata()
    updated_metadata = dict(stored_metadata)
    updated_count = 0
    for skill_id, skill in registry.items():
        embed_text = _build_embed_text(skill)
        current_hash = _hash_content(embed_text)
        if current_hash == stored_metadata.get(skill_id) and skill_id in embedded_in_graph:
            continue
        embedding = await compute_embedding(embed_text, label=skill_id)
        await client.set_skill_embedding(skill_id, embedding)
        updated_metadata[skill_id] = current_hash
        updated_count += 1
    save_index_metadata(updated_metadata)
    logger.info("Embeddings: %d updated, %d skipped.", updated_count, len(registry) - updated_count)
