# registry/cli.py
"""
CLI entry point for the Skill Graph Registry loader.

Usage:
    python -m registry.cli load <skills_dir> [--dry-run]
    python -m registry.cli reindex

Examples:
    python -m registry.cli load skills/
    python -m registry.cli load skills/ --dry-run
    python -m registry.cli reindex
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="registry.cli",
        description="Skill Graph — Registry loader CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    load_cmd = sub.add_parser("load", help="Load skill YAML files into Neo4j.")
    load_cmd.add_argument(
        "skills_dir",
        type=Path,
        help="Directory containing .yaml/.yml skill files.",
    )
    load_cmd.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate files only; do not write to Neo4j.",
    )
    load_cmd.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Path to skills/schema.json (auto-detected if omitted).",
    )

    sub.add_parser(
        "reindex",
        help=(
            "Drop and recreate the Neo4j vector index, then clear index_metadata.json. "
            "Required after changing embedding model or dimensions. "
            "Follow with: python -m registry.cli load <skills_dir>"
        ),
    )

    return parser


async def _run_load(skills_dir: Path, schema_path: Path, dry_run: bool) -> None:
    from registry import loader, embedder

    if dry_run:
        # Dry-run: validate without Neo4j.
        import json

        yaml_files = sorted(
            [p for p in skills_dir.iterdir() if p.suffix in {".yaml", ".yml"}]
        )
        valid_count = 0
        for yaml_path in yaml_files:
            try:
                loader.load_skill_file(yaml_path, schema_path)
                valid_count += 1
            except (ValueError, Exception) as exc:
                logger.error("INVALID %s: %s", yaml_path.name, exc)
                sys.exit(1)

        print(f"Validated {valid_count} skills OK.")
        return

    # Real load: connect to Neo4j.
    from neo4j import AsyncGraphDatabase
    from server.graph.neo4j_client import Neo4jClient

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "skillgraph")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)

    try:
        await loader.load_skills_directory(
            skills_dir=skills_dir,
            schema_path=schema_path,
            client=client,
            embedder_module=embedder,
            dry_run=False,
        )
        logger.info("Load complete.")
    finally:
        await driver.close()


async def _run_reindex() -> None:
    from neo4j import AsyncGraphDatabase
    from server.graph.neo4j_client import Neo4jClient
    from registry.embedder import _metadata_path

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "skillgraph")

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)
    try:
        await client.reset_vector_index()
        metadata_file = _metadata_path()
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info("Cleared %s.", metadata_file)
        logger.info("Reindex complete. Run 'load <skills_dir>' to re-embed all skills.")
    finally:
        await driver.close()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "reindex":
        asyncio.run(_run_reindex())
        return

    skills_dir: Path = args.skills_dir.resolve()
    if not skills_dir.is_dir():
        logger.error("skills_dir '%s' is not a directory.", skills_dir)
        sys.exit(1)

    # Auto-detect schema.json relative to the skills directory.
    if args.schema:
        schema_path = args.schema.resolve()
    else:
        schema_path = skills_dir / "schema.json"
        if not schema_path.exists():
            # Fallback: look one level up.
            schema_path = skills_dir.parent / "skills" / "schema.json"
        if not schema_path.exists():
            logger.error(
                "Cannot find schema.json. Pass --schema explicitly."
            )
            sys.exit(1)

    asyncio.run(_run_load(skills_dir, schema_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
