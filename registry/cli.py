# registry/cli.py
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="registry.cli", description="Skill Graph Registry loader CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    load_cmd = sub.add_parser("load", help="Load skill YAML files into Neo4j.")
    load_cmd.add_argument("skills_dir", type=Path)
    load_cmd.add_argument("--dry-run", action="store_true", default=False)
    load_cmd.add_argument("--schema", type=Path, default=None)
    sub.add_parser("reindex", help="Drop and recreate the Neo4j vector index (required after changing embedding dims).")
    return parser


async def _run_load(skills_dir: Path, schema_path: Path, dry_run: bool) -> None:
    from registry import loader, embedder
    if dry_run:
        yaml_files = sorted([p for p in skills_dir.iterdir() if p.suffix in {".yaml", ".yml"}])
        for yaml_path in yaml_files:
            try:
                loader.load_skill_file(yaml_path, schema_path)
            except Exception as exc:
                logger.error("INVALID %s: %s", yaml_path.name, exc)
                sys.exit(1)
        print(f"Validated {len(yaml_files)} skills OK.")
        return
    from neo4j import AsyncGraphDatabase
    from core.neo4j_client import Neo4jClient
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "skillgraph")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)
    try:
        await loader.load_skills_directory(
            skills_dir=skills_dir, schema_path=schema_path,
            client=client, embedder_module=embedder, dry_run=False,
        )
        logger.info("Load complete.")
    finally:
        await driver.close()


async def _run_reindex() -> None:
    from neo4j import AsyncGraphDatabase
    from core.neo4j_client import Neo4jClient
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "skillgraph")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    client = Neo4jClient(driver)
    try:
        await client.reset_vector_index()
        logger.info("Reindex complete.")
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
    if args.schema:
        schema_path = args.schema.resolve()
    else:
        schema_path = skills_dir / "schema.json"
        if not schema_path.exists():
            schema_path = skills_dir.parent / "skills" / "schema.json"
        if not schema_path.exists():
            logger.error("Cannot find schema.json. Pass --schema explicitly.")
            sys.exit(1)
    asyncio.run(_run_load(skills_dir, schema_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
