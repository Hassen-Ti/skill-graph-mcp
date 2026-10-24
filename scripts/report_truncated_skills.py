#!/usr/bin/env python3
"""
scripts/report_truncated_skills.py

Static report (no OpenAI calls) of which staged skills would have their
embedding input truncated by core.embeddings._MAX_TOKENS. Uses the exact
same text-assembly logic as registry.embedder._build_embed_text, so the
count matches what actually gets sent to the embedding API.

Usage:
    python scripts/report_truncated_skills.py [--staging-dir staging/skills]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tiktoken
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.embeddings import _MAX_TOKENS, EMBEDDING_MODEL  # noqa: E402
from registry.embedder import _build_embed_text  # noqa: E402

DEFAULT_STAGING_DIR = Path(__file__).resolve().parents[1] / "staging" / "skills"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    args = parser.parse_args()

    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    over_limit: list[tuple[str, int]] = []

    for yaml_path in sorted(args.staging_dir.glob("*.yaml")):
        skill = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(skill, dict):
            continue
        text = _build_embed_text(skill)
        token_count = len(enc.encode(text))
        if token_count > _MAX_TOKENS:
            over_limit.append((yaml_path.stem, token_count))

    if not over_limit:
        print("No skills exceed the embedding token limit.")
        return

    print(
        f"{len(over_limit)} skill(s) exceed the {_MAX_TOKENS}-token embedding limit "
        f"({EMBEDDING_MODEL}) and are silently truncated:\n"
    )
    for skill_id, count in sorted(over_limit, key=lambda item: -item[1]):
        print(f"  {skill_id:50s} {count:6d} tokens ({count - _MAX_TOKENS} over)")


if __name__ == "__main__":
    main()
