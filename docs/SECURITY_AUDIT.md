# Security Audit — Skill Graph MCP v1.0

**Date:** 2026-05-09  
**Scope:** Full codebase — server, scripts, configuration

## Summary

| Severity | Count | Status |
|---|---|---|
| CRITICAL | 3 | Fixed in v1.0 |
| HIGH | 7 | Fixed in v1.0 |
| MEDIUM | 1 | Fixed 2026-08-27 |

All CRITICAL and HIGH findings resolved before first public release.

## CRITICAL findings — Fixed

**[CRIT-1]** `get_knowledge` used `startswith()` for path confinement → bypass via directory-name prefix collision. Fixed: `Path.relative_to()`.

**[CRIT-2]** `navigate()` tool accepted `edge_type` without validation at MCP boundary → Cypher interpolation risk. Fixed: explicit whitelist in `navigate()` before any DB call.

**[CRIT-3]** `.env` not in `.gitignore` → credentials exposure on push. Fixed: `.gitignore` added.

## HIGH findings — Fixed

**[HIGH-1]** Neo4j password `skillgraph` as default fallback in code. Fixed: fail-fast `EnvironmentError` if env vars missing.

**[HIGH-2]** No validation on `search_skills.query` length → unbounded OpenAI cost. Fixed: 2000-char limit.

**[HIGH-3]** Race condition on `get_skill` rate limit counter. Fixed: increment position moved.

**[HIGH-4]** Operator precedence bug: `exclude_tools` only applied to parent tools. Fixed: explicit parentheses.

**[HIGH-5]** `context.payload.tools` → `AttributeError` when payload is None. Fixed: None guard.

**[HIGH-6]** `os.environ["OPENAI_API_KEY"]` → opaque `KeyError`. Fixed: descriptive error message.

**[HIGH-7]** Absolute Windows path hardcoded in `convert_skills.py`. Fixed: `SKILLS_LIB_PATH` env var.

## MEDIUM findings — Fixed

**[MED-1]** No dependency lock file — `pyproject.toml` only declared loose `>=` version constraints (e.g. `neo4j>=5.0.0`, `openai>=1.0.0`), so two installs at different times could silently resolve different transitive dependency versions, making builds non-reproducible and widening supply-chain exposure to any breaking/malicious release of a transitive dependency. Fixed 2026-08-27: committed `uv.lock` (generated via `uv lock`), pinning exact resolved versions for all 58 packages. `uv sync` now installs an identical, reproducible environment every time.

## Not found

- SQL/Cypher injection (parameterised queries throughout)
- Hardcoded API keys in source
- Sensitive data in logs
- SSRF
- Auth bypass (stdio = local only)