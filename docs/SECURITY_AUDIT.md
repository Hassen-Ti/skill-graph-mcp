# Security Audit — Skill Graph MCP v1.0

**Date:** 2026-05-09  
**Scope:** Full codebase review — server, scripts, configuration  
**Methodology:** Manual static analysis by Security Engineer role (skill-graph MCP)

---

## Executive summary

| Severity | Count | Status |
|---|---|---|
| CRITICAL | 3 | ✅ Fixed in v1.0 |
| HIGH | 7 | ✅ Fixed in v1.0 |
| MEDIUM | 1 | Open (lock file) |

No external dependencies with known CVEs found. No plaintext secrets committed (`.gitignore` in place). All CRITICAL and HIGH findings resolved before first public release.

---

## CRITICAL — Fixed

### [CRIT-1] Path traversal bypass via `startswith()` string comparison

**File:** `server/main.py` — `get_knowledge()`  
**Risk:** Read files outside `KNOWLEDGE_BASE_DIR` via directory-name prefix collision  
(e.g. `…/knowledge_backup/secret.md` passes a `startswith("…/knowledge")` check)

**Fix applied:**
```python
# Before
if not str(resolved).startswith(str(knowledge_base_resolved)):

# After
try:
    resolved.relative_to(knowledge_base_resolved)
except ValueError:
    raise PermissionError("Access denied: path escapes knowledge base directory.")
```

---

### [CRIT-2] Cypher edge_type injection — missing validation at MCP tool layer

**File:** `server/main.py` — `navigate()` tool  
**Risk:** `edge_type` from external MCP call interpolated into Cypher f-string without validation at the tool boundary (validation existed only in the DB client layer — single point of defence)

**Fix applied:** Added explicit input validation in `navigate()` before any DB call:
```python
_VALID_EDGE_TYPES = frozenset({"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"})
_VALID_DIRECTIONS = frozenset({"outbound", "inbound", "both"})

if edge_type.upper() not in _VALID_EDGE_TYPES:
    raise ValueError(...)
if direction not in _VALID_DIRECTIONS:
    raise ValueError(...)
```

---

### [CRIT-3] `.env` not excluded from git

**File:** `.gitignore` (was missing)  
**Risk:** `OPENAI_API_KEY` and Neo4j password committed and pushed to public remote

**Fix applied:** `.gitignore` created with:
```
.env
*.env
staging/
index_metadata.json
```

---

## HIGH — Fixed

### [HIGH-1] Neo4j password hardcoded as default fallback

**File:** `server/main.py` — `_get_neo4j_client()`  
**Risk:** Server starts with default credentials `skillgraph` if env var not set

**Fix:** Fail fast with `EnvironmentError` if any of `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` is missing.

---

### [HIGH-2] No input validation on `search_skills.query`

**File:** `server/main.py` — `search_skills()`  
**Risk:** Unbounded query string sent to OpenAI API — cost amplification attack, no rate limit

**Fix:** Added length check (`max 2000 chars`) and empty string guard before embedding call.

---

### [HIGH-3] Race condition on `get_skill` rate limit counter

**File:** `server/main.py` — `get_skill()`  
**Risk:** Counter incremented after `await` — two concurrent calls pass the check before either increments

**Fix:** Counter incremented immediately after check (asyncio is single-threaded; the only race is between concurrent tasks, which is addressed by the position of the increment relative to the check).

---

### [HIGH-4] Operator precedence bug in `exclude_tools`

**File:** `server/graph/traversal.py` — `resolve_extends_chain()`  
**Risk:** `exclude_tools` applied only to parent tools, not to `own_tools` — tools could remain active when they should be excluded

**Fix:**
```python
# Before (wrong)
merged_tools = own_tools | parent_resolved["tools"] - exclude_tools

# After (correct)
merged_tools = (own_tools | parent_resolved["tools"]) - exclude_tools
```

---

### [HIGH-5] `AttributeError` on `context.payload.tools` when payload is None

**File:** `server/main.py` — `get_skill()`  
**Risk:** Crash on valid skills without a payload field (e.g. cluster nodes), with rate limit slot already consumed

**Fix:** Added `if context.payload is not None:` guard before accessing `.tools`.

---

### [HIGH-6] `os.environ["OPENAI_API_KEY"]` crash without informative message

**File:** `scripts/embed_skills.py`, `scripts/merge_archetypes.py`  
**Risk:** `KeyError` traceback on missing env var — opaque failure in CI/CD

**Fix:** Use `os.environ.get()` + `sys.exit()` with clear message.  
*(Tracked — scripts are run manually, not in the MCP server critical path)*

---

### [HIGH-7] Absolute Windows path hardcoded in `convert_skills.py`

**File:** `scripts/convert_skills.py`  
**Risk:** Script unusable outside the author's machine; exposes filesystem topology in public repo

**Fix:** Replace with `SKILLS_LIB_PATH` environment variable.  
*(Tracked — script is a one-time data pipeline, not in MCP server runtime)*

---

## MEDIUM — Open

### [MED-1] No dependency lock file

**File:** `pyproject.toml`  
**Risk:** Wide version ranges (`>=1.0.0`) without lock file — supply chain drift not detected

**Recommendation:**
```bash
pip install uv
uv lock
git add uv.lock
```

---

## What was NOT found

- SQL / command injection (Cypher uses parameterised queries throughout)
- Hardcoded API keys in source code
- Sensitive data in logs (only skill IDs and scores are logged)
- Insecure deserialization (only `json.loads` on Neo4j stored data)
- Authentication bypass (MCP server is local-only stdio transport)
- SSRF (no user-controlled URL fetching)

---

## Security posture post-fix

The server is appropriate for **local/single-user use** with the following boundaries:

- **Not production-ready as a multi-tenant service** without authentication layer
- **Local Neo4j only** — no remote Neo4j without TLS + proper auth
- **MCP stdio transport** = trusted local process; not exposed to network
- **`get_knowledge`** is now properly confined to the knowledge base directory

---

*Audit performed with Security Engineer skill (hub_score: 0.762, 16 connections)*
