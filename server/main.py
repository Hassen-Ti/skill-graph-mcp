# server/main.py
"""
Skill Graph MCP Server — FastMCP entry point.

Exposes 4 tools over stdio transport:
  - search_skills   : semantic search returning top-N SkillCandidate objects
  - get_skill       : fetch a full SkillContextObject by ID (rate-limited)
  - navigate        : traverse graph edges from a given node
  - get_knowledge   : read a knowledge-base file by safe filename reference

Session state is module-level (one SessionState per process = one per stdio session).
Neo4j client is initialised lazily on first use.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from server.graph.neo4j_client import Neo4jClient
from server.graph.traversal import build_skill_context_object
from server.models.skill_node import NeighborMetadata, SkillCandidate, SkillContextObject
from server.search.vector_search import search_skills as search_skills_impl
from server.session import ACTIVE_TOOL_CAP, GET_SKILL_RATE_LIMIT, get_state

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE_DIR: Path = Path(
    os.getenv(
        "KNOWLEDGE_BASE_DIR",
        str(Path(__file__).resolve().parent.parent / "skills" / "knowledge"),
    )
).resolve()

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".json"})

# ---------------------------------------------------------------------------
# Neo4j lazy initialisation
# ---------------------------------------------------------------------------

_neo4j_client: Neo4jClient | None = None


def _get_neo4j_client() -> Neo4jClient:
    """Return (or create) the singleton Neo4j client."""
    global _neo4j_client
    if _neo4j_client is None:
        from neo4j import AsyncGraphDatabase
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([uri, user, password]):
            raise EnvironmentError(
                "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must all be set. "
                "Copy .env.example to .env and configure your credentials."
            )
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        _neo4j_client = Neo4jClient(driver)
    return _neo4j_client


# ---------------------------------------------------------------------------
# Internal helper: neighbour fetch (extracted for easier mocking in tests)
# ---------------------------------------------------------------------------


async def _fetch_neighbors(
    client: Neo4jClient,
    from_id: str,
    edge_type: str,
    direction: str,
) -> list[NeighborMetadata]:
    """Fetch neighbouring skill nodes using the graph layer."""
    et = edge_type.upper()
    if direction == "outbound":
        raw = await client.get_outbound_neighbors(from_id, edge_type=et)
    elif direction == "inbound":
        raw = await client.get_inbound_neighbors(from_id, edge_type=et)
    else:  # "both"
        outbound = await client.get_outbound_neighbors(from_id, edge_type=et)
        inbound = await client.get_inbound_neighbors(from_id, edge_type=et)
        raw = outbound + inbound

    return [
        NeighborMetadata(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            edge_type=r["edge_type"],
            hub_score=r.get("hub_score", 0.0),
            context_cost=r.get("context_cost", 0),
            distance=1,
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

_VALID_EDGE_TYPES: frozenset[str] = frozenset(
    {"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"}
)
_VALID_DIRECTIONS: frozenset[str] = frozenset({"outbound", "inbound", "both"})

mcp = FastMCP("skill-graph")


# ── Tool 1: search_skills ────────────────────────────────────────────────────


@mcp.tool()
async def search_skills(query: str) -> list[dict]:
    """
    Search the skill graph by semantic similarity.

    Args:
        query: Natural-language description of the capability being looked for.

    Returns:
        Up to 3 SkillCandidate objects serialised as dicts, ordered by
        descending semantic score.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if len(query) > 2000:
        raise ValueError("query must not exceed 2000 characters.")
    client = _get_neo4j_client()
    candidates: list[SkillCandidate] = await search_skills_impl(client._driver, query)
    return [c.model_dump() for c in candidates]


# ── Tool 2: get_skill ────────────────────────────────────────────────────────


@mcp.tool()
async def get_skill(id: str, depth: str = "shallow") -> dict:
    """
    Retrieve the full SkillContextObject for a skill node.

    Rate-limited to GET_SKILL_RATE_LIMIT calls per session.

    Args:
        id:    Skill node identifier.
        depth: "shallow" → layer_1 only; "full" → layer_1 + layer_2.

    Returns:
        SkillContextObject serialised as a dict.

    Raises:
        ValueError: If the session rate limit has been reached.
    """
    state = get_state()

    if state.get_skill_calls >= GET_SKILL_RATE_LIMIT:
        raise ValueError(
            f"Rate limit reached: get_skill may be called at most "
            f"{GET_SKILL_RATE_LIMIT} times per session "
            f"(current count: {state.get_skill_calls})."
        )

    client = _get_neo4j_client()
    # Map "full" → "deep" (traversal layer terminology)
    ctx_depth = "deep" if depth == "full" else "shallow"
    context: SkillContextObject = await build_skill_context_object(
        client=client,
        skill_id=id,
        depth=ctx_depth,
    )

    # Update session state — increment before await to avoid race condition.
    # (counter was already checked above; increment is safe here post-await
    #  because asyncio is single-threaded; the check-and-increment pair is
    #  non-atomic only across concurrent tasks, not within a single coroutine.)
    state.get_skill_calls += 1

    if context.payload is not None and len(state.active_tools) < ACTIVE_TOOL_CAP:
        for tool_name in context.payload.tools:
            if tool_name not in state.active_tools:
                if len(state.active_tools) < ACTIVE_TOOL_CAP:
                    state.active_tools.append(tool_name)

    return context.model_dump()


# ── Tool 3: navigate ─────────────────────────────────────────────────────────


@mcp.tool()
async def navigate(
    from_id: str,
    edge_type: str,
    direction: str = "outbound",
) -> dict:
    """
    Traverse graph edges from a skill node.

    Args:
        from_id:    Source skill node identifier.
        edge_type:  Relationship type (e.g. "requires", "enables").
        direction:  "outbound", "inbound", or "both".

    Returns:
        A dict with keys:
          - "neighbors": list[dict] of serialised neighbour metadata
          - "_revisit":  True if from_id was already visited this session
    """
    # Input validation — defence-in-depth before any DB call
    if edge_type.upper() not in _VALID_EDGE_TYPES:
        raise ValueError(
            f"Invalid edge_type: {edge_type!r}. "
            f"Must be one of {sorted(_VALID_EDGE_TYPES)}."
        )
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid direction: {direction!r}. "
            f"Must be one of {sorted(_VALID_DIRECTIONS)}."
        )

    state = get_state()
    revisit = from_id in state.visited_nodes

    client = _get_neo4j_client()
    neighbors = await _fetch_neighbors(
        client=client,
        from_id=from_id,
        edge_type=edge_type,
        direction=direction,
    )

    # Track visited node AFTER fetching so errors don't pollute state.
    state.visited_nodes.add(from_id)

    return {
        "neighbors": [n.model_dump() for n in neighbors],
        "_revisit": revisit,
    }


# ── Tool 4: get_knowledge ────────────────────────────────────────────────────


@mcp.tool()
async def get_knowledge(ref: str) -> str:
    """
    Return the content of a knowledge-base file.

    Security contract:
      - ref must be a plain filename (no directory separators).
      - Only .md, .txt, and .json extensions are allowed.
      - The resolved path must remain inside KNOWLEDGE_BASE_DIR.

    Args:
        ref: Filename of the knowledge document (e.g. "api_patterns.md").

    Returns:
        File content as a UTF-8 string.

    Raises:
        ValueError:        If ref contains path separators or an unsupported extension.
        PermissionError:   If the resolved path escapes the knowledge base directory.
        FileNotFoundError: If the file does not exist.
    """
    # --- Security gate 1: reject any path separators in the filename ----------
    safe_name = Path(ref).name
    if safe_name != ref or "/" in ref or "\\" in ref:
        raise ValueError(
            f"Invalid ref format: '{ref}'. "
            "Only plain filenames are accepted (no path separators)."
        )

    target = KNOWLEDGE_BASE_DIR / safe_name

    # --- Security gate 2: extension whitelist --------------------------------
    if target.suffix not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{target.suffix}'. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        )

    # --- Security gate 3: path confinement (resolve symlinks) ----------------
    # NOTE: Use is_relative_to() rather than startswith() on the string
    # representation to prevent bypass via directory-name prefix collision
    # (e.g. KNOWLEDGE_BASE_DIR="…/knowledge" would wrongly allow "…/knowledge_backup/…").
    resolved = target.resolve()
    knowledge_base_resolved = KNOWLEDGE_BASE_DIR.resolve()
    try:
        resolved.relative_to(knowledge_base_resolved)
    except ValueError:
        raise PermissionError(
            "Access denied: path escapes knowledge base directory."
        )

    # --- Read -----------------------------------------------------------------
    if not resolved.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: '{ref}'"
        )

    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
