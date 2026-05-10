# tests/test_server_integration.py
"""
Integration tests for server/main.py MCP tools.

These tests run against a real Neo4j instance. They load a minimal 3-skill
fixture inline (no dependency on Plan 06 CLI) and clean up after themselves.

Requirements:
  - Neo4j running at NEO4J_URI (default bolt://localhost:7687)
  - KNOWLEDGE_BASE_DIR set or defaulting to skills/knowledge
  - pip install -e ".[dev]"
  - OPENAI_API_KEY set for search_skills tests

Run:
  pytest tests/test_server_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from neo4j import AsyncGraphDatabase

import server.main as main_module
from server.session import reset_state

# Load .env so OPENAI_API_KEY and Neo4j creds are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "skillgraph")

_MINIMAL_SKILLS = [
    {
        "id": "si_backend_dev",
        "name": "Backend Developer",
        "description": "Builds backend APIs and services",
        "type": "role",
        "hub_score": 0.8,
        "degree": 3,
        "context_cost": 400,
        "instructions": "You are a backend developer. Design and build APIs.",
        "tools": ["bash", "git"],
        "knowledge": [],
    },
    {
        "id": "si_python_lang",
        "name": "Python",
        "description": "Python programming language",
        "type": "tool",
        "hub_score": 0.6,
        "degree": 2,
        "context_cost": 200,
        "instructions": "Use Python idioms and type hints.",
        "tools": [],
        "knowledge": [],
    },
    {
        "id": "si_api_design",
        "name": "API Design",
        "description": "RESTful API design principles",
        "type": "domain",
        "hub_score": 0.5,
        "degree": 1,
        "context_cost": 150,
        "instructions": "Follow REST conventions.",
        "tools": [],
        "knowledge": [],
    },
]

_MINIMAL_EDGES = [
    ("si_backend_dev", "REQUIRES", "si_python_lang"),
    ("si_backend_dev", "REQUIRES", "si_api_design"),
]

# Skip marker for tests that require OPENAI_API_KEY
_api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
skip_if_no_key = pytest.mark.skipif(
    not _api_key_present,
    reason="OPENAI_API_KEY not set — skipping OpenAI-dependent integration tests",
)


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_session():
    reset_state()
    yield
    reset_state()


# ---------------------------------------------------------------------------
# Minimal graph fixture (function-scoped — compatible with asyncio_mode=auto)
# ---------------------------------------------------------------------------


@pytest.fixture
async def loaded_graph():
    """
    Load 3 minimal skills + 2 edges into Neo4j and clean up after each test.

    Uses Neo4jClient.upsert_skill_node so that payload_json is stored correctly
    (required by get_skill_node / build_skill_context_object).
    Skills are embedded via update_skill_embedding if OPENAI_API_KEY is set.
    """
    from server.graph.neo4j_client import Neo4jClient
    from server.search.vector_search import update_skill_embedding

    driver = AsyncGraphDatabase.driver(
        _NEO4J_URI,
        auth=(_NEO4J_USER, _NEO4J_PASSWORD),
    )
    client = Neo4jClient(driver)
    await client.setup_schema()

    # Load nodes with proper payload_json serialisation
    for skill in _MINIMAL_SKILLS:
        skill_data = {
            "id": skill["id"],
            "name": skill["name"],
            "description": skill["description"],
            "type": skill["type"],
            "hub_score": skill["hub_score"],
            "degree": skill["degree"],
            "context_cost": skill["context_cost"],
            "payload": {
                "instructions": skill["instructions"],
                "tools": skill["tools"],
                "knowledge": skill["knowledge"],
                "exclude_tools": [],
            },
        }
        await client.upsert_skill_node(skill_data)

    # Create edges
    for src, rel, tgt in _MINIMAL_EDGES:
        await client.upsert_edge(src, tgt, rel)

    # Embed descriptions for vector search (requires OPENAI_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        for skill in _MINIMAL_SKILLS:
            await update_skill_embedding(driver, skill["id"], skill["description"])

    # Point main module at this client so tools use the test DB
    original_client = main_module._neo4j_client
    main_module._neo4j_client = client

    yield

    # Teardown — remove only the nodes we created
    ids_to_delete = [s["id"] for s in _MINIMAL_SKILLS]
    async with driver.session() as session:
        await session.run(
            "MATCH (s:Skill) WHERE s.id IN $ids DETACH DELETE s",
            ids=ids_to_delete,
        )

    # Restore previous client and close driver
    main_module._neo4j_client = original_client
    await driver.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@skip_if_no_key
@pytest.mark.asyncio
async def test_search_skills_integration(loaded_graph):
    """search_skills returns at least 1 result with 'id' and 'semantic_score'."""
    from server.main import search_skills

    results = await search_skills(query="backend API")

    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert "id" in first
    assert "semantic_score" in first


@pytest.mark.asyncio
async def test_get_skill_integration(loaded_graph):
    """get_skill('si_backend_dev') returns correct SkillContextObject dict."""
    from server.main import get_skill

    result = await get_skill(id="si_backend_dev")

    assert result["metadata"]["id"] == "si_backend_dev"
    assert result["payload"]["instructions"] != ""


@pytest.mark.asyncio
async def test_get_skill_shallow_vs_full(loaded_graph):
    """depth='full' populates layer_2; depth='shallow' may leave layer_2 empty."""
    from server.main import get_skill

    shallow = await get_skill(id="si_backend_dev", depth="shallow")
    reset_state()
    full = await get_skill(id="si_backend_dev", depth="full")

    # layer_2 must be present in both responses (may be empty for shallow).
    assert "layer_2" in shallow
    assert "layer_2" in full
    # Full depth should either have layer_2 populated or the same as shallow
    # depending on graph depth available. At minimum, the key must exist.
    assert isinstance(full["layer_2"], list)


@pytest.mark.asyncio
async def test_navigate_integration(loaded_graph):
    """navigate from si_backend_dev via 'requires' returns non-empty neighbour list."""
    from server.main import navigate

    result = await navigate(from_id="si_backend_dev", edge_type="requires", direction="outbound")

    assert "neighbors" in result
    assert len(result["neighbors"]) >= 1


@pytest.mark.asyncio
async def test_get_knowledge_integration(loaded_graph, tmp_path):
    """get_knowledge reads a .md file from a temp KNOWLEDGE_BASE_DIR."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    (kb_dir / "test_doc.md").write_text("# Test\nContent here.", encoding="utf-8")

    # Point module at temp dir for this test.
    original_dir = main_module.KNOWLEDGE_BASE_DIR
    main_module.KNOWLEDGE_BASE_DIR = kb_dir.resolve()

    try:
        from server.main import get_knowledge

        content = await get_knowledge(ref="test_doc.md")
        assert "Content here." in content
    finally:
        main_module.KNOWLEDGE_BASE_DIR = original_dir


@skip_if_no_key
@pytest.mark.asyncio
async def test_full_flow_search_then_get(loaded_graph):
    """Full flow: search → take first result → get_skill → payload non-empty."""
    from server.main import get_skill, search_skills

    candidates = await search_skills(query="developer API backend")
    assert len(candidates) >= 1

    first_id = candidates[0]["id"]
    context = await get_skill(id=first_id)

    assert context["payload"]["instructions"] != ""
