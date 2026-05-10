# tests/test_system.py
"""
System integration tests for Skill Graph v1.

Prerequisites:
  - Neo4j running (docker-compose up -d)
  - Seeds loaded (python -m registry.cli load skills/)
  - .env configured (NEO4J_URI, NEO4J_PASSWORD, OPENAI_API_KEY)

These tests exercise the full stack: Neo4j -> traversal -> MCP tools -> response.
No mocking. Real database, real embeddings.

Run: pytest tests/test_system.py -v
"""

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Autouse fixture — set KNOWLEDGE_BASE_DIR and reset session between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def setup_environment():
    """Point the server module at skills/knowledge/, reset session state and Neo4j client."""
    import server.main as main_module
    from server.session import reset_state

    skills_knowledge = (Path(__file__).parent.parent / "skills" / "knowledge").resolve()
    original_dir = main_module.KNOWLEDGE_BASE_DIR
    original_client = main_module._neo4j_client

    main_module.KNOWLEDGE_BASE_DIR = skills_knowledge
    # Reset client so each test creates a fresh driver in its own event loop
    main_module._neo4j_client = None
    reset_state()

    yield

    main_module.KNOWLEDGE_BASE_DIR = original_dir
    main_module._neo4j_client = original_client
    reset_state()


# ---------------------------------------------------------------------------
# Test 1: search -> get_skill full flow
# ---------------------------------------------------------------------------


class TestSearchThenGetFullFlow:
    """
    System test: search_skills followed by get_skill.
    Validates that search returns meaningful candidates and get_skill returns
    a populated SkillContextObject.
    """

    async def test_search_then_get_full_flow(self):
        from server.main import search_skills, get_skill

        results = await search_skills("implement a REST API endpoint with input validation")

        assert isinstance(results, list)
        assert len(results) >= 1, "search_skills returned no results"

        top = results[0]
        top_id = top["id"] if isinstance(top, dict) else top.id

        skill_ctx = await get_skill(top_id)

        assert skill_ctx is not None
        payload = skill_ctx["payload"] if isinstance(skill_ctx, dict) else skill_ctx.payload
        instructions = (
            payload["instructions"] if isinstance(payload, dict) else payload.instructions
        )
        assert len(instructions) > 50, (
            f"instructions too short ({len(instructions)} chars) — "
            f"skill '{top_id}' payload not loaded"
        )


# ---------------------------------------------------------------------------
# Test 2: extends resolution — tools union
# ---------------------------------------------------------------------------


class TestExtendsResolutionSystem:
    """
    System test: get_skill on backend_dev must return a payload whose
    tools list is the union of backend_dev's own tools AND software_engineer's
    tools (resolved server-side via extends chain).

    backend_dev.tools = [bash, git, docker]
    software_engineer.tools = [bash, git]
    union = [bash, git, docker]  (docker is the discriminating element)
    """

    async def test_extends_resolution_system(self):
        from server.main import get_skill

        skill_ctx = await get_skill("backend_dev")

        assert skill_ctx is not None
        payload = skill_ctx["payload"] if isinstance(skill_ctx, dict) else skill_ctx.payload
        tools = payload["tools"] if isinstance(payload, dict) else payload.tools
        instructions = (
            payload["instructions"] if isinstance(payload, dict) else payload.instructions
        )

        tools_set = set(tools)
        assert "git" in tools_set, (
            "git (inherited from software_engineer) missing from backend_dev payload"
        )
        assert "docker" in tools_set, (
            "docker (backend_dev own tool) missing from resolved payload"
        )
        assert "backend developer" in instructions.lower() or "backend" in instructions.lower(), (
            "instructions should be backend_dev's own (substitution)"
        )


# ---------------------------------------------------------------------------
# Test 3: navigate finds security via requires edge
# ---------------------------------------------------------------------------


class TestNavigateFindsSecurity:
    """
    System test: navigate from backend_dev along 'requires' outbound edges
    must return at least one neighbor whose id == 'security'.

    Edge defined in seed: backend_dev --[requires]--> security
    """

    async def test_navigate_finds_security(self):
        from server.main import navigate

        # navigate returns {"neighbors": [NeighborMetadata.model_dump()], "_revisit": bool}
        result = await navigate("backend_dev", "requires", "outbound")

        assert "neighbors" in result
        neighbors = result["neighbors"]
        assert len(neighbors) >= 1, (
            "navigate('backend_dev', 'requires', 'outbound') returned empty neighbors"
        )

        ids = [n["id"] for n in neighbors]
        assert "security" in ids, (
            f"Expected 'security' in navigate results, got: {ids}"
        )


# ---------------------------------------------------------------------------
# Test 4: knowledge content accessible
# ---------------------------------------------------------------------------


class TestKnowledgeContentAccessible:
    """
    System test: get_knowledge on auth_best_practices.md must return the
    raw file content as a string containing 'JWT'.
    """

    async def test_knowledge_content_accessible(self):
        from server.main import get_knowledge

        content = await get_knowledge("auth_best_practices.md")

        assert isinstance(content, str)
        assert len(content) > 100, "auth_best_practices.md content too short"
        assert "JWT" in content, "auth_best_practices.md must contain 'JWT'"


# ---------------------------------------------------------------------------
# Test 5: cluster node navigable
# ---------------------------------------------------------------------------


class TestClusterNodeNavigable:
    """
    System test: web_dev_cluster is a cluster-type node. get_skill on it
    must return layer_1 with at least the nodes reachable via 'enables' edges.
    """

    async def test_cluster_node_navigable(self):
        from server.main import get_skill

        skill_ctx = await get_skill("web_dev_cluster")

        assert skill_ctx is not None
        layer_1 = (
            skill_ctx["layer_1"] if isinstance(skill_ctx, dict) else skill_ctx.layer_1
        )

        assert isinstance(layer_1, list)
        assert len(layer_1) >= 1, (
            "web_dev_cluster.layer_1 is empty — enables edges not traversed"
        )

        neighbor_ids = [
            (n["id"] if isinstance(n, dict) else n.id) for n in layer_1
        ]
        assert any(nid in ("software_engineer", "security") for nid in neighbor_ids), (
            f"Expected software_engineer or security in layer_1, got: {neighbor_ids}"
        )
