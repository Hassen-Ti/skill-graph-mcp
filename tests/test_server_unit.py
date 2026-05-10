# tests/test_server_unit.py
"""
Unit tests for server/session.py and server/main.py.

All external dependencies (Neo4j, OpenAI embeddings) are replaced by mocks.
No network or database required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import server.session as session_module
from server.session import (
    ACTIVE_TOOL_CAP,
    GET_SKILL_RATE_LIMIT,
    get_state,
    reset_state,
)


# ---------------------------------------------------------------------------
# Autouse fixture — reset session state between every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_session():
    """Reset module-level session state before each test."""
    reset_state()
    yield
    reset_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(id: str = "skill_a", name: str = "Skill A") -> MagicMock:
    """Return a mock SkillCandidate with a working .model_dump()."""
    c = MagicMock()
    c.id = id
    c.name = name
    c.semantic_score = 0.9
    c.hub_score = 0.7
    c.model_dump.return_value = {
        "id": id,
        "name": name,
        "semantic_score": 0.9,
        "hub_score": 0.7,
    }
    return c


def _make_context_object(
    skill_id: str = "skill_a",
    tools: list[str] | None = None,
) -> MagicMock:
    """Return a mock SkillContextObject with a working .model_dump()."""
    tools = tools or []
    ctx = MagicMock()
    ctx.metadata.id = skill_id
    ctx.payload.tools = tools
    ctx.model_dump.return_value = {
        "metadata": {"id": skill_id},
        "payload": {"tools": tools, "instructions": "Do things.", "knowledge": []},
        "layer_1": [],
        "layer_2": [],
    }
    return ctx


# ---------------------------------------------------------------------------
# search_skills
# ---------------------------------------------------------------------------


class TestSearchSkillsTool:
    """test_search_skills_returns_serialized_candidates"""

    @pytest.mark.asyncio
    async def test_search_skills_returns_serialized_candidates(self):
        """search_skills tool serializes SkillCandidate list to list[dict]."""
        candidates = [_make_candidate("a"), _make_candidate("b"), _make_candidate("c")]

        with patch(
            "server.main.search_skills_impl",
            new=AsyncMock(return_value=candidates),
        ):
            from server.main import search_skills

            result = await search_skills(query="backend API")

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, dict)
            assert "id" in item
            assert "semantic_score" in item


# ---------------------------------------------------------------------------
# get_skill
# ---------------------------------------------------------------------------


class TestGetSkillTool:
    """Tests for the get_skill MCP tool."""

    @pytest.mark.asyncio
    async def test_get_skill_increments_call_counter(self):
        """Each call to get_skill increments state.get_skill_calls by 1."""
        ctx = _make_context_object("my_skill")

        with (
            patch("server.main._get_neo4j_client", return_value=MagicMock()),
            patch(
                "server.main.build_skill_context_object",
                new=AsyncMock(return_value=ctx),
            ),
        ):
            from server.main import get_skill

            await get_skill(id="my_skill")

        assert get_state().get_skill_calls == 1

    @pytest.mark.asyncio
    async def test_get_skill_rate_limit_enforced(self):
        """When get_skill_calls == GET_SKILL_RATE_LIMIT, raise ValueError."""
        get_state().get_skill_calls = GET_SKILL_RATE_LIMIT

        with pytest.raises(ValueError, match="Rate limit"):
            from server.main import get_skill

            await get_skill(id="any_skill")

    @pytest.mark.asyncio
    async def test_get_skill_tracks_active_tools(self):
        """Tools listed in payload are added to state.active_tools."""
        ctx = _make_context_object("skill_b", tools=["bash", "git"])

        with (
            patch("server.main._get_neo4j_client", return_value=MagicMock()),
            patch(
                "server.main.build_skill_context_object",
                new=AsyncMock(return_value=ctx),
            ),
        ):
            from server.main import get_skill

            await get_skill(id="skill_b")

        state = get_state()
        assert "bash" in state.active_tools
        assert "git" in state.active_tools

    @pytest.mark.asyncio
    async def test_get_skill_active_tool_cap(self):
        """When active_tools already has ACTIVE_TOOL_CAP entries, no new tools added."""
        # Pre-fill active_tools to the cap with unique dummy names.
        get_state().active_tools = [f"tool_{i}" for i in range(ACTIVE_TOOL_CAP)]
        ctx = _make_context_object("skill_c", tools=["new_tool"])

        with (
            patch("server.main._get_neo4j_client", return_value=MagicMock()),
            patch(
                "server.main.build_skill_context_object",
                new=AsyncMock(return_value=ctx),
            ),
        ):
            from server.main import get_skill

            await get_skill(id="skill_c")

        state = get_state()
        assert "new_tool" not in state.active_tools
        assert len(state.active_tools) == ACTIVE_TOOL_CAP


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------


class TestNavigateTool:
    """Tests for the navigate MCP tool."""

    @pytest.mark.asyncio
    async def test_navigate_tracks_visited_nodes(self):
        """After navigate(), from_id is added to state.visited_nodes."""
        neighbors = [_make_context_object("neighbor_1")]

        with (
            patch("server.main._get_neo4j_client", return_value=MagicMock()),
            patch(
                "server.main._fetch_neighbors",
                new=AsyncMock(return_value=neighbors),
            ),
        ):
            from server.main import navigate

            await navigate(from_id="root_skill", edge_type="requires")

        assert "root_skill" in get_state().visited_nodes

    @pytest.mark.asyncio
    async def test_navigate_flags_revisit(self):
        """Second navigate() on the same from_id sets _revisit=True in result."""
        neighbors = [_make_context_object("neighbor_1")]

        with (
            patch("server.main._get_neo4j_client", return_value=MagicMock()),
            patch(
                "server.main._fetch_neighbors",
                new=AsyncMock(return_value=neighbors),
            ),
        ):
            from server.main import navigate

            # First call — not a revisit
            result_1 = await navigate(from_id="root_skill", edge_type="requires")
            assert result_1.get("_revisit") is not True

            # Second call — must be flagged
            result_2 = await navigate(from_id="root_skill", edge_type="requires")
            assert result_2.get("_revisit") is True


# ---------------------------------------------------------------------------
# get_knowledge
# ---------------------------------------------------------------------------


class TestGetKnowledgeTool:
    """Tests for the get_knowledge MCP tool."""

    @pytest.fixture
    def knowledge_dir(self, tmp_path: Path):
        """Create a temporary KNOWLEDGE_BASE_DIR and point the module at it."""
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        with patch.dict(os.environ, {"KNOWLEDGE_BASE_DIR": str(kdir)}):
            # Force re-evaluation of the module constant if needed.
            import server.main as main_module
            main_module.KNOWLEDGE_BASE_DIR = kdir
            yield kdir

    @pytest.mark.asyncio
    async def test_get_knowledge_valid_ref(self, knowledge_dir: Path):
        """A valid .md filename returns the file content."""
        (knowledge_dir / "guide.md").write_text("# Guide\nSome content.", encoding="utf-8")

        from server.main import get_knowledge

        result = await get_knowledge(ref="guide.md")
        assert "Some content." in result

    @pytest.mark.asyncio
    async def test_get_knowledge_path_traversal_blocked(self, knowledge_dir: Path):
        """ref containing path separators raises ValueError."""
        from server.main import get_knowledge

        with pytest.raises(ValueError, match="Invalid ref"):
            await get_knowledge(ref="../../etc/passwd")

    @pytest.mark.asyncio
    async def test_get_knowledge_unsupported_extension(self, knowledge_dir: Path):
        """ref with unsupported extension raises ValueError."""
        from server.main import get_knowledge

        with pytest.raises(ValueError, match="Unsupported"):
            await get_knowledge(ref="script.sh")

    @pytest.mark.asyncio
    async def test_get_knowledge_file_not_found(self, knowledge_dir: Path):
        """ref pointing to non-existent file raises FileNotFoundError."""
        from server.main import get_knowledge

        with pytest.raises(FileNotFoundError):
            await get_knowledge(ref="nonexistent.md")


# ---------------------------------------------------------------------------
# Session reset integrity
# ---------------------------------------------------------------------------


class TestSessionResetBetweenTests:
    """Verify that the autouse fixture truly isolates tests from each other."""

    @pytest.mark.asyncio
    async def test_session_reset_between_tests_a(self):
        """Dirty the state — the next test must start clean."""
        state = get_state()
        state.get_skill_calls = 7
        state.visited_nodes.add("dirty_node")
        state.active_tools.append("dirty_tool")
        # No assertion — this test's sole purpose is to pollute state.

    @pytest.mark.asyncio
    async def test_session_reset_between_tests_b(self):
        """State must be pristine regardless of what the previous test did."""
        state = get_state()
        assert state.get_skill_calls == 0
        assert len(state.visited_nodes) == 0
        assert len(state.active_tools) == 0
