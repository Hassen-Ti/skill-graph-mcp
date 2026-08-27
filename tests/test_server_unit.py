# tests/test_server_unit.py
"""
Unit tests for server/session.py and server/main.py.

Adapted from the version recovered off `master` (94b6815): get_skill's rate
limiting moved from a flat call counter (GET_SKILL_RATE_LIMIT) to a
cumulative token budget (CONTEXT_BUDGET_TOKENS / total_context_cost) this
session — those tests are rewritten below, everything else is unchanged.

All external dependencies (Neo4j, OpenAI embeddings) are mocked. No network
or database required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.session import (
    ACTIVE_TOOL_CAP,
    CONTEXT_BUDGET_TOKENS,
    get_state,
    reset_state,
)


@pytest.fixture(autouse=True)
def reset_session():
    reset_state()
    yield
    reset_state()


def _make_candidate(id: str = "skill_a", name: str = "Skill A") -> MagicMock:
    c = MagicMock()
    c.id = id
    c.name = name
    c.semantic_score = 0.9
    c.hub_score = 0.7
    c.model_dump.return_value = {
        "id": id, "name": name, "semantic_score": 0.9, "hub_score": 0.7,
    }
    return c


def _make_context_object(
    skill_id: str = "skill_a",
    tools: list[str] | None = None,
    context_cost: int = 100,
) -> MagicMock:
    tools = tools or []
    ctx = MagicMock()
    ctx.metadata.id = skill_id
    ctx.metadata.context_cost = context_cost
    ctx.payload.tools = tools
    ctx.model_dump.return_value = {
        "metadata": {"id": skill_id},
        "payload": {"tools": tools, "instructions": "Do things.", "knowledge": []},
        "layer_1": [], "layer_2": [],
    }
    return ctx


# ---------------------------------------------------------------------------
# search_skills
# ---------------------------------------------------------------------------

class TestSearchSkillsTool:
    @pytest.mark.asyncio
    async def test_search_skills_returns_serialized_candidates(self):
        candidates = [_make_candidate("a"), _make_candidate("b"), _make_candidate("c")]

        with patch("server.main.search_skills_impl", new=AsyncMock(return_value=candidates)), \
             patch("server.main._get_neo4j_client", return_value=MagicMock()):
            from server.main import search_skills
            result = await search_skills(query="backend API")

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, dict)
            assert "id" in item
            assert "semantic_score" in item

    @pytest.mark.asyncio
    async def test_search_skills_rejects_empty_query(self):
        from server.main import search_skills
        with pytest.raises(ValueError):
            await search_skills(query="   ")

    @pytest.mark.asyncio
    async def test_search_skills_rejects_oversized_query(self):
        from server.main import search_skills
        with pytest.raises(ValueError):
            await search_skills(query="x" * 2001)


# ---------------------------------------------------------------------------
# get_skill — token-budget rate limiting
# ---------------------------------------------------------------------------

class TestGetSkillTool:
    @pytest.mark.asyncio
    async def test_get_skill_accumulates_context_cost(self):
        """Each call adds the fetched skill's context_cost to the session total."""
        ctx = _make_context_object("my_skill", context_cost=150)

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main.build_skill_context_object", new=AsyncMock(return_value=ctx)):
            from server.main import get_skill
            await get_skill(id="my_skill")

        assert get_state().total_context_cost == 150

    @pytest.mark.asyncio
    async def test_get_skill_rejected_when_budget_already_exhausted(self):
        get_state().total_context_cost = CONTEXT_BUDGET_TOKENS

        with pytest.raises(ValueError, match="budget"):
            from server.main import get_skill
            await get_skill(id="any_skill")

    @pytest.mark.asyncio
    async def test_get_skill_rejected_when_single_skill_exceeds_remaining_budget(self):
        """A single oversized skill is rejected outright, even on the very first call."""
        get_state().total_context_cost = 0
        ctx = _make_context_object("huge_skill", context_cost=CONTEXT_BUDGET_TOKENS + 1)

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main.build_skill_context_object", new=AsyncMock(return_value=ctx)):
            from server.main import get_skill
            with pytest.raises(ValueError, match="exceed"):
                await get_skill(id="huge_skill")

        # rejected cost must not be counted against the budget
        assert get_state().total_context_cost == 0

    @pytest.mark.asyncio
    async def test_get_skill_tracks_active_tools(self):
        ctx = _make_context_object("skill_b", tools=["bash", "git"])

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main.build_skill_context_object", new=AsyncMock(return_value=ctx)):
            from server.main import get_skill
            await get_skill(id="skill_b")

        state = get_state()
        assert "bash" in state.active_tools
        assert "git" in state.active_tools

    @pytest.mark.asyncio
    async def test_get_skill_active_tool_cap(self):
        get_state().active_tools = [f"tool_{i}" for i in range(ACTIVE_TOOL_CAP)]
        ctx = _make_context_object("skill_c", tools=["new_tool"])

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main.build_skill_context_object", new=AsyncMock(return_value=ctx)):
            from server.main import get_skill
            await get_skill(id="skill_c")

        state = get_state()
        assert "new_tool" not in state.active_tools
        assert len(state.active_tools) == ACTIVE_TOOL_CAP


# ---------------------------------------------------------------------------
# navigate
# ---------------------------------------------------------------------------

class TestNavigateTool:
    @pytest.mark.asyncio
    async def test_navigate_tracks_visited_nodes(self):
        neighbors = [_make_context_object("neighbor_1")]

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main._fetch_neighbors", new=AsyncMock(return_value=neighbors)):
            from server.main import navigate
            await navigate(from_id="root_skill", edge_type="requires")

        assert "root_skill" in get_state().visited_nodes

    @pytest.mark.asyncio
    async def test_navigate_flags_revisit(self):
        neighbors = [_make_context_object("neighbor_1")]

        with patch("server.main._get_neo4j_client", return_value=MagicMock()), \
             patch("server.main._fetch_neighbors", new=AsyncMock(return_value=neighbors)):
            from server.main import navigate
            result_1 = await navigate(from_id="root_skill", edge_type="requires")
            assert result_1.get("_revisit") is not True
            result_2 = await navigate(from_id="root_skill", edge_type="requires")
            assert result_2.get("_revisit") is True

    @pytest.mark.asyncio
    async def test_navigate_rejects_invalid_edge_type(self):
        from server.main import navigate
        with pytest.raises(ValueError):
            await navigate(from_id="root_skill", edge_type="not_a_real_type")

    @pytest.mark.asyncio
    async def test_navigate_rejects_invalid_direction(self):
        from server.main import navigate
        with pytest.raises(ValueError):
            await navigate(from_id="root_skill", edge_type="requires", direction="sideways")


# ---------------------------------------------------------------------------
# get_knowledge — path confinement (security-critical, see docs/SECURITY_AUDIT.md)
# ---------------------------------------------------------------------------

class TestGetKnowledgeTool:
    @pytest.fixture
    def knowledge_dir(self, tmp_path: Path):
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        with patch.dict(os.environ, {"KNOWLEDGE_BASE_DIR": str(kdir)}):
            import server.main as main_module
            main_module.KNOWLEDGE_BASE_DIR = kdir
            yield kdir

    @pytest.mark.asyncio
    async def test_get_knowledge_valid_ref(self, knowledge_dir: Path):
        (knowledge_dir / "guide.md").write_text("# Guide\nSome content.", encoding="utf-8")
        from server.main import get_knowledge
        content = await get_knowledge("guide.md")
        assert "Some content." in content

    @pytest.mark.asyncio
    async def test_get_knowledge_rejects_path_traversal(self, knowledge_dir: Path):
        from server.main import get_knowledge
        with pytest.raises(ValueError):
            await get_knowledge("../secrets.md")

    @pytest.mark.asyncio
    async def test_get_knowledge_rejects_absolute_path(self, knowledge_dir: Path):
        from server.main import get_knowledge
        with pytest.raises(ValueError):
            await get_knowledge("/etc/passwd")

    @pytest.mark.asyncio
    async def test_get_knowledge_rejects_disallowed_extension(self, knowledge_dir: Path):
        (knowledge_dir / "script.py").write_text("print(1)", encoding="utf-8")
        from server.main import get_knowledge
        with pytest.raises(ValueError):
            await get_knowledge("script.py")

    @pytest.mark.asyncio
    async def test_get_knowledge_missing_file(self, knowledge_dir: Path):
        from server.main import get_knowledge
        with pytest.raises(FileNotFoundError):
            await get_knowledge("nope.md")
