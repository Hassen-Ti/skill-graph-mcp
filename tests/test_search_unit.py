# tests/test_search_unit.py
"""
Unit tests for server/search/vector_search.py.
All OpenAI calls are mocked — no network required.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(dims: int = 1536) -> list[float]:
    """Return a list of `dims` floats (all 0.1) that mimics an OpenAI embedding."""
    return [0.1] * dims


def _make_openai_response(embedding: list[float]):
    """Build a minimal mock that matches openai.types.CreateEmbeddingResponse shape."""
    embedding_obj = MagicMock()
    embedding_obj.embedding = embedding
    response = MagicMock()
    response.data = [embedding_obj]
    return response


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_text_returns_1536_floats(mocker):
    """embed_text must return exactly 1536 floats regardless of input length."""
    fake_vec = _fake_embedding(1536)
    mock_create = AsyncMock(return_value=_make_openai_response(fake_vec))

    mocker.patch(
        "server.search.vector_search._get_openai_client",
        return_value=MagicMock(
            embeddings=MagicMock(create=mock_create)
        ),
    )

    from server.search.vector_search import embed_text

    result = await embed_text("hello world")

    assert isinstance(result, list)
    assert len(result) == 1536
    assert all(isinstance(v, float) for v in result)


@pytest.mark.asyncio
async def test_embed_text_calls_correct_model(mocker):
    """embed_text must call the OpenAI API with model='text-embedding-3-small'."""
    fake_vec = _fake_embedding(1536)
    mock_create = AsyncMock(return_value=_make_openai_response(fake_vec))

    mocker.patch(
        "server.search.vector_search._get_openai_client",
        return_value=MagicMock(
            embeddings=MagicMock(create=mock_create)
        ),
    )

    from server.search.vector_search import embed_text, EMBEDDING_MODEL

    await embed_text("some query text")

    mock_create.assert_awaited_once()
    call_kwargs = mock_create.call_args
    assert call_kwargs.kwargs.get("model") == EMBEDDING_MODEL or (
        len(call_kwargs.args) > 0 and EMBEDDING_MODEL in call_kwargs.args
    ), f"Expected model={EMBEDDING_MODEL!r}, got call: {call_kwargs}"


# ---------------------------------------------------------------------------
# search_skills
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_skills_returns_skill_candidates(mocker):
    """search_skills must return a list[SkillCandidate] from mocked Neo4j records."""
    from server.models.skill_node import SkillCandidate

    fake_vec = _fake_embedding(1536)

    # Mock embed_text so we avoid hitting OpenAI
    mocker.patch(
        "server.search.vector_search.embed_text",
        new=AsyncMock(return_value=fake_vec),
    )

    # Build fake Neo4j records
    def _make_record(id_, name, score, hub):
        rec = MagicMock()
        rec.__getitem__ = lambda self, key: {
            "id": id_, "name": name, "semantic_score": score, "hub_score": hub
        }[key]
        return rec

    records = [
        _make_record("backend_dev", "Backend Dev", 0.9231, 0.75),
        _make_record("api_design", "API Design", 0.8812, 0.60),
        _make_record("devops", "DevOps", 0.7543, 0.50),
    ]

    mock_result = AsyncMock()
    mock_result.__aiter__ = AsyncMock(return_value=iter(records))

    async def _fake_aiter(self):
        for r in records:
            yield r

    mock_result.__aiter__ = _fake_aiter

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    from server.search.vector_search import search_skills

    results = await search_skills(mock_driver, "REST API development", top_n=3)

    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, SkillCandidate) for r in results)
    assert results[0].id == "backend_dev"
    assert results[0].name == "Backend Dev"


@pytest.mark.asyncio
async def test_search_skills_semantic_score_rounded(mocker):
    """semantic_score on each SkillCandidate must be rounded to 4 decimal places."""
    from server.models.skill_node import SkillCandidate

    fake_vec = _fake_embedding(1536)
    mocker.patch(
        "server.search.vector_search.embed_text",
        new=AsyncMock(return_value=fake_vec),
    )

    # Score with many decimals — must be rounded to 4
    raw_score = 0.923123456789

    def _make_record():
        rec = MagicMock()
        rec.__getitem__ = lambda self, key: {
            "id": "test_skill", "name": "Test", "semantic_score": raw_score, "hub_score": 0.5
        }[key]
        return rec

    records = [_make_record()]

    async def _fake_aiter(self):
        for r in records:
            yield r

    mock_result = AsyncMock()
    mock_result.__aiter__ = _fake_aiter

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    from server.search.vector_search import search_skills

    results = await search_skills(mock_driver, "query", top_n=1)

    assert len(results) == 1
    score = results[0].semantic_score
    # Verify rounded to at most 4 decimal places
    assert score == round(raw_score, 4), f"Expected {round(raw_score, 4)}, got {score}"


@pytest.mark.asyncio
async def test_search_skills_top_3_hard_cap(mocker):
    """Even if Neo4j returns more than top_n records, search_skills must return at most top_n."""
    fake_vec = _fake_embedding(1536)
    mocker.patch(
        "server.search.vector_search.embed_text",
        new=AsyncMock(return_value=fake_vec),
    )

    # Simulate Neo4j returning 10 records
    def _make_record(i):
        rec = MagicMock()
        rec.__getitem__ = lambda self, key: {
            "id": f"skill_{i}", "name": f"Skill {i}",
            "semantic_score": round(0.9 - i * 0.01, 4), "hub_score": 0.5
        }[key]
        return rec

    records = [_make_record(i) for i in range(10)]

    async def _fake_aiter(self):
        for r in records:
            yield r

    mock_result = AsyncMock()
    mock_result.__aiter__ = _fake_aiter

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    from server.search.vector_search import search_skills

    results = await search_skills(mock_driver, "some query", top_n=3)

    # Must never exceed top_n even if Neo4j sends more
    assert len(results) <= 3, f"Expected at most 3 results, got {len(results)}"


# ---------------------------------------------------------------------------
# update_skill_embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_skill_embedding_calls_neo4j(mocker):
    """update_skill_embedding must SET s.embedding on the correct skill node."""
    fake_vec = _fake_embedding(1536)
    mocker.patch(
        "server.search.vector_search.embed_text",
        new=AsyncMock(return_value=fake_vec),
    )

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock()

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    from server.search.vector_search import update_skill_embedding

    await update_skill_embedding(mock_driver, skill_id="backend_dev", description="Python backend API developer")

    # session.run must have been called at least once
    mock_session.run.assert_awaited_once()

    # The Cypher call must include skill_id="backend_dev" as a parameter
    call_args = mock_session.run.call_args
    # cypher is the first positional arg
    cypher: str = call_args.args[0] if call_args.args else ""
    params: dict = call_args.kwargs if call_args.kwargs else {}

    # Check that the SET embedding statement is present
    assert "SET" in cypher and "embedding" in cypher, (
        f"Expected Cypher to contain 'SET ... embedding', got: {cypher!r}"
    )

    # Check that skill_id is passed as a parameter (not interpolated into the string)
    assert "backend_dev" in str(params.values()), (
        f"Expected skill_id='backend_dev' in query params, got: {params}"
    )
