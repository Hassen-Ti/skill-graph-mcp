# tests/test_search_integration.py
"""
Integration tests for server/search/vector_search.py.

Requirements:
  - Neo4j 5.x running (docker-compose up -d)
  - OPENAI_API_KEY set in .env (python-dotenv loaded via conftest.py or pytest.ini)

Run with:
    pytest tests/test_search_integration.py -v -m integration

Skip if no API key:
    pytest tests/test_search_integration.py -v -m "not integration"
"""

import asyncio
import os
import pytest
from neo4j import AsyncGraphDatabase

# Load .env so OPENAI_API_KEY and Neo4j creds are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional — keys may already be in environment

# ---------------------------------------------------------------------------
# Skip marker: skip all tests in this file if OPENAI_API_KEY is absent
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

_api_key_present = bool(os.environ.get("OPENAI_API_KEY"))

skip_if_no_key = pytest.mark.skipif(
    not _api_key_present,
    reason="OPENAI_API_KEY not set — skipping integration tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def driver():
    """Provide a connected Neo4j AsyncDriver and close it after the test."""
    drv = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "skillgraph"),
        ),
    )
    yield drv
    await drv.close()


@pytest.fixture
async def ensure_vector_index(driver):
    """
    Create the vector index if it does not exist, then wait until ONLINE.
    Safe to call multiple times (IF NOT EXISTS).
    """
    async with driver.session() as session:
        result = await session.run(
            """
            CREATE VECTOR INDEX skill_description_embedding IF NOT EXISTS
            FOR (s:Skill) ON s.embedding
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """
        )
        await result.consume()  # DDL must be consumed to take effect

    # Poll until the index is ONLINE (Neo4j builds indexes asynchronously)
    for _ in range(30):
        async with driver.session() as session:
            check = await session.run(
                'SHOW INDEXES WHERE name = "skill_description_embedding"'
            )
            records = await check.data()
            if records and records[0].get("state") == "ONLINE":
                break
        await asyncio.sleep(1)
    yield


# ---------------------------------------------------------------------------
# Test 1 : update then search
# ---------------------------------------------------------------------------

@skip_if_no_key
@pytest.mark.asyncio
async def test_update_and_search_skill(driver, ensure_vector_index):
    """
    Full round-trip:
      1. Create a Skill node with id='search_test_skill'
      2. Call update_skill_embedding — persists the OpenAI vector on the node
      3. Call search_skills with a semantically similar query
      4. Assert 'search_test_skill' appears in the top results
      5. Cleanup: DETACH DELETE the node
    """
    from server.search.vector_search import update_skill_embedding, search_skills

    skill_id = "search_test_skill"
    description = "backend API developer"

    # Step 1 — Create node
    async with driver.session() as session:
        await session.run(
            "MERGE (s:Skill {id: $id}) SET s.name = $name, s.hub_score = 0.5",
            id=skill_id,
            name="Search Test Skill",
        )

    try:
        # Step 2 — Embed and persist
        await update_skill_embedding(driver, skill_id=skill_id, description=description)

        # Step 3 — Search with semantically similar query
        results = await search_skills(driver, "REST API development", top_n=3)

        # Step 4 — Assert the skill appears in results
        result_ids = [r.id for r in results]
        assert skill_id in result_ids, (
            f"Expected '{skill_id}' in search results, got: {result_ids}"
        )

        # Bonus assertions
        assert len(results) <= 3
        matching = next(r for r in results if r.id == skill_id)
        assert 0.0 < matching.semantic_score <= 1.0, (
            f"semantic_score out of range: {matching.semantic_score}"
        )

    finally:
        # Step 5 — Cleanup
        async with driver.session() as session:
            await session.run(
                "MATCH (s:Skill {id: $id}) DETACH DELETE s",
                id=skill_id,
            )


# ---------------------------------------------------------------------------
# Test 2 : semantic ranking — most relevant first
# ---------------------------------------------------------------------------

@skip_if_no_key
@pytest.mark.asyncio
async def test_search_returns_most_relevant_first(driver, ensure_vector_index):
    """
    Verify ranking order:
      1. Create 'python_skill' (desc: "Python programming") and
         'security_skill' (desc: "network security firewalls")
      2. Generate embeddings for both
      3. search_skills(driver, "Python code", top_n=2)
      4. Assert python_skill has a higher semantic_score than security_skill
      5. Cleanup
    """
    from server.search.vector_search import update_skill_embedding, search_skills

    python_id = "inttest_python_skill"
    security_id = "inttest_security_skill"

    # Step 1 — Create nodes
    async with driver.session() as session:
        await session.run(
            "MERGE (s:Skill {id: $id}) SET s.name = $name, s.hub_score = 0.5",
            id=python_id,
            name="Python Skill",
        )
        await session.run(
            "MERGE (s:Skill {id: $id}) SET s.name = $name, s.hub_score = 0.5",
            id=security_id,
            name="Security Skill",
        )

    try:
        # Step 2 — Generate embeddings
        await update_skill_embedding(driver, skill_id=python_id, description="Python programming")
        await update_skill_embedding(driver, skill_id=security_id, description="network security firewalls")

        # Step 3 — Search with a generous top_n so both test skills can appear
        # alongside any production skills that may also match.
        results = await search_skills(driver, "Python code", top_n=10)

        # Collect scores for the two test skills specifically.
        scores = {r.id: r.semantic_score for r in results}

        # Step 4 — Assert python_skill is more relevant than security_skill.
        # Both must appear somewhere in the top-10.
        assert python_id in scores, (
            f"Expected '{python_id}' in top-10 results. Got: {list(scores.keys())}"
        )
        assert security_id in scores, (
            f"Expected '{security_id}' in top-10 results. Got: {list(scores.keys())}"
        )
        assert scores[python_id] > scores[security_id], (
            f"Expected python_skill ({scores[python_id]}) > "
            f"security_skill ({scores[security_id]})"
        )

        # Bonus: results must be sorted descending by semantic_score
        score_list = [r.semantic_score for r in results]
        assert score_list == sorted(score_list, reverse=True), (
            f"Results not sorted by semantic_score descending: {score_list}"
        )

    finally:
        # Step 5 — Cleanup
        async with driver.session() as session:
            await session.run(
                "MATCH (s:Skill) WHERE s.id IN $ids DETACH DELETE s",
                ids=[python_id, security_id],
            )
