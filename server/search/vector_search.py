# server/search/vector_search.py
"""
Vector Search module for Skill Graph.

Responsibilities:
  - embed_text         : generate an OpenAI embedding vector for any text string
  - search_skills      : query the Neo4j vector index and return top-N SkillCandidate
  - update_skill_embedding : persist a skill's embedding vector back to Neo4j

OpenAI client is lazily initialised as a module-level singleton on first call.
OPENAI_API_KEY must be set in the environment (loaded from .env by the caller).
"""

import os
import tiktoken
from openai import AsyncOpenAI
from neo4j import AsyncDriver

from server.models.skill_node import SkillCandidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072
EMBEDDING_MAX_TOKENS = 8191
INDEX_NAME = "skill_description_embedding"

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _truncate_to_token_limit(text: str) -> str:
    tokens = _tokenizer.encode(text)
    if len(tokens) <= EMBEDDING_MAX_TOKENS:
        return text
    return _tokenizer.decode(tokens[:EMBEDDING_MAX_TOKENS])

_CYPHER_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
RETURN node.id AS id, node.name AS name,
       score AS semantic_score, node.hub_score AS hub_score
"""

_CYPHER_SET_EMBEDDING = """
MATCH (s:Skill {id: $skill_id})
SET s.embedding = $embedding
"""

# ---------------------------------------------------------------------------
# Singleton OpenAI client (lazy init)
# ---------------------------------------------------------------------------

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """Return the module-level AsyncOpenAI singleton, creating it on first call."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your key."
            )
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def embed_text(text: str) -> list[float]:
    """
    Generate an OpenAI embedding for `text`.

    Returns a list of exactly EMBEDDING_DIMS (3072) floats.
    Raises EnvironmentError if OPENAI_API_KEY is missing.
    Raises openai.OpenAIError on API failure.
    """
    client = _get_openai_client()
    response = await client.embeddings.create(
        input=_truncate_to_token_limit(text),
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


async def search_skills(
    driver: AsyncDriver,
    query: str,
    top_n: int = 3,
) -> list[SkillCandidate]:
    """
    Semantic search: embed `query`, query the Neo4j vector index, return top-N skills.

    Args:
        driver:  A connected neo4j.AsyncDriver instance.
        query:   Natural-language task description from the LLM.
        top_n:   Hard cap on results (default 3, never exceeded).

    Returns:
        list[SkillCandidate] sorted by semantic_score descending.
        semantic_score and hub_score are rounded to 4 decimal places.
    """
    embedding = await embed_text(query)

    async with driver.session() as session:
        result = await session.run(
            _CYPHER_VECTOR_SEARCH,
            index=INDEX_NAME,
            k=top_n,
            embedding=embedding,
        )
        candidates: list[SkillCandidate] = []
        async for record in result:
            candidates.append(
                SkillCandidate(
                    id=record["id"],
                    name=record["name"],
                    semantic_score=round(float(record["semantic_score"]), 4),
                    hub_score=round(float(record["hub_score"] or 0.0), 4),
                )
            )

    # Hard cap: never return more than top_n even if Neo4j somehow sends more
    return candidates[:top_n]


async def update_skill_embedding(
    driver: AsyncDriver,
    skill_id: str,
    content: str,
) -> None:
    """
    Generate an embedding for `content` and persist it on the Neo4j Skill node.

    Args:
        driver:   A connected neo4j.AsyncDriver instance.
        skill_id: The `id` property of the target Skill node.
        content:  Text to embed (description + instructions combined).
    """
    embedding = await embed_text(content)

    async with driver.session() as session:
        await session.run(
            _CYPHER_SET_EMBEDDING,
            skill_id=skill_id,
            embedding=embedding,
        )
