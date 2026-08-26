# server/search/vector_search.py
import logging
from neo4j import AsyncDriver
from server.models.skill_node import SkillCandidate
from core.embeddings import embed_text, INDEX_NAME

logger = logging.getLogger(__name__)

_CYPHER_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
RETURN node.id AS id, node.name AS name,
       score AS semantic_score, node.hub_score AS hub_score
"""


async def search_skills(
    driver: AsyncDriver, query: str, top_n: int = 3,
) -> list[SkillCandidate]:
    embedding = await embed_text(query)
    async with driver.session() as session:
        result = await session.run(
            _CYPHER_VECTOR_SEARCH, index=INDEX_NAME, k=top_n, embedding=embedding
        )
        candidates: list[SkillCandidate] = []
        async for record in result:
            candidates.append(SkillCandidate(
                id=record["id"], name=record["name"],
                semantic_score=round(float(record["semantic_score"]), 4),
                hub_score=round(float(record["hub_score"] or 0.0), 4),
            ))
    return candidates[:top_n]
