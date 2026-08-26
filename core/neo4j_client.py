# core/neo4j_client.py
from __future__ import annotations
import json
import logging
from typing import Any
from neo4j import AsyncDriver

from core.embeddings import EMBEDDING_DIMS, INDEX_NAME

logger = logging.getLogger(__name__)

SKILL_LABEL = "Skill"
VECTOR_INDEX_NAME = INDEX_NAME
VECTOR_DIMENSIONS = EMBEDDING_DIMS


class Neo4jClient:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def _run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()

    async def setup_schema(self) -> None:
        await self._run(
            "CREATE CONSTRAINT skill_id IF NOT EXISTS "
            "FOR (s:Skill) REQUIRE s.id IS UNIQUE"
        )
        await self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            f"FOR (s:Skill) ON (s.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )

    async def reset_vector_index(self) -> None:
        """Drop and recreate the vector index — required when changing embedding dimensions."""
        await self._run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
        await self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            f"FOR (s:Skill) ON (s.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        logger.info("Vector index reset to %d dimensions.", VECTOR_DIMENSIONS)

    async def upsert_skill_node(self, data: dict) -> None:
        node_data = dict(data)
        if "payload" in node_data and isinstance(node_data["payload"], dict):
            node_data["payload_json"] = json.dumps(node_data.pop("payload"))
        await self._run(
            "MERGE (s:Skill {id: $id}) SET s += $props",
            {"id": data["id"], "props": node_data},
        )

    async def get_skill_node(self, skill_id: str) -> dict | None:
        records = await self._run(
            "MATCH (s:Skill {id: $id}) RETURN s{.*} AS node", {"id": skill_id}
        )
        if not records:
            return None
        node = records[0]["node"]
        if "payload_json" in node:
            node["payload"] = json.loads(node.pop("payload_json"))
        return node

    async def delete_skill_node(self, skill_id: str) -> None:
        await self._run("MATCH (s:Skill {id: $id}) DETACH DELETE s", {"id": skill_id})

    async def get_all_skill_ids(self) -> list[str]:
        records = await self._run("MATCH (s:Skill) RETURN s.id AS id")
        return [r["id"] for r in records]

    async def upsert_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        allowed = {"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"}
        if edge_type.upper() not in allowed:
            raise ValueError(f"Invalid edge_type: {edge_type!r}. Must be one of {allowed}.")
        edge_type = edge_type.upper()
        query = (
            f"MATCH (a:Skill {{id: $from_id}}), (b:Skill {{id: $to_id}}) "
            f"MERGE (a)-[:{edge_type}]->(b)"
        )
        await self._run(query, {"from_id": from_id, "to_id": to_id})

    async def get_outbound_neighbors(self, skill_id: str, edge_type: str | None = None) -> list[dict]:
        if edge_type is not None:
            allowed = {"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"}
            et = edge_type.upper()
            if et not in allowed:
                raise ValueError(f"Invalid edge_type: {edge_type!r}")
            query = (
                f"MATCH (s:Skill {{id: $id}})-[r:{et}]->(n:Skill) "
                f"RETURN n.id AS id, n.name AS name, n.description AS description, "
                f"type(r) AS edge_type, coalesce(n.hub_score, 0.0) AS hub_score, "
                f"coalesce(n.context_cost, 0) AS context_cost"
            )
        else:
            query = (
                "MATCH (s:Skill {id: $id})-[r]->(n:Skill) "
                "RETURN n.id AS id, n.name AS name, n.description AS description, "
                "type(r) AS edge_type, coalesce(n.hub_score, 0.0) AS hub_score, "
                "coalesce(n.context_cost, 0) AS context_cost"
            )
        return await self._run(query, {"id": skill_id})

    async def get_inbound_neighbors(self, skill_id: str, edge_type: str | None = None) -> list[dict]:
        if edge_type is not None:
            allowed = {"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"}
            et = edge_type.upper()
            if et not in allowed:
                raise ValueError(f"Invalid edge_type: {edge_type!r}")
            query = (
                f"MATCH (n:Skill)-[r:{et}]->(s:Skill {{id: $id}}) "
                f"RETURN n.id AS id, n.name AS name, n.description AS description, "
                f"type(r) AS edge_type, coalesce(n.hub_score, 0.0) AS hub_score, "
                f"coalesce(n.context_cost, 0) AS context_cost"
            )
        else:
            query = (
                "MATCH (n:Skill)-[r]->(s:Skill {id: $id}) "
                "RETURN n.id AS id, n.name AS name, n.description AS description, "
                "type(r) AS edge_type, coalesce(n.hub_score, 0.0) AS hub_score, "
                "coalesce(n.context_cost, 0) AS context_cost"
            )
        return await self._run(query, {"id": skill_id})

    async def get_skill_payload(self, skill_id: str) -> dict | None:
        records = await self._run(
            "MATCH (s:Skill {id: $id}) RETURN s.payload_json AS payload_json", {"id": skill_id}
        )
        if not records or records[0]["payload_json"] is None:
            return None
        return json.loads(records[0]["payload_json"])

    async def set_skill_payload(self, skill_id: str, payload: dict) -> None:
        await self._run(
            "MATCH (s:Skill {id: $id}) SET s.payload_json = $payload_json",
            {"id": skill_id, "payload_json": json.dumps(payload)},
        )

    async def set_skill_embedding(self, skill_id: str, embedding: list[float]) -> None:
        await self._run(
            "MATCH (s:Skill {id: $id}) SET s.embedding = $embedding",
            {"id": skill_id, "embedding": embedding},
        )

    async def get_all_skill_embeddings(self) -> dict[str, list[float]]:
        records = await self._run(
            "MATCH (s:Skill) WHERE s.embedding IS NOT NULL RETURN s.id AS id, s.embedding AS embedding"
        )
        return {r["id"]: r["embedding"] for r in records}

    async def vector_knn(self, embedding: list[float], k: int) -> list[dict]:
        return await self._run(
            f"CALL db.index.vector.queryNodes($index, $k, $embedding) "
            f"YIELD node, score "
            f"RETURN node.id AS id, score AS semantic_score",
            {"index": VECTOR_INDEX_NAME, "k": k, "embedding": embedding},
        )

    async def edge_exists(self, from_id: str, to_id: str) -> bool:
        records = await self._run(
            "MATCH (a:Skill {id: $from_id}), (b:Skill {id: $to_id}) "
            "RETURN EXISTS { (a)-[]-(b) } AS exists",
            {"from_id": from_id, "to_id": to_id},
        )
        return bool(records and records[0]["exists"])

    async def similarity_edge_exists_or_create(self, from_id: str, to_id: str, score: float) -> bool:
        records = await self._run(
            "MATCH (a:Skill {id: $from_id}), (b:Skill {id: $to_id}) "
            "WHERE NOT EXISTS { (a)-[]-(b) } "
            "CREATE (a)-[r:COLLABORATES_WITH {source: 'similarity', score: $score}]->(b) "
            "RETURN r",
            {"from_id": from_id, "to_id": to_id, "score": score},
        )
        return len(records) > 0

    async def delete_similarity_edges(self) -> None:
        await self._run("MATCH ()-[r:COLLABORATES_WITH {source: 'similarity'}]->() DELETE r")

    async def recompute_hub_scores(self) -> None:
        await self._run("""
        MATCH (s:Skill)
        WITH s, size([(s)-[]->(n) | n]) as out_degree
        WITH max(out_degree) as max_d, collect({s: s, d: out_degree}) as nodes
        UNWIND nodes as item
        SET item.s.degree = item.d,
            item.s.hub_score = CASE max_d WHEN 0 THEN 0.0 ELSE toFloat(item.d) / max_d END
        """)

    async def detect_cycles(self) -> list[str]:
        query = """
        MATCH path = (a:Skill)-[:REQUIRES|ENABLES|USES|PART_OF*2..]->(a)
        RETURN a.id as id LIMIT 10
        """
        records = await self._run(query)
        return [r["id"] for r in records]
