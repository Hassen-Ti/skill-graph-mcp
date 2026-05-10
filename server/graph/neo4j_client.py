"""
Async Neo4j client for the Skill Graph project.

Wraps the neo4j AsyncDriver and exposes typed helpers for:
- Schema setup (uniqueness constraint + vector index)
- Node and edge upsert / read / delete
- Neighbor queries (outbound and inbound)
- Payload read / write
- Hub score recomputation (degree centrality)
- Cycle detection (asymmetric edges only)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# Neo4j label and relationship constants
SKILL_LABEL = "Skill"
VECTOR_INDEX_NAME = "skill_description_embedding"
VECTOR_DIMENSIONS = 3072  # text-embedding-3-large


class Neo4jClient:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a Cypher query and return all records as plain dicts."""
        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
        return records

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    async def setup_schema(self) -> None:
        """Create uniqueness constraint and vector index if they do not exist."""
        # Uniqueness constraint on Skill.id
        await self._run(
            "CREATE CONSTRAINT skill_id IF NOT EXISTS "
            "FOR (s:Skill) REQUIRE s.id IS UNIQUE"
        )

        # Vector index on Skill.embedding (dimensions = 3072, cosine similarity)
        await self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            f"FOR (s:Skill) ON (s.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        logger.info("Neo4j schema setup complete.")

    async def reset_vector_index(self) -> None:
        """Drop and recreate the vector index — required when changing embedding dimensions."""
        await self._run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
        await self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} "
            f"FOR (s:Skill) ON (s.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, "
            f"`vector.similarity_function`: 'cosine'}}}}"
        )
        logger.info("Vector index reset to %d dimensions.", VECTOR_DIMENSIONS)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def upsert_skill_node(self, data: dict) -> None:
        """Create or update a Skill node. `data` must contain at least 'id'."""
        node_data = dict(data)
        if "payload" in node_data and isinstance(node_data["payload"], dict):
            node_data["payload_json"] = json.dumps(node_data.pop("payload"))

        await self._run(
            "MERGE (s:Skill {id: $id}) "
            "SET s += $props",
            {"id": data["id"], "props": node_data},
        )

    async def get_skill_node(self, skill_id: str) -> dict | None:
        """Return a Skill node as a plain dict, or None if not found."""
        records = await self._run(
            "MATCH (s:Skill {id: $id}) RETURN s{.*} AS node",
            {"id": skill_id},
        )
        if not records:
            return None
        node = records[0]["node"]
        if "payload_json" in node:
            node["payload"] = json.loads(node.pop("payload_json"))
        return node

    async def delete_skill_node(self, skill_id: str) -> None:
        """Delete a Skill node and all its relationships."""
        await self._run(
            "MATCH (s:Skill {id: $id}) DETACH DELETE s",
            {"id": skill_id},
        )

    async def get_all_skill_ids(self) -> list[str]:
        """Return a list of all Skill node ids."""
        records = await self._run("MATCH (s:Skill) RETURN s.id AS id")
        return [r["id"] for r in records]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def upsert_edge(self, from_id: str, to_id: str, edge_type: str) -> None:
        """Create or update a typed directed relationship between two Skill nodes.

        edge_type must be one of: REQUIRES, ENABLES, USES, PART_OF, EXTENDS, COLLABORATES_WITH.
        """
        allowed = {"REQUIRES", "ENABLES", "USES", "PART_OF", "EXTENDS", "COLLABORATES_WITH"}
        if edge_type.upper() not in allowed:
            raise ValueError(f"Invalid edge_type: {edge_type!r}. Must be one of {allowed}.")

        edge_type = edge_type.upper()
        query = (
            f"MATCH (a:Skill {{id: $from_id}}), (b:Skill {{id: $to_id}}) "
            f"MERGE (a)-[:{edge_type}]->(b)"
        )
        await self._run(query, {"from_id": from_id, "to_id": to_id})

    # ------------------------------------------------------------------
    # Neighbor queries
    # ------------------------------------------------------------------

    async def get_outbound_neighbors(
        self,
        skill_id: str,
        edge_type: str | None = None,
    ) -> list[dict]:
        """Return outbound neighbors of a Skill node."""
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

    async def get_inbound_neighbors(
        self,
        skill_id: str,
        edge_type: str | None = None,
    ) -> list[dict]:
        """Return inbound neighbors of a Skill node."""
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

    # ------------------------------------------------------------------
    # Payload operations
    # ------------------------------------------------------------------

    async def get_skill_payload(self, skill_id: str) -> dict | None:
        """Return the payload dict of a Skill node, or None if not found."""
        records = await self._run(
            "MATCH (s:Skill {id: $id}) RETURN s.payload_json AS payload_json",
            {"id": skill_id},
        )
        if not records or records[0]["payload_json"] is None:
            return None
        return json.loads(records[0]["payload_json"])

    async def set_skill_payload(self, skill_id: str, payload: dict) -> None:
        """Overwrite the payload of an existing Skill node."""
        await self._run(
            "MATCH (s:Skill {id: $id}) SET s.payload_json = $payload_json",
            {"id": skill_id, "payload_json": json.dumps(payload)},
        )

    # ------------------------------------------------------------------
    # Hub score recomputation
    # ------------------------------------------------------------------

    async def recompute_hub_scores(self) -> None:
        """Recompute hub_score = outbound_degree / max_outbound_degree for all Skill nodes."""
        query = """
        MATCH (s:Skill)
        WITH s, size([(s)-[]->(n) | n]) as out_degree
        WITH max(out_degree) as max_d, collect({s: s, d: out_degree}) as nodes
        UNWIND nodes as item
        SET item.s.degree = item.d,
            item.s.hub_score = CASE max_d WHEN 0 THEN 0.0 ELSE toFloat(item.d) / max_d END
        """
        await self._run(query)
        logger.info("Hub scores recomputed.")

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    async def detect_cycles(self) -> list[str]:
        """Detect cycles in dependency edges (excludes COLLABORATES_WITH and EXTENDS).

        EXTENDS is an inheritance edge; A-EXTENDS->B while B-ENABLES->A is intentional
        and not a problematic cycle.
        """
        query = """
        MATCH path = (a:Skill)-[:REQUIRES|ENABLES|USES|PART_OF*2..]->(a)
        RETURN a.id as id LIMIT 10
        """
        records = await self._run(query)
        return [r["id"] for r in records]
