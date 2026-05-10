import os
import pytest
from neo4j import AsyncGraphDatabase


@pytest.fixture
async def neo4j_driver():
    driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "skillgraph")),
    )
    yield driver
    await driver.close()


@pytest.fixture
def sample_skill_dict():
    return {
        "id": "test_skill",
        "name": "Test Skill",
        "description": "A test skill for unit tests.",
        "author": "test",
        "version": "1.0.0",
        "type": "role",
        "priority": 1,
        "prerequisites": [],
        "edges": [],
        "payload": {
            "instructions": "You are a test agent.",
            "tools": ["bash"],
            "knowledge": [],
        },
    }
