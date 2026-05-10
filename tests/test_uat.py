# tests/test_uat.py
"""
User Acceptance Tests (UAT) for Skill Graph v1.

8 UAT criteria derived from the business and system requirements.

Prerequisites:
  - Neo4j running with the 4 seed skills loaded
  - skills/knowledge/ directory with api_patterns.md and auth_best_practices.md
  - .env configured

Run all UAT:   pytest tests/test_uat.py -v -m uat
Run one UAT:   pytest tests/test_uat.py::TestUAT01Search -v
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def pytest_configure(config):
    config.addinivalue_line("markers", "uat: User Acceptance Tests")


pytestmark = [pytest.mark.asyncio, pytest.mark.uat]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_session_and_env():
    """
    Reset module-level session state, Neo4j client, and knowledge directory
    before each test so each test creates a fresh driver in its own event loop.
    """
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


@pytest.fixture
async def neo4j_driver():
    """Live Neo4j driver — requires docker-compose up with seeds loaded."""
    from neo4j import AsyncGraphDatabase

    driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "skillgraph"),
        ),
    )
    yield driver
    await driver.close()


# ---------------------------------------------------------------------------
# UAT-01 : SEARCH — Semantic relevance
# ---------------------------------------------------------------------------


class TestUAT01Search:
    """
    UAT-01: Semantic search relevance.

    Given: The graph contains 4 skills with embeddings.
    When:  search_skills("fix authentication bug in REST API") is called.
    Then:  "backend_dev" or "security" appears in the top 3 results.
           Each result has semantic_score > 0.5 and hub_score in [0.0, 1.0].
    """

    async def test_uat01_search_relevance(self):
        from server.main import search_skills

        results = await search_skills("fix authentication bug in REST API")

        assert isinstance(results, list), "search_skills must return a list"
        assert len(results) >= 1, "search_skills returned no results"

        result_dicts = [r if isinstance(r, dict) else r.model_dump() for r in results]

        for r in result_dicts:
            assert "semantic_score" in r, f"Missing semantic_score in result: {r}"
            assert "hub_score" in r, f"Missing hub_score in result: {r}"
            assert r["semantic_score"] > 0.5, (
                f"semantic_score {r['semantic_score']} <= 0.5 for {r['id']}"
            )
            assert 0.0 <= r["hub_score"] <= 1.0, (
                f"hub_score {r['hub_score']} out of [0.0, 1.0] for {r['id']}"
            )

        top_ids = [r["id"] for r in result_dicts[:3]]
        assert "backend_dev" in top_ids or "security" in top_ids, (
            f"Expected backend_dev or security in top 3. Got: {top_ids}. "
            "Check that embeddings are generated and the vector index is populated."
        )


# ---------------------------------------------------------------------------
# UAT-02 : GET_SKILL — Full context loaded
# ---------------------------------------------------------------------------


class TestUAT02GetSkill:
    """
    UAT-02: get_skill returns a complete context object.

    Given: The graph is loaded.
    When:  get_skill("backend_dev") is called.
    Then:  payload.instructions is non-empty (> 50 characters).
           layer_1 contains at least 1 neighbor.
           Session call counter incremented to 1.
    """

    async def test_uat02_get_skill_full_context(self):
        from server.main import get_skill
        from server.session import get_state

        result = await get_skill("backend_dev")

        assert result is not None

        def get_field(obj, *keys):
            current = obj
            for key in keys:
                current = current[key] if isinstance(current, dict) else getattr(current, key)
            return current

        instructions = get_field(result, "payload", "instructions")
        assert len(instructions) > 50, (
            f"payload.instructions too short: {len(instructions)} chars"
        )

        layer_1 = get_field(result, "layer_1")
        assert isinstance(layer_1, list)
        assert len(layer_1) >= 1, "layer_1 must contain at least 1 neighbor"

        # Rate limit counter must have been incremented
        state = get_state()
        assert state.get_skill_calls == 1, (
            f"Expected get_skill_calls == 1 after 1 call, got {state.get_skill_calls}"
        )


# ---------------------------------------------------------------------------
# UAT-03 : EXTENDS — Server-side inheritance resolution
# ---------------------------------------------------------------------------


class TestUAT03Extends:
    """
    UAT-03: extends inheritance resolved server-side.

    Given: backend_dev extends software_engineer in the YAML seed.
    When:  get_skill("backend_dev") is called.
    Then:  payload.tools contains the union of backend_dev AND software_engineer tools.
           payload.instructions is backend_dev's own (full substitution).
    """

    async def test_uat03_extends_tools_union(self):
        from server.main import get_skill

        result = await get_skill("backend_dev")

        payload = result["payload"] if isinstance(result, dict) else result.payload
        tools = payload["tools"] if isinstance(payload, dict) else payload.tools
        instructions = (
            payload["instructions"] if isinstance(payload, dict) else payload.instructions
        )

        tools_set = set(tools)
        assert "git" in tools_set, "git (from software_engineer) not in resolved tools union"
        assert "docker" in tools_set, "docker (from backend_dev) not in resolved tools union"
        assert "bash" in tools_set, "bash missing from tools union"

        assert "software engineer" not in instructions.lower() or "backend" in instructions.lower(), (
            "Instructions appear to be software_engineer's — extends substitution failed"
        )
        assert len(instructions) > 50, "instructions are empty after extends resolution"

    async def test_uat03_extends_instructions_substitution(self):
        from server.main import get_skill
        from server.session import reset_state

        parent_result = await get_skill("software_engineer")

        reset_state()

        child_result = await get_skill("backend_dev")

        def get_instructions(r):
            p = r["payload"] if isinstance(r, dict) else r.payload
            return p["instructions"] if isinstance(p, dict) else p.instructions

        parent_instructions = get_instructions(parent_result)
        child_instructions = get_instructions(child_result)

        assert child_instructions != parent_instructions, (
            "backend_dev instructions are identical to software_engineer's — "
            "extends substitution not applied"
        )


# ---------------------------------------------------------------------------
# UAT-04 : NAVIGATE — Directed traversal
# ---------------------------------------------------------------------------


class TestUAT04Navigate:
    """
    UAT-04: navigate performs directed traversal.

    Given: backend_dev --[requires]--> security edge exists in the graph.
    When:  navigate("backend_dev", "requires", "outbound") is called.
    Then:  neighbors contains a node with id == "security".
           backend_dev is recorded in visited_nodes after the call.
    """

    async def test_uat04_navigate_directed_traversal(self):
        from server.main import navigate

        result = await navigate("backend_dev", "requires", "outbound")

        assert isinstance(result, dict), "navigate must return a dict"
        neighbors = result["neighbors"]
        assert isinstance(neighbors, list), "navigate result must have a 'neighbors' list"
        assert len(neighbors) >= 1, (
            "navigate('backend_dev', 'requires', 'outbound') returned empty — "
            "check that the requires edge was loaded correctly"
        )

        result_ids = [n["id"] for n in neighbors]
        assert "security" in result_ids, (
            f"Expected 'security' in navigate results. Got: {result_ids}"
        )

    async def test_uat04_navigate_first_visit_tracked(self):
        """
        After the first navigate from backend_dev, it must be recorded
        in the session's visited_nodes set.
        """
        from server.main import navigate
        from server.session import get_state

        await navigate("backend_dev", "requires", "outbound")

        state = get_state()
        assert "backend_dev" in state.visited_nodes, (
            "backend_dev not recorded in visited_nodes after navigate"
        )


# ---------------------------------------------------------------------------
# UAT-05 : GET_KNOWLEDGE — Content fetched on demand
# ---------------------------------------------------------------------------


class TestUAT05GetKnowledge:
    """
    UAT-05: get_knowledge fetches document content on demand.

    Given: auth_best_practices.md exists in skills/knowledge/.
    When:  get_knowledge("auth_best_practices.md") is called.
    Then:  Returns a string containing "JWT" or "bcrypt" or "OAuth".
           Path traversal attempts are rejected.
    """

    async def test_uat05_knowledge_content_fetched(self):
        from server.main import get_knowledge

        content = await get_knowledge("auth_best_practices.md")

        assert isinstance(content, str), "get_knowledge must return a string"
        assert len(content) > 50, "auth_best_practices.md content is unexpectedly short"

        has_jwt = "JWT" in content
        has_bcrypt = "bcrypt" in content
        has_oauth = "OAuth" in content

        assert has_jwt or has_bcrypt or has_oauth, (
            "auth_best_practices.md content must contain at least one of: "
            "JWT, bcrypt, OAuth. None found."
        )

    async def test_uat05_path_traversal_blocked(self):
        """Path traversal attempts must be rejected before any filesystem access."""
        from server.main import get_knowledge

        with pytest.raises((PermissionError, ValueError)) as exc_info:
            await get_knowledge("../../etc/passwd")

        err_str = str(exc_info.value).lower()
        assert len(err_str) > 0, "Exception message should not be empty"

    async def test_uat05_nonexistent_file_raises(self):
        """get_knowledge on a valid filename that does not exist must raise FileNotFoundError."""
        from server.main import get_knowledge

        with pytest.raises(FileNotFoundError):
            await get_knowledge("nonexistent_file_xyz.md")


# ---------------------------------------------------------------------------
# UAT-06 : RATE LIMIT — Server protection
# ---------------------------------------------------------------------------


class TestUAT06RateLimit:
    """
    UAT-06: get_skill is rate-limited to 10 calls per session.

    Given: A fresh session (calls reset to 0 by autouse fixture).
    When:  get_skill is called 11 times.
    Then:  The first 10 return valid dicts.
           The 11th raises ValueError containing "rate limit" (case-insensitive).
    """

    async def test_uat06_rate_limit_enforced(self):
        from server.main import get_skill

        skill_ids = [
            "backend_dev", "security", "software_engineer", "web_dev_cluster",
            "backend_dev", "security", "software_engineer", "web_dev_cluster",
            "backend_dev", "security",
        ]
        assert len(skill_ids) == 10, "Must have exactly 10 IDs for this test"

        for i, skill_id in enumerate(skill_ids):
            result = await get_skill(skill_id)
            assert result is not None, (
                f"Call {i+1} to get_skill('{skill_id}') returned None"
            )

        with pytest.raises(ValueError) as exc_info:
            await get_skill("backend_dev")

        error_msg = str(exc_info.value).lower()
        assert "rate limit" in error_msg, (
            f"11th get_skill call raised ValueError but message does not contain "
            f"'rate limit'. Got: '{exc_info.value}'"
        )


# ---------------------------------------------------------------------------
# UAT-07 : YAML LOAD — CLI loading
# ---------------------------------------------------------------------------


class TestUAT07YamlLoad:
    """
    UAT-07: CLI loads YAML seeds correctly.

    Assumes seeds are already loaded (run: python -m registry.cli load skills/).
    v2 ships 14 skill YAML files (4 original + 10 new).
    """

    async def test_uat07_skill_count_in_neo4j(self, neo4j_driver):
        """Neo4j must contain exactly 14 :Skill nodes after v2 seed load."""
        async with neo4j_driver.session() as session:
            result = await session.run("MATCH (s:Skill) RETURN count(s) AS cnt")
            record = await result.single()
            count = record["cnt"]

        assert count == 14, (
            f"Expected 14 :Skill nodes in Neo4j (v2 ships 14), found {count}. "
            "Run: python -m registry.cli load skills/"
        )

    async def test_uat07_web_dev_cluster_hub_score_positive(self, neo4j_driver):
        """web_dev_cluster has outbound enables edges -> hub_score must be > 0."""
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (s:Skill {id: 'web_dev_cluster'}) RETURN s.hub_score AS hs"
            )
            record = await result.single()
            assert record is not None, "web_dev_cluster not found in Neo4j"
            hub_score = record["hs"]

        assert hub_score is not None, "web_dev_cluster.hub_score is None"
        assert hub_score > 0, (
            f"web_dev_cluster.hub_score must be > 0 (has enables edges). Got: {hub_score}"
        )

    def test_uat07_index_metadata_has_14_entries(self):
        """index_metadata.json must contain exactly 14 entries after v2 load."""
        import json

        candidates = [
            Path("index_metadata.json"),
            Path("registry/index_metadata.json"),
            Path("topic_2/prod/skill-graph/index_metadata.json"),
        ]

        metadata_path = next((c for c in candidates if c.exists()), None)
        assert metadata_path is not None, (
            "index_metadata.json not found. Run: python -m registry.cli load skills/"
        )

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert len(metadata) == 14, (
            f"Expected 14 entries in index_metadata.json (v2 ships 14 skills), found {len(metadata)}. "
            f"Keys: {list(metadata.keys())}"
        )

        # All v2 skill IDs must be present.
        expected_keys = {
            "web_dev_cluster", "software_engineer", "backend_dev", "security",
            "frontend_dev", "tech_lead", "code_reviewer", "architecture_designer",
            "devops_engineer", "sre", "data_engineer", "ml_engineer",
            "data_cluster", "devops_cluster",
        }
        actual_keys = set(metadata.keys())
        assert actual_keys == expected_keys, (
            f"index_metadata.json keys mismatch. "
            f"Missing: {expected_keys - actual_keys}. Extra: {actual_keys - expected_keys}"
        )


# ---------------------------------------------------------------------------
# UAT-08 : STARTUP — System startup
# ---------------------------------------------------------------------------


class TestUAT08Startup:
    """
    UAT-08: MCP server starts and shuts down cleanly.

    Given: docker-compose up -d (Neo4j healthy).
    When:  python -m server.main is started as a subprocess.
    Then:  Process starts without error within 5 seconds.
           Termination causes a clean exit.
    """

    @pytest.mark.slow
    def test_uat08_server_starts_cleanly(self):
        """
        Start server.main as a subprocess, wait 5 seconds for it to stay alive,
        then terminate and verify clean exit.
        """
        python = sys.executable

        proc = subprocess.Popen(
            [python, "-m", "server.main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        startup_ok = False
        deadline = time.time() + 5.0

        try:
            while time.time() < deadline:
                ret = proc.poll()
                if ret is not None:
                    if ret != 0:
                        # Non-zero: actual crash
                        stderr = proc.stderr.read()
                        pytest.fail(
                            f"server.main crashed during startup (exit code {ret}). "
                            f"stderr: {stderr[:500]}"
                        )
                    else:
                        # Exit code 0: stdio EOF clean exit — startup was successful
                        startup_ok = True
                    break
                time.sleep(0.1)
            else:
                startup_ok = True
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        assert startup_ok, "server.main did not stay alive for 5 seconds."

        exit_code = proc.returncode
        acceptable = {0, 1, -15}
        if hasattr(signal, "SIGTERM"):
            acceptable.add(-signal.SIGTERM)
        assert exit_code in acceptable, (
            f"server.main terminated with unexpected exit code: {exit_code}"
        )
