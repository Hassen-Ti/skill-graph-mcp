# Architecture — Skill Graph MCP

Technical deep-dive into the system design, data pipeline, and runtime behavior.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (one-time setup)                                     │
│                                                                     │
│  sklills_lib/                                                       │
│  └── 713 SKILL.md files                                             │
│         │                                                           │
│         ▼ scripts/convert_skills.py                                 │
│  staging/skills/                                                     │
│  └── 718 YAML files (713 skills + 5 bundle clusters)               │
│         │                                                           │
│         ▼ scripts/embed_skills.py                                   │
│  staging/skills/ (enriched)                                         │
│  └── +2,865 semantic edges (cosine similarity > 0.55, top-8)        │
│         │                                                           │
│         ▼ scripts/merge_archetypes.py                               │
│  staging/skills/ (final)                                            │
│  └── +138 archetype→skill enables edges                             │
│         │                                                           │
│         ▼ python -m registry.cli load staging/skills/               │
│  Neo4j 5.x                                                          │
│  └── 732 nodes, 3,364 edges, hub_scores, vector index               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  RUNTIME (MCP server)                                               │
│                                                                     │
│  Claude (LLM)                                                       │
│     │  stdio / MCP protocol                                         │
│     ▼                                                               │
│  server/main.py (FastMCP)                                           │
│  ├── search_skills(query)                                           │
│  │     ├── embed_text(query)   → OpenAI API (1536-dim vector)       │
│  │     └── vector index query  → Neo4j (top-3 cosine similarity)    │
│  ├── get_skill(id, depth)                                           │
│  │     ├── fetch Skill node + payload_json                          │
│  │     └── traverse L1 (+ L2 if depth="full") neighbors             │
│  ├── navigate(from_id, edge_type, direction)                        │
│  │     └── MATCH (s)-[:TYPE]->(n) / (n)-[:TYPE]->(s)                │
│  └── get_knowledge(ref)                                             │
│        └── safe file read (3-layer security gate)                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Neo4j data model

### Node schema

```cypher
(:Skill {
  id:           String,   // unique PK, lowercase, hyphens allowed (regex ^[a-z0-9_-]+$)
  name:         String,
  description:  String,   // 1-500 chars, used for embedding
  type:         String,   // "domain" | "tool" | "role" | "cluster"
  author:       String,
  version:      String,
  priority:     Integer,  // 1 (cluster) | 2 (role/domain) | 3 (tool)
  hub_score:    Float,    // out_degree / max_out_degree in [0.0, 1.0]
  degree:       Integer,  // raw outbound degree
  context_cost: Integer,  // estimated token count of payload.instructions
  embedding:    Float[],  // 1536-dim OpenAI vector (text-embedding-3-small)
  payload_json: String    // JSON: {instructions, tools, knowledge, exclude_tools}
})
```

### Relationship types

| Type | Direction | Meaning |
|---|---|---|
| `REQUIRES` | A → B | A cannot function without B |
| `ENABLES` | A → B | A unlocks or significantly enhances B |
| `COLLABORATES_WITH` | A → B | A and B work well together (semantic similarity) |
| `USES` | A → B | A uses B as a dependency or tool |
| `EXTENDS` | A → B | A is a specialisation of B |
| `PART_OF` | A → B | A is a member of cluster B |

### Vector index

```cypher
CREATE VECTOR INDEX skill_description_embedding
FOR (s:Skill) ON (s.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
```

---

## Data pipeline detail

### Sprint 1 — Converter (`scripts/convert_skills.py`)

**Input:** 713 SKILL.md files from sklills_lib  
**Output:** 718 YAML files in `staging/skills/`

Key transformations:
- Frontmatter extraction (name, description, requires.mcp)
- Type classification by heuristic:
  ```python
  # Tool: ID contains "-pro", "-automation", "bash", "git", "docker", etc.
  # Role: description starts with "You are a..."
  # Domain: everything else
  ```
- "Works well with" section parser → `COLLABORATES_WITH` explicit edges
- 5 bundle cluster nodes synthesized from category metadata
- Produces `staging/conversion_report.json` (stats + schema validation)

**Result:** 718 files, 0 schema failures, 107 explicit edges

---

### Sprint 2 — Embedding pipeline (`scripts/embed_skills.py`)

**Model:** `text-embedding-3-small` (1536 dims)  
**Cost:** ~$0.002 for 718 descriptions  
**Parameters:**
- Similarity threshold: 0.55 (below 0.65 → 82% isolated nodes)
- Top-K neighbors per skill: 8

**Algorithm:**
1. Generate embeddings for all 718 descriptions (batches of 100)
2. Compute full 718×718 cosine similarity matrix (pure Python, O(n²))
3. For each skill: keep top-8 neighbors above threshold, excluding self and existing edges
4. Inject as `COLLABORATES_WITH` edges directly into YAML files
5. Update `index_metadata.json` (SHA256 drift detection per skill)

**Result:** 2,865 semantic edges, avg degree 8.8, 90 isolated nodes (12.5%)

---

### Sprint 3 — Archetype merger (`scripts/merge_archetypes.py`)

**Problem:** 14 hand-crafted archetype skills (Software Engineer, Security Expert...) are isolated from the 718 domain skills.

**Solution:** Separate embedding similarity pass with lower threshold (0.45) because archetype descriptions are shorter and more generic than domain skill descriptions.

**Algorithm:**
1. Embed 11 non-cluster archetypes (14 - 3 cluster types)
2. Embed all 718 staging skills
3. For each archetype: top-15 matches above 0.45 → `ENABLES` edges
4. Back-edges from staging skills to archetypes → `COLLABORATES_WITH`

**Result:** 138 `ENABLES` edges, 136 back-edges. Each archetype: 8–17 connections to domain skills.

---

### Sprint 4 — Neo4j load (`python -m registry.cli`)

**Key design:** Archetypes copied into `staging/skills/` before load because the registry CLI's orphan check rejects edges whose targets aren't in the same batch.

**Load sequence:**
1. Dry-run validation (schema + orphan check) → 0 errors
2. Real load: MERGE all 732 nodes, CREATE all edges
3. `recompute_hub_scores()`: hub_score = out_degree / max_out_degree
4. Vector embeddings generated per node on load
5. `detect_cycles()`: 0 cycles in REQUIRES/ENABLES/USES/PART_OF

---

## Runtime architecture

### Session state

Because stdio transport = 1 process per session, state is module-level:

```python
@dataclass
class SessionState:
    get_skill_calls: int = 0        # rate limit counter
    visited_nodes: set[str]         # for _revisit flag in navigate()
    active_tools: list[str]         # tools registered from loaded skills
```

Limits:
- `get_skill`: max 10 calls/session (prevent context overflow)
- `active_tools`: max 15 (prevent tool overload)

### Search flow

```
search_skills(query: str) -> list[SkillCandidate]
  │
  ├─ embed_text(query)
  │    └─ AsyncOpenAI.embeddings.create(model="text-embedding-3-small")
  │         → 1536-dim float list
  │
  └─ Neo4j vector query
       CALL db.index.vector.queryNodes("skill_description_embedding", 3, $embedding)
       YIELD node, score
       RETURN node.id, node.name, score AS semantic_score, node.hub_score
```

### get_skill depth levels

```
depth="shallow"
  └─ Skill node + payload_json
  └─ Direct neighbors (distance=1) via MATCH (s)-[r]->(n)

depth="full"  (depth="deep" internally)
  └─ Skill node + payload_json
  └─ L1 neighbors (distance=1)
  └─ L2 neighbors (distance=2, sampled to avoid explosion)
```

### get_knowledge security gates

```
Input: ref = "api_patterns.md"

Gate 1: Path separator check
  Path(ref).name == ref → pass
  "/" or "\\" in ref → ValueError

Gate 2: Extension whitelist
  suffix in {".md", ".txt", ".json"} → pass
  otherwise → ValueError

Gate 3: Path confinement
  resolved = (KNOWLEDGE_BASE_DIR / ref).resolve()
  str(resolved).startswith(str(KNOWLEDGE_BASE_DIR.resolve())) → pass
  otherwise → PermissionError (symlink escape)
```

---

## Hub score interpretation

`hub_score` is a lightweight proxy for centrality (not PageRank, just degree centrality):

```
hub_score = out_degree / max_out_degree_in_graph
```

| hub_score | Meaning |
|---|---|
| 0.9+ | Cluster node (core-dev-cluster, security-core-cluster...) |
| 0.5-0.9 | Archetype (Software Engineer, Security Expert...) |
| 0.1-0.5 | Domain skill with many connections |
| 0.0-0.1 | Niche or isolated skill |

Claude uses `hub_score` to decide whether to `get_skill` on a candidate — high hub_score = likely more useful context.

---

## Performance characteristics

| Operation | Typical latency | Notes |
|---|---|---|
| `search_skills` | 200–500ms | OpenAI embed (100ms) + Neo4j vector (50ms) |
| `get_skill` shallow | 50–100ms | Single Neo4j round-trip |
| `get_skill` full | 100–200ms | Two Neo4j round-trips |
| `navigate` | 30–80ms | Single Cypher MATCH |
| `get_knowledge` | < 5ms | Local file read |

---

## Extension points

### Add a new skill
1. Create `skills/my-skill.yaml` following `skills/schema.json`
2. `python -m registry.cli load skills/my-skill.yaml --schema skills/schema.json`
3. Embedding is generated on load; skill is searchable immediately

### Sync with updated skill library
```bash
# Only re-embeds skills whose SHA256 description hash changed
python scripts/embed_skills.py
python -m registry.cli load staging/skills/ --schema skills/schema.json
```

### Improve edge types
Current limitation: 90% of edges are `COLLABORATES_WITH` (semantic, undirected-ish). To improve:
- LLM batch classification: feed each edge pair to Claude → assign REQUIRES/ENABLES/USES
- Improves `navigate` quality significantly
