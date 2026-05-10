# User Guide — Skill Graph MCP

> How Claude automatically becomes an expert using the knowledge graph — without any manual installation.

---

## The concept in one sentence

When you give Claude a task, it searches a graph of 732 expert skills, loads the most relevant ones, and applies that knowledge immediately — all without you typing a single `/skill` command.

---

## How it works (under the hood)

```
You type:  "Help me build a secure login system with JWT"
              │
              ▼
Claude thinks: "This needs auth expertise — let me check the graph"
              │
              ▼
Claude calls: search_skills("secure authentication JWT login")
              │
              ▼
Graph returns: auth-implementation-patterns [score: 0.81]
               api-security-best-practices  [score: 0.77]
              │
              ▼
Claude calls: get_skill("auth-implementation-patterns")
              │
              ▼
Claude now has: Full JWT implementation guide, OAuth2 patterns,
               session management best practices, code examples
              │
              ▼
Claude answers: with expert-level depth, not just general knowledge
```

**You see none of this.** It happens silently before Claude's response.

---

## What triggers automatic skill lookup?

Claude searches the graph proactively when your request involves:

| Domain | Example prompts |
|---|---|
| Backend / APIs | "Build a REST API", "Add authentication", "Design a webhook" |
| Frontend | "Create a React dashboard", "Add real-time updates", "Optimize performance" |
| DevOps | "Deploy to Kubernetes", "Set up CI/CD", "Configure Nginx" |
| Data | "Build a data pipeline", "Set up Airflow", "Write dbt models" |
| Security | "Audit my code", "Implement OAuth2", "Prevent XSS" |
| Payments | "Integrate Stripe", "Add subscriptions", "Handle refunds" |
| AI/ML | "Fine-tune a model", "Build a RAG system", "Deploy to production" |

---

## Prerequisites

For this to work, you need:

1. **Docker Desktop running** — Neo4j runs in a container
2. **Skill Graph MCP connected** — visible in your Claude MCP list
3. **Claude Code or Claude Desktop restarted** after MCP configuration

### Verify the MCP is active

In Claude Code terminal:
```bash
claude mcp list
# Expected output:
# skill-graph: ✓ Connected
```

In Claude Desktop: look for `skill-graph` in the MCP panel (same location as other connected servers).

---

## Example sessions

### Example 1 — E-commerce project

**You:** "I want to build an e-commerce store with Next.js and Stripe payments"

**What Claude does silently:**
- `search_skills("e-commerce Next.js Stripe")` → finds `stripe-integration`, `nextjs`, `payment-integration`
- `get_skill("stripe-integration")` → loads complete Stripe guide (checkout, webhooks, PCI compliance)
- `get_skill("nextjs")` → loads Next.js App Router patterns, SSR, API routes

**Result:** Claude designs your full stack with production-ready Stripe integration, not a generic "here's how Stripe works."

---

### Example 2 — Security review

**You:** "Review my API for security issues"

**What Claude does silently:**
- `search_skills("API security review vulnerabilities")` → finds `api-security-best-practices`, `auth-implementation-patterns`
- Loads OWASP Top 10 coverage, injection prevention, auth patterns

**Result:** Structured finding report with CRITICAL/HIGH/MEDIUM/LOW classification, not a generic "check your inputs" answer.

---

### Example 3 — Graph traversal

**You:** "I'm building a Kubernetes monitoring dashboard"

**What Claude does silently:**
- `search_skills("Kubernetes monitoring dashboard")` → finds `observability-monitoring-monitor-setup`
- `get_skill("observability-monitoring-monitor-setup")` → loads full observability stack
- `navigate("observability-monitoring-monitor-setup", "COLLABORATES_WITH")` → discovers related skills: `grafana`, `prometheus`, `kpi-dashboard-design`

**Result:** Claude proposes a coherent Grafana + Prometheus + AlertManager stack with actual config examples.

---

## Troubleshooting

### The MCP doesn't appear

1. Check Docker Desktop is running
2. Verify Neo4j container is up: open `http://localhost:7474` in your browser
3. Restart Claude Desktop / Claude Code completely
4. Run `claude mcp list` to confirm connection

### Claude isn't using the graph for my task

The graph is searched when tasks are technical and specific. If your prompt is very vague ("help me with my project"), Claude may not trigger a search. Be more specific:

- ❌ "Help me with authentication"
- ✅ "Implement JWT authentication with refresh tokens in a FastAPI app"

### Error: "Rate limit reached"

`get_skill` is limited to **10 calls per session** to prevent excessive context consumption. This resets when you start a new Claude session. If you hit the limit, continue in a new conversation.

### Neo4j connection error

```bash
# Restart Neo4j
docker-compose restart neo4j
# Wait 30 seconds, then try again
```

---

## Privacy & costs

- **Your queries are sent to OpenAI** to generate the search embedding (~10 tokens per query, ~$0.000002)
- **Your conversation is never stored** in the graph — only the query vector is used transiently
- **Neo4j runs locally** — your data stays on your machine
- **No skill data leaves your machine** except the query embedding

---

## Advanced: navigating the graph yourself

You can also ask Claude to explore the graph directly:

```
"What skills are related to 'react'?"
→ Claude calls navigate("react", "COLLABORATES_WITH")

"What does 'software_engineer' enable?"
→ Claude calls navigate("software_engineer", "ENABLES")

"Show me the full context for the stripe-integration skill"
→ Claude calls get_skill("stripe-integration", depth="full")
```

---

## The graph at a glance

```
732 nodes total:
  ├── 713 domain/tool/role skills (converted from real skill library)
  ├── 14  archetype nodes (senior roles: Software Engineer, Security Expert...)
  └── 5   cluster nodes (Core Dev, Security, Data, Ops, K8s)

3,364 edges total:
  ├── COLLABORATES_WITH  — semantic similarity (embedding cosine > 0.55)
  ├── ENABLES            — archetypes → domain skills
  ├── REQUIRES           — hard dependencies
  └── PART_OF            — cluster membership
```
