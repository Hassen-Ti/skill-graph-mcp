# Architecture Patterns

## Decision Records (ADRs)
Every significant architectural choice needs an ADR:
- Context: what forces are at play?
- Decision: what did we choose?
- Consequences: what does this enable or foreclose?
Store ADRs in the repository (`docs/architecture/`) alongside the code they describe.

## C4 Model
Use the 4 levels to communicate at different zoom levels:
1. **Context**: the system and its external actors
2. **Container**: deployable units (services, databases, CDN)
3. **Component**: significant internal building blocks within a container
4. **Code**: class/module diagrams (generate from code, don't maintain manually)

## Service Boundaries
- Draw boundaries around business capabilities, not technical layers.
- A service owns its data. No service reads another service's database directly.
- Shared databases create hidden coupling. Use events or APIs to cross boundaries.
- When in doubt, build a monolith first. Extract services when you can name the pain.

## Scalability Heuristics
- Identify the bottleneck before sharding anything. Profile under realistic load.
- Stateless services scale horizontally; stateful services need coordination — push state to the edge (CDN) or center (DB).
- Read-heavy workloads: add caching layers (Redis, CDN). Write-heavy: partition data or use append-only logs.
- Async by default for non-critical path operations. Queues decouple producers from consumers.

## Failure Modes
- Design for failure. Every external call can fail — add timeouts, retries with exponential backoff, and circuit breakers.
- Define SLOs before writing code. "99.9% availability" means < 9 hours/year downtime — what does your design allow?
- Chaos test in staging before going to production. Inject latency and failures intentionally.

## Technology Selection
- Prefer boring technology for the core (proven databases, well-understood runtimes).
- Reserve experimental technology for the edges where failure is isolated.
- Every new dependency is a maintenance commitment. Evaluate operational complexity, not just feature set.
- The best architecture is the one your team can operate at 3 a.m.
