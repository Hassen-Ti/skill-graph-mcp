# API Design Patterns — Reference

## REST Fundamentals

Use nouns for resources, verbs for HTTP methods. A resource is a thing, not an action.

Good:  GET /users/{id}
Bad:   GET /getUser?id=123

HTTP method semantics:
- GET    — read, idempotent, cacheable, no body
- POST   — create or non-idempotent operation
- PUT    — full replacement of a resource (idempotent)
- PATCH  — partial update (not always idempotent — document your semantics)
- DELETE — remove a resource (idempotent)

Always return the created/updated resource in the response body for POST/PUT/PATCH. Clients should not need a second request to see what changed.

## Pagination

Never return unbounded lists. Choose one strategy and document it in your API contract.

**Cursor-based (preferred for large or frequently updated datasets):**
```json
{
  "data": [...],
  "next_cursor": "eyJpZCI6MTIzfQ==",
  "has_more": true
}
```
Cursor encodes position (e.g., base64-encoded last-seen ID). Stable across inserts and deletes. Use for feeds, activity logs, infinite scroll.

**Offset-based (acceptable for small, stable datasets):**
```json
{
  "data": [...],
  "total": 842,
  "page": 3,
  "page_size": 20
}
```
Breaks on concurrent inserts. Use only for admin panels and reporting.

**Hard limits:** Always enforce a server-side maximum page size (e.g., 100 items). Never allow `page_size=0` or negative values.

## Error Format

Use a consistent error envelope for all 4xx and 5xx responses.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Field 'email' is required.",
    "field": "email",
    "request_id": "req_a1b2c3d4"
  }
}
```

Rules:
- `code` — machine-readable string constant (SCREAMING_SNAKE_CASE). Clients switch on this.
- `message` — human-readable, English, not for display to end-users.
- `field` — present only for field-level validation errors.
- `request_id` — always present, enables log correlation.
- Never include stack traces or internal state in error responses.
- Never return HTML error pages from a JSON API.

HTTP status code guidance:
- 400 Bad Request — malformed input or validation failure
- 401 Unauthorized — missing or invalid authentication
- 403 Forbidden — authenticated but not authorized
- 404 Not Found — resource does not exist (or you are hiding its existence)
- 409 Conflict — state conflict (duplicate, concurrent modification)
- 422 Unprocessable Entity — well-formed but semantically invalid
- 429 Too Many Requests — rate limit hit (always include Retry-After header)
- 500 Internal Server Error — unexpected failure (always log, never expose details)

## Rate Limiting

Every public API endpoint must be rate limited. Expose the limit in response headers.

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1714000000
Retry-After: 30
```

Strategy options:
- **Token bucket** (preferred): smooth burst handling, configurable fill rate.
- **Fixed window**: simple but has cliff effect at window boundary.
- **Sliding window log**: most accurate, highest memory cost.

When the limit is hit, return 429 with a `Retry-After` header. Never return 503 for rate limits — that implies service unavailability, not client throttling.

Implement rate limiting at the gateway layer, not inside individual service handlers.

## Versioning

Version in the URL path for breaking changes: `/v1/`, `/v2/`.
Use headers (`Accept: application/vnd.api.v2+json`) only for fine-grained content negotiation.

A breaking change is: removing a field, renaming a field, changing a field's type, changing auth requirements, changing error codes that clients switch on.

Maintain the previous major version for a minimum deprecation window (90 days recommended). Announce via a `Deprecation` and `Sunset` header on all v1 responses once v2 is live.
