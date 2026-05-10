# Data Pipeline Patterns

## Design Principles
- **Idempotency**: re-running a pipeline must produce the same result. Prefer UPSERT over INSERT; use surrogate keys.
- **Append-only source of truth**: never update historical data. Add a new record with a corrected value and a reason.
- **Bounded context**: each pipeline owns its data. Don't share mutable state across pipelines.
- **Fail fast**: validate data at ingestion. Bad data that enters the warehouse corrupts downstream consumers.

## Batch vs Streaming
- Batch: simplest to reason about, cheapest to run, acceptable when latency > 1 hour is fine.
- Micro-batch (Spark Structured Streaming, Flink): latency in seconds/minutes; complex state management.
- Streaming (Kafka + consumer): true event-time processing; requires exactly-once semantics infrastructure.
- Default to batch. Only move to streaming when a business requirement demands sub-minute latency.

## Schema Evolution
- Use a schema registry (Confluent, AWS Glue) for event-based pipelines.
- Allowed without migration: adding a nullable column, adding a new enum value.
- Requires migration: renaming a column, changing a type, removing a field.
- Version your schemas. Never modify a schema in place — create a new version.

## Data Quality
- Define expectations at pipeline entry points: row counts, null rate, value range, referential integrity.
- Use Great Expectations, dbt tests, or custom SQL assertions to encode expectations as code.
- Alert when quality checks fail — don't just log them.
- Quarantine bad records: don't silently drop them. Log the record + reason to a dead-letter store.

## Partitioning and Performance
- Partition time-series data by date (year/month/day). Queries almost always filter on time.
- Avoid small files: compact to 128–512 MB parquet files in object storage.
- Use columnar formats (Parquet, ORC) for analytical workloads — 10–100× faster than row-based formats for aggregations.
- Avoid SELECT *: column-pruning is free in columnar storage; in row-based storage it forces full row reads.

## Lineage and Documentation
- Document every dataset: what it contains, who owns it, where it comes from, how fresh it is.
- Track lineage automatically with OpenLineage/Marquez or dbt's graph.
- A dataset without a known owner is a dataset no one is responsible for — assign ownership at creation.
