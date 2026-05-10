# Observability

## The Three Pillars
- **Metrics**: numeric aggregates over time (request rate, error rate, latency percentiles). Use them for dashboards and alerts.
- **Logs**: structured event records. Always use JSON. Include trace_id, user_id, and service name on every log line.
- **Traces**: distributed request flows across services. Instrument with OpenTelemetry; store in Tempo or Jaeger.

## SLIs, SLOs, Error Budgets
- A **Service Level Indicator** (SLI) is the metric that matters: availability, latency p99, error rate.
- A **Service Level Objective** (SLO) is the target: "99.9% of requests succeed in < 500ms, measured over 30 days."
- The **error budget** = 1 - SLO. If your SLO is 99.9%, you have 0.1% = ~43 min/month to burn.
- When the error budget is depleted, freeze feature work and focus on reliability.
- Track error budget burn rate, not just current SLO status. A 14× burn rate means you'll be out of budget in 2 days.

## Alerting Principles
- Alert on symptoms, not causes. "Latency p99 > 1s" is a symptom. "CPU > 80%" is not.
- Every alert must have a runbook. If the on-call can't act on it, it's noise.
- Target < 5 pages per engineer per on-call shift. More than that = alert fatigue.
- Distinguish between: page (customer impact, now), ticket (degraded, but within SLO), log (interesting event).

## Dashboards
- Design dashboards for the four golden signals: latency, traffic, errors, saturation.
- Top row: SLO status and error budget. Second row: the four golden signals. Third row: service health details.
- Never use averages for latency. Use p50, p95, p99. Averages hide the tail.

## Incident Response
- Declare an incident early. It's cheaper to stand down than to miss a growing problem.
- Assign roles: incident commander, communication lead, technical responders.
- Communicate externally every 30 minutes, even if there's nothing new to report.
- Run blameless postmortems within 48 hours. Focus on system factors, not human error.
- Track action items with owners and deadlines. Follow up in 30 days.
