# Canonical Event Table (For Labels and Validation)

This table is a single source of truth for operations events that can explain
performance shifts and support supervised label generation.

## Why this table exists

The model currently has an all-cause performance proxy. To move from proxy-only
monitoring to reliable supervised targets, we need dated events from operations.

Examples:
- cleaning campaigns,
- inverter outages,
- planned maintenance,
- curtailment,
- sensor downtime.

Without this table, high-loss days may be mislabeled as soiling when the cause
was something else.

## File

- Template path: `docs/templates/canonical_event_log.csv`

## Column definitions

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique stable ID for each event. |
| `event_type` | string | Event class (`cleaning`, `inverter_outage`, `curtailment`, `sensor_outage`, `maintenance`, etc.). |
| `scope` | string | Impact scope (`plant`, `block`, `inverter`, `sensor`). |
| `scope_id` | string | Specific target (`B1`, inverter ID, sensor ID). |
| `start_date` | date | Event start date (local plant date). |
| `end_date` | date | Event end date (local plant date). |
| `coverage_type` | string | `full` or `partial` coverage confidence for event period completeness. |
| `source` | string | Source of truth (`manual_report`, `SCADA_log`, `CMMS`, etc.). |
| `confidence` | string | `high`, `medium`, `low` confidence in timing/scope. |
| `notes` | string | Free-form operational context. |

## How to use in pipeline

1. Load this table by date range.
2. Expand each event window to daily rows.
3. Left-join event flags to `daily_model_input.csv` by `day`.
4. Use event flags for:
- target construction,
- error analysis,
- exclusion of ambiguous training periods.

## Current status

- Initial cleaning windows are prefilled from provided records.
- Coverage is partial for the full model horizon.
- Sensor outage validation from SCADA logs is still pending.
