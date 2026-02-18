# Data Cleaning and Preprocessing Plan

## Objective

Build a deterministic preprocessing layer that converts raw telemetry exports
into quality-gated daily features for:
- soiling-loss modeling,
- anomaly detection,
- cross-plant inference support.

## Scope

Input files:
- `data/inverters_2025_to_current_10min_avg_si.csv`
- `data/irradiance_2025_to_current_15min_sum_si.csv`
- `data/power_generation_2025_to_current_1day_none_si.csv`

Output files:
- `artifacts/preprocessed/inverters_clean.csv`
- `artifacts/preprocessed/irradiance_clean.csv`
- `artifacts/preprocessed/generation_clean.csv`
- `artifacts/preprocessed/generation_daily_clean.csv`
- `artifacts/preprocessed/daily_model_input.csv`
- `artifacts/preprocessed/preprocessing_summary.md`

## Plan

1. Standardize schema
- Parse `Date` to datetime.
- Cast value columns to numeric.
- Drop rows with invalid dates.

2. Deduplicate by telemetry timestamp
- Group by `Timestamp`.
- Keep first `Date`.
- Average numeric duplicates.

3. Sanity filtering
- Inverter power: valid range `[0, 300000]` W.
- Inverter current: valid range `[0, 250]` A.
- Irradiance: no negative values.
- Generation: valid range `[0, 360000000000]` J.
- Invalid values become NaN.

4. Daily aggregation
- Inverter subset energy from power times 600 seconds.
- Irradiance daily sums and record coverage.
- Generation daily value using latest value (fallback max).

5. Feature engineering
- `normalized_output` with irradiance floor guard.
- rolling clean baseline (30-day, 95th percentile on clear days).
- soiling proxy percentage.
- B1/B2 mismatch metrics.

6. Transfer-readiness gating
- Sensor and coverage flags.
- Block mismatch flag.
- Transfer quality score and tier.

## Quality Gates

- preprocessing script exits successfully.
- output files are generated.
- `soiling_loss_pct_proxy` remains in `[0, 100]`.
- `normalized_output` is capped to prevent outlier corruption.
- transfer readiness fields are populated.
