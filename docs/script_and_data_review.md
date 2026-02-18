# Script and Data Review

## Scope

Reviewed files:
- `scripts/inverter_data_fetch.py`
- `scripts/irradiance_data_fetch.py`
- `scripts/power_generation_data_fetch.py`
- `scripts/jwt_fetch.py`
- `data/inverters_2025_to_current_10min_avg_si.csv`
- `data/irradiance_2025_to_current_15min_sum_si.csv`
- `data/power_generation_2025_to_current_1day_none_si.csv`

## High-Priority Findings

1. Generation feed is not truly daily in shape.
- File: `data/power_generation_2025_to_current_1day_none_si.csv`
- Impact: multiple intraday records per day can break daily KPI logic if consumed directly.
- Action: aggregate to one trusted daily value (for example, latest or max per day after validation).

2. Generation conversion logic has hard-coded clipping assumptions.
- File: `scripts/power_generation_data_fetch.py`
- Impact: extreme values are force-clamped to a fixed constant, which can hide genuine quality issues.
- Action: replace hard clamp with explicit invalid flag/NaN and preserve raw value for traceability.

3. Inverter coverage is partial (8 of 34 inverters).
- File: `data/inverters_2025_to_current_10min_avg_si.csv`
- Impact: direct comparison to plant-level generation can look inconsistent unless scaled.
- Action: always report subset-vs-plant scaling factor in dashboards and models.

## Medium-Priority Findings

1. Timestamp irregularity and large gaps are present.
- Inverter median interval is correct (600s) but max gap is large.
- Irradiance median interval is correct (900s) but has many long gaps.
- Action: build gap-aware features (`availability`, `gap_count`) and avoid naive interpolation across long outages.

2. Missingness is non-trivial in inverter power channels.
- Some inverter power columns have high null ratios.
- Action: add per-channel data quality thresholds before using a period for model training.

3. Fetch scripts are not fully standardized.
- Different retry behavior, limits, and post-processing style across scripts.
- Action: unify common concerns (timeout, retry, parse/date helpers, output path conventions).

## Lower-Priority Findings

1. Naming and comments can be more production-grade.
- Example: ad-hoc phrasing in comments.
- Action: keep comments operational and domain-specific.

2. Output path behavior should be explicit.
- Some scripts save to current working directory rather than a defined data target.
- Action: write exports to `data/` or a configured output directory.

## Implemented in This Repo Update

- Added `scripts/data_quality_audit.py` to:
  - Standardize loading/parsing.
  - Quantify interval irregularity and missingness.
  - Produce daily merged features and anomaly flags.
  - Generate EDA plots and an audit summary.
- Updated fetch scripts to:
  - Write exports to a configurable output directory (`TB_OUTPUT_DIR`, default `data`).
  - Use request timeouts (`TB_REQUEST_TIMEOUT_S`).
  - Treat invalid generation values as missing instead of forcing fixed constants.
- Added `README.md` with stakeholder-centric objectives and workflow.
- Added `requirements.txt` and tightened `.gitignore`.

## Recommended Next Engineering Actions

1. Validate generation units against ThingsBoard key semantics and update conversion logic accordingly.
2. Add a canonical preprocessing module used by both fetch and training pipelines.
3. Define model targets explicitly:
- `cleaning_needed_within_7d`
- `underperformance_event_within_24h`
- `real_time_anomaly`
