# Domain Context and Concern Resolution

This document captures plant-specific operational context, modeling assumptions,
and how previously raised concerns are handled in the current pipeline.

## Operational Context (Plant-Specific)

- Plant size: 10-15 MW.
- Full plant inverters: 34.
- Current sampled inverters in dataset: 8 (4 from `B1`, 4 from `B2`).
- Cleaning operation is not instantaneous across the full site.
- Reported washing campaign window (2025-2026):
  - 2025-09-20 to 2025-09-30
  - 2025-10-20 to 2025-10-30
  - 2025-11-20 to 2025-11-30
  - 2025-12: no washing
  - 2026-01: no washing
- Field note: cleaning one end of the plant to the other can take about 10-30 days.

Implication:
- Recovery in performance proxies can be gradual through a washing window.
- Do not interpret every recovery as a one-day step change.

## What Current Pipeline Is (and Is Not)

Current stage focuses on:
- data quality checks,
- deterministic cleaning,
- stable daily preprocessing features.

Current stage is not:
- final causal attribution of all losses to soiling,
- final supervised target engineering for failure labels,
- a full physical PV model.

## Concern-by-Concern Resolution

### 1) Why low irradiance days are treated carefully

Concern:
- "Low irradiance can happen in morning/evening/cloudy periods. Why discount days?"

Resolution:
- The threshold is applied on daily irradiance sum (`irradiance_tilted_sum`), not on
  minute-level values. Morning/evening low values are normal and already included in
  daily totals.
- Very low daily totals with simultaneous inverter production are treated as likely
  sensor issues, not normal weather.
- These days are excluded only from baseline construction to avoid corrupting the
  reference curve.

Where implemented:
- `scripts/daily_features.py`:
  - `MIN_IRRADIANCE_FOR_BASELINE`
  - `compute_performance_features()`
  - `compute_quality_flags()` (`flag_sensor_suspect_irradiance`)

### 2) Why 14-day median is used (not mean)

Concern:
- "Why `M14_t` median, not mean? Why no sorting?"

Resolution:
- Median is used for robustness against outliers, telemetry spikes, and missing data.
- Mean is more sensitive to single bad days.
- Explicit manual sorting is not needed; rolling median internally computes the median
  of the window values.
- The 14-day median is used as a short-term "typical" reference, not a long-term trend.

Where implemented:
- `scripts/daily_features.py`: `normalized_output_14d_median`

### 3) Data leak concern for "soiling loss" feature

Concern:
- "Is loss feature a leak? Is naming as soiling loss safe?"

Resolution:
- Naming was changed to `performance_loss_pct_proxy` to avoid claiming pure soiling.
- It is treated as an all-cause proxy deficit versus a rolling clean-like baseline.
- It should not be used as ground-truth soiling label without validation.

Where implemented:
- `scripts/daily_features.py`: `performance_loss_pct_proxy`
- pipeline outputs and docs renamed accordingly

### 4) Is clear-day loss equal to soiling only?

Concern:
- "Can we assume only soiling affects output on clear days?"

Resolution:
- No. Current proxy is intentionally all-cause.
- Possible confounders include curtailment, inverter limits, thermal effects,
  sensor error, communication gaps, and temporary electrical issues.
- This is why quality flags and block-mismatch checks are retained.

Current recommendation:
- Keep proxy for monitoring/ranking.
- Build validated labels using O&M logs before supervised failure modeling.
- Introduce physical normalization (for example with pvlib) in a later modeling stage.

### 5) Environmental drivers and available data

Concern:
- Wind, humidity/dew, rainfall, PM, and weather context should be considered.

Resolution:
- Solcast 10-minute standard files are integrated and aggregated to daily features.
- Added daily PM2.5/PM10, precipitation, humidity, wind, dewpoint, temperature,
  cloud, pressure, and modelled irradiance metrics.

Plant weather-station telemetry noted:
- Device: `f81ed490-9cd2-11ef-9b7d-ef27f6a9abbc`
- `wstn1_humidity` (RH%, 1-minute)
- `wstn1_wind_speed` (m/s, 1-minute)
- `wstn1_wind_dir` (deg from north, 1-minute)

Current status:
- These station channels are acknowledged as important and should be integrated as
  first-party measurements in a future fetch/preprocess pass.

Note:
- Solcast is modelled data, not direct plant sensor telemetry. Use it as context and
  cross-check source, not blind replacement.

Gap-fill policy guidance:
- Prefer direct plant telemetry as primary.
- Use Solcast as secondary reference to validate suspicious days.
- If filling missing values, record an explicit provenance flag per field/day.

### 6) "4.3% typical but up to 81%" plausibility

Concern:
- "Can changes be that drastic?"

Resolution:
- Large proxy values can be real on some days, but can also result from outages,
  curtailment, sensor issues, or data gaps.
- This is why interpretation must be filtered by quality flags and availability.

Practical rule:
- Treat high proxy days with no quality flags as higher-confidence candidates.

### 7) B1 data unavailability and whether labels are wrong

Concern:
- "Inverters are verified and correctly labelled. Why B1 gap?"

Resolution:
- Investigation indicates B1 channels and B2-04 share low-availability windows
  with strong correlation (`r=0.98`, 64 shared low days), suggesting a shared
  communication/ingestion path issue rather than mislabeling.
- Availability improved in Feb 2026, indicating intermittent behavior.

Where documented:
- `scripts/investigate_b1_gap.py`
- `artifacts/b1_investigation/b1_investigation_report.md`

### 8) Candidate alternate inverters for future fetches

Provided candidates:
- `B1-01` (`783e5900-45af-11ef-b4ce-d5aee9e495ad`)
- `B1-05` (`7886fac0-45af-11ef-b4ce-d5aee9e495ad`)
- `B1-16` (`793fc370-45af-11ef-b4ce-d5aee9e495ad`)
- `B1-12` (`790b9410-45af-11ef-b4ce-d5aee9e495ad`)

Constraint:
- Keep exactly 4 devices per block in analysis sets for block comparability.

## Cross-Plant Inference Guidance

This repository is built on one plant but intended to inform others.

Rules:
- transfer quality-gate records before cross-plant inference,
- recalibrate thresholds for each target site,
- separate transferable patterns from plant-specific operational behavior.

## Open Items and Resolution Status

### ✅ Resolved: Native unit of `EnergyMeter_dailyGeneration`

- Confirmed: native unit is **kWh**.
- The fetch script (`scripts/power_generation_data_fetch.py`) already converts
  kWh → Joules at fetch time (`raw_kwh * KWH_TO_JOULES`).
- Downstream pipeline variables (`daily_generation_j`, `subset_energy_j`) and
  conversions (`/ 3.6e9` for MWh) are therefore correct.

### ⏸ Deferred: Validate sensor outage windows with maintenance/SCADA logs

- Not feasible at this time — SCADA/maintenance logs are not available.
- Mitigation: `flag_sensor_suspect_irradiance` and `flag_coverage_gap` identify
  candidate outage days from the data itself.

### ✅ Partially addressed: Cleaning/O&M event logs for label generation

- "Canonical event table" means: a structured file mapping dates to known operational
  events (cleaning, outages, curtailment) that can serve as supervised labels.
- Currently available: cleaning campaign windows (Sep/Oct/Nov 2025, ~20th-30th each).
  These are documented in README and this file (§ Operational Context).
- Not available: events for the full data period, or non-cleaning events.
- Next step: populate `docs/templates/canonical_event_log.csv` with known dates,
  and integrate as label source when building supervised targets.
- See also: `docs/canonical_event_table.md` for the schema definition.

### 🔄 In progress: Decide on B1 sample inverters

- Current plan: test candidate alternates (B1-01, B1-05, B1-16, B1-12) to compare
  data availability against current set (B1-04, B1-08, B1-13, B1-17).
- Constraint: maintain exactly 4 inverters per block.
- Action: run availability comparison on alternate IDs, then decide whether to swap.

### ⏸ Future: Integrate plant weather station telemetry

- Device `f81ed490-9cd2-11ef-9b7d-ef27f6a9abbc` provides 1-minute:
  - `wstn1_humidity` (RH%)
  - `wstn1_wind_speed` (m/s)
  - `wstn1_wind_dir` (bearing from north)
- These are first-party measurements and should be preferred over Solcast modelled data.
- Requires a new fetch script pass and integration into daily aggregation.
