# Domain Context and Concern Resolution

This document captures plant-specific operational context, modeling assumptions,
and how previously raised concerns are handled in the current pipeline.

## Operational Context (Plant-Specific)

- Site location: 8.561510736689941, 80.65921384406597
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
- Temperature correction is now implemented via `pvlib.temperature.sapm_cell()`
  in `scripts/daily_features.py`: `compute_temperature_corrected_pr()`.
  The `pr_temperature_corrected` column partially removes thermal effects
  from the performance loss proxy.

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

### 9) Ground irradiance sensor soiling detection

Concern:
- The ground irradiance sensor (`irradiance_tilted_sum`) may itself accumulate
  dirt over time. If the sensor is soiled, all normalized output and performance
  loss calculations are biased — the denominator is artificially low, making
  panels appear to perform better than they actually are (or masking real
  soiling trends).

Resolution:
- Cross-check Solcast satellite GTI (`solcast_gti_sum`) against the ground-sensor
  `irradiance_tilted_sum`. Persistent divergence where satellite reads
  consistently higher than the ground sensor suggests sensor soiling.
- Sudden convergence after a known cleaning campaign would further confirm this.
- Note: some divergence is expected due to measurement methodology differences
  (satellite model vs pyranometer). The diagnostic is based on *trend* in the
  ratio, not absolute level.

Where to implement:
- Check during EDA (bivariate plot of `solcast_gti_sum` vs
  `irradiance_tilted_sum` over time, and their ratio trend).
- If confirmed as a recurring issue, add a daily quality flag
  (e.g., `flag_sensor_soiling_suspect`) based on a rolling ratio threshold.

## Cross-Plant Inference Guidance

This repository is built on one plant but intended to inform others.

Rules:
- transfer quality-gate records before cross-plant inference,
- recalibrate thresholds for each target site,
- separate transferable patterns from plant-specific operational behavior.

## Open Items and Resolution Status

### ⚠️ Known: Ground irradiance sensor unit ambiguity

- The ground irradiance sensor values (labelled `W·s/m²` in the data dictionary)
  are actually sums of sub-minute W/m² readings produced by ThingsBoard's `SUM`
  aggregation over 15-minute windows — not true energy-density values.
- Cross-referencing against Solcast GTI (confirmed W/m² → W·s/m² via `gti_w_m2 *
  600s`), the ground sensor values are ~140× smaller than true W·s/m².
- **Impact on EDA:** `normalized_output` and `performance_loss_pct_proxy` use
  the ground sensor value as a ratio denominator. The **relative trends**
  (sawtooth patterns, cleaning jumps, soiling slopes) are scale-independent
  and unaffected by this issue.
- **Impact on PR:** Per-inverter `{inv}_pr` and `subset_pr` values are
  systematically inflated (~200×) because the PR formula assumes the irradiance
  is in true W·s/m². These values are useful for **relative trend analysis**
  (sawtooth shape, cleaning detection) but not for industry-standard benchmarking
  until the irradiance scale is calibrated.
- **Calibration path:** Use `solcast_gti_sum / irradiance_tilted_sum` ratio
  to derive a day-level correction factor, or switch the PR denominator to
  Solcast GTI for a properly-scaled reference PR.

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

### ✅ Resolved: Tropical weather contamination of soiling metrics

The site (~8.5°N, Sri Lanka) has tropical climate with high baseline cloud
cover (mean 36% opacity on HQ days), frequent rain (>40% of days), and
narrow temperature range (24-31°C). This creates three problems for soiling
analysis:

1. **Cloud-driven metric spikes**: Cloud opacity has r = -0.35 with the
   performance loss proxy. Cloudy days depress normalised output against the
   clear-sky baseline, producing false "soiling" spikes that are actually
   weather artefacts.
2. **Equipment failures passing quality filters**: 11 zero-output days
   (equipment shutdowns on sunny days) survived the original HQ filter,
   creating 100% loss proxy spikes.
3. **Rain carry-over**: Days immediately following rain are typically cloudy,
   masking any panel-cleaning benefit and making the rain recovery signal
   (Signal 3) fail in statistical tests.

**Mitigations applied**:

- `flag_zero_output`: New quality flag catches equipment shutdowns where
  output = 0 despite sufficient irradiance.
- Cloud-opacity guard on baseline: The rolling 95th-percentile baseline now
  excludes days with cloud > 40%, preventing cloudy days from inflating it.
- Clear-Sky Analyzable (CSA) filter: A boolean column `is_clear_sky_analyzable`
  marks 57 of 235 HQ days (24%) where cloud < 35%, rain < 1 mm, equipment OK,
  and >= 1 day since rain. On CSA days, cumulative dust features achieve
  statistically significant positive correlations with loss proxy
  (cumulative PM2.5: r = +0.36, p = 0.01; days since rain: r = +0.36,
  p = 0.01) that are hidden in the unfiltered data.
- EDA C-series plots visualise the contrast between all-HQ and CSA analyses.

**Efficiency factors quantified** (variance contribution to loss proxy):

| Factor | Impact | Notes |
|---|---|---|
| Cloud opacity | Dominant (~r = -0.35) | Drives most non-soiling variance |
| Equipment shutdown | 11 days at 100% loss | Now flagged |
| Temperature | Minor (r ~ -0.10) | Narrow tropical range limits effect |
| Rain carry-over cloud | Masks recovery signal | Causes Signal 3 failure |
| Irradiance sensor soiling | Unknown magnitude | Cross-check via Solcast ratio |

### ⏸ Future: Integrate plant weather station telemetry

- Device `f81ed490-9cd2-11ef-9b7d-ef27f6a9abbc` provides 1-minute:
  - `wstn1_humidity` (RH%)
  - `wstn1_wind_speed` (m/s)
  - `wstn1_wind_dir` (bearing from north)
- These are first-party measurements and should be preferred over Solcast modelled data.
- Requires a new fetch script pass and integration into daily aggregation.
