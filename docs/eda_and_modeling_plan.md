# EDA and Modeling Plan

This document captures the research direction, methodology decisions, go/no-go
signals, feature engineering needs, pipeline improvements, pvlib integration
strategy, and modeling approach hierarchy for the soiling loss prediction project.

It was produced after a methodology review comparing this pipeline against
industry-standard PV O&M soiling analysis practices and soiling research
literature.

## Research Objective

The EDA phase exists to confirm whether the following three predictions are
achievable with the data currently available:

1. **Soiling rate prediction** — how fast panels get dirty (% loss/day).
2. **Cleaning trigger recommendation** — when to clean, based on estimated
   recoverable energy exceeding cleaning cost.
3. **Daily soiling loss %** — the fraction of daily output lost specifically to
   soiling (as opposed to other all-cause performance deficits).

If the EDA confirms that soiling signal is extractable from the available
features, the project proceeds to modeling. If not, the research direction
should be re-evaluated before investing in model engineering.

## Go / No-Go Signals

The EDA must confirm three signals before modeling is worth pursuing. If all
three are visible, soiling is a dominant and predictable component of the
performance loss proxy. If none are visible, the proxy is likely dominated by
equipment issues and data gaps, and the research direction needs revision.

### Signal 1: Sawtooth pattern in performance loss proxy

Plot `performance_loss_pct_proxy` (or `t1_performance_loss_pct_proxy`) over
time. Look for:

- **Gradual upward drift** (soiling accumulation between events).
- **Sudden drops** at:
  - Rain events (`rain_day == True`, especially heavy ones).
  - Known cleaning campaigns (Sep 20-30, Oct 20-30, Nov 20-30 of 2025).

If this sawtooth is visible, soiling is a dominant component of the proxy.
Temperature and equipment effects do not produce sawtooth signatures.

### Signal 2: Correlation between PM/dust and loss rate

`pm10_mean` and `pm25_mean` should correlate positively with
`perf_loss_rate_14d_pct_per_day` (worsening rate). If dustier days lead to
faster performance decline, that is direct evidence of a soiling signal in the
data.

### Signal 3: Rain causes measurable recovery

Days after significant rainfall should show a drop in
`performance_loss_pct_proxy`. If `precipitation_total_mm > threshold` on the
previous day predicts a loss proxy decrease, that confirms the natural cleaning
mechanism is visible in the data.

## Pipeline Improvements Required Before EDA

Gaps identified from methodology comparison, ordered by priority.

### High priority

**Peak-hour filtering.** ✅ Implemented in `scripts/daily_features.py`:
`filter_peak_hours()`. Default window 10:00–14:00 local time, applied to both
inverter and irradiance records before daily aggregation. Disable with
`--no-peak-filter` flag in `data_preprocess.py`.

**Per-timestamp irradiance threshold.** ✅ Implemented in
`scripts/daily_features.py`: `filter_irradiance_threshold()`. Threshold is
derived from data at runtime (10th percentile of positive peak-hour irradiance
values), auto-tuning for the sensor's unit conventions. An explicit override
threshold can also be passed.

**Proper Performance Ratio.** ✅ Implemented in `scripts/daily_features.py`:
`compute_performance_ratio()`. Uses configurable `P_NOM_KWP` (placeholder:
75 kWp). PR = E_actual / (P_nom × H_POA / G_STC), producing a dimensionless
value. Replace `P_NOM_KWP` with confirmed value from asset owner.

### Medium priority

**Per-inverter daily metrics.** ✅ Implemented in `scripts/daily_features.py`:
`aggregate_per_inverter_daily()`. Computes per-inverter energy, PR, and
normalized output for all 6 tiered inverters. Enables per-inverter sawtooth
visualization and cleaning event detection during EDA.

**Temperature-corrected PR via pvlib.** ✅ Implemented in
`scripts/daily_features.py`: `compute_temperature_corrected_pr()`. Uses
`pvlib.temperature.sapm_cell()` with Solcast air temperature, wind speed, and
irradiance. Produces `pr_temperature_corrected` column.

**Financial/energy loss quantification.** Deferred — not required for EDA.
Compute recoverable MWh and estimated cost savings from cleaning, using the
rolling clean baseline as the reference. This is a downstream output needed
for the cleaning trigger recommendation.

### Retained design decisions

- **Keep the `rolling_clean_baseline` (30-day 95th percentile).** It is more
  robust and automated than the alternative approach of manually identifying
  peak PR after each cleaning event.
- **Peer-to-peer clean-vs-dirty inverter comparison is infeasible for this
  plant.** Progressive cleaning takes 10–30 days end-to-end, so there is no
  reliable window where one inverter is definitively "just cleaned" while an
  adjacent one is definitively "dirty." Block mismatch ratio partially captures
  this signal at a coarser level.

## Feature Engineering for EDA and Modeling

Features to derive from existing `daily_model_eda.csv` columns before or during
EDA. These are the features that soiling literature identifies as the strongest
predictors.

| Feature | Source | Description |
|---|---|---|
| `days_since_last_rain` | `rain_day` | Count of consecutive days since last `rain_day == True`. |
| `days_since_significant_rain` | `precipitation_total_mm` | Days since last day with precipitation above a threshold (e.g., 2–5 mm). |
| `cumulative_pm10_since_rain` | `pm10_mean`, `rain_day` | Running sum of `pm10_mean` reset to zero on each rain day. Dust exposure proxy. |
| `cumulative_pm25_since_rain` | `pm25_mean`, `rain_day` | Same for PM2.5. |
| `humidity_x_pm10` | `humidity_mean`, `pm10_mean` | Interaction feature: high humidity + high dust → cementation (sticky soiling harder to wash). |
| `wind_speed_10m_rolling_7d` | `wind_speed_10m_mean` | Rolling 7-day mean wind speed. |
| `month` | `day` | Calendar month (1–12) for seasonal encoding. |
| `season` | `day` | Season category derived from month (e.g., dry/wet or monsoon/inter-monsoon for this tropical site). |
| `pvlib_soiling_ratio` | Solcast rainfall, PM, tilt, cleaning schedule | Physics-based soiling ratio from `pvlib.soiling.hsu()` or `pvlib.soiling.kimber()`. |

## pvlib Integration Strategy

pvlib provides two soiling estimation modules that can produce a physics-based
soiling ratio from environmental data:

- `pvlib.soiling.hsu()` — uses rainfall, PM2.5, PM10, tilt angle, and cleaning
  interval to estimate a daily soiling ratio.
- `pvlib.soiling.kimber()` — uses rainfall and a soiling rate assumption to
  estimate soiling ratio with rain-triggered resets.

### Implementation plan

1. Compute a pvlib soiling ratio from Solcast data (rainfall from
   `precipitation_total_mm`, PM from `pm25_mean`/`pm10_mean`) plus site tilt
   angle and the documented cleaning schedule (Sep/Oct/Nov 2025 windows).
2. Compare the pvlib soiling estimate against `performance_loss_pct_proxy`. If
   they correlate well, this isolates the soiling component from other causes
   in the all-cause proxy.
3. Use the pvlib soiling ratio as a feature alongside environmental data in ML
   models (Tier 4 hybrid approach).
4. Use pvlib temperature correction (`pvlib.temperature` models) to partially
   remove thermal effects from the performance proxy, using Solcast
   `air_temp_mean`, `wind_speed_10m_mean`, and irradiance.

### Site parameters for pvlib

- Site location: 8.561510736689941, 80.65921384406597
- Tilt angle: to be confirmed from asset owner (assumed plane-of-array from
  irradiance sensor orientation).
- Azimuth: to be confirmed (likely south-facing for Northern Hemisphere at
  latitude ~8.5°N, but close to equator so may differ).

## Quality Gating for Training Data

When filtering data for model training, apply these rules:

- `transfer_quality_tier == "high"` — only high-confidence days.
- `flag_count == 0` — no quality flags raised.
- Equipment issues are already captured by `flag_low_output_high_irr` and
  `flag_block_mismatch`.
- **Sensor dirt detection**: compare Solcast satellite GTI (`solcast_gti_sum`)
  against ground-sensor `irradiance_tilted_sum`. Persistent divergence where
  satellite reads higher than ground suggests the ground irradiance sensor
  itself is dirty, biasing all normalized output calculations. This should be
  checked during EDA and potentially added as a daily quality flag.

## EDA Workflow

The 7-step workflow, refined for this plant's soiling detection context.
Input table: `artifacts/preprocessed/daily_model_eda.csv` (overlap-filtered,
peak-hour filtered 10:00–14:00).

1. **Univariate distributions.**
   Distribution of `t1_performance_loss_pct_proxy`, `precipitation_total_mm`,
   `pm10_mean`. Check for skewness, outliers, and structural zeros.

2. **Bivariate relationships.**
   - Scatter: `pm10_mean` vs `t1_performance_loss_pct_proxy`.
   - Time-series: loss proxy with rain events and cleaning campaigns overlaid.
   - Scatter: `solcast_gti_sum` vs `irradiance_tilted_sum` (sensor dirt check).

3. **Soiling cycle detection.**
   Identify sawtooth pattern in loss proxy: gradual increase (soiling
   accumulation) punctuated by sudden drops (rain cleaning or wash campaigns).
   Overlay known cleaning windows (Sep/Oct/Nov 2025) and heavy rain days.

4. **Feature importance.**
   Correlation matrix of all weather/environmental features vs
   `t1_performance_loss_pct_proxy`. Include the engineered features
   (`days_since_last_rain`, `cumulative_pm10_since_rain`, `humidity_x_pm10`).

5. **Temporal stationarity.**
   Compare loss proxy distributions across months and seasons. Check whether
   soiling rates differ between dry and wet periods (expected: faster soiling
   in dry months, resets during monsoon).

6. **Tier validation.**
   Compare `t1_*` and `t2_*` loss trends. If a soiling model trained on Tier-1
   (B2) also predicts Tier-2 (B1) patterns, it generalizes across blocks.
   The existing `tier_loss_correlation` (median r = 0.98) is a strong indicator.

7. **Quality gating.**
   Visualize `transfer_quality_score` distribution. Decide final training
   filter threshold. Check how many days survive
   `transfer_quality_tier == "high"` AND `flag_count == 0`.

## Modeling Approach Hierarchy

For reference. Not yet implemented — these are the candidate approaches to
pursue after EDA confirms viability.

### Tier 1: Physics-based (pvlib)

Use pvlib soiling estimate as a standalone baseline prediction. Cheapest to
implement and provides a physically grounded reference.

### Tier 2: Feature-engineered ML (XGBoost / LightGBM)

- Target: `t1_performance_loss_pct_proxy` (or its 14-day rate of change).
- Features: `cumulative_pm10_since_rain`, `days_since_rain`,
  `humidity_x_pm10`, `wind_speed_10m_rolling_7d`, `season`, `month`.
- Train on Tier-1 (B2), validate on Tier-2 (B1).
- ~300 clean high-quality days is sufficient for gradient boosted trees.

### Tier 3: Time-series with exogenous regressors

Prophet or ARIMA with weather regressors on the loss proxy. Can explicitly
model soiling accumulation trend + seasonal effects + cleaning/rain resets.
Good for forecasting "when will loss reach X% if no cleaning?"

### Tier 4: Hybrid physics + ML

Use pvlib soiling estimate as a feature alongside environmental data. Let ML
learn the residual between physics prediction and actual loss. Best expected
accuracy but most complex.

## Accuracy Expectations

Realistic ranges based on published soiling studies with similar data:

| Prediction Target | Expected R-squared | Notes |
|---|---|---|
| Soiling rate (% loss/day) | 0.4–0.7 | Strong environmental feature set |
| Cleaning trigger (when to clean) | Achievable | Threshold/policy problem, not pure regression |
| Daily soiling loss % | 0.3–0.5 | Without physical model correction; higher with pvlib hybrid |

These ranges assume the go/no-go signals are confirmed during EDA. If the
sawtooth is not visible or PM/rain correlations are weak, the ceiling is lower.
