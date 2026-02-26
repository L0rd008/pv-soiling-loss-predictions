# EDA Feature Guide - `daily_model_eda.csv`

Comprehensive reference for all 80 features in the EDA-ready daily feature table. Each column is documented with its meaning, type, unit, value range, null behavior, and relevance to soiling detection and prediction.

**Table shape:** 383 rows x 80 columns (days with all 3 core sources present)

---

## 1. Index

| # | Column | Group |
|---|---|---|
| 0 | `day` | [Time](#2-time) |
| 1-6 | `subset_energy_j` ... `inverter_coverage_ratio` | [Combined Inverter](#3-combined-inverter-aggregates) |
| 7-12 | `b1_energy_j` ... `block_mismatch_ratio_rolling_median` | [Block-Level](#4-block-level-b1-vs-b2) |
| 13-15 | `t1_energy_j` ... `t1_data_availability` | [Tier-1 (B2 Training)](#5-tier-1-b2-training) |
| 16-18 | `t2_energy_j` ... `t2_data_availability` | [Tier-2 (B1 Validation)](#6-tier-2-b1-validation) |
| 19-22 | `irradiance_horizontal_sum` ... `irradiance_coverage_ratio` | [Irradiance](#7-irradiance) |
| 23-28 | `daily_generation_j_latest` ... `generation_intraday_spread_j` | [Generation](#8-generation) |
| 29-46 | `pm25_mean` ... `dominant_weather` | [Solcast Weather / Air Quality](#9-solcast-weather--air-quality) |
| 47-49 | `solcast_ghi_sum` ... `solcast_dni_mean` | [Solcast Irradiance](#10-solcast-satellite-irradiance) |
| 50-52 | `subset_energy_mwh` ... `plant_to_subset_energy_ratio` | [Derived Energy Ratios](#11-derived-energy-ratios) |
| 53-57 | `normalized_output` ... `perf_loss_rate_14d_pct_per_day` | [Combined Performance](#12-combined-performance-metrics) |
| 58-62 | `t1_normalized_output` ... `t1_perf_loss_rate_14d_pct_per_day` | [Tier-1 Performance](#13-tier-1-performance-metrics) |
| 63-67 | `t2_normalized_output` ... `t2_perf_loss_rate_14d_pct_per_day` | [Tier-2 Performance](#14-tier-2-performance-metrics) |
| 68-70 | `tier_loss_correlation` ... `tier_agreement_flag` | [Cross-Tier Correlation](#15-cross-tier-correlation) |
| 71 | `in_common_overlap` | [Overlap Gate](#16-overlap-gate) |
| 72-76 | `flag_sensor_suspect_irradiance` ... `flag_count` | [Quality Flags](#17-quality-flags) |
| 77-79 | `transfer_quality_score` ... `cross_plant_inference_ready` | [Transfer Readiness](#18-transfer-readiness) |

---

## 2. Time

| Column | Type | Example | Description |
|---|---|---|---|
| `day` | `object` (date string) | `2025-01-01` | Calendar date. Index key - one row per day. Parse with `pd.to_datetime()`. |

**Soiling relevance:** Time axis for all trend and seasonal analyses. Soiling accumulates between rain events, so ordering matters.

---

## 3. Combined Inverter Aggregates

Aggregated from all 6 tiered inverters' 10-minute Active Power readings.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `subset_energy_j` | float64 | Joules | 1.1e9 - 4.5e10 | 0% | Total daily energy from all 6 inverters (sum power x 600s). |
| `subset_power_w_p95` | float64 | Watts | 91k - 1.7M | 0% | 95th-percentile of 10-min combined power. Captures near-peak output. |
| `subset_data_availability_mean` | float64 | ratio 0-1 | 0.46 - 1.0 | 0% | Mean row-level power completeness (fraction of inverters reporting). |
| `subset_data_availability_p10` | float64 | ratio 0-1 | 0.0 - 1.0 | 0% | 10th-percentile availability. Detects days with sustained gaps. |
| `inverter_records` | float64 | count | 43 - 144 | 0% | Number of 10-min records received. Expected: 144 (24h x 6). |
| `inverter_coverage_ratio` | float64 | ratio 0-1 | 0.30 - 1.0 | 0% | `inverter_records / 144`, clipped at 1.0. |

**Soiling relevance:**
- `subset_energy_j` is the **primary target signal** - declining energy relative to irradiance indicates soiling.
- `subset_data_availability_mean` gates data quality - days below 0.5 should be treated with caution.
- `subset_power_w_p95` captures peak achievable output, which drops as soiling accumulates.

---

## 4. Block-Level (B1 vs B2)

Separated by inverter naming prefix. B2 = Tier-1 (training), B1 = Tier-2 (validation).

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `b1_energy_j` | float64 | Joules | 9.0e7 - 2.3e10 | 0% | B1 block total daily energy. |
| `b2_energy_j` | float64 | Joules | 1.0e9 - 2.3e10 | 0% | B2 block total daily energy. |
| `b1_data_availability` | float64 | ratio 0-1 | 0.10 - 1.0 | 0% | Fraction of B1 power values non-null. |
| `b2_data_availability` | float64 | ratio 0-1 | 0.84 - 1.0 | 0% | Fraction of B2 power values non-null. Higher than B1. |
| `block_mismatch_ratio` | float64 | ratio | 0.09 - 1.29 | 0% | `b1_energy_j / b2_energy_j`. Near 1.0 = balanced. |
| `block_mismatch_ratio_rolling_median` | float64 | ratio | 0.96 - 1.03 | 1% | 14-day rolling median of mismatch ratio. Smoothed baseline. |

**Soiling relevance:**
- `block_mismatch_ratio` can reveal **differential soiling** between blocks (if one block is cleaned but not the other).
- Sudden mismatch deviations from the rolling median flag anomalous events (cleaning, shading, equipment faults).
- B2 has consistently higher availability (0.84-1.0) than B1 (0.10-1.0), validating the tiered strategy.

---

## 5. Tier-1 (B2 Training)

Computed from **B2-08, B2-13, B2-17 only**. These are the high-availability training inverters.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `t1_energy_j` | float64 | Joules | 1.0e9 - 2.3e10 | 0% | Same as `b2_energy_j` - Tier-1 is the B2 block. |
| `t1_power_w_p95` | float64 | Watts | 49k - 872k | 0% | 95th-percentile power for Tier-1 inverters. |
| `t1_data_availability` | float64 | ratio 0-1 | 0.77 - 1.0 | 0% | Tier-1 data availability. Never below 0.77. |

**Soiling relevance:**
- **Use these for model training** - highest data quality and fewest gaps.
- `t1_data_availability` >= 0.77 on all days means Tier-1 data is always trustworthy.

---

## 6. Tier-2 (B1 Validation)

Computed from **B1-08, B1-01, B1-13 only**. Cross-block validation set.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `t2_energy_j` | float64 | Joules | 9.0e7 - 2.3e10 | 0% | Tier-2 daily energy. |
| `t2_power_w_p95` | float64 | Watts | 42k - 896k | 0% | 95th-percentile power for Tier-2. |
| `t2_data_availability` | float64 | ratio 0-1 | 0.10 - 1.0 | 0% | B1 availability - lower and more variable than Tier-1. |

**Soiling relevance:**
- **Use for validation** - if a soiling model trained on Tier-1 also predicts Tier-2 loss patterns, the model generalizes.
- Low availability days (t2 < 0.5) should be excluded from validation metrics.

---

## 7. Irradiance

Aggregated from 15-minute SUM irradiance records.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `irradiance_horizontal_sum` | float64 | W/m^2 (daily sum) | 0 - 413k | 0% | Global Horizontal Irradiance daily total. |
| `irradiance_tilted_sum` | float64 | W/m^2 (daily sum) | 0 - 434k | 0% | Tilted (plane-of-array) irradiance daily total. **Primary irradiance metric.** |
| `irradiance_records` | float64 | count | 4 - 96 | 0% | 15-min records received. Expected: 96 (24h x 4). |
| `irradiance_coverage_ratio` | float64 | ratio 0-1 | 0.04 - 1.0 | 0% | `irradiance_records / 96`. |

**Soiling relevance:**
- `irradiance_tilted_sum` is the **denominator for normalized output** - the single most important feature for soiling isolation.
- Days with near-zero irradiance (overcast/rainy) are natural cleaning events - rain washes panels.
- Low `irradiance_coverage_ratio` makes the irradiance sum unreliable and should be flagged.

---

## 8. Generation

Plant-level cumulative energy meter readings, aggregated to daily.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `daily_generation_j_latest` | float64 | Joules | 0 - 3.2e11 | 0% | Last meter reading of the day. Preferred daily value. |
| `daily_generation_j_max` | float64 | Joules | 0 - 3.2e11 | 0% | Maximum meter reading of the day (fallback). |
| `daily_generation_j_min` | float64 | Joules | 0 - 3.2e11 | 0% | Minimum meter reading. |
| `generation_records` | float64 | count | 1 - 2603 | 0% | Intraday records received. Varies wildly. |
| `daily_generation_j` | float64 | Joules | 0 - 3.2e11 | 0% | Best daily energy estimate: `latest` if available, else `max`. |
| `generation_intraday_spread_j` | float64 | Joules | 0 - 2.5e11 | 0% | `max - min`. High spread on days with many intraday records. |

**Soiling relevance:**
- `daily_generation_j` is the **plant-level ground truth** - covers all 34 inverters, not just our 6.
- `plant_to_subset_energy_ratio` (col 52) validates that our subset tracks the full plant.
- Large `generation_intraday_spread_j` indicates days with irregular reporting.

---

## 9. Solcast Weather / Air Quality

Daily weather and air quality from Solcast satellite data.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `pm25_mean` | float64 | ug/m^3 | 3.2 - 47.7 | 0% | Daily mean PM2.5 concentration. |
| `pm25_max` | float64 | ug/m^3 | 4.4 - 72.9 | 0% | Daily max PM2.5. |
| `pm10_mean` | float64 | ug/m^3 | 4.4 - 138.5 | 0% | Daily mean PM10 (coarse dust). |
| `pm10_max` | float64 | ug/m^3 | 5.8 - 261.4 | 0% | Daily max PM10. |
| `precipitation_total_mm` | float64 | mm | 0 - 161.6 | 0% | Total daily precipitation. |
| `rain_day` | bool | - | True/False | 0% | Whether any precipitation occurred. |
| `humidity_mean` | float64 | % | 61.5 - 98.6 | 0% | Daily mean relative humidity. |
| `humidity_max` | float64 | % | 77.8 - 100.0 | 0% | Daily max humidity. |
| `wind_speed_10m_mean` | float64 | m/s | 0.8 - 7.9 | 0% | Mean wind speed at 10m height. |
| `wind_speed_10m_max` | float64 | m/s | 1.6 - 9.8 | 0% | Max wind speed at 10m. |
| `wind_speed_100m_mean` | float64 | m/s | 1.3 - 12.3 | 0% | Mean wind speed at 100m. |
| `dewpoint_mean` | float64 | C | 19.1 - 25.3 | 0% | Daily mean dewpoint temperature. |
| `air_temp_mean` | float64 | C | 23.6 - 30.7 | 0% | Daily mean air temperature. |
| `air_temp_min` | float64 | C | 19.2 - 27.6 | 0% | Daily minimum temperature. |
| `air_temp_max` | float64 | C | 25.5 - 37.0 | 0% | Daily maximum temperature. |
| `cloud_opacity_mean` | float64 | % | 0.5 - 85.1 | 0% | Mean cloud opacity. |
| `pressure_mean` | float64 | hPa | 985.6 - 996.6 | 0% | Mean surface pressure. |
| `dominant_weather` | object | - | text categories | 0% | Dominant weather category (e.g., `RAIN`, `PARTLY CLOUDY`, `CLEAR`). |

**Soiling relevance - these are critical predictors:**
- **`pm25_mean` / `pm10_mean`**: Direct proxy for airborne particulates that cause soiling. High PM -> faster soiling.
- **`precipitation_total_mm` / `rain_day`**: Rain is the primary **natural cleaning** mechanism. Soiling resets after heavy rain.
- **`humidity_mean`**: High humidity + dust -> cementation (sticky soiling harder to wash). Key interaction feature.
- **`wind_speed_10m_mean`**: High wind resuspends dust (reduces soiling) but can also deposit more (direction-dependent).
- **`dewpoint_mean`**: Condensation overnight glues dust to panels -> accelerated soiling.
- **`air_temp_mean`**: Temperature affects soiling adhesion and panel efficiency.
- **`cloud_opacity_mean`**: Cloudy days have lower irradiance, making soiling impact harder to isolate.
- **`dominant_weather`**: Categorical feature for weather regime segmentation.

> [!TIP]
> **Best soiling predictors** from literature: PM10 (deposition), precipitation (cleaning), humidityxPM interaction (cementation), and days-since-last-rain (accumulation proxy).

---

## 10. Solcast Satellite Irradiance

Satellite-derived irradiance from Solcast (independent of ground sensors).

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `solcast_ghi_sum` | float64 | J/m^2/day | 2.9M - 26.3M | 0% | Global Horizontal Irradiance (satellite). |
| `solcast_gti_sum` | float64 | J/m^2/day | 2.8M - 28.1M | 0% | Global Tilted Irradiance (satellite). Compare with ground-measured `irradiance_tilted_sum`. |
| `solcast_dni_mean` | float64 | W/m^2 | 0 - 369.5 | 0% | Direct Normal Irradiance daily mean. |

**Soiling relevance:**
- **Cross-validate ground irradiance sensors** - if ground tilted and Solcast GTI diverge, the ground sensor may be dirty too.
- `solcast_gti_sum` can serve as an **alternative denominator** for normalized output on days where ground sensors are unreliable.

---

## 11. Derived Energy Ratios

Convenience conversions and cross-validation ratios.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `subset_energy_mwh` | float64 | MWh | 0.30 - 12.6 | 0% | `subset_energy_j / 3.6e9`. Human-readable energy. |
| `generation_mwh` | float64 | MWh | 0 - 90.0 | 0% | `daily_generation_j / 3.6e9`. Plant-level energy. |
| `plant_to_subset_energy_ratio` | float64 | ratio | 0 - 197.4 | 0% | `generation_mwh / subset_energy_mwh`. Expected ~5.7 (34/6 inverters). |

**Soiling relevance:**
- `plant_to_subset_energy_ratio` should be roughly constant (~5-7). Deviations indicate either our subset or the plant meter is anomalous.
- Extreme values (0 or >10) indicate data quality issues on one side.

---

## 12. Combined Performance Metrics

Computed from **all 6 inverters combined** energy vs. irradiance.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `normalized_output` | float64 | J/J (dimensionless) | 27k - 500k | 8.9% | `subset_energy_j / irradiance_tilted_sum`. NaN when irradiance is below baseline threshold. Clipped at 500k. |
| `normalized_output_14d_median` | float64 | J/J | 77k - 500k | 6.0% | 14-day rolling median of normalized output. Smoothed trend. |
| `rolling_clean_baseline` | float64 | J/J | 104k - 163k | 8.1% | 95th percentile of normalized output on clear days (rolling 30-day window). Represents "best achievable" output. |
| `performance_loss_pct_proxy` | float64 | % | 0 - 80.9 | 13.8% | `100 x (1 - normalized_output / rolling_clean_baseline)`. All-cause loss including soiling. Clipped [0, 100]. |
| `perf_loss_rate_14d_pct_per_day` | float64 | %/day | -5.8 - 5.8 | 22.7% | 14-day loss acceleration. Positive = worsening; negative = improving (rain cleaning). |

**Soiling relevance - these are the core target variables:**
- **`performance_loss_pct_proxy`** is the **primary soiling signal**. A slow upward trend = soiling accumulation; sudden drop = cleaning event.
- **`perf_loss_rate_14d_pct_per_day`** captures **soiling velocity** - how fast panels are getting dirtier.
- **`rolling_clean_baseline`** defines "what output should be if panels were clean". Seasonal variation is expected.
- NaN values occur on low-irradiance days where normalized output is unreliable.

> [!IMPORTANT]
> `performance_loss_pct_proxy` is an **all-cause** deficit, not pure soiling. It includes temperature effects, equipment degradation, shading, and sensor errors. EDA must isolate soiling from these confounders.

---

## 13. Tier-1 Performance Metrics

Same calculations as [Section 12](#12-combined-performance-metrics), but using **Tier-1 (B2) energy only**.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `t1_normalized_output` | float64 | J/J | 14k - 297k | 8.9% | Tier-1 specific normalized output. |
| `t1_normalized_output_14d_median` | float64 | J/J | 41k - 254k | 6.0% | 14-day rolling median. |
| `t1_rolling_clean_baseline` | float64 | J/J | 51k - 81k | 8.1% | Tier-1 clean reference. |
| `t1_performance_loss_pct_proxy` | float64 | % | 0 - 80.7 | 13.8% | Tier-1 all-cause loss. **Best training target.** |
| `t1_perf_loss_rate_14d_pct_per_day` | float64 | %/day | -5.8 - 5.8 | 22.7% | Tier-1 soiling velocity. |

**Soiling relevance:**
- **These are the recommended training targets** - Tier-1 has the highest and most consistent data availability (0.77-1.0).
- Use `t1_performance_loss_pct_proxy` as the label your model learns to predict.

---

## 14. Tier-2 Performance Metrics

Same calculations using **Tier-2 (B1) energy only**.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `t2_normalized_output` | float64 | J/J | 13k - 300k | 8.9% | Tier-2 specific normalized output. |
| `t2_normalized_output_14d_median` | float64 | J/J | 38k - 253k | 6.0% | 14-day rolling median. |
| `t2_rolling_clean_baseline` | float64 | J/J | 52k - 83k | 8.1% | Tier-2 clean reference. |
| `t2_performance_loss_pct_proxy` | float64 | % | 0 - 81.1 | 13.8% | Tier-2 all-cause loss. |
| `t2_perf_loss_rate_14d_pct_per_day` | float64 | %/day | -5.8 - 5.8 | 22.7% | Tier-2 soiling velocity. |

**Soiling relevance:**
- **Validation set** - if your model predicts T2 loss from T1-trained weights, it generalizes across blocks.
- T2 and T1 are highly correlated (r = 0.98), confirming soiling is plant-wide.

---

## 15. Cross-Tier Correlation

Measures agreement between Tier-1 and Tier-2 performance loss.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `tier_loss_correlation` | float64 | Pearson r | 0.67 - 1.0 | 10.4% | Rolling 30-day correlation between T1 and T2 loss proxies. Median: **0.98**. |
| `tier_loss_delta` | float64 | percentage points | -61.5 - 17.4 | 13.8% | `t1_loss - t2_loss`. Positive = T1 cleaner than T2. |
| `tier_agreement_flag` | bool | - | True/False | 0% | Both tiers trending same direction over 7 days. |

**Soiling relevance:**
- `tier_loss_correlation` near 1.0 **confirms plant-wide soiling** (not localized to one block).
- `tier_loss_delta` near 0 means both blocks soil at the same rate - useful for validating that any detected soiling is real and not a data artifact.
- `tier_agreement_flag = False` days may indicate localized events (partial cleaning, shading, equipment faults).

---

## 16. Overlap Gate

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `in_common_overlap` | bool | - | Always `True` in this file | 0% | Day has non-null `subset_energy_j`, `irradiance_tilted_sum`, and `daily_generation_j`. |

**Note:** In `daily_model_eda.csv` this is always `True` by construction (the file is filtered to overlap-valid days). In `daily_model_input.csv` it can be `False`.

---

## 17. Quality Flags

Boolean flags for data quality issues. Flags use **Tier-1 (B2) fields** preferentially, so B1 data gaps don't inflate flag counts.

| Column | Type | Range | Nulls | Description |
|---|---|---|---|---|
| `flag_sensor_suspect_irradiance` | bool | True/False | 0% | Low irradiance but non-trivial Tier-1 output -> possible sensor fault. |
| `flag_coverage_gap` | bool | True/False | 0% | Inverter or irradiance coverage below 30%. |
| `flag_block_mismatch` | bool | True/False | 0% | B1/B2 ratio deviates >15% from rolling median. |
| `flag_low_output_high_irr` | bool | True/False | 0% | Tier-1 normalized output < 70% of 14-day median on a high-irradiance day. |
| `flag_count` | int64 | 0 - 2 | 0% | Sum of all boolean flags above. |

**Soiling relevance:**
- `flag_low_output_high_irr` is the most direct **soiling alert** - high sun but low output.
- `flag_count >= 2` days should likely be excluded from model training.
- `flag_sensor_suspect_irradiance` indicates days where normalized output is unreliable.

---

## 18. Transfer Readiness

Composite quality scoring for cross-plant model transfer.

| Column | Type | Unit | Range | Nulls | Description |
|---|---|---|---|---|---|
| `transfer_quality_score` | float64 | points 0-100 | 20 - 100 | 0% | Starts at 100, penalized for flags and gaps. Uses Tier-1 fields. Median: 100. |
| `transfer_quality_tier` | object | category | `low` / `medium` / `high` | 0% | `>=80 -> high`, `>=60 -> medium`, `<60 -> low`. |
| `cross_plant_inference_ready` | bool | - | True/False | 0% | `high` or `medium` tier -> ready for cross-plant use. |

**Soiling relevance:**
- Filter to `transfer_quality_tier == "high"` for the most reliable training data.
- `cross_plant_inference_ready == True` marks days suitable for training a model transferable to other plants.

---

## Null Summary

| Null % | Columns | Reason |
|---|---|---|
| **0%** | 62 columns | Complete data - always present on overlap days. |
| **1%** | `block_mismatch_ratio_rolling_median` | 14-day rolling window warm-up. |
| **6-9%** | `normalized_output`, `*_14d_median`, `rolling_clean_baseline` | Low-irradiance days where normalization is undefined. |
| **10%** | `tier_loss_correlation` | 30-day rolling window warm-up. |
| **14%** | `*_performance_loss_pct_proxy`, `tier_loss_delta` | Requires both normalized output and baseline. |
| **23%** | `*_perf_loss_rate_14d_pct_per_day` | Requires 14-day diff + baseline. |

> [!NOTE]
> NaN values in performance columns are **structurally expected** - they occur on low-irradiance days where computing normalized output would be misleading. Do not impute - instead, consider these as "no soiling signal available" days.

---

## Recommended EDA Workflow

This workflow has been implemented as `scripts/eda_soiling_signals.py`.
See `docs/pipeline_replication/05_eda_soiling_signals.md` for run
instructions and `docs/eda_output_interpretation.md` for how to read
each output plot.

1. **Univariate:** Distribution of `t1_performance_loss_pct_proxy`, `precipitation_total_mm`, `pm10_mean`
2. **Bivariate:** Scatter `pm10_mean` vs. `t1_performance_loss_pct_proxy`; time-series of loss proxy with rain events overlaid
3. **Soiling cycle detection:** Identify sawtooth pattern - gradual loss increase -> sudden drop (rain cleaning)
4. **Feature importance:** Correlation matrix of weather features vs. `t1_performance_loss_pct_proxy`
5. **Temporal stationarity:** Compare loss distributions across months/seasons
6. **Tier validation:** Compare `t1_*` and `t2_*` loss trends to confirm generalizability
7. **Quality gating:** Visualize `transfer_quality_score` distribution and decide training filter threshold
