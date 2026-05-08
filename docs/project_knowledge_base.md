# Comprehensive Project Knowledge Base: PV Soiling Loss Predictions

> **Purpose**: This mega-document consolidates all context, observations, hypotheses, latest findings, knowledge, inferences, suggestions, reasoning, and debunks for the soiling loss prediction project. It serves as the single source of truth for the project's data interpretation and EDA phase.

## 1. Project Overview & Objectives

This project builds AI/ML workflows for a **10–15 MW utility-scale solar plant in Sri Lanka** (~8.5°N latitude) to detect soiling (dust accumulation on panels), predict performance loss rates, and optimize cleaning decisions. Data is collected from **ThingsBoard** telemetry exports.

### Plant Context
- **Plant size**: 10–15 MW, 34 inverters total
- **Monitored inverters**: 6 tiered (3 B2 Tier-1 training + 3 B1 Tier-2 validation)
- **Climate**: Tropical — high humidity (78–98%), frequent rain (>40% of days), mean cloud opacity 36%
- **Cleaning campaigns**: Sep 20–30, Oct 20–30, Nov 20–30 (2025); none in Dec 2025 or Jan 2026
- **Known issue**: B1 channels have lower data availability due to shared communication path issues

### The Go/No-Go Objective
The EDA phase must determine if **soiling loss** is detectable and predictable from the available telemetry.

- **S1: Sawtooth**: Does performance loss increase gradually between cleanings/rain, then reset? (Target: ≥50% of dry spells show positive soiling slope)
- **S2: PM/Dust Correlation**: Does particulate matter correlate with observed performance decline? (Target: Statistically significant partial correlation)
- **S3: Rain Recovery**: Does rain measurably reduce the loss proxy? (Target: Wilcoxon p < 0.05 for post-rain loss < pre-rain loss)

**Current Overall Verdict**: **CONDITIONAL GO** (2/3 signals confirmed).

---

## 2. Pipeline Architecture & Feature Engineering

### 6-Stage Pipeline
1. **1_fetch**: Pulls from ThingsBoard API (Inverter power, irradiance, generation).
2. **2_organize**: Splits inverters into primary (6 tiered) and secondary files.
3. **3_preprocess**: Deterministic cleaning (dedup, sanity filtering, peak-hour filter 10AM-2PM). Assembles `daily_model_eda.csv`.
4. **4_audit**: Independent data quality validation.
5. **5_eda**: Tests the three go/no-go signals.
6. **[6_modeling]**: Future ML modeling.

### Core Unit Conventions — CRITICAL
- **Energy**: `subset_energy_j`, `t1_energy_j` (Joules) — Sum of (10-min AVG power × 600s), NOT cumulative meter.
- **Irradiance**: `irradiance_tilted_sum` (W/m²) — Raw sum of 15-min readings. Causes inflated absolute Performance Ratios, but relative trends are perfectly valid.
- **Normalized Output**: `normalized_output` (J/J) — Unitless ratio of energy to irradiance.
- **Loss Proxy**: `performance_loss_pct_proxy` (%) — All-cause loss vs a 30-day rolling 95th-percentile clean baseline. *(0=clean, 100=total loss)*.
- **Cycle Deviation**: `cycle_deviation_pct` (%) — Within-cycle deviation from cycle max, reset at every rain/cleaning. **(Best soiling metric)**.
- **PM10, PM25**: µg/m³. Usually, *cumulative* sums since the last rain (`cumulative_pm25_since_rain`) are used.

---

## 3. Latest Findings & Interpretations (The "Debunks & Reasoning")

Based on the deepest statistical analyses (including `llm_eda_summary.json`), we have revised several initial hypotheses.

### ⚠️ Corrected Hypothesis: The Zero-Inflation Floor Effect
**Debunk**: It was initially suspected that the old-source loss proxy was dominated by inverter trips, causing binary (0/100) behavior.
**Truth**: All 6 monitored inverters have **0% zero-output days** on HQ days. They are healthy. The 60.6% "zero loss" days in the loss proxy are a **mathematical baseline floor effect**. The formula `loss = 100 × (1 - normalized_output / rolling_95th_percentile_baseline)` yields zero whenever today's output meets or exceeds the baseline.
**Inference**: The loss proxy requires specialized zero-inflated modeling (e.g., hurdle models) or we should switch to predicting `cycle_deviation_pct`.

### ⚠️ Corrected Hypothesis: The PM10 Power Correlation Anomaly
**Debunk**: PM10 showed a confusing *positive* correlation with power at reference irradiance (more dust = more power?).
**Truth**: This is caused by **seasonal confounding**. PM10 is high during the dry season, which is also the high-irradiance/high-generation season. The dry season boosts average power output more than PM10 suppresses it.
**Inference**: When mathematically controlling for cloud opacity + temperature (partial correlations), the relationship normalizes. PM2.5 (especially *cumulative* PM2.5 since rain) is the true dominant predictor of within-cycle deviation, isolated from this seasonal effect.

### 💡 Discovery: Clear-Sky Analyzable (CSA) Days Reveal the True Signal
**Observation**: CSA days show a HIGHER mean loss (11.7%) than non-CSA days (6.7%).
**Reasoning**: This is physically correct. On non-CSA (cloudy) days, cloud-driven irradiance drops suppress both output and irradiance readings, masking panel degradation (loss ≈ 0). CSA days strip away the weather noise, allowing the true accumulated dirt to become visible.
**Inference**: CSA days (53 days, ~24% of the dataset) are the most informative days for training soiling models.

---

## 4. EDA Signal Results & Statistics

### S1: Sawtooth Detection ➔ PASS (Marginal)
- **Result**: Median soiling rate +0.738 %/day. 50% of qualifying dry spells show a positive slope.
- **Caveat**: Only 10 qualifying dry spells (≥3 days) exist because of frequent tropical rain. The statistical power is very low.

### S2: PM/Dust Correlation ➔ PASS (Strong)
- **Result**: Highest partial correlation is cumulative PM2.5 vs cycle deviation (partial r = +0.328, p<0.001).
- **Finding**: **PM2.5 × Days Dry** is the strongest feature interaction (r=0.439). The soiling effect of PM2.5 amplifies with elapsed dry time.

### S3: Rain Recovery ➔ FAIL
- **Result**: Event-study Wilcoxon p = 0.395.
- **Reasoning**: The all-cause loss proxy mixes soiling + clouds. Post-rain days are usually cloudy, which suppresses normalized output, completely masking the panel-cleaning benefit in the metrics. Furthermore, tropical light rain (<1mm) can *cement* dust rather than clean it.

---

## 5. Data Quality Diagnostics (DQ) Commentary

| Metric/Plot | Deep Analysis & Resolution |
|---|---|
| **DQ1: Irr vs Gen** | T1 gen vs onsite irr has low correlation (r=0.099) because T1 represents a small subset (3 inverters) over a truncated period (10AM-2PM). Full-plant correlation is much higher (r=0.512). Not a physics failure; it's a subset variance issue. |
| **DQ2: New Telemetry** | `daily_generated_electricity` shows high stability, but correlates poorly (r=-0.034) with old active-power integral. **Suggestion**: Run a parallel pipeline using the new metric, but expect loss of sub-daily resolution. |
| **DQ3: Gen/Irr Ratio** | Ratio varies wildly (CV 42.3%) but rarely aligns perfectly with pure irradiance fluctuations because the true daily meter smooths intra-day spikes. Monthly medians DO follow the expected dry/wet seasonal sawtooth. |
| **DQ4: Power@Ref Irr** | Confirmed the PM10 seasonal confounder. Plot gets noisy starting Jan 2026 due to seasonal shift away from the annual median reference irradiance. |
| **DQ5: Old vs New Source** | Old sources have wider variance (0-95%) but much of it is weather noise. New sources have narrower variance (10-80%) and lower correlations, but potentially better Signal-to-Noise Ratio (SNR). |
| **DQ6: Performance Index** | Median=0.643. Values >1.0 occur (12% of days) as a normal statistical effect of the 95th percentile baseline. Scatter plots vs soiling show a "triangle envelope" (upper bound capped by soiling). |

---

## 6. Recommended Next Steps & Modeling Strategy

### High Priority
1. **Target Variable Pivot**: Shift modeling focus from `performance_loss_pct_proxy` to `cycle_deviation_pct`. The proxy is heavily confounded by baseline calculations and zero-inflation. Cycle deviation correctly isolates within-cycle degradation.
2. **Handle Zero-Inflation**: If keeping the loss proxy, use hurdle/two-part modeling (predicting probability of zero loss, then modeling the magnitude of non-zero loss).
3. **Exploit Interactions**: Include `PM2.5 × Days Dry` as a primary engineered feature in all future ML models.
4. **Focus on CSA Days**: Treat the 53 CSA days as the gold-standard training instances where the pure soiling signal is actually visible.

### Medium Priority
1. **Parallel New-Source Pipeline**: Evaluate standardizing on `#21 Gen/Irr Ratio` using the recently collected full-day tracking data to see if it provides better SNR than the T1 subset.
2. **Quantile Regression**: Apply 90th percentile quantile regression for the Performance Index to capture real soiling envelope constraints, rather than simple linear regression which fails on the triangle distribution.
3. **Investigate Zero-Gen Days**: Investigate the 6 sunny days where `daily_generated_electricity` dropped to zero to determine if they are meter resets or real shutdowns.

### Documentation Feedback Addressed
> **User Question**: Is the explanation about `new_performance_index` exceeding 1.0 valid?
**Answer**: Yes, it is completely valid. The values > 1.0 are mathematically expected because the 95th percentile baseline inherently leaves ~5% of values above it, and changing conditions (e.g., temperature improvements, brief high irradiance spikes) can push today's value past a trailing historical baseline. The suggested mitigations (like using `min_periods=4` and forward-filling) are standard practices in sliding-window baselines.

---

## 7. Actions Taken & Technical Executions

During this EDA Deep Dive, the following pipeline enhancements were implemented based on observations:

*   **DQ1 Enhancements**: Normalized variables to [0, 1] range and added a 7-day rolling median overlay for `plot_irradiance_vs_generation`. Added a correlation matrix that combines T1+T2 generation (using `subset_energy_j`) vs irradiance to improve the signal-to-noise ratio and CSA markers.
*   **Zero-Gen Debugging**: Investigated 6 instances of sunny zero-gen days (`2025-02-18, 2025-05-09, 2025-07-08, 2025-07-14, 2025-09-07, 2026-01-03`). They had normal to high irradiance indicating equipment or communication shutdowns, aside from 2025-07-08 which had high cloud opacity. These are now explicitly excluded from the Transfer Readiness Tier and QA gated using a new `flag_zero_subset_gen` in `daily_features.py`.
*   **Sensor Drift/Recalibration Detection**: Added `flag_sensor_recalibrated` using a rolling 14-day median check comparing the `solcast_gti_peak_sum` vs `irradiance_tilted_sum` ratio. Step changes larger than 15% are flagged automatically in `daily_features.py`.
*   **Parallel New Telemetry Pipeline**: Configured `main()` in `soiling_signals.py` to seamlessly execute the three signal tests against the new daily telemetry (`new_performance_loss_pct_proxy`) if available, generating outputs separately in `plots_new_telemetry/` directory.
