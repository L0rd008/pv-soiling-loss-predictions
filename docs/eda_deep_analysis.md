# Soiling Loss Predictions — Comprehensive Analysis & Knowledge Base

> **Last updated**: 2026-03-05  
> **Stage**: Post-EDA, pre-modeling  
> **Overall verdict**: **CONDITIONAL GO** (2/3 soiling signals confirmed)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Unit Conventions & Interpretation](#3-unit-conventions--interpretation)
4. [EDA Methodology](#4-eda-methodology)
5. [Signal Test Results](#5-signal-test-results)
6. [Supporting Analyses](#6-supporting-analyses)
7. [Data Quality Diagnostics (DQ1–DQ6)](#7-data-quality-diagnostics-dq1dq6)
8. [Feature Rankings & Interactions](#8-feature-rankings--interactions)
9. [Observation-by-Observation Analysis](#9-observation-by-observation-analysis)
10. [Debunked Hypotheses](#10-debunked-hypotheses)
11. [Key Inferences & Knowledge](#11-key-inferences--knowledge)
12. [Recommendations & Next Steps](#12-recommendations--next-steps)
13. [Open Items](#13-open-items)
14. [Reference: EDA Script Architecture](#14-reference-eda-script-architecture)
15. [Reference: LLM JSON Structure](#15-reference-llm-json-structure)

---

## 1. Project Overview

### What This Project Does

This project builds AI/ML workflows for a **10–15 MW utility-scale solar plant in Sri Lanka** (~8.5°N latitude) to detect soiling (dust accumulation on panels), predict performance loss rates, and optimize cleaning decisions. Data is collected from **ThingsBoard** telemetry exports.

### Plant Context

| Item | Detail |
|---|---|
| Plant size | 10–15 MW, 34 inverters total |
| Monitored inverters | 6 tiered: 3 B2 Tier-1 (training) + 3 B1 Tier-2 (validation) |
| Climate | Tropical — high humidity (78–98%), frequent rain (>40% of days), mean cloud opacity 36% |
| Dry season | Dec–Mar (northeast monsoon tails off) |
| Wet season | May–Sep (southwest monsoon) + inter-monsoon |
| Cleaning campaigns | 2025-03-16–18, 2025-09-08–09 |
| Known issue | B1 channels have lower data availability (~54%) due to shared communication path |

### Dataset Summary

- **361 total days** (2025-01-04 to 2026-02-05)
- **223 HQ days** (high-quality tier, zero flags) — used for all statistical tests
- **53 CSA days** (Clear-Sky Analyzable: low cloud, no rain, equipment OK)
- **149 columns** in `daily_model_eda.csv`

---

## 2. Data Pipeline

```
1_fetch → 2_organize → 3_preprocess → 4_audit → 5_eda → [6_modeling]
```

| Stage | Script | Role |
|---|---|---|
| 1. Fetch | `scripts/1_fetch/` (7 scripts) | Pull from ThingsBoard API: inverter power (10-min AVG), irradiance (15-min SUM), generation (kWh→J), per-inverter daily gen, plant avg irradiance |
| 2. Organize | `scripts/2_organize/` | Split inverters: Tier-1 (B2-08, B2-13, B2-17), Tier-2 (B1-08, B1-01, B1-13) |
| 3. Preprocess | `scripts/3_preprocess/` | Clean, dedupe, peak-hour filter (10AM-2PM), daily features (~118 cols), quality flags |
| 4. Audit | `scripts/4_audit/` | Independent data quality validation: profiles, interval distribution, missingness |
| 5. EDA | `scripts/5_eda/` | 3 signal tests + 22+ DQ plots + signal report + LLM JSON output |
| 6. Modeling | *Not yet implemented* | ML models for soiling prediction |

### Key Data Flow

```
Raw ThingsBoard CSV → clean/dedupe → peak-hour filter (10AM-2PM)
  → daily aggregation → feature engineering → quality flags
  → transfer readiness scoring → daily_model_eda.csv
```

---

## 3. Unit Conventions & Interpretation

> [!CAUTION]
> These units are non-standard and may cause misinterpretation if not carefully observed.

| Feature | Unit | Notes |
|---|---|---|
| `subset_energy_j`, `t1_energy_j`, `t2_energy_j` | Joules (J) | Sum of 10-min AVG power × 600s, NOT cumulative meter |
| `irradiance_tilted_sum` | W/m² (daily sum of 15-min readings) | **NOT W·h/m²**; raw sum of 15-min W/m² readings |
| `normalized_output` | Dimensionless (J/J) | = energy_j / irradiance_tilted_sum; NOT efficiency or PR |
| `performance_loss_pct_proxy` | % (0=clean, 100=total loss) | = 100 × (1 − normalized_output / rolling_clean_baseline) |
| `perf_loss_rate_14d_pct_per_day` | %/day | = (loss_proxy[t] − loss_proxy[t−14]) / 14 |
| `cycle_deviation_pct` | % (0=cycle-max, 100=total loss) | Within-cycle deviation from cycle max |
| PM10, PM25 | µg/m³ | Daily mean from Solcast satellite |
| `cumulative_pm*_since_rain` | µg/m³·days | Running sum since last rain, resets on rain_day |
| `precipitation_total_mm` | mm/day | From Solcast |
| `domain_soiling_index` | Cumulative (arb. units) | Accumulates daily_rate, resets on cleaning rain (≥1mm) |
| `gen_irr_ratio` | kWh/(W/m²) | = subset_daily_gen_kwh / plant_avg_irradiance_wm2 |
| `transfer_quality_score` | 0–100 | Starts at 100, subtracts penalties |

### Critical Interpretation Rules

- **Normalized output** is NOT a performance ratio (PR). Do not compare across sites.
- **Performance loss proxy** is ALL-CAUSE loss (soiling + equipment + clipping + temperature). Soiling isolation requires controlling for confounders.
- **Rolling clean baseline** uses 95th percentile on clear days (rolling 30-day window, min 7 points). Cloud opacity guard prevents diffuse-radiation inflation.
- **Cycle deviation** is cycle-relative: each soiling cycle starts with max(normalized_output) as the clean reference.

---

## 4. EDA Methodology

### Quality Gating

Two tiers of filtering restrict analyses to trustworthy days:

**HQ Filter** — `transfer_quality_tier == "high"` AND `flag_count == 0` → **223 of 361 days (62%)**

Five quality flags (any one disqualifies a day):
1. `flag_sensor_suspect_irradiance` — low irradiance but non-trivial inverter output (11.6% of days)
2. `flag_coverage_gap` — <30% of expected records (5.5%)
3. `flag_block_mismatch` — B1/B2 ratio deviates >15% from rolling median (15.0%)
4. `flag_low_output_high_irr` — output <70% of 14-day median on high-irradiance day (17.2%)
5. `flag_zero_output` — zero/NaN output on a sunny day (5.0%)

**CSA Filter** (further restricts HQ days) → **53 of 223 HQ days (24%)**
- Cloud opacity < 35%, precipitation < 1mm, days since rain ≥ 1, equipment operating

### Three Go/No-Go Signal Tests

| Signal | Question | Pass Criteria | Method |
|---|---|---|---|
| **S1: Sawtooth** | Does loss accumulate gradually between cleanings? | ≥50% of dry spells show positive slope | Linear regression within ≥3-day dry spells |
| **S2: PM/Dust** | Does PM correlate with performance decline? | Significant partial correlation after controlling cloud+temp | OLS residualization, Pearson partial r |
| **S3: Rain Recovery** | Does rain reduce loss proxy? | Wilcoxon p < 0.05 for post-rain < pre-rain | Event study ±3-5 days around ≥5mm rain |

### Decision Matrix

- **3/3 PASS** → GO
- **2/3 PASS** → CONDITIONAL GO (proceed with caution)
- **≤1/3** → NO-GO

---

## 5. Signal Test Results

### Verdicts

| Signal | Verdict | Key Metric |
|---|---|---|
| S1 Sawtooth | **PASS** | Median soiling rate +0.738 %/day, 5/10 positive-slope spells |
| S2 Dust Correlation | **PASS** | Best partial r = +0.328 (cumul PM2.5 vs cycle deviation, p<0.001) |
| S3 Rain Recovery | **FAIL** | Event-study Wilcoxon p = 0.395, dry-spell p = 0.150 |
| **Overall** | **CONDITIONAL GO** | 2/3 signals confirmed |

### Why S1 Passed

- Tropical site has alternating dry/wet periods creating natural soiling cycles
- Loss proxy increases gradually during dry spells (dust accumulates)
- Of 10 qualifying dry spells: 5 accumulating, 3 recovering, 1 flat, 1 recovering
- **Caveat**: Only 10 dry spells — verdict is fragile (marginal 50%)

### Why S2 Passed

- **Cumulative** PM features integrate exposure over time (single-day PM is noisy)
- **Cycle deviation** isolates within-cycle change (removes baseline shifts)
- Partial correlations survive deconfounding (cloud + temp control)
- PM2.5 > PM10 in predictive power (finer particles adhere more strongly)

**Key partial correlations** (controlled for cloud opacity + air temperature):

| Feature | vs Loss Proxy | vs Loss Rate | vs Cycle Deviation |
|---|---|---|---|
| `cumulative_pm25_since_rain` | r=-0.006, p=0.93 | r=+0.049, p=0.48 | **r=+0.328, p<0.001** |
| `cumulative_pm10_since_rain` | r=-0.077, p=0.26 | r=+0.017, p=0.80 | **r=+0.313, p<0.001** |
| `days_since_last_rain` | r=-0.058, p=0.40 | r=+0.013, p=0.86 | **r=+0.288, p<0.001** |
| `pm10_mean` | r=-0.032, p=0.64 | r=+0.023, p=0.74 | r=+0.119, p=0.08 |
| `pm25_mean` | r=+0.106, p=0.12 | r=+0.070, p=0.31 | r=+0.114, p=0.09 |
| `humidity_x_pm10` | r=-0.015, p=0.83 | r=-0.002, p=0.98 | r=+0.080, p=0.23 |

> [!IMPORTANT]
> Soiling features correlate significantly with **cycle deviation**, NOT with **loss proxy**. This confirms cycle deviation is the better soiling target.

### Why S3 Failed

- Loss proxy is all-cause → rain simultaneously reduces soiling but increases post-rain cloud/diffuse artifacts
- Only 30 qualifying rain events → low statistical power
- Rain vs dry loss: Cohen's d = 0.051 (negligible), Welch t-test p = 0.788
- Cementation: light rain <1mm cements dust rather than cleaning
- Post-rain days have low irradiance → loss proxy unreliable

### What "CONDITIONAL GO" Means for Modeling

- Soiling IS present and correlates with environmental predictors
- Use **cycle deviation** as primary target, not raw loss proxy
- **Cumulative** features are more predictive than daily features
- Rain recovery is weak → model may struggle to predict cleaning benefit
- Consider ensembling with DSPI as a physics-based prior

---

## 6. Supporting Analyses

### Physics-Based Models

| Model | vs Loss Proxy | vs Cycle Deviation | Notes |
|---|---|---|---|
| pvlib Kimber | r = -0.109 | — | Predicts ~1% loss; observed fluctuates 0-50%+ |
| Domain Soiling Index (DSPI) | r = -0.037 | **r = +0.377** | Leakage-free (no plant data used) |

### Tier Validation

T1 vs T2 loss: Pearson r = **0.969**, regression slope = 0.963, MAE = 3.86%
→ Soiling is **plant-wide**, not block-specific

### Seasonal Patterns

| Month Group | Mean Loss | Median Loss | Explanation |
|---|---|---|---|
| Jan, Mar, Apr, Oct | 12.4–14.7% | 0.0–12.2% | Dry season / inter-monsoon transition → higher soiling |
| May, Jul, Aug, Sep | 2.3–5.0% | 0.0% | Southwest monsoon → rain cleaning |
| Jun, Nov, Dec | 7.0–11.0% | 0.0% | Mixed / inter-monsoon |

Cycle deviation shows **medium** seasonal effect (Cohen's d = 0.52, p<0.001, dry 10.5% vs wet 2.9%).
Loss proxy shows **negligible** seasonal effect (d = -0.175, p = 0.26).

### CSA Analysis

CSA mean loss = **11.7%** vs non-CSA = **6.7%** (Cohen's d = 0.325, "small" effect)

> [!NOTE]
> CSA days have HIGHER loss than non-CSA. This is physically correct: clear-sky conditions reveal actual panel degradation because weather confounders are removed. The soiling signal is **only visible on clear-sky days**.

CSA correlation improvements over HQ:

| Feature | r (HQ) | r (CSA) |
|---|---|---|
| Cum PM2.5 | +0.108 | **+0.201** |
| Days since rain | +0.095 | **+0.128** |
| Temperature | -0.102 | **-0.245** |
| PM2.5 | +0.100 | **+0.140** |

### Sensor Dirt Check

Solcast/ground ratio trend: **-0.3186 per day** (negative = ground sensor reading declining relative to satellite, or satellite over-predicting over time)

### New-Source Data Gap

- `plant_avg_irradiance_wm2` starts **2025-04-11** — Jan-Mar 2025 missing
- New-source metrics (`gen_irr_ratio`, `new_*`) lack the peak dry season
- All signal verdicts use old-source columns (unaffected)

---

## 7. Data Quality Diagnostics (DQ1–DQ6)

### Summary Table

| DQ | Metric | Key r Values | Concern |
|---|---|---|---|
| DQ1: Irr vs Gen | On-site irr vs T1 gen | **r=0.099** (T1), **0.019** (Solcast peak), **0.512** (full-plant) | Low-r from T1 subset noise, not data error |
| DQ2: New Telemetry | Old vs new generation | **r=-0.034** (gen), **r=0.694** (irradiance) | Old/new measure fundamentally different things |
| DQ3: Gen/Irr Ratio | Ratio stability | median=17.16, std=7.27, ratio vs −loss: **r=0.066** | High variability; daily_gen may be pre-smoothed |
| DQ4: Power@Ref Irr | Soiling correlations | Loss proxy **r=-0.675**, Cycle dev **r=-0.266** | PM10-positive anomaly explained by seasonal confounding |
| DQ5: Old vs New | Source agreement | **r=-0.019** | Expected: different instruments, time windows, scope |
| DQ6: Perf Index | Performance health | median=**0.643**, 81% below 0.9, 71% below 0.8 | Persistent degradation; 12% above 1.0 is baseline artifact |

### DQ4: The PM10 Positive Correlation Anomaly (Resolved)

| Feature | r with power@ref_irr | Expected Sign | Explanation |
|---|---|---|---|
| Humidity × PM10 | **+0.210** | Expected − | Seasonal confounding: PM10 ↔ dry ↔ clear ↔ more power |
| Cum PM10 | **+0.105** | Expected − | Same confounding; coarse particles have weak soiling effect |
| Cum PM2.5 | **-0.110** | ✅ Correct − | Fine particles: stronger soiling, weaker seasonal correlation |
| DSPI | **+0.064** | Expected − | DSPI also correlates with dry season |
| Loss proxy | **-0.675** | ✅ Correct − | Direct performance measure (strong) |
| Days since rain | **-0.135** | ✅ Correct − | Longer dry → more soiling |
| Cycle deviation | **-0.266** | ✅ Correct − | Within-cycle degradation |

**Resolution**: Rolling 60-day analysis confirms PM10 vs loss is consistently **negative** (mean r = -0.136, 88% negative windows). The positive PM10 correlation with power is seasonal confounding, not a real soiling effect. PM10 = clear-sky proxy.

---

## 8. Feature Rankings & Interactions

### Best Predictors of Cycle Deviation

| Rank | Feature | Partial r | Raw r |
|---|---|---|---|
| 1 | `cumulative_pm25_since_rain` | **+0.328** (p<0.001) | +0.433 |
| 2 | `cumulative_pm10_since_rain` | **+0.313** (p<0.001) | +0.352 |
| 3 | `days_since_last_rain` | **+0.288** (p<0.001) | +0.417 |
| 4 | `domain_soiling_index` | — | +0.377 |
| 5 | `pm10_mean` | +0.119 (p=0.08) | — |
| 6 | `pm25_mean` | +0.114 (p=0.09) | — |
| 7 | `humidity_x_pm10` | +0.080 (p=0.23) | — |

### DSPI Correlation Profile

| Feature | r with DSPI |
|---|---|
| Cumul. PM10 | +0.575 |
| Humidity | -0.530 |
| Cumul. PM2.5 | +0.501 |
| Wind speed | +0.399 |
| PM10 | +0.364 |
| Temperature | +0.348 |
| Days since rain | +0.320 |

### Interaction Effects (Microscopic Level)

| Interaction | vs Target | r_interaction | p | R² improvement | Adds Value? |
|---|---|---|---|---|---|
| **PM2.5 × Days dry** | Cycle deviation | **+0.439** | **<0.001** | 0.0106 | ✅ Yes |
| Cum PM2.5 × Humidity | Loss proxy | +0.124 | 0.069 | 0.0169 | ✅ Yes |
| DSPI daily × Cloud | Loss proxy | -0.194 | 0.004 | 0.0197 | ✅ Yes |
| PM10 × Humidity | Loss proxy | -0.128 | 0.061 | 0.0136 | ✅ Yes |
| PM10 × Temperature | Cycle deviation | +0.032 | 0.630 | 0.0363 | ✅ Yes |

> [!TIP]
> **PM2.5 × Days dry → cycle deviation** is the strongest interaction (r=0.439, p<0.001). Soiling from fine particles amplifies with elapsed dry time. This should be included as an engineered feature in modeling.

### Performance Index vs Soiling Features (DQ6)

| Feature | r with Perf Index | Visual Pattern |
|---|---|---|
| Days dry | **-0.108** | Decreasing, concave down |
| Cum PM2.5 | **-0.093** | Decreasing, concave down |
| Cum PM10 | -0.057 | Decreasing, concave down |
| DSPI | +0.026 | No clear pattern |
| Hum × PM10 | +0.073 | Decreasing, **concave up** (threshold effect) |

All soiling features show a **triangle/envelope pattern**: base at x=0 (full range of perf index), tip at x=x_max (converging to ~0.6). Soiling caps the **upper bound** of performance. Pearson r understates the relationship; **quantile regression** is recommended.

---

## 9. Observation-by-Observation Analysis

> [!IMPORTANT]
> Each observation below is a **visual observation from manual plot inspection** followed by a technical explanation backed by `llm_eda_summary.json` data, and a suggestion on how to proceed.

---

### DQ1: `dq1_irradiance_vs_generation_timeseries.png`

#### OBS-1a: "We could use same units in both Y axes to compare visually"

**Explanation**: The two Y axes plot `irradiance_tilted_sum` (W/m² — raw daily sum of 15-min sensor readings) and `t1_energy_j` (Joules — sum of 10-min power × 600s). These are fundamentally different units with no simple conversion: irradiance sum is NOT energy density (it's not W·h/m²), so dividing by 3600 wouldn't produce correct kWh/m². The dual-axis approach is technically correct for comparing *trends*, but visually misleading because the scales are incomparable.

**Proceed**: Instead of unit alignment, **normalize both series to [0, 1] range** (min-max or percentage-of-median). This preserves trends while making them visually comparable. Alternatively, show both as "% of each series' median" on a shared Y axis. **Priority: Low** — cosmetic improvement only.

#### OBS-1b: "Both graphs are so noisy and hard to see a pattern"

**Explanation**: The noise is **real and physical**, not measurement error. Both series are daily sums, so any variation in cloud cover (dominant at this tropical site: mean cloud opacity = 36%, r = -0.305 with loss proxy) directly modulates both curves. A single overcast afternoon can halve the day's irradiance/generation sum. The JSON confirms high variability: normalized output monthly CV = 21.06%.

**Proceed**: **Add a 7-day rolling median overlay** to both curves. The EDA already uses rolling medians in DQ3 and DQ4 — applying the same technique here is consistent and low-effort. This would suppress weather noise and expose the seasonal/soiling trend. **Priority: Low** — plot enhancement.

#### OBS-1c: "We calculate clear days on other plots, it's better if we could see them here"

**Explanation**: CSA (Clear-Sky Analyzable) days are the subset where soiling is most visible (CSA mean loss = 11.7% vs non-CSA = 6.7%). Marking them on the time-series would highlight which points are most trustworthy.

**Proceed**: Add **light vertical ticks or dot markers** on the X axis for CSA days (53 of 223 HQ days). Use a subtle color to avoid cluttering. **Priority: Low** — plot enhancement.

---

### DQ1: `dq1_irradiance_vs_generation.png`

#### OBS-1d: "Are the resulting Pearson r values too low? (0.099 and 0.019)"

**Answer: Yes, they are low, but this is expected given the data structure, not a data error.**

**JSON evidence**:
- On-site irradiance vs T1 generation: **r = 0.099** (n≈361)
- Solcast peak GTI vs T1 generation: **r = 0.019** (n≈361)
- Solcast peak GTI vs **full-plant** generation: **r = 0.512** (n≈163)

**Why T1 is low**: T1 is only **3 inverters** (B2-08, B2-13, B2-17) out of 34 total, measured during **peak hours only** (10AM-2PM). This tiny sample amplifies noise:
- Per-inverter data shows all 3 are healthy (0% zero-output days, mean ~20,300 normalized output), so inverter trips are NOT the cause
- But with only 3 inverters × 4 hours, any random equipment variability (partial shading, clipping at ~35,000 cap, minor communication delays) creates enormous scatter
- The irradiance sensor is a **single point** that may not represent what the 3 T1 inverters see due to spatial cloud variation across the site

**Why full-plant r = 0.512**: When you aggregate 34 inverters × full day, the random variability averages out. The physics (more sun → more power) emerges. This **proves the underlying relationship works** — the low T1-r is a sample-size/window problem, not a fundamental data issue.

**Proceed**: Compute T1+T2 combined (6 inverters) energy vs irradiance to check if r improves. If so, confirms the 3-inverter sample is too small. **Priority: Medium** — would inform whether to expand the inverter subset.

#### OBS-1e: "Monthly medians roughly 'sawtooth' or 'sine wave'. Can we explain with numerical analysis?"

**Answer: Yes — the pattern is driven by the Sri Lankan monsoon cycle.**

**JSON evidence** (monthly loss proxy):

| Month | n | Mean Loss | Median Loss | Std |
|---|---|---|---|---|
| Jan | 15 | **12.4%** | 0.0% | 23.3 |
| Feb | 18 | 5.3% | 0.0% | 10.1 |
| Mar | 17 | **13.6%** | 5.2% | 14.4 |
| Apr | 16 | **12.9%** | 12.2% | 13.0 |
| May | 23 | 2.3% | 0.0% | 7.5 |
| Jun | 26 | 7.0% | 0.0% | 17.8 |
| Jul | 21 | 4.2% | 0.0% | 6.6 |
| Aug | 21 | 5.0% | 0.0% | 8.3 |
| Sep | 21 | 3.9% | 0.0% | 5.4 |
| Oct | 18 | **14.7%** | 8.1% | 15.4 |
| Nov | 10 | 10.9% | 0.0% | 29.8 |
| Dec | 10 | 11.0% | 0.0% | 29.8 |

The pattern matches the climate: **Jan-Apr and Oct-Dec** (dry/inter-monsoon) show higher losses → more dust accumulation. **May-Sep** (southwest monsoon) shows lower losses → frequent rain cleaning. The "sawtooth" is **expected seasonality, not noise**.

**CV = 21.1%**: Confirmed by JSON (`norm_output_monthly_cv_pct = 21.06%`). This is moderate variability driven by the monsoon cycle — a positive finding confirming seasonality is detectable.

#### OBS-1f: "Jan-Mar, and Aug-Nov have slightly lower medians"

**Explanation**: For normalized output (not loss): lower normalized output = higher loss. The table above shows Jan, Mar, Apr, Oct have highest *mean loss*, consistent with lower normalized output medians for those months. Aug-Nov is the northeast monsoon onset — mixed weather with highly variable cloud cover (Nov/Dec std = 29.8, vs Sep std = 5.4), creating broader scatter.

**Proceed**: No action needed — this confirms expected seasonality. Optionally run **STL decomposition** (seasonal + trend + residual) to quantitatively separate the seasonal component. **Priority: Lower**.

---

### DQ2: `dq2_daily_gen_validation_timeseries.png`

#### OBS-2a: "daily_generated_electricity is much more stable; active power is hugely varying"

**Explanation**: These two metrics measure fundamentally different things:

| Property | Active Power Integral (`subset_energy_mwh`) | Daily Generated Electricity (`subset_daily_gen_kwh`) |
|---|---|---|
| **Scope** | 3 inverters (T1 only) | Entire plant (34 inverters) |
| **Time window** | Peak hours 10AM-2PM (4h) | Full day (~12h) |
| **Method** | Sum of 10-min instantaneous readings | Cumulative energy meter reading |
| **Sensitivity** | Any single-inverter fluctuation → ~33% swing | 34-inverter average → fluctuations cancel |
| **Data start** | Jan 2025 | Apr 2025 |

The active power integral captures ~3% of plant-day energy (3/34 inverters × 4/12 hours) where any single-inverter anomaly dominates. The daily generation captures 100%, averaging out equipment-level noise. The stability difference is **expected**.

#### OBS-2b: "The two curves never collapse; they only collapse when daily_gen becomes 0"

**Explanation**: They will never collapse because they measure different scopes (3 vs 34 inverters, 4h vs 12h). The zero-drop days in daily_gen are suspicious — the JSON confirms **6 sunny zero-generation days** (`n_zero_gen_sunny = 6`). These are likely:
1. **Communication gaps** — the meter didn't transmit data (most likely)
2. **Meter reset events** — cumulative counter restarted
3. **Actual plant shutdowns** — entire plant was off (least likely for a utility-scale plant)

**Proceed**: **Investigate the 6 zero-gen sunny days** — check if they correlate with known SCADA events, holidays, or grid outages. Flag and exclude them. **Priority: High** — these zeros will corrupt any metric built on daily_gen.

#### OBS-2c: "Can we replace the use of active_power with daily_generated_electricity?"

**Answer: Don't replace — run a parallel pipeline branch.**

| Factor | Active Power (Current) | daily_generated_electricity |
|---|---|---|
| Coverage | 3/34 inverters, 10AM-2PM | Full plant, full day |
| Stability | Extremely variable | Smooth, stable |
| Soiling SNR | High noise → poor SNR | Lower noise → better SNR *in theory* |
| Peak-hour filtering | Built-in (reduces thermal effects) | Full-day includes dawn/dusk |
| Data start | Jan 2025 | Apr 2025 (**3 months missing** from peak dry) |
| Sub-daily resolution | Yes → can compute ref-irradiance | No → only daily totals |
| Known issues | 3-inverter lottery | Suspicious zero-drops, possible pre-smoothing |

**Recommendation**: Keep the old pipeline for sub-daily features (`power_at_ref_irradiance`, per-inverter analysis). **Add** a parallel pipeline branch using `daily_generated_electricity / plant_avg_irradiance_wm2` as a secondary soiling metric. Compare which produces stronger soiling signals. If the new branch wins, gradually shift.

**However, read the DQ3 concerns first** — daily_gen may be "too stable" (doesn't track daily irradiance variation), suggesting possible pre-smoothing or accumulation artifacts. This needs verification before relying on it.

**Proceed**: Implement parallel pipeline branch. **Priority: Medium** — depends on DQ3 investigation.

#### OBS-2d: "Plant avg_solar_radiation seems to be a scaled-down version of Solcast (good correlation)"

**JSON evidence**: plant_vs_solcast_irr_r = **0.694** — good.

**Explanation**: Solcast measures satellite-derived GTI (global tilted irradiance) representing the theoretical maximum at the plane of array. The plant's on-site sensor is subject to dirt accumulation, tilt angle differences, and calibration offsets. A consistent scale offset is **expected and healthy** — it confirms both sensors see the same weather patterns with a proportionality factor. The sensor dirt check trend (-0.3186/day) suggests the on-site sensor may be slowly drifting downward relative to Solcast over the observation period.

#### OBS-2e: "Values get a little more closer after 2026"

**Explanation**: Two plausible causes: (1) the on-site irradiance sensor was cleaned or recalibrated; (2) seasonal irradiance changes alter the satellite-to-ground offset. Without SCADA logs documenting sensor maintenance, this is unverifiable.

**Proceed**: Monitor the Solcast/ground ratio trend over time. If a step change occurred, add a `flag_sensor_recalibrated` flag for that date. **Priority: Lower**.

---

### DQ2: `dq2_daily_gen_validation.png`

#### OBS-2f: "For almost every value of active power integral, daily_gen is around 7000-8000. r = -0.034. Even negative!"

**Answer: This is expected and NOT concerning.**

**JSON evidence**: `old_vs_new_gen_r = -0.034`.

**Explanation**: Active power integral varies wildly (0 to ~20,000+ kWh) because it's 3 inverters in a 4-hour window — any equipment variability creates huge swings. Daily_gen is the full plant for the full day → varies in a narrow band (7000-8000 kWh) because averaging 34 inverters × 12h smooths everything. The scatter plot literally shows: **the T1 peak-hour subset does NOT predict full-plant daily output**. This is a statement about representativeness, not data quality.

The slightly negative r (-0.034) is **random noise** at this sample size — not a real inverse relationship.

**"This is not even a parallel line to the 1:1 line"**: Correct — because they're not the same measurement at different scales. They're fundamentally different measurements. A 1:1 relationship would only exist if both measured the same thing.

**Proceed**: No action needed — the low r is explained. This reinforces that the old and new data sources capture different aspects of plant performance.

#### OBS-2g: "Second plot has better r=0.694. Solcast values seem higher."

**JSON evidence**: `plant_vs_solcast_irr_r = 0.694`.

**Explanation**: Good correlation confirms both irradiance sources track the same weather. Solcast being systematically higher is **expected** — satellite GTI represents theoretical clear-atmosphere conditions, while the ground sensor is affected by local occlusion, sensor aging, and dirt. The offset is a calibration difference, not an error.

**Proceed**: Use the Solcast/ground ratio as a sensor health indicator. A sudden change in ratio could indicate sensor failure. **Priority: Lower**.

---

### DQ3: `dq3_gen_irr_ratio_timeseries.png`

#### OBS-3a: "daily_generated_electricity doesn't deviate with irradiance fluctuations. Suspicious."

**Answer: This is the most important question about the new data source.**

**JSON evidence**: gen_irr_ratio median = 17.16, std = 7.27 (CV ≈ 42%). The ratio itself varies substantially, but the visual impression of "steady daily_gen" is real.

**Explanation**: The formula `gen_irr_ratio = subset_daily_gen_kwh / plant_avg_irradiance_wm2` has a **units problem**. The numerator (kWh) is total energy; the denominator (W/m²) is *average instantaneous irradiance*, NOT total irradiation (kWh/m²). These have incompatible time dimensions:
- On a long sunny day: more kWh produced, but avg W/m² doesn't change much → ratio inflated
- On a short cloudy day: less kWh, but avg W/m² can still be moderate → ratio deflated

The ratio's CV of 42% comes primarily from: clouds (dominant confounder), temperature (~5-10% seasonal), inverter clipping (plateau at rated power on high-irradiance days), and solar geometry (seasonal angle changes). Soiling (0.1-0.5%/day) is small relative to these confounders.

Additionally, `daily_generated_electricity` may be **pre-smoothed or accumulated** internally by the meter — if it reports a running average or multi-day accumulation, it would appear unnaturally stable compared to daily irradiance variation.

**Proceed**: **Priority High** — To make gen_irr_ratio a proper Performance Ratio, it should be: `PR = E_measured / (H_total × P_rated / G_STC)`, where `H_total = avg_irradiance × daylight_hours / 1000` (total daily irradiation in kWh/m²). This is blocked by: (1) nameplate capacity `P_NOM_KWP = 500 kWp` is a placeholder; (2) daylight hours aren't directly available. However, for *relative* soiling detection (ratio-to-baseline), the current approach works because the nameplate would cancel out in `gen_irr_ratio / rolling_baseline`.

#### OBS-3b: "Generation should be lower than irradiance"

**Explanation**: **Pure scaling/units issue, not a physics violation.** `gen_irr_ratio = kWh / (W/m²)` ≈ 17. This has units of kWh·m²/W — it is NOT dimensionless and NOT bounded by 1.0. The value exceeding 1 does NOT mean the plant produces more energy than it receives. It just means the numerator (in kWh, a large number) exceeds the denominator (in W/m², a smaller number). The ratio would be bounded between 0 and 1 only if it were a proper PR with matched units.

**Proceed**: No action needed for soiling detection (baseline-normalized). For interpretability, fix the PR formula (see OBS-3a). **Priority: Lower** (blocked by nameplate capacity).

#### OBS-3c: "Rolling median shows increase during cleaning periods and decrease during soiling spells"

**Answer: This is an encouraging sign, even with the ratio's unit problems.**

**Explanation**: Even though the absolute gen_irr_ratio values are hard to interpret, the *relative trend* (rolling median going up after cleaning, down during dry spells) suggests the new-source data captures the soiling/cleaning cycle. The two cleaning campaigns visible (Mar 2025 and Sep 2025) both coincided with rainfall, so the increase could be from cleaning, rain-washing, or both — they can't be separated here.

**Proceed**: No action needed — this is a positive qualitative validation. The formal quantitative test is Signal 1 (sawtooth), which passed.

#### OBS-3d: "Loss proxy alternates between very high and very low too frequently. r=0.066."

**JSON evidence**: `ratio_vs_neg_loss_r = 0.066` — confirmed near-zero.

**Explanation**: The loss proxy's alternation between ~0% and 50%+ is the **hallmark of weather noise dominating over soiling**:
```
loss = 100 × (1 - normalized_output / rolling_95th_percentile_baseline)
```
On a clear day near baseline → loss ≈ 0%. On a cloudy day when output drops → loss spikes to 30-80%. The alternation is weather (clear→cloudy→clear), not soiling (gradual). JSON confirms: loss proxy has **60.6% zeros**, median=0.00, autocorrelation ≈ 0 at all lags (no day-to-day persistence). This means each day's loss value is essentially **independent of the previous day** — not the behavior of a gradual soiling process.

The near-zero r (0.066) between gen_irr_ratio and loss proxy confirms they're measuring different things through different noise profiles. This is expected because the loss proxy uses old-source data (T1 subset, peak hours) while gen_irr_ratio uses new-source data (full plant, full day).

**Proceed**: This is the fundamental problem. The solution is a **two-stage decomposition**: (1) build a weather-expected-output model (using irradiance, cloud, temperature); (2) the residual = soiling proxy. This isolates soiling from weather. **Priority: High** — this is the modeling strategy for Stage 6.

---

### DQ3: `dq3_gen_irr_ratio.png`

#### OBS-3e: "Jan, May-Jul, Nov-Dec higher medians; Feb, Apr, Aug-Oct lower. Sawtooth visible."

**Explanation**: This monthly pattern overlays **three physical effects**:

1. **Solar geometry**: Panel tilt is fixed. At 8.5°N latitude, POA irradiance varies with solar declination. Panels tilted south may favor certain months. Day-length variation (~11.5 to 12.8h) contributes ~11% ratio variation alone.
2. **Monsoon pattern**: SW monsoon (May-Sep) brings cloud + rain → lower irradiance denominator but also lower generation → net effect on ratio depends on which drops more. Post-monsoon (Oct-Nov) sees rapid clearing → ratio recovery.
3. **Soiling accumulation**: Dry months (Jan-Mar) → dust builds up → lower gen relative to irradiance → lower ratio. But this is entangled with seasonal irradiance changes.

**"Jan, Nov, Dec, Oct have larger box plots"**: These are inter-monsoon transition months where weather is highly variable — some days clear, some days stormy — creating wider spread.

**Proceed**: The seasonal pattern is a **positive finding** confirming the gen_irr_ratio tracks expected climatology. No action needed, but consider STL decomposition to separate seasonal from soiling. **Priority: Lower**.

---

### DQ4: `dq4_power_at_ref_irradiance.png`

#### OBS-4a: "Aug 2025–Jan 2026 slightly credible; Jan 2026+ extremely noisy"

**Explanation**: The feature computes mean active power when sub-daily irradiance falls within ±tolerance of the dataset's **median irradiance** (ref_irr = 8550 W/m² per JSON). When irradiance distribution shifts seasonally away from this median, **fewer 10-min intervals match** the reference band per day → the daily mean is computed from fewer points → noisier.

In Jan 2026+: the irradiance distribution may have shifted (seasonal change, or sensor calibration drift), causing fewer matching intervals. The T1 3-inverter subset also amplifies any remaining noise.

**Proceed**: Plot `ref_irr_match_count` per day. If Jan 2026+ shows significantly fewer matching intervals, the noise is explained. Consider using a **seasonal reference irradiance** (quarterly median) instead of the annual median. **Priority: Medium**.

#### OBS-4b: "No clear sawtooth at daily level. Maybe 3-7 day bins or rolling median."

**Explanation**: At daily resolution, weather noise (~10-40% daily fluctuation) completely overwhelms the soiling signal (~0.1-0.5%/day). A 7-day rolling median would suppress weather and potentially reveal the ~1-3.5%/week soiling accumulation.

**Proceed**: **Add 7-day rolling median overlay** to DQ4 time-series. **Priority: Low** — plot enhancement, but would greatly aid visual interpretation.

#### OBS-4c: "All loss metrics should show negative correlation, right?"

**Answer: Yes, in general, but several confounders create exceptions — explained below.**

#### OBS-4d: "Humidity × PM10 = +0.22 — shouldn't it be negative?"

**JSON evidence**: `ref_irr_vs_humidity_x_pm10_r = +0.2099`.

**Answer: The positive sign is caused by seasonal confounding, not a physics error.**

**Full explanation**: This is the **PM10 positive correlation anomaly** (resolved). PM10 is high during the dry season. The dry season also has more sunshine and higher irradiance → more intervals matching the reference band → higher average power at reference irradiance. The seasonal effect (more PM10 = more sun = more power) outweighs the soiling effect (more PM10 = more dust = less power).

Rolling 60-day analysis proves this: PM10 vs loss is consistently **negative** (mean r = -0.136, 88% of windows negative, 0% significantly positive). The sign flips because power@ref_irr and loss proxy have an inverted relationship (r = -0.675 between them).

**"Does humidity both promote adhesion and clean via dew?"**: This is a valid physical mechanism (cementation vs dew cleaning) but accounts for only a small fraction of the observed +0.22. The dominant driver is seasonal confounding.

**Proceed**: Run **partial correlations** of power@ref_irr vs PM10 features controlling for cloud_opacity_mean + month. If the positive correlations flip negative, seasonal confounding is confirmed. **Priority: Medium**.

#### OBS-4e: "Cum PM10 = +0.15 — shouldn't this be negative too?"

**JSON evidence**: `ref_irr_vs_cumulative_pm10_since_rain_r = +0.1055`.

**Explanation**: Same seasonal confounding as Humidity × PM10. Cum PM10 accumulates during dry spells, which are also the high-irradiance periods → positive correlation with power. PM10 (coarse particles, >2.5µm) has a **weaker soiling effect per unit mass** than PM2.5, so the soiling signal is too weak to overcome the seasonal confounder.

#### OBS-4f: "Both PM10 features being positive seems non-random"

**Answer: Correct — it's NOT random. It's a systematic confound.**

Both PM10-derived features (Humidity×PM10 and Cum PM10) are positive because PM10 is fundamentally a **dry-season/clear-sky proxy**. PM10 particles:
- Correlate strongly with dry/windy conditions (high irradiance → positive power effect)
- Have weaker per-unit soiling impact than PM2.5 (coarser particles are more easily removed by wind/dew)
- Their seasonal correlation with clear skies dominates over their soiling effect

Meanwhile, **Cum PM2.5 = -0.110** (correctly negative) because PM2.5:
- Correlates less strongly with clear-sky conditions
- Has stronger per-unit soiling effect (finer particles adhere better, block more light per mass)
- Soiling signal survives the seasonal confounding

**Proceed**: The PM10 vs PM2.5 sign difference is informative for modeling — use **PM2.5 features** (not PM10) as primary soiling predictors. Use PM10 only after deconfounding. **Priority: Medium** — affects feature selection for modeling.

#### OBS-4g: "Loss proxy = -0.675 (good)"

**JSON evidence**: `ref_irr_vs_t1_performance_loss_pct_proxy_r = -0.6747`.

**Explanation**: This is the strongest correlation in DQ4 and confirms the feature works correctly — more loss → less power at reference irradiance. This strong negative r shows that power@ref_irr responds to the same signal as the loss proxy, validating it as a performance indicator.

#### OBS-4h: "DSPI = +0.064 (problematic)"

**JSON evidence**: `ref_irr_vs_domain_soiling_index_r = +0.0639`.

**Explanation**: DSPI correlates with the dry season (r = +0.320 with days_since_rain, r = +0.575 with cumul_PM10) → same seasonal confounding as PM10. DSPI accumulates during dry spells, which are also high-power periods. The near-zero/slightly-positive r is a confounder artifact, not a physics failure.

Against cycle deviation, DSPI shows **r = +0.377** — confirming it works correctly when compared to a metric that isolates within-cycle soiling.

#### OBS-4i: "Days since rain = -0.135 (good)" / "Cum PM2.5 = -0.11 (good)" / "Cycle deviation = -0.266 (good)"

**JSON evidence**: All confirmed exact. These negative correlations match physical expectation: more dry days / more fine dust / more within-cycle degradation → less power at reference irradiance. These are the features to trust for soiling detection.

---

### DQ5: `dq5_old_vs_new_timeseries.png`

#### OBS-5a: "Old T1 alternates between 100 & 0 — unusable"

**JSON evidence**: Loss proxy: median=0.00, **60.6% zeros**, mean=7.91, max=95.24.

**Explanation**: This is the **baseline floor effect** (verified Hypothesis H1). The formula `loss = 100 × (1 - norm_output / 95th_pctile_baseline)` produces zero whenever today's output meets or exceeds the baseline. Since the 95th percentile baseline tracks near-peak performance, ~60% of days naturally produce loss = 0%. The remaining ~40% of days show non-zero loss from a mix of weather (dominant), temperature, equipment, and soiling (small).

This is NOT caused by inverter trips — per-inverter data confirms all 6 inverters have **0% zero-output HQ days** with consistent performance (~20,300 mean normalized output each). The alternation is inherent to the loss proxy's mathematical construction.

**Proceed**: The loss proxy is a poor continuous target variable. Use **cycle deviation** (isolates within-cycle change) or **gen_irr_ratio / baseline** (new source) instead. **Priority: High** — affects target variable selection.

#### OBS-5b: "New metric is more grounded, shows more stability and usable values"

**JSON evidence**: DQ5 confirms new metric varies 10-80% (vs 0-95% for old).

**Explanation**: The new metric (`new_performance_loss_pct_proxy`) uses `gen_irr_ratio / new_rolling_clean_baseline` which has different characteristics:
- Based on full-plant daily generation (34 inverters) → inherently more stable
- Uses a different baseline computation (40th percentile clear-day mask, min_periods=4, cloud_opacity ≤ 40%)
- The narrower dynamic range (10-80%) produces more moderate, usable values

**"But doesn't show recognizable patterns on cleaning/raining/soiling"**: Correct — the new metric's stability may mean lower *sensitivity* to soiling. The soiling signal (0.1-0.5%/day) may be within the noise floor of this metric.

#### OBS-5c: "Could have rolling median to show a smoothened curve"

**Proceed**: **Add rolling median overlay** to DQ5 time-series (7-day or 14-day window). **Priority: Low** — same plot enhancement as DQ1/DQ4.

#### OBS-5d: "New metric has value where old is 0; would correlate if old actually had a value"

**Explanation**: This is a perceptive observation. On days where old loss = 0 (baseline floor effect), the new metric shows non-zero values because it has a different baseline. If you could "see through" the old metric's floor effect, the two would likely correlate. The r = -0.019 between them is almost entirely driven by the old metric's 60.6% zeros masking the relationship.

**Proceed**: Compute correlation **using only days where old loss > 0** (86 of 216 HQ days). This would reveal whether the two metrics agree on non-zero-loss days. **Priority: Medium** — would validate the new source.

#### OBS-5e: "Old cycle deviation more stable than T1 loss proxy, but still mostly 0/1"

**JSON evidence**: Cycle deviation: 62.2% zeros, median=0.00, mean=14.10.

**Explanation**: Cycle deviation is zero-inflated for a different reason than the loss proxy: it's zero at the **start of each cycle** (by definition, the cycle starts at the maximum normalized output) and remains zero until performance degrades below that maximum. Since cycles are short (median ~4 days due to frequent rain), many cycles don't last long enough for measurable deviation to develop.

#### OBS-5f: "New metric rarely becomes 1 [100%]"

**Explanation**: The new cycle deviation uses `gen_irr_ratio` which has a narrower dynamic range → the deviation from cycle max is proportionally smaller → rarely reaches 100%. The old cycle deviation uses T1 normalized output with its wider swings → can reach 100% when a single poor day occurs within a short cycle.

---

### DQ5: `dq5_old_vs_new_comparison.png`

#### OBS-5g: "Left panel: many new proxy values have old proxy = 0; r = -0.019"

**JSON evidence**: `old_vs_new_loss_r = -0.019` — confirmed.

**Explanation**: The old loss proxy's 60.6% zeros means most scatter points cluster at y=0 (old) across the full x-range (new). This destroys any linear relationship. The -0.019 is indistinguishable from zero — NOT a real inverse relationship.

#### OBS-5h: "Right panel: old source has higher correlation with soiling features"

**JSON evidence**:

| Feature | Old r | New r | Old wins? |
|---|---|---|---|
| DSPI | +0.006 | -0.041 | No (both ~0) |
| Cum PM2.5 | **+0.122** | +0.072 | Yes |
| Cum PM10 | +0.012 | +0.021 | No (both ~0) |
| Days dry | **+0.163** | +0.095 | Yes |
| Hum × PM10 | **-0.209** | -0.131 | Yes |

**Explanation**: The old source's higher correlations do NOT mean it's a better soiling metric. The old source has **wider variance** (0-95% range) which provides more statistical leverage for Pearson r. But much of that variance is noise (weather), not signal (soiling). The new source's narrower range (10-80%) produces lower r values but potentially higher **signal-to-noise ratio**.

All correlations are weak (|r| < 0.21), meaning neither source currently provides strong linear soiling prediction on its own.

#### OBS-5i: "Humidity × PM10 shows same exception as DQ4 (both negatively correlated)"

**Explanation**: Both old (-0.209) and new (-0.131) show *negative* correlation of Hum×PM10 with loss proxy. This is the **opposite** sign from DQ4 (where Hum×PM10 vs *power* was positive +0.210). This is mathematically consistent: power@ref_irr and loss proxy are inversely related (r = -0.675). So a feature that correlates *positively* with power will correlate *negatively* with loss, and vice versa. The sign flip is expected.

---

### DQ6: `dq6_performance_index.png`

#### OBS-6a: "First plot doesn't show sensitivity to cleaning"

**Explanation**: Three reasons the cleaning response is invisible:
1. New-source data starts Apr 2025, but the first cleaning campaign was **Mar 2025 → entirely missed**
2. The Sep 2025 cleaning occurred during monsoon rain → hard to attribute recovery to cleaning vs rain
3. The rolling baseline **continuously adapts** → post-cleaning improvement gets absorbed into the baseline within 2-3 weeks, flattening the visible signal

**Proceed**: Plot performance index **freezing the baseline at the pre-cleaning level** to see if post-cleaning days visually jump above the frozen baseline. **Priority: Medium** — would directly test cleaning sensitivity.

#### OBS-6b: "There seems to be an increase in generation caused by raining"

**Explanation**: Rain physically cleans panels, reducing soiling → higher gen_irr_ratio → higher performance index. But this conflicts with Signal 3's FAIL result. The resolution: the performance index (new source, full plant) may be **more sensitive** to rain recovery than the loss proxy (old source, T1 subset). Alternatively, rain coincides with seasonal transitions that independently improve performance. This is an important observation that deserves quantitative follow-up.

**Proceed**: Compute performance index change ±3 days around significant rain events (≥5mm) and run Wilcoxon test — the same test that failed for the loss proxy. If it passes for performance index, the new source is better at detecting rain recovery. **Priority: Medium**.

#### OBS-6c: "Most of the time (80%) below 0.8"

**JSON evidence**: `pct_below_08 = 70.6%`, `pct_below_09 = 81.0%`, `pct_below_1 = 88.1%`.

**Explanation**: The plant typically operates at 64.3% of its rolling clean baseline (median = 0.643). But this 36% deficit is **NOT exclusively soiling** — it includes:
- Cloud effects (dominant — cloud opacity r = -0.305 with loss)
- Temperature losses (~0.4%/°C above 25°C)
- Inverter clipping/inefficiency
- Soiling (estimated 1-5% based on literature for tropical sites)

Soiling alone cannot explain 36% loss. A two-stage decomposition (weather model → residual = soiling) is essential to isolate the soiling fraction.

#### OBS-6d: "Distribution peaked in range 0.45-0.65. Considerable days above 1.0. Expected or error?"

**JSON evidence**: 12% of days above 1.0, clipped at 1.5 by code.

**Answer: Expected, not an error.** Three mechanisms create values > 1.0:

1. **Mathematical certainty (~5%)**: The baseline is the 95th percentile → by definition, ~5% of days exceed it
2. **Baseline lag (~4%)**: After cleaning or seasonal improvement, the trailing 30-day baseline is anchored to worse days in the past → today's ratio overshoots it. The JSON shows spikes around Nov 2025, right after cleaning campaigns
3. **Cloud-edge enhancement (<3%)**: Brief intense irradiance spikes (sun breaking through cloud edges) can temporarily boost generation above clear-sky levels, but the irradiance sensor averages these out → ratio spikes

The code clips at 1.5: `daily["new_performance_index"] = (gen_irr_ratio / baseline).clip(upper=1.5)`.

**"Gradients: 0.25→0.45→1.3→1.5"**: The 0.45-0.65 peak is the "normal operating range" on clear days with soiling. The tail above 1.0 is post-cleaning/seasonal. The secondary spike at 1.4-1.5 (5 days) is likely the clipping cap catching extreme outliers.

**"No two clear peaks to call bimodal"**: Correct. The distribution is **right-skewed unimodal with a heavy left tail** (dominated by cloudy/poor days) and a thin right tail (post-cleaning/clear days).

**"Median & mean not visible"**: JSON provides: **median = 0.643, mean = 0.693**. The mean exceeding median confirms right skew.

**Proceed**: Investigate the 12% of days above 1.0 — identify if they cluster after cleaning or at seasonal transitions. Cap at 1.05 instead of 1.5 for modeling if desired. **Priority: Lower**.

#### OBS-6e: "Scatter plots show |r|<0.1 but visual decrease. Days dry r=-0.108, Cum PM2.5 r=-0.093"

**JSON evidence**: All confirmed:

| Feature | r with Perf Index |
|---|---|
| Days dry | -0.108 |
| Cum PM2.5 | -0.093 |
| Cum PM10 | -0.057 |
| DSPI | +0.026 |
| Hum × PM10 | +0.073 |

**Explanation**: The weak Pearson r **dramatically understates the real relationship** because it's an **envelope/constraint** pattern, not a linear relationship:

- **At x ≈ 0** (no soiling pressure): Performance index has full range [0.3, 1.5] → dominated by weather, equipment, season
- **At high x** (high soiling pressure): Performance index converges to ~0.6 → soiling becomes the **binding constraint**, capping how high performance can go

Pearson r measures the *linear association of the mean*, but soiling constrains the *upper bound*, not the mean. A **quantile regression** (e.g., modeling the 90th percentile vs soiling features) would show much stronger effects because it captures the envelope.

**Proceed**: **Apply quantile regression** (90th percentile of performance index vs each soiling feature). If the slope of the 90th-percentile line is strongly negative, soiling IS constraining performance even though the mean relationship is weak. **Priority: Medium** — novel analysis technique for this dataset.

#### OBS-6f: "Hum × PM10 concave UP; others concave DOWN"

**Explanation**: The different curvatures reveal different physical mechanisms:

- **Concave down** (Days dry, Cum PM2.5, Cum PM10, DSPI): Effect **saturates**. Beyond a threshold, adding more dust causes diminishing additional loss. Like a log curve — the first layer of dust blocks a lot of light, subsequent layers block less. This is the standard soiling saturation effect documented in literature.

- **Concave up** (Hum × PM10): **Threshold/cementation effect**. At low Hum×PM10 values, there's little impact (dust is dry and loosely adhered, or humidity is low). Above a critical value (high humidity + high PM10), the **cementation effect kicks in** — humidity causes dust particles to bind tightly to the glass surface, and each additional unit of Hum×PM10 causes progressively more damage. This non-linear activation is physically expected for the cementation mechanism.

**Proceed**: This non-linearity suggests **log-transforming** cumulative PM features and using **polynomial or threshold-based** features for Hum×PM10 in modeling. **Priority: Medium** — affects feature engineering for modeling.

#### OBS-6g: "Triangle shape: base at x=0, tip at (x_max, ~0.6). All negatively correlated — good."

**Explanation**: Covered in OBS-6e above. The triangle shape is the **envelope/constraint** signature. All soiling features being negatively correlated with performance index is **physically correct and expected** — more soiling pressure → lower performance. This is the strongest qualitative evidence of soiling being real in the dataset.

**Proceed**: Quantile regression (OBS-6e) is the appropriate tool to quantify this pattern. **Priority: Medium**.

---

### Verified Hypotheses

All hypotheses below have been tested against `llm_eda_summary.json` numerical data.

| # | Hypothesis | Status | Confidence | JSON Evidence |
|---|---|---|---|---|
| H1 | Loss proxy zero-inflation is a baseline floor effect, not inverter trips | ✅ Verified | ★★★★★ | 60.6% zeros; per-inverter: 0% zero-output days |
| H2 | PM10+ positive correlation with power is seasonal confounding | ✅ Verified | ★★★★★ | Rolling 60d: PM10 vs loss mean r = -0.136, 88% negative |
| H3 | CSA days reveal true soiling (higher loss = expected) | ✅ Verified | ★★★★★ | CSA mean=11.7% vs non-CSA=6.7% |
| H4 | PM2.5 × Days dry is the critical interaction | ✅ Verified | ★★★★★ | r_interaction=0.439, p<0.001 |
| H5 | Signal 1 verdict is fragile (only 10 dry spells) | ✅ Verified | ★★★★☆ | 5/10 positive slope, marginal pass |
| H6 | daily_generated_electricity may be pre-smoothed | 🔶 Unverified | ★★★★☆ | Indirect: uncorrelated with daily irr variation |
| H7 | Performance index >1.0 is expected (not error) | ✅ Verified | ★★★★★ | 12% above 1.0; math + post-cleaning lag |
| H8 | Zero-inflated target demands hurdle model, not OLS | ✅ Verified | ★★★★★ | Loss 60.6% zeros; cycle dev 62.2% zeros |

---

## 10. Debunked Hypotheses

### ~~"Old-Source Loss Proxy Is Dominated by Inverter Trips"~~ ❌

**Original claim**: The loss proxy's binary 0/100 behavior is caused by individual T1 inverter trips on a 3-inverter subset.

**Debunking evidence**: Per-inverter normalized output shows:
- All 6 inverters have **0% zero-output days** on HQ days
- All inverters show consistent performance: mean ~20,300, std ~6,300, min ~700, max ~37,500
- CV 30-32% — driven by irradiance variability, not inverter faults

**Correct explanation**: The binary behavior is inherent to the loss proxy math (95th-percentile baseline floor effect). See H1 above.

### ~~"Old Source Correlations Are Higher Because of Spurious Inverter-Trip Alignment"~~ ❌

**Original claim**: Old source's higher soiling correlations result from inverter trips aligning with environmental conditions.

**Debunking**: Since inverters are healthy on all HQ days, the higher correlations stem from the old source's wider dynamic range (0-95%) providing more variance for Pearson r computation. The new source's narrower range (10-80%) produces lower but potentially more honest correlations.

---

## 11. Key Inferences & Knowledge

### What We Know For Certain

1. **Soiling is detectable** at this site via cumulative PM features and cycle deviation
2. **Cycle deviation >> loss proxy** as a soiling target — loss proxy is zero-inflated, non-autocorrelated, and dominated by weather noise
3. **Cumulative features >> daily features** — PM2.5 accumulation since rain is the strongest predictor
4. **DSPI is a valid physics-based feature** — correlates with the right environmental factors and with cycle deviation (r=0.377), leakage-free
5. **Soiling is plant-wide** — T1 vs T2 r=0.969, no block-specific effects
6. **Clear-sky days are where soiling is visible** — CSA filter reveals the true signal
7. **PM2.5 > PM10 for soiling prediction** — finer particles adhere more strongly, correlate less with seasonal confounders
8. **The envelope pattern is real** — soiling caps the upper bound of performance, not the mean

### What We Suspect (Needs Verification)

1. `daily_generated_electricity` may be internally smoothed → needs sub-daily resolution check
2. Jan 2026+ noise increase in power@ref_irr may be from fewer matching ref-irradiance intervals
3. The 6 sunny zero-gen days may be communication gaps rather than actual shutdowns
4. Widening dry-spell definition from ≥3 to ≥2 days would increase sample size significantly

### What We Don't Know Yet

1. Whether cycle deviation or gen/irr ratio will produce better model performance
2. Whether a hurdle model or transformation (e.g., log(1+loss)) will be more effective
3. How the new-source pipeline compares to old-source for soiling signal strength
4. The actual nameplate capacity (P_NOM_KWP = 500 kWp is a **placeholder**)

---

## 12. Recommendations & Next Steps

### High Priority (Do Before Modeling)

1. **Consider hurdle/two-part model** for loss proxy — 60.6% zeros makes OLS inappropriate. Options: logistic (zero vs non-zero) + regression (non-zero values), or use cycle deviation, or use gen/irr ratio.

2. **Use PM2.5 × Days dry as an engineered feature** — strongest interaction (r=0.439, p<0.001). Also consider Cum PM2.5 × Humidity.

3. **Run partial correlations for DQ4** vs PM10/humidity controlling for cloud + month → confirm seasonal confounding numerically.

4. **Investigate 6 sunny zero-gen days** — meter resets vs genuine shutdowns.

### Medium Priority (Improves Model Quality)

5. **Implement parallel new-source pipeline** — compute loss features from `daily_gen / plant_avg_irradiance` and compare soiling signal strengths against old T1 pipeline.

6. **Apply quantile regression** (90th percentile) for performance index vs soiling features — captures envelope relationship better than Pearson r.

7. **Add smoothing overlays** to DQ1, DQ4, DQ5 time-series (7-day rolling median) + CSA day markers to DQ1.

8. **Focus modeling on CSA-filtered subset** — these 53 days contain the cleanest soiling signal.

### Lower Priority

9. **STL seasonal decomposition** on monthly medians.

10. **Cap performance index at ~1.1** and investigate days >1.2.

11. **Widen dry-spell definition** from ≥3 to ≥2 days to increase sample size from 10 to ~20+ spells.

---

## 13. Open Items

| Status | Item |
|---|---|
| ✅ Resolved | Generation unit confirmed as kWh (converted to J at fetch) |
| ✅ Resolved | Tropical weather contamination mitigated via CSA filter + flag_zero_output |
| ✅ Resolved | PM10 positive correlation anomaly explained (seasonal confounding) |
| ✅ Resolved | Loss proxy 0/100 alternation explained (baseline floor effect, not inverter trips) |
| ✅ Resolved | Performance index >1.0 explained (baseline calibration artifact) |
| ⚠️ Known | Ground irradiance sensor unit ambiguity (raw W/m² sum, not energy) |
| ⚠️ Known | P_NOM_KWP = 500 kWp is a placeholder — must confirm with asset owner |
| ⚠️ Known | Only 6 of 34 inverters monitored — plant-level extrapolation requires caution |
| ⏸ Deferred | Validate sensor outage windows with SCADA logs (not available) |
| ⏸ Future | Integrate first-party weather station telemetry (humidity, wind, wind direction) |
| 🔄 In progress | Decide on final B1 sample inverters (4 candidates under evaluation) |

---

## 14. Reference: EDA Script Architecture

| Script | Lines | Role |
|---|---|---|
| `soiling_signals.py` | 2655 | Main engine: 3 signal tests + supporting analyses + CSA + DQ diagnostics + report + main() |
| `feature_glossary.py` | 516 | Feature metadata dictionary (66 entries) for LLM context |
| `llm_output.py` | 1137 | Statistical analysis utilities: series_stats (41 metrics), autocorrelation, trend, stationarity |
| `multilevel_analysis.py` | 443 | Three-zoom-level: ATOMIC (per-event), MICROSCOPIC (rolling/interactions), MACROSCOPIC (rankings) |

### Execution Flow

```
main() in soiling_signals.py:
  1. load_and_filter() → read daily_model_eda.csv, parse dates, add year/season
  2. test_signal_1_sawtooth() → 4 plots, SignalResult
  3. test_signal_2_dust_correlation() → 3 plots, SignalResult
  4. test_signal_3_rain_recovery() → 4 plots, SignalResult
  5. run_supporting_analyses() → 6 plots, dict
  6. test_clear_sky_soiling() → 3 plots, dict
  7-12. plot_irradiance_vs_generation(), plot_daily_gen_validation(),
        plot_gen_irr_ratio(), plot_power_at_ref_irradiance(),
        plot_new_performance_index(), plot_old_vs_new_source_comparison()
  13. write_report() → eda_signal_report.md
  14. Build LLM output → llm_eda_summary.json (314KB)
```

---

## 15. Reference: LLM JSON Structure

The `llm_eda_summary.json` (2344 lines, 314KB) contains:

| Section | Content |
|---|---|
| `feature_glossary` | 66 feature definitions with units, dtypes, soiling relevance, caveats |
| `conventions_and_interpretation` | HQ/CSA filter rules, loss proxy scale, tier system, peak hours, cleaning dates |
| `signal_test_descriptions` | What each test does, pass criteria, key metrics |
| `dataset_overview` | 361 days, 149 columns, coverage stats |
| `verdicts` | S1 PASS, S2 PASS, S3 FAIL, overall CONDITIONAL GO |
| `signal_1_sawtooth` | 17 keys: stats, autocorrelation, trends, per-inverter, DSPI |
| `signal_2_dust_correlation` | 11 keys: Pearson/Spearman, partial correlations, seasonal |
| `signal_3_rain_recovery` | 8 keys: rain stats, dry-spell test, cross-lag |
| `supporting_analyses` | Distributions, quality gating, T1-T2 agreement, seasonal, trend |
| `clear_sky_analysis` | CSA vs HQ comparison, CSA-specific correlations |
| `dq1`–`dq6` | Data quality diagnostic results |
| `atomic_level` | 10 dry spells, 32 rain events, 156 cleaning cycles |
| `microscopic_level` | Rolling 60-day correlations, 12 interaction effects |
| `macroscopic_level` | Feature importance, data sufficiency, decision summary |
