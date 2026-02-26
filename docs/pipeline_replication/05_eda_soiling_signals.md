# Stage 5: EDA Soiling Signal Analysis

This guide reproduces the exploratory data analysis that tests three go/no-go
signals for soiling loss prediction.

## Prerequisites

Stages 1-4 must be complete. The input file
`artifacts/preprocessed/daily_model_eda.csv` must exist with all 114 columns
(including per-inverter metrics, soiling features, pvlib estimates, and
cycle-aware deviation).

Install `scipy` if not already present:

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/eda_soiling_signals.py
```

Custom paths:

```bash
python scripts/eda_soiling_signals.py --input path/to/daily_model_eda.csv --out-dir artifacts/eda
```

## Inputs

| File | Role |
|---|---|
| `artifacts/preprocessed/daily_model_eda.csv` | Daily feature table (361 rows, 114 columns) |

The script also imports constants from `scripts/daily_features.py`:
`CLEANING_CAMPAIGN_DATES`, `SIGNIFICANT_RAIN_MM`, `SITE_LAT`, `SITE_LON`.

## What It Does

The script tests three go/no-go signals that determine whether soiling loss is
predictable from the available data:

1. **Signal 1 (Sawtooth)**: Checks for gradual performance loss accumulation
   between rain events, punctuated by sudden drops — the soiling fingerprint.
2. **Signal 2 (PM/Dust)**: Tests whether particulate matter correlates with
   soiling rate after controlling for weather confounders (cloud opacity,
   temperature).
3. **Signal 3 (Rain Recovery)**: Tests whether significant rainfall causes a
   measurable decrease in the loss proxy.

It also runs six supporting analyses (univariate distributions, pvlib
comparison, sensor dirt check, tier validation, seasonal patterns, quality
gating).

All analyses are restricted to high-quality, zero-flag days (246 of 361) for
statistical tests, while time-series plots show the full dataset.

## Outputs

### Report

`artifacts/eda/eda_signal_report.md` — Quantitative findings for all three
signals with per-signal verdicts (PASS / WEAK / FAIL) and an overall go/no-go
decision.

### Plots

All saved to `artifacts/eda/plots/`:

| Plot | Signal | Description |
|---|---|---|
| `s1_loss_proxy_timeseries.png` | 1 | Loss proxy time-series with rain and cleaning overlays |
| `s1_per_inverter_output.png` | 1 | Per-inverter normalised output (6 panels) |
| `s1_cycle_deviation.png` | 1 | Cycle-aware deviation time-series |
| `s1_dryspell_slopes.png` | 1 | Linear soiling rates fitted within dry spells |
| `s2_pm10_scatter_panels.png` | 2 | PM10 vs loss rate — raw and clear-sky |
| `s2_cumulative_pm10_vs_deviation.png` | 2 | Cumulative dust exposure vs cycle deviation |
| `s2_feature_heatmap.png` | 2 | Feature correlation matrix |
| `s3_rain_event_study.png` | 3 | Loss trajectory around significant rain events |
| `s3_dryspell_start_end.png` | 3 | Paired comparison: dry-spell start vs end loss |
| `s3_recovery_vs_precipitation.png` | 3 | Recovery magnitude vs rainfall amount |
| `s3_rain_event_study_seasonal.png` | 3 | Seasonal split of rain event study |
| `s4_univariate_distributions.png` | Support | Histograms of loss proxy, precipitation, PM10 |
| `s4_pvlib_vs_observed.png` | Support | pvlib soiling estimate vs observed proxy |
| `s4_sensor_dirt_check.png` | Support | Solcast/ground sensor ratio trend |
| `s4_tier_validation.png` | Support | T1 vs T2 loss proxy overlay |
| `s4_seasonal_boxplots.png` | Support | Monthly loss distribution box plots |
| `s4_quality_gating.png` | Support | Quality score distribution and tier counts |

## Validation Checks

Verify output completeness:

```powershell
python -c "import os; plots=[f for f in os.listdir('artifacts/eda/plots') if f.endswith('.png')]; print(f'{len(plots)} plots'); assert len(plots) >= 14, 'Expected at least 14 plots'"
```

Verify the report exists and contains all signal verdicts:

```powershell
python -c "t=open('artifacts/eda/eda_signal_report.md').read(); assert 'Signal 1' in t and 'Signal 2' in t and 'Signal 3' in t and 'Overall Go/No-Go' in t; print('Report OK')"
```

## Interpreting Results

See `docs/eda_output_interpretation.md` for a detailed guide on how to read
each plot and what patterns to look for.

The overall verdict logic:

| Signals passing | Verdict |
|---|---|
| 3/3 | **Strong go** — proceed to modeling |
| 2/3 | **Conditional go** — proceed with caution |
| 1/3 or 2+ weak | **Weak go** — consider additional data |
| 0/3 | **No-go** — re-evaluate research direction |
