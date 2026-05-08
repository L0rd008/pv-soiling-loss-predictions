# Stage 2: Cleaning and Preprocessing Guide

This guide reproduces deterministic cleaning and daily feature assembly.

## Run

Default run:

```bash
python scripts/3_preprocess/preprocess.py --data-dir data --out-dir artifacts/preprocessed
```

Optional overlap trim:

```bash
python scripts/3_preprocess/preprocess.py --data-dir data --out-dir artifacts/preprocessed --trim-to-overlap
```

Physical-PR options (defaults shown):

```bash
python scripts/3_preprocess/preprocess.py \
  --inverter-capacity-kw 330 \
  --plant-inverter-count 34 \
  --runtime-min-hours 6 \
  --runtime-max-hours 18 \
  --runtime-csv data/time_series_chart_time_series_chart.csv \
  --disable-new-source-if-clipped-pct 5
```

## Input Resolution Rules

`data_preprocess.py` uses:

1. Inverters:
   - Prefer `data/inverters_tiered_primary_10min.csv`
   - Fallback `data/inverters_2025_to_current_10min_avg_si.csv`
2. Irradiance:
   - `data/irradiance_2025_to_current_15min_sum_si.csv`
3. Generation:
   - `data/power_generation_2025_to_current_1day_none_si.csv`
4. Solcast (optional):
   - `data/soiling_2025_to_current_10min_none_std.csv`
   - `data/irradiance_2025_to_current_10min_none_std.csv`

If Solcast files are absent, preprocessing continues without those columns.

5. Per-inverter daily generation (optional):
   - `data/inverters_daily_gen_2025_to_current_none_si.csv`
6. Plant average solar radiation (optional):
   - `data/plant_avg_irradiance_2025_to_current_none_si.csv`

If the new telemetry files (5, 6) are absent, preprocessing continues without
those columns. No existing features are affected.

## Cleaning Logic Implemented

Inverters (`clean_inverters`):

- Parse `Date`, coerce numeric columns
- Deduplicate by `Timestamp` (mean on collisions)
- Invalid values to NaN:
  - Power `< 0` or `> 300000 W`
  - Current `< 0` or `> 250 A`
- Add helper columns:
  - `subset_power_w`
  - `row_power_completeness`

Irradiance (`clean_irradiance`):

- Parse and deduplicate by `Timestamp`
- Negative irradiance set to NaN

Generation (`clean_generation`):

- Parse and deduplicate by `Timestamp`
- Invalid values to NaN:
  - `< 0` or `> 360000000000 J`
- Daily aggregation:
  - `daily_generation_j_latest`
  - fallback `daily_generation_j_max`
- Add `generation_intraday_spread_j`

Inverter daily generation (`clean_inverter_daily_gen`, optional):

- Parse and deduplicate by `Timestamp`
- Reject negatives only (`< 0` -> NaN); no destructive upper clipping
- Aggregate per inverter/day with `last`, `max`, `p99`
- Daily selection rule: use `max`, unless `max > 1.25 * p99`, then use `p99`
- Post-aggregation outlier fence per inverter: `Q3 + 3*IQR` on selected daily values
- Convert kWh to Joules (`* 3,600,000`)
- Output columns: `{inv_label}_daily_gen_j`
- Audit export: `artifacts/preprocessed/inverter_daily_gen_audit.csv`

Plant average irradiance (`clean_plant_avg_irradiance`, optional):

- Parse and deduplicate by `Timestamp`
- Group by day, take last reading (end-of-day = daily average)
- Invalid values to NaN: `< 0` or `> 1500 W/m^2`
- Output column: `plant_avg_irradiance_wm2`

## Daily Feature Assembly

`build_daily_model_table` merges daily aggregates and computes:

- Combined performance proxy columns
- Tier-1 performance proxy columns (`t1_*`)
- Tier-2 performance proxy columns (`t2_*`)
- Block features (`b1_*`, `b2_*`, `block_mismatch_ratio`)
- Quality flags (`flag_*`)
- Transfer readiness (`transfer_quality_score`, `transfer_quality_tier`, `cross_plant_inference_ready`)
- Common overlap marker (`in_common_overlap`)
- Per-inverter daily generation (`{inv}_daily_gen_j`, `subset_daily_gen_j`, `subset_daily_gen_kwh`) — if new telemetry is present
- Plant average irradiance (`plant_avg_irradiance_wm2`) — if new telemetry is present
- Runtime merge (`runtime_h`, `runtime_source`) using runtime CSV first, Solcast daylight fallback second
- Physical irradiation (`irradiation_kwh_m2 = plant_avg_irradiance_wm2 * runtime_h / 1000`)
- Physical PR fields:
  - `subset_pr_physical_raw`, `subset_pr_physical_outlier`, `subset_pr_physical_interp`
  - `plant_pr_physical_raw`, `plant_pr_physical_outlier`
- Backward-compatible aliases:
  - `gen_irr_ratio` = `subset_pr_physical_raw`
  - `gen_irr_ratio_smoothed` = 7-day median of `gen_irr_ratio`
- Power at reference irradiance (`power_at_ref_irradiance_w`, tier variants) — from existing active power matched with on-site irradiance at the dataset median level
- New-source performance loss proxy (`new_normalized_output`, `new_rolling_clean_baseline`, `new_performance_loss_pct_proxy`, `new_perf_loss_rate_14d_pct_per_day`, `new_normalized_output_14d_median`) — same pipeline as old-source but now built from physical PR alias
- New-source cycle deviation (`new_cycle_id`, `new_soiling_index_x`, `new_cycle_max_x`, `new_cycle_deviation_pct`) — cycle-aware deviation using `gen_irr_ratio` directly

Important interpretation:

- `performance_loss_pct_proxy` is an all-cause performance deficit proxy, not pure soiling truth.
- `gen_irr_ratio` is now a physical PR alias: `subset_daily_gen_kwh / (subset_capacity_kw * irradiation_kwh_m2)`.
- `power_at_ref_irradiance_w` controls for irradiance variation by extracting active power only when irradiance is near the dataset median (+/-15%).

## Outputs

Always written:

- `artifacts/preprocessed/inverters_clean.csv`
- `artifacts/preprocessed/irradiance_clean.csv`
- `artifacts/preprocessed/generation_clean.csv`
- `artifacts/preprocessed/generation_daily_clean.csv`
- `artifacts/preprocessed/daily_model_input.csv`
- `artifacts/preprocessed/daily_model_eda.csv`
- `artifacts/preprocessed/preprocessing_summary.md`

Behavior note:

- `daily_model_eda.csv` is always overlap-filtered (`in_common_overlap == True`).
- `daily_model_input.csv` is full by default; it is trimmed only when `--trim-to-overlap` is passed.

## Validation Checks

Check shapes:

```powershell
python -c "import pandas as pd; print('input', pd.read_csv('artifacts/preprocessed/daily_model_input.csv').shape); print('eda', pd.read_csv('artifacts/preprocessed/daily_model_eda.csv').shape)"
```

Check overlap integrity:

```powershell
python -c "import pandas as pd; d=pd.read_csv('artifacts/preprocessed/daily_model_input.csv'); print('overlap_days', int(d['in_common_overlap'].sum()))"
```
