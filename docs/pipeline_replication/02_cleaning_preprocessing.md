# Stage 2: Cleaning and Preprocessing Guide

This guide reproduces deterministic cleaning and daily feature assembly.

## Run

Default run:

```bash
python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed
```

Optional overlap trim:

```bash
python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed --trim-to-overlap
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

## Daily Feature Assembly

`build_daily_model_table` merges daily aggregates and computes:

- Combined performance proxy columns
- Tier-1 performance proxy columns (`t1_*`)
- Tier-2 performance proxy columns (`t2_*`)
- Block features (`b1_*`, `b2_*`, `block_mismatch_ratio`)
- Quality flags (`flag_*`)
- Transfer readiness (`transfer_quality_score`, `transfer_quality_tier`, `cross_plant_inference_ready`)
- Common overlap marker (`in_common_overlap`)

Important interpretation:

- `performance_loss_pct_proxy` is an all-cause performance deficit proxy, not pure soiling truth.

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
