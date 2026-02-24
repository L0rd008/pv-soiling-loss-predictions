# Pipeline Replication Guide

This is the entry point for reproducing the project pipeline exactly, from ThingsBoard fetch to EDA-ready daily tables.

Detailed stage guides:

1. `docs/pipeline_replication/01_fetching.md`
2. `docs/pipeline_replication/02_cleaning_preprocessing.md`
3. `docs/pipeline_replication/03_audit_validation.md`
4. `docs/pipeline_replication/04_eda_guide_daily_model_eda.md`

## Why This Is Split

The previous single file mixed fetching, cleaning, preprocessing, and audit details in one place. Splitting by stage keeps each section maintainable and aligned with the scripts.

## End-to-End Order

1. Fetch ThingsBoard telemetry files.
2. Build tiered inverter files (`primary` and `secondary`).
3. Run deterministic cleaning and preprocessing.
4. Run audit and validation outputs.

## Quick Run

```bash
python scripts/inverter_data_fetch.py
python scripts/irradiance_data_fetch.py
python scripts/power_generation_data_fetch.py
python scripts/split_inverter_tiers.py
python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed
python scripts/data_quality_audit.py --data-dir data --out-dir artifacts/audit
```

Optional:

```bash
python scripts/b1_availability_comparison.py
```

## Reproducibility Notes

- `scripts/data_preprocess.py` and `scripts/data_quality_audit.py` prefer `data/inverters_tiered_primary_10min.csv` if it exists; otherwise they fall back to `data/inverters_2025_to_current_10min_avg_si.csv`.
- Solcast inputs are optional and must be located at:
  - `data/soiling_2025_to_current_10min_none_std.csv`
  - `data/irradiance_2025_to_current_10min_none_std.csv`
- Output row counts change as new days are fetched. Treat row/column counts as run snapshots, not fixed constants.
