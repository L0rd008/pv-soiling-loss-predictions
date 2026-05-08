# Pipeline Replication Guide

This is the entry point for reproducing the project pipeline exactly, from
ThingsBoard fetch through EDA signal analysis.

Detailed stage guides:

1. `docs/pipeline/01_fetching.md`
2. `docs/pipeline/02_cleaning_preprocessing.md`
3. `docs/pipeline/03_audit_validation.md`
4. `docs/pipeline/04_eda_features.md`
5. `docs/pipeline/05_eda_soiling_signals.md`

## Why This Is Split

The previous single file mixed fetching, cleaning, preprocessing, and audit
details in one place. Splitting by stage keeps each section maintainable and
aligned with the scripts.

## End-to-End Order

1. Fetch ThingsBoard telemetry files.
2. Build tiered inverter files (`primary` and `secondary`).
3. Run deterministic cleaning and preprocessing.
4. Run audit and validation outputs.
5. Run EDA soiling signal analysis.

## Quick Run

```bash
python scripts/1_fetch/inverter_power.py
python scripts/1_fetch/irradiance.py
python scripts/1_fetch/power_generation.py
python scripts/1_fetch/inverter_daily_gen.py
python scripts/1_fetch/plant_avg_irradiance.py
python scripts/2_organize/split_tiers.py
python scripts/3_preprocess/preprocess.py --data-dir data --out-dir artifacts/preprocessed
python scripts/4_audit/audit.py --data-dir data --out-dir artifacts/audit
python scripts/5_eda/soiling_signals.py
```

Optional:

```bash
python scripts/utils/b1_availability_comparison.py
```

## Reproducibility Notes

- `scripts/3_preprocess/preprocess.py` and `scripts/4_audit/audit.py` prefer `data/inverters_tiered_primary_10min.csv` if it exists; otherwise they fall back to `data/inverters_2025_to_current_10min_avg_si.csv`.
- Solcast inputs are optional and must be located at:
  - `data/soiling_2025_to_current_10min_none_std.csv`
  - `data/irradiance_2025_to_current_10min_none_std.csv`
- Output row counts change as new days are fetched. Treat row/column counts as run snapshots, not fixed constants.
