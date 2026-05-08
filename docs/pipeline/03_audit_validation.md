# Stage 3: Audit and Validation Guide

This guide reproduces the independent audit pipeline and validation artifacts.

## Run Audit

```bash
python scripts/data_quality_audit.py --data-dir data --out-dir artifacts/audit
```

## Input Resolution Rules

Audit uses the same inverter preference as preprocessing:

- Prefer `data/inverters_tiered_primary_10min.csv`
- Fallback `data/inverters_2025_to_current_10min_avg_si.csv`

Required files:

- `data/irradiance_2025_to_current_15min_sum_si.csv`
- `data/power_generation_2025_to_current_1day_none_si.csv`

Optional Solcast files (auto-detected):

- `data/soiling_2025_to_current_10min_none_std.csv`
- `data/irradiance_2025_to_current_10min_none_std.csv`

## Outputs

CSV artifacts:

- `artifacts/audit/dataset_profile.csv`
- `artifacts/audit/interval_distribution.csv`
- `artifacts/audit/missingness_by_column.csv`
- `artifacts/audit/daily_features.csv`
- `artifacts/audit/daily_flags.csv`

Report and plots:

- `artifacts/audit/quality_summary.md`
- `artifacts/audit/interval_histograms.png`
- `artifacts/audit/normalized_output_and_performance_proxy.png`
- `artifacts/audit/availability_and_imbalance.png`

## Optional Availability Investigation

```bash
python scripts/b1_availability_comparison.py
```

Outputs:

- `artifacts/b1_availability/b1_all_availability.csv`
- `artifacts/b1_availability/b1_availability_report.md`

Use this for inverter-level telemetry reliability ranking and B1 vs B2 gap diagnosis.

## Consistency Checks Between Pipelines

Run both pipelines and compare key metrics:

```powershell
python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed
python scripts/data_quality_audit.py --data-dir data --out-dir artifacts/audit
```

Recommended checks:

1. `flag_coverage_gap` count should match between preprocessed and audit daily tables.
2. `performance_loss_pct_proxy` should remain in `[0, 100]`.
3. `normalized_output` should not exceed `500000`.
4. Coverage logic should reflect the `< 30% expected records` threshold.

Quick check command:

```powershell
python -c "import pandas as pd; p=pd.read_csv('artifacts/preprocessed/daily_model_input.csv'); a=pd.read_csv('artifacts/audit/daily_features.csv'); print('pre_cov', int(p['flag_coverage_gap'].sum())); print('audit_cov', int(a['flag_coverage_gap'].sum())); print('perf_bounds', p['performance_loss_pct_proxy'].min(), p['performance_loss_pct_proxy'].max())"
```
