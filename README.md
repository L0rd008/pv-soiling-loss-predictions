# Soiling Loss Predictions for Utility-Scale PV

This repository is for building AI/ML workflows that help PV stakeholders detect failures early, run real-time anomaly detection, estimate soiling/degradation trends, and optimize cleaning/O&M decisions.

Current plant context:
- Plant size: ~10-15 MW
- Inverters in plant: 34
- Inverters in this dataset: 8 (4 from block `B1`, 4 from block `B2`)
- Data source: ThingsBoard telemetry exports

## Stakeholder-Focused Outcomes

`Control Center Operators`
- Real-time anomaly alerts (unexpected underproduction, inverter block drops, current imbalance).
- Daytime-only incident filtering to reduce false alarms.

`Performance Engineers`
- Soiling loss proxy trend (% loss vs rolling clean baseline).
- Degradation proxy trend (long-term normalized-output slope).
- Inverter/block-level underperformance ranking.

`Field Service / O&M`
- Cleaning trigger recommendations (date window + expected recoverable energy).
- Prioritized field checks (likely sensor fault, inverter derating, cable/phase issue).

`Asset Owners / Investors`
- Daily and monthly KPI summaries with confidence flags.
- Lost-energy estimate due to soiling/anomalies.

`Portfolio Managers`
- Plant-comparable normalized KPIs and alert severity buckets.
- Data quality scorecards to qualify reporting confidence.

## Available Data

1. `data/inverters_2025_to_current_10min_avg_si.csv`
- 10-minute aggregated inverter telemetry (`AVG`).
- Power and phase currents (SI units).
- 8 inverters sampled.

2. `data/irradiance_2025_to_current_15min_sum_si.csv`
- 15-minute weather station telemetry (`SUM`).
- Horizontal and tilted irradiance.

3. `data/power_generation_2025_to_current_1day_none_si.csv`
- Raw asset generation telemetry (`NONE`).
- Stored as energy in Joules after script-side scaling.
- Contains many intraday rows and irregular timing, not strictly one row per day.

## Known Data Characteristics

- Timestamps are irregular and not perfectly aligned to expected clock boundaries.
- Sampling gaps exist due to connectivity/device issues and nighttime no-sun periods.
- Invalid or missing values exist (NaNs/empties/possible fallback substitutions).
- The generation file has multiple records per day and irregular intervals.
- Inverter file includes only a subset (8/34) of plant inverters; direct plant-level comparisons must account for this.

## Repository Workflow

1. Fetch telemetry data from ThingsBoard using scripts in `scripts/`.
2. Run data-quality audit + preprocessing for ML feature readiness.
3. Use curated daily features for model training and anomaly logic.
4. Generate stakeholder-facing KPI and alert outputs.

## Repository Structure

- `data/`: raw exported telemetry CSVs.
- `scripts/`: data fetchers and quality/EDA tooling.
- `docs/`: engineering notes and review findings.
- `artifacts/`: generated audit outputs (ignored by git).

## Environment Configuration

Base environment variables are documented in `.env.example`.

Additional optional controls:
- `TB_OUTPUT_DIR`: export location for fetched CSV files (default `data`).
- `TB_REQUEST_TIMEOUT_S`: HTTP timeout for ThingsBoard calls (default `30`).
- `TB_GEN_MAX_J`: upper sanity threshold for generation values in Joules.

## Data Quality + EDA Script

Run:

```bash
python scripts/data_quality_audit.py --data-dir data --out-dir artifacts/audit
```

Outputs (under `artifacts/audit/`):
- `dataset_profile.csv`: row/column/time-range/interval stats.
- `interval_distribution.csv`: interval frequency distributions by dataset.
- `missingness_by_column.csv`: null-rate profile.
- `daily_features.csv`: merged daily features and proxy KPIs.
- `daily_flags.csv`: anomaly-oriented daily flags.
- `quality_summary.md`: concise audit report for review.

Plots (if `matplotlib` is installed):
- `interval_histograms.png`
- `normalized_output_and_soiling_proxy.png`
- `availability_and_imbalance.png`

## Suggested Modeling Tracks

1. Failure Prediction (ahead of time)
- Input: rolling window of normalized output, irradiance context, current imbalance, availability.
- Output: probability of near-term underperformance/fault event (next 1-7 days).

2. Real-Time Anomaly Detection
- Input: live inverter power/current + irradiance.
- Method: robust residual or isolation-based anomaly detector over daytime periods.
- Output: alert with reason code (`underpower`, `phase_imbalance`, `sensor_suspect`, `data_gap`).

3. Cleaning Optimization
- Input: soiling-loss proxy trend + cleaning cost assumptions + energy price.
- Output: recommended cleaning date and expected payback window.

## Additional Predictions Derivable From Current Data

- Soiling rate (% loss/day) from normalized-output drift vs rolling clean baseline.
- Block-level mismatch index (`B1` vs `B2` relative output under similar irradiance).
- Inverter health score (availability, variability, imbalance, anomaly counts).
- Sensor consistency score (horizontal vs tilted irradiance coherence).
- Recoverable-energy estimate for prioritized maintenance actions.

## Engineering Notes

- Keep secrets out of version control (`.env` is ignored).
- Keep generated artifacts out of git (`artifacts/` ignored).
- Keep raw CSV exports out of git by default (`data/*.csv` ignored).
- Prefer reproducible scripts over manual spreadsheet edits.
- Treat raw exports as immutable and write processed outputs to separate locations.
- See `docs/script_and_data_review.md` for current script/data risk findings.

## Immediate Next Steps

1. Run `scripts/data_quality_audit.py` and review `artifacts/audit/quality_summary.md`.
2. Validate generation-unit conversion assumptions in `scripts/power_generation_data_fetch.py` against ThingsBoard key semantics.
3. Build the first training dataset from `daily_features.csv` with clear target definitions.
