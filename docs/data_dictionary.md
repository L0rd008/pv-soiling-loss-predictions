# Data Dictionary

Reference for all columns in the raw and generated CSV files.

---

## Raw Data Files

### `inverters_2025_to_current_10min_avg_si.csv`

10-minute AVG-aggregated inverter telemetry for 8 inverters (4 from Block 1, 4 from Block 2).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| Timestamp | int | ms (epoch) | Unix timestamp in milliseconds |
| Date | datetime | — | Local timestamp (IST, UTC+05:30) |
| `<inv_name> Active Power (W)` | float | W | 10-min average active power output |
| `<inv_name> Current A (A)` | float | A | Phase A current (converted from mA) |
| `<inv_name> Current B (A)` | float | A | Phase B current (converted from mA) |
| `<inv_name> Current C (A)` | float | A | Phase C current (converted from mA) |

**Inverter naming**: `B1_INV_01` … `B1_INV_04` (Block 1), `B2_INV_01` … `B2_INV_04` (Block 2).

> **Known issue**: B1 inverters have ~43-45% missing power data; B2 is ~2-5% missing.
> Only 8 of 34 plant inverters are included.

### `irradiance_2025_to_current_15min_sum_si.csv`

15-minute SUM-aggregated weather station irradiance.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| Timestamp | int | ms (epoch) | Unix timestamp in milliseconds |
| Date | datetime | — | Local timestamp (IST) |
| `<sensor> Horizontal Irradiance (SUM)` | float | W·s/m² | Horizontal irradiance sum over 15 min |
| `<sensor> Tilted Irradiance (SUM)` | float | W·s/m² | Tilted (plane-of-array) irradiance sum over 15 min |

> **Known issue**: Dec 2025 has sensor outage/gaps (4–12 records/day vs normal ~90+). Treat as invalid.

### `power_generation_2025_to_current_1day_none_si.csv`

Raw asset-level generation telemetry. **Not strictly daily** — contains multiple intraday records.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| Timestamp | int | ms (epoch) | Unix timestamp in milliseconds |
| Date | datetime | — | Local timestamp (IST) |
| `Energy Meter Daily Generation (J)` | float | J (Joules) | Daily generation converted from kWh (`raw_kWh × 3,600,000`) |

> **Unit conversion**: ThingsBoard stores `energymeter_dailygeneration` in kWh. Multiply by 1000 × 3600 = 3,600,000 for Joules.

---

## Generated Files (`artifacts/audit/`)

### `daily_features.csv`

Daily aggregated features computed by `data_quality_audit.py`.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| day | date | — | Calendar date |
| subset_energy_j | float | J | Total energy from 8 sampled inverters |
| subset_power_w_p95 | float | W | 95th percentile of combined inverter power |
| subset_data_availability_mean | float | ratio (0–1) | Mean fraction of non-null power columns |
| subset_data_availability_p10 | float | ratio (0–1) | 10th percentile availability |
| phase_imbalance_mean | float | ratio | Mean phase current imbalance |
| phase_imbalance_p95 | float | ratio | 95th percentile imbalance |
| inverter_records | int | count | Number of 10-min records in the day |
| irradiance_horizontal_sum | float | W·s/m² | Daily sum of horizontal irradiance |
| irradiance_tilted_sum | float | W·s/m² | Daily sum of tilted irradiance |
| irradiance_records | int | count | Number of 15-min irradiance records |
| daily_generation_j_latest | float | J | Last reported generation value for the day |
| daily_generation_j_max | float | J | Max generation value for the day |
| daily_generation_j_min | float | J | Min generation value for the day |
| generation_records | int | count | Number of generation telemetry points |
| generation_intraday_spread_j | float | J | Max − Min generation within the day |
| subset_energy_mwh | float | MWh | Subset energy converted to MWh |
| generation_mwh_latest | float | MWh | Latest generation converted to MWh |
| plant_to_subset_energy_ratio | float | ratio | Plant generation / subset energy |
| b1_energy_j | float | J | Block 1 daily energy (if available) |
| b2_energy_j | float | J | Block 2 daily energy (if available) |
| b1_data_availability | float | ratio (0–1) | Block 1 data availability |
| b2_data_availability | float | ratio (0–1) | Block 2 data availability |
| block_mismatch_ratio | float | ratio | B1 energy / B2 energy |
| block_mismatch_ratio_rolling_median | float | ratio | 14-day rolling median of mismatch ratio |
| normalized_output | float | J/(W·s/m²) | Energy / irradiance (capped at 500,000) |
| normalized_output_14d_median | float | J/(W·s/m²) | 14-day rolling median |
| rolling_clean_baseline | float | J/(W·s/m²) | 30-day rolling 95th percentile of clear-day output |
| soiling_loss_pct_proxy | float | % (0–100) | `100 × (1 − output/baseline)` |
| soiling_rate_14d_pct_per_day | float | %/day | Rate of soiling change over 14 days |

### `daily_flags.csv`

Binary anomaly flags per day.

| Flag | Description |
|------|-------------|
| flag_low_data_availability | Availability < 50% |
| flag_high_phase_imbalance | 95th percentile imbalance > 0.12 |
| flag_zero_irr_nontrivial_gen | Irradiance ≤ 1 W·s/m² but generation > 5 MWh |
| flag_low_output_high_irr | Output < 70% of 14-day median under high irradiance |
| flag_large_generation_intraday_spread | Spread exceeds 95th percentile |
| flag_sensor_suspect_irradiance | Irradiance below 50,000 W·s/m² with non-trivial inverter output |
| flag_coverage_gap | Day has < 50% of expected inverter records (72 of 144) |
| flag_block_mismatch | B1/B2 ratio deviates > 15% from its rolling median |
| flag_count | Total flags triggered |
