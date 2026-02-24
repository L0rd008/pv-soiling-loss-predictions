# Pipeline Replication Guide

How to replicate the full data pipeline from scratch — fetching raw telemetry through to EDA-ready daily features.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip packages | `pandas`, `numpy`, `requests`, `python-dotenv` |

Install dependencies:

```bash
pip install pandas numpy requests python-dotenv
```

## 1. Environment Setup

Create a `.env` file in the project root with these variables:

```ini
# ThingsBoard connection
TB_URL=https://your-thingsboard-instance.com
TB_TOKEN=your_jwt_bearer_token

# Inverters (name:device_uuid pairs, comma-separated)
TB_INVERTERS=B1-04:uuid1,B1-08:uuid2,B1-13:uuid3,B1-17:uuid4,B2-04:uuid5,B2-08:uuid6,B2-13:uuid7,B2-17:uuid8
TB_INV_KEYS=active_power,current_a,current_b,current_c

# Weather station
TB_WSTN_ID=weather_station_device_uuid
TB_IRR_KEYS=wstn_horiz_irr,wstn_tilted_irr

# Plant-level generation
TB_PLNT_ID=plant_asset_uuid
TB_GEN_KEYS=EnergyMeter_dailyGeneration

# Optional overrides
TB_OUTPUT_DIR=data              # default: data
TB_TZ_OFFSET=+05:30            # default: +05:30
TB_START_DATE=2025-01-01        # default: 2025-01-01
TB_REQUEST_TIMEOUT_S=30         # default: 30
TB_GEN_MAX_J=360000000000       # sanity cap for generation (J)
```

> [!IMPORTANT]
> The `TB_TOKEN` is a JWT Bearer token from ThingsBoard. It expires periodically — refresh it before each fetch session via `scripts/jwt_fetch.py` or the ThingsBoard UI.

## 2. Data Fetching

All fetch scripts use `scripts/tb_client.py` — a shared client that handles chunked HTTP requests with retry/pacing, environment loading, and CSV export.

Run in this order (steps 1–3 are independent and can run in parallel):

### Step 1: Fetch Inverter Telemetry

```bash
python scripts/inverter_data_fetch.py
```

| Detail | Value |
|---|---|
| **Output** | `data/inverters_2025_to_current_10min_avg_si.csv` |
| **Aggregation** | 10-minute `AVG` |
| **Chunk size** | 3-day windows |
| **Unit conversions** | Current: mA → A (÷ 1000) |
| **Sanity caps** | Power: 300 kW, Current: 250 A |
| **Night handling** | Negative values during 19:00–05:00 → 0; during day → NaN |
| **Columns per inverter** | Active Power (W), Current A/B/C (A) |
| **Runtime** | ~5–10 minutes |

### Step 2: Fetch Irradiance Telemetry

```bash
python scripts/irradiance_data_fetch.py
```

| Detail | Value |
|---|---|
| **Output** | `data/irradiance_2025_to_current_15min_sum_si.csv` |
| **Aggregation** | 15-minute `SUM` |
| **Chunk size** | 5-day windows |
| **Night handling** | Same as inverters (negative → 0 at night, NaN during day) |
| **Columns** | Horizontal Irradiance (SUM), Tilted Irradiance (SUM) |
| **Runtime** | ~2–3 minutes |

### Step 3: Fetch Generation Telemetry

```bash
python scripts/power_generation_data_fetch.py
```

| Detail | Value |
|---|---|
| **Output** | `data/power_generation_2025_to_current_1day_none_si.csv` |
| **Aggregation** | `NONE` (raw data, single request) |
| **Unit conversions** | kWh → Joules (× 3,600,000) |
| **Entity type** | `ASSET` (plant-level, not device) |
| **Sanity cap** | `TB_GEN_MAX_J` env var (default ~360 GJ) |
| **Runtime** | ~1 minute |

### Step 4: Solcast Data (Manual Download)

Solcast soiling index and irradiance data is **downloaded manually** from the [Solcast API portal](https://solcast.com/) and placed in `data/solcast/`:

```
data/solcast/
├── solcast_soiling.csv         # Soiling loss index
└── solcast_irradiance.csv      # Satellite-derived GHI/GTI
```

These files are optional — the pipeline runs without them but produces fewer features.

## 3. Inverter Tier Split

After fetching, split the 8-inverter raw file into tiered subsets:

```bash
python scripts/split_inverter_tiers.py
```

This merges the raw inverter CSV with `data/b1_candidates/inverters_2025_to_current_10min_avg_si.csv` (4 additional B1 inverters fetched separately), then splits into:

| Output | Inverters | Purpose |
|---|---|---|
| `data/inverters_tiered_primary_10min.csv` | B2-08, B2-13, B2-17, B1-08, B1-01, B1-13 | Pipeline input (6 tiered) |
| `data/inverters_secondary_10min_avg_si.csv` | B1-04, B1-17, B2-04, B1-05, B1-12, B1-16 | Reserve/augmentation |

The primary file gains a `tier` column:

| Tier | Block | Inverters | Role |
|---|---|---|---|
| 1 | B2 | B2-08, B2-13, B2-17 | Training (high availability) |
| 2 | B1 | B1-08, B1-01, B1-13 | Validation (cross-block) |

> [!NOTE]
> The raw 8-inverter file is **preserved unmodified**. The split script writes to a separate derived path. If the raw file is ever accidentally overwritten, use `python scripts/recover_raw_inverters.py` to reconstruct it from the split outputs.

## 4. Cleaning & Preprocessing

```bash
python scripts/data_preprocess.py [--trim-to-overlap]
```

### What It Does

The pipeline reads the tiered primary inverter CSV (falls back to raw CSV if tiered doesn't exist), irradiance, generation, and Solcast data, then applies deterministic cleaning and feature engineering.

### 4a. Cleaning (Per-Source)

**Inverters** (`clean_inverters`):
1. Load CSV, parse `Date`, coerce all value columns to numeric
2. Deduplicate by `Timestamp` (average numeric columns on collision)
3. Sanity-cap: power > 300 kW or < 0 → NaN; current > 250 A or < 0 → NaN
4. Compute `subset_power_w` (sum of all Active Power columns) and `row_power_completeness`

**Irradiance** (`clean_irradiance`):
1. Load, deduplicate by Timestamp
2. Negative irradiance → NaN

**Generation** (`clean_generation`):
1. Load, deduplicate by Timestamp
2. Values < 0 or > 360 GJ → NaN
3. Aggregate to daily: take `last` value per day (cumulative meter), fall back to `max`
4. Compute `generation_intraday_spread_j` (max − min per day)

### 4b. Daily Feature Assembly (`build_daily_model_table`)

All sub-daily data is aggregated to daily and merged on `day`:

```
Inverter (10-min) ─┐
                   ├─→ Merge on day ─→ Performance features ─→ Quality flags ─→ Transfer readiness
Irradiance (15-min)┤
                   ├─→ Tier-1/Tier-2 split features
Generation (daily) ┤
                   └─→ Cross-block correlation
Solcast (daily) ───┘
```

Key feature groups (80 columns total):

| Group | Example Columns | Source |
|---|---|---|
| Inverter energy | `subset_energy_j`, `subset_power_w_p95`, `subset_data_availability_mean` | Aggregated from 10-min |
| Block energy | `b1_energy_j`, `b2_energy_j`, `block_mismatch_ratio` | B1/B2 column split |
| Irradiance | `irradiance_tilted_sum`, `irradiance_coverage_ratio` | Aggregated from 15-min |
| Generation | `daily_generation_j`, `generation_intraday_spread_j` | Daily aggregation |
| **Tier-1 performance** | `t1_energy_j`, `t1_normalized_output`, `t1_performance_loss_pct_proxy` | B2 inverters only |
| **Tier-2 performance** | `t2_energy_j`, `t2_normalized_output`, `t2_performance_loss_pct_proxy` | B1 inverters only |
| Cross-tier | `tier_loss_correlation`, `tier_loss_delta`, `tier_agreement_flag` | 30-day rolling Pearson r |
| Combined performance | `normalized_output`, `rolling_clean_baseline`, `performance_loss_pct_proxy` | All inverters |
| Quality flags | `flag_sensor_suspect_irradiance`, `flag_coverage_gap`, `flag_low_output_high_irr` | Tier-1 preferred |
| Transfer readiness | `transfer_quality_score`, `transfer_quality_tier`, `cross_plant_inference_ready` | Tier-1 preferred |
| Overlap | `in_common_overlap` | Non-null check on 3 core columns |
| Solcast | `solcast_soiling_index`, `solcast_ghi_sum`, etc. | Solcast CSV |

### 4c. Outputs

| File | Rows | Description |
|---|---|---|
| `artifacts/preprocessed/daily_model_input.csv` | 418 | Full daily table (all days) |
| `artifacts/preprocessed/daily_model_eda.csv` | 383 | Overlap-filtered (all 3 sources present) |
| `artifacts/preprocessed/inverters_clean.csv` | — | Cleaned 10-min inverter data |
| `artifacts/preprocessed/irradiance_clean.csv` | — | Cleaned 15-min irradiance data |
| `artifacts/preprocessed/generation_clean.csv` | — | Cleaned raw generation data |
| `artifacts/preprocessed/generation_daily_clean.csv` | — | Daily-aggregated generation |
| `artifacts/preprocessed/preprocessing_summary.md` | — | Human-readable summary report |

The `--trim-to-overlap` flag filters `daily_model_input.csv` to only common-overlap days (same as the EDA table).

## 5. Data Quality Audit

```bash
python scripts/data_quality_audit.py
```

Runs the same feature pipeline independently and produces:

| Output | Description |
|---|---|
| `artifacts/audit/daily_features.csv` | Audit-pipeline feature table (73 cols) |
| `artifacts/audit/data_quality_summary.md` | Quality report with flag counts |

## 6. Availability Comparison (Optional)

```bash
python scripts/b1_availability_comparison.py
```

Produces per-inverter daily availability rankings across all fetched inverters:

| Output | Description |
|---|---|
| `artifacts/b1_availability/b1_all_availability.csv` | Per-inverter daily availability |
| `artifacts/b1_availability/b1_availability_report.md` | Ranked report with tier validation |

## Complete Run (Copy-Paste)

```bash
# 1. Fetch (requires valid .env with ThingsBoard credentials)
python scripts/inverter_data_fetch.py
python scripts/irradiance_data_fetch.py
python scripts/power_generation_data_fetch.py

# 2. Split into tiers
python scripts/split_inverter_tiers.py

# 3. Clean & preprocess
python scripts/data_preprocess.py

# 4. Audit
python scripts/data_quality_audit.py

# 5. Availability comparison (optional)
python scripts/b1_availability_comparison.py
```

## File Map

```
data/
├── inverters_2025_to_current_10min_avg_si.csv      ← Raw fetch (8 inverters)
├── inverters_tiered_primary_10min.csv               ← Split: 6 tiered (Tier 1+2)
├── inverters_secondary_10min_avg_si.csv             ← Split: 6 secondary
├── irradiance_2025_to_current_15min_sum_si.csv      ← Raw fetch
├── power_generation_2025_to_current_1day_none_si.csv ← Raw fetch
├── b1_candidates/
│   └── inverters_2025_to_current_10min_avg_si.csv   ← Separate B1 candidate fetch
└── solcast/                                          ← Manual download

scripts/
├── tb_client.py                    ← Shared ThingsBoard HTTP client
├── jwt_fetch.py                    ← JWT token refresh
├── inverter_data_fetch.py          ← Step 1
├── irradiance_data_fetch.py        ← Step 2
├── power_generation_data_fetch.py  ← Step 3
├── split_inverter_tiers.py         ← Step 4
├── data_preprocess.py              ← Step 5 (main pipeline)
├── daily_features.py               ← Shared feature functions
├── data_quality_audit.py           ← Step 6
├── b1_availability_comparison.py   ← Step 7 (optional)
└── recover_raw_inverters.py        ← Recovery utility

artifacts/
├── preprocessed/
│   ├── daily_model_input.csv       ← PRIMARY OUTPUT (418 × 80)
│   ├── daily_model_eda.csv         ← EDA TABLE (383 × 80)
│   └── ...
└── audit/
    └── ...
```
