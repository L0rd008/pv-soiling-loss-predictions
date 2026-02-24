# Stage 1: Fetching Guide

This guide replicates raw data collection exactly as implemented by the fetch scripts.

## Prerequisites

- Python 3.10+
- Dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create `.env` from `.env.example` and fill real IDs/tokens.

Required keys used by fetch scripts:

- `TB_URL`
- `TB_TOKEN`
- `TB_INVERTERS`
- `TB_INV_KEYS`
- `TB_WSTN_ID`
- `TB_IRR_KEYS`
- `TB_PLNT_ID`
- `TB_GEN_KEYS`

Optional keys:

- `TB_OUTPUT_DIR` (default `data`)
- `TB_TZ_OFFSET` (default `+05:30`)
- `TB_START_DATE` (default `2025-01-01`)
- `TB_REQUEST_TIMEOUT_S` (default `30`)
- `TB_GEN_MAX_J` (default `360000000000`)

Important:

- `TB_GEN_KEYS` should be `EnergyMeter_dailyGeneration`.
- Raw `EnergyMeter_dailyGeneration` is in kWh and converted to Joules in `scripts/power_generation_data_fetch.py` using `KWH_TO_JOULES = 3,600,000`.

## Step 1: Fetch Inverter Telemetry

```bash
python scripts/inverter_data_fetch.py
```

Behavior:

- Aggregation: `AVG`
- Interval: 10 minutes
- Chunking: 3-day windows
- Current conversion: mA to A (`/1000`)
- Sanity caps: power > 300,000 W to NaN, current > 250 A to NaN
- Negative values: set to 0 for night hours (19:00-05:00), else NaN

Output:

- `data/inverters_2025_to_current_10min_avg_si.csv`

## Step 2: Fetch Irradiance Telemetry

```bash
python scripts/irradiance_data_fetch.py
```

Behavior:

- Aggregation: `SUM`
- Interval: 15 minutes
- Chunking: 5-day windows
- Negative irradiance: set to 0 at night (19:00-05:00), else NaN

Output:

- `data/irradiance_2025_to_current_15min_sum_si.csv`

## Step 3: Fetch Plant Generation Telemetry

```bash
python scripts/power_generation_data_fetch.py
```

Behavior:

- Entity type: `ASSET`
- Aggregation: `NONE` (raw)
- Single request (no chunking)
- Converts kWh to J
- Invalid values: negative or above `TB_GEN_MAX_J` set to NaN

Output:

- `data/power_generation_2025_to_current_1day_none_si.csv`

## Step 4: Fetch B1 Candidate Inverters (If Needed)

If `data/b1_candidates/inverters_2025_to_current_10min_avg_si.csv` is missing or stale, fetch it separately.

PowerShell example:

```powershell
$env:TB_OUTPUT_DIR = "data/b1_candidates"
$env:TB_INVERTERS = "B1-01:<uuid>,B1-05:<uuid>,B1-12:<uuid>,B1-16:<uuid>"
python scripts/inverter_data_fetch.py
$env:TB_OUTPUT_DIR = "data"
```

Keep exactly 4 B1 candidates in this file for the split step.

## Step 5: Build Tiered Inverter Files

```bash
python scripts/split_inverter_tiers.py
```

Inputs:

- `data/inverters_2025_to_current_10min_avg_si.csv`
- `data/b1_candidates/inverters_2025_to_current_10min_avg_si.csv`

Outputs:

- `data/inverters_tiered_primary_10min.csv`
- `data/inverters_secondary_10min_avg_si.csv`

Primary tier design:

- Tier-1 (B2 training): `B2-08`, `B2-13`, `B2-17`
- Tier-2 (B1 validation): `B1-08`, `B1-01`, `B1-13`

## Quick Validation

```powershell
Get-ChildItem data | Select-Object Name,Length
```

You should see all five files present before preprocessing:

- `inverters_2025_to_current_10min_avg_si.csv`
- `inverters_tiered_primary_10min.csv`
- `inverters_secondary_10min_avg_si.csv`
- `irradiance_2025_to_current_15min_sum_si.csv`
- `power_generation_2025_to_current_1day_none_si.csv`
