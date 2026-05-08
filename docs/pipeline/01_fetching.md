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
- `TB_INV_DAILY_GEN_KEYS` (default `daily_generated_electricity`) — for Step 6
- `TB_INV_DAILY_GEN_WARN_KWH` (default `1000`) — warning-only threshold for Step 6
- `TB_PLNT_IRR_KEYS` (default `avg_solar_radiation`) — for Step 7

Important:

- `TB_GEN_KEYS` should be `EnergyMeter_dailyGeneration`.
- Raw `EnergyMeter_dailyGeneration` is in kWh and converted to Joules in `scripts/1_fetch/power_generation.py` using `KWH_TO_JOULES = 3,600,000`.

## Step 1: Fetch Inverter Telemetry

```bash
python scripts/1_fetch/inverter_power.py
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
python scripts/1_fetch/irradiance.py
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
python scripts/1_fetch/power_generation.py
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
python scripts/1_fetch/inverter_power.py
$env:TB_OUTPUT_DIR = "data"
```

Keep exactly 4 B1 candidates in this file for the split step.

## Step 5: Build Tiered Inverter Files

```bash
python scripts/2_organize/split_tiers.py
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

## Step 6: Fetch Per-Inverter Daily Generated Electricity (Optional)

```bash
python scripts/1_fetch/inverter_daily_gen.py
```

Requires env vars: `TB_INV_DAILY_GEN_KEYS=daily_generated_electricity`

Behavior:

- Entity type: `DEVICE` (same inverters from `TB_INVERTERS`)
- Aggregation: `NONE` (raw cumulative values that reset at midnight)
- Chunking: 30-day windows
- Negative values are dropped; non-negative values are preserved (no destructive upper clipping)
- Values above `TB_INV_DAILY_GEN_WARN_KWH` are counted as warnings only
- No unit conversion in fetch (kWh preserved; preprocessing converts to J)
- Writes fetch audit sidecar: `data/inverters_daily_gen_fetch_audit.json`

Output:

- `data/inverters_daily_gen_2025_to_current_none_si.csv`

Note: This provides a direct daily kWh total per inverter, independent of the
active-power-based energy integral. The end-of-day cumulative value equals
the daily generation total.

## Step 7: Fetch Plant Average Solar Radiation (Optional)

```bash
python scripts/1_fetch/plant_avg_irradiance.py
```

Requires env vars: `TB_PLNT_IRR_KEYS=avg_solar_radiation`

Behavior:

- Entity type: `ASSET` (same plant asset as generation, `TB_PLNT_ID`)
- Aggregation: `NONE` (raw running-average values that reset at midnight)
- Single request (no chunking)
- Sanity cap: values > 1500 W/m^2 or negative set to NaN
- No unit conversion (W/m^2 preserved as-is)

Output:

- `data/plant_avg_irradiance_2025_to_current_none_si.csv`

Note: The `avg_solar_radiation` key stores a running daily average (W/m^2).
The end-of-day value equals the true daily-average irradiance.

## Quick Validation

```powershell
Get-ChildItem data | Select-Object Name,Length
```

You should see all five core files present before preprocessing:

- `inverters_2025_to_current_10min_avg_si.csv`
- `inverters_tiered_primary_10min.csv`
- `inverters_secondary_10min_avg_si.csv`
- `irradiance_2025_to_current_15min_sum_si.csv`
- `power_generation_2025_to_current_1day_none_si.csv`

Optional (new telemetry) files:

- `inverters_daily_gen_2025_to_current_none_si.csv`
- `plant_avg_irradiance_2025_to_current_none_si.csv`
