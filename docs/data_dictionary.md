# Data Dictionary

Reference for raw telemetry, cleaned outputs, and daily modeling features.

## Context

- Source data comes from one 10-15 MW plant.
- This project also targets cross-plant inference for plants not yet analyzed.
- For cross-plant use, rely on quality-gated features and transfer-readiness fields.

## Raw Data Files (`data/`)

### `inverters_2025_to_current_10min_avg_si.csv`

10-minute AVG inverter telemetry for 6 tiered inverters (3 B2 Tier-1 + 3 B1 Tier-2, subset of 34 total).

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | integer | ms since epoch | Telemetry timestamp |
| `Date` | datetime | local time | Human-readable local timestamp |
| `<inverter> Active Power (W)` | float | W | Active power |
| `<inverter> Current A (A)` | float | A | Phase A current |
| `<inverter> Current B (A)` | float | A | Phase B current |
| `<inverter> Current C (A)` | float | A | Phase C current |

Known issue:
- Block B1 channels show significantly higher missingness than B2.

### `irradiance_2025_to_current_15min_sum_si.csv`

15-minute SUM weather-station irradiance telemetry.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | integer | ms since epoch | Telemetry timestamp |
| `Date` | datetime | local time | Human-readable local timestamp |
| `<sensor> Horizontal Irradiance (SUM)` | float | W*s/m^2 | Horizontal irradiance sum |
| `<sensor> Tilted Irradiance (SUM)` | float | W*s/m^2 | Tilted irradiance sum |

Known issue:
- Late 2025 includes low-coverage periods and likely sensor outages.

### `power_generation_2025_to_current_1day_none_si.csv`

Asset-level generation feed (`agg=NONE`) with irregular intraday records.

| Column | Type | Unit | Description |
|---|---|---|---|
| `Timestamp` | integer | ms since epoch | Telemetry timestamp |
| `Date` | datetime | local time | Human-readable local timestamp |
| `Energy Meter Daily Generation (J)` | float | J | Converted generation value |

Unit note:
- `EnergyMeter_dailyGeneration` is treated as kWh and converted as `kWh * 3,600,000 = J`.

## Preprocessing Outputs (`artifacts/preprocessed/`)

### `inverters_clean.csv`

Cleaned inverter telemetry after deduplication and sanity filtering.

Extra columns:
- `day` (date key)
- `subset_power_w`
- `row_power_completeness`

### `irradiance_clean.csv`

Cleaned irradiance telemetry after deduplication and negative-value handling.

Extra columns:
- `day` (date key)

### `generation_clean.csv`

Cleaned generation telemetry after deduplication and range filtering.

Extra columns:
- `day` (date key)

### `generation_daily_clean.csv`

Daily generation view derived from raw generation telemetry.

| Column | Description |
|---|---|
| `day` | Day key |
| `daily_generation_j_latest` | Last value seen in day |
| `daily_generation_j_max` | Max value seen in day |
| `daily_generation_j_min` | Min value seen in day |
| `generation_records` | Number of telemetry records in day |
| `daily_generation_j` | Preferred daily generation (`latest` then fallback to `max`) |
| `generation_intraday_spread_j` | `max - min` within day |

### `daily_model_input.csv`

Daily modeling table for training, monitoring, and cross-plant transfer gating.

Interpretation:
- `performance_loss_pct_proxy` is an all-cause proxy (not a direct soiling label).
- Use quality flags and operational context before treating high-loss days as
  soiling-only events.

Core fields:
- `subset_energy_j`
- `subset_power_w_p95`
- `subset_data_availability_mean`
- `inverter_coverage_ratio`
- `irradiance_horizontal_sum`
- `irradiance_tilted_sum`
- `irradiance_coverage_ratio`
- `daily_generation_j`
- `plant_to_subset_energy_ratio`
- `normalized_output`
- `rolling_clean_baseline`
- `performance_loss_pct_proxy`
- `perf_loss_rate_14d_pct_per_day`

Block fields:
- `b1_energy_j`, `b2_energy_j`
- `b1_data_availability`, `b2_data_availability`
- `block_mismatch_ratio`
- `block_mismatch_ratio_rolling_median`

Environmental features (from Solcast):
- `pm25_mean`, `pm25_max` — fine dust (µg/m³)
- `pm10_mean`, `pm10_max` — coarse dust (µg/m³)
- `precipitation_total_mm` — daily rainfall
- `humidity_mean`, `humidity_max` — relative humidity (%)
- `wind_speed_10m_mean`, `wind_speed_10m_max` — surface wind (m/s)
- `wind_speed_100m_mean` — regional wind (m/s)
- `dewpoint_mean` — dewpoint temperature (°C)
- `air_temp_mean`, `air_temp_min`, `air_temp_max` — air temperature (°C)
- `cloud_opacity_mean` — cloud cover (%)
- `pressure_mean` — surface pressure (hPa)
- `solcast_ghi_sum`, `solcast_gti_sum` — modelled irradiance (W·s/m²)
- `solcast_dni_mean` — direct normal irradiance (W/m²)
- `rain_day` — boolean: any precipitation > 0.1 mm/h
- `dominant_weather` — most frequent weather type string

Quality flags:
- `flag_sensor_suspect_irradiance`
- `flag_coverage_gap`
- `flag_block_mismatch`

Cross-plant transfer fields:
- `transfer_quality_score` (0-100)
- `transfer_quality_tier` (`high`/`medium`/`low`)
- `cross_plant_inference_ready` (boolean)

## Raw Solcast Data (`data/`)

### `soiling_2025_to_current_10min_none_std.csv`

10-minute Solcast environmental data for performance-relevant variables (PM, weather, wind, etc.).

| Column | Type | Unit | Description |
|---|---|---|---|
| `air_temp_celcius` | float | °C | Ambient air temperature |
| `cloud_opacity_percentage` | float | % | Cloud cover percentage |
| `dewpoint_temp_celcius` | float | °C | Dewpoint temperature |
| `precipitable_water_kg_m2` | float | kg/m² | Column precipitable water |
| `precipitation_rate_mm_h` | float | mm/h | Precipitation rate |
| `relative_humidity_percentage` | float | % | Relative humidity |
| `surface_pressure_hpa` | float | hPa | Surface pressure |
| `wind_direction_100m_deg_from_north` | float | ° | Wind direction at 100m |
| `wind_direction_10m_deg_from_north` | float | ° | Wind direction at 10m |
| `wind_speed_100m_m_s` | float | m/s | Wind speed at 100m |
| `wind_speed_10m_m_s` | float | m/s | Wind speed at 10m |
| `weather_type_str` | string | — | Weather type category |
| `min_air_temp_celcius` | float | °C | Min temperature in period |
| `max_air_temp_celcius` | float | °C | Max temperature in period |
| `pm10_micro_g_m3` | float | µg/m³ | PM10 particulate matter |
| `pm2.5_micro_g_m3` | float | µg/m³ | PM2.5 particulate matter |
| `period_end` | datetime | ISO 8601 | Period end timestamp |
| `period` | string | ISO 8601 | Duration of period (PT10M) |

### `irradiance_2025_to_current_10min_none_std.csv`

10-minute Solcast modelled irradiance data.

| Column | Type | Unit | Description |
|---|---|---|---|
| `albedo_frac` | float | fraction | Ground albedo |
| `azimuth_deg_from_north` | float | ° | Solar azimuth angle |
| `dhi_w_m2` | float | W/m² | Diffuse horizontal irradiance |
| `dni_w_m2` | float | W/m² | Direct normal irradiance |
| `ghi_w_m2` | float | W/m² | Global horizontal irradiance |
| `gti_w_m2` | float | W/m² | Global tilted irradiance |
| `zenith_deg_from_vertical` | float | ° | Solar zenith angle |
| `period_end` | datetime | ISO 8601 | Period end timestamp |
| `period` | string | ISO 8601 | Duration of period (PT10M) |

## Audit Outputs (`artifacts/audit/`)

`scripts/data_quality_audit.py` produces:
- `dataset_profile.csv`
- `interval_distribution.csv`
- `missingness_by_column.csv`
- `daily_features.csv`
- `daily_flags.csv`
- `quality_summary.md`
