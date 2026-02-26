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

## Per-Inverter Metrics (in `daily_model_input.csv`)

Computed by `aggregate_per_inverter_daily()` for each of the 6 tiered inverters.
Column names use the pattern `{inv_label}_*` where `inv_label` is the lowercase
inverter name with hyphens replaced by underscores (e.g. `b2_08`).

| Column Pattern | Type | Unit | Description |
|---|---|---|---|
| `{inv}_energy_j` | float64 | J | Daily energy for individual inverter |
| `{inv}_pr` | float64 | dimensionless | Performance Ratio using `P_NOM_KWP` placeholder |
| `{inv}_normalized_output` | float64 | J/(W-s/m^2) | Per-inverter normalized output |

Combined PR:

| Column | Type | Unit | Description |
|---|---|---|---|
| `subset_pr` | float64 | dimensionless | Combined PR for all tiered inverters |

## Soiling Features (in `daily_model_input.csv`)

Computed by `compute_soiling_features()`. Require Solcast columns to be present.

| Column | Type | Unit | Description |
|---|---|---|---|
| `days_since_last_rain` | int | days | Consecutive days since last `rain_day == True` |
| `days_since_significant_rain` | int | days | Days since `precipitation_total_mm >= 5.0 mm` |
| `cumulative_pm10_since_rain` | float64 | ug/m^3-days | Running sum of `pm10_mean` reset on rain days |
| `cumulative_pm25_since_rain` | float64 | ug/m^3-days | Running sum of `pm25_mean` reset on rain days |
| `humidity_x_pm10` | float64 | %*ug/m^3 | `humidity_mean * pm10_mean` (cementation proxy) |
| `wind_speed_10m_rolling_7d` | float64 | m/s | 7-day rolling mean of `wind_speed_10m_mean` |
| `month` | int | 1-12 | Calendar month |
| `season` | string | category | `dry` (Jan-Mar, Jun-Sep) or `wet` (Apr-May, Oct-Dec) |

## Domain Soiling Pressure Index (in `daily_model_input.csv`)

Computed by `compute_domain_soiling_index()`. A physics-based soiling estimate
built entirely from environmental satellite data. No plant performance data is
used. Weights are calibrated via constrained optimisation that maximises
positive correlation with PM and negative with rainfall while penalising
correlation with non-soiling factors.

| Column | Type | Unit | Description |
|---|---|---|---|
| `domain_soiling_daily` | float64 | composite | Daily soiling pressure rate: `(w_pm25*PM2.5 + w_pm10*PM10) * humidity_factor * dew_factor * cementation_factor` |
| `domain_soiling_index` | float64 | composite (cumul.) | Cumulative sum of `domain_soiling_daily`, resets on rain >= 1 mm or cleaning campaigns |

## pvlib Soiling Estimates (in `daily_model_input.csv`)

Computed by `compute_pvlib_soiling_ratio()` from 10-min Solcast data.

| Column | Type | Unit | Description |
|---|---|---|---|
| `pvlib_soiling_ratio_hsu` | float64 | ratio 0-1 | Daily mean HSU soiling ratio (1 = clean) |
| `pvlib_soiling_loss_kimber` | float64 | fraction 0-0.3 | Kimber soiling loss fraction (0 = clean) |

## Temperature Correction (in `daily_model_input.csv`)

Computed by `compute_temperature_corrected_pr()` using pvlib SAPM model.

| Column | Type | Unit | Description |
|---|---|---|---|
| `pr_temperature_corrected` | float64 | % | Performance loss proxy with thermal effects removed |

## Cycle-Aware Deviation (in `daily_model_input.csv`)

Computed by `compute_cycle_deviation()`. Cycles are delimited by rain or cleaning events.

| Column | Type | Unit | Description |
|---|---|---|---|
| `cycle_id` | int | count | Cycle identifier, increments at each rain/cleaning reset |
| `soiling_index_x` | float64 | J*s | X = energy / mean_irradiance_rate |
| `cycle_max_x` | float64 | J*s | Max X within the current cycle |
| `cycle_deviation_pct` | float64 | % 0-100 | `100 * (1 - X / cycle_max_x)` |

## Audit Outputs (`artifacts/audit/`)

`scripts/data_quality_audit.py` produces:
- `dataset_profile.csv`
- `interval_distribution.csv`
- `missingness_by_column.csv`
- `daily_features.csv`
- `daily_flags.csv`
- `quality_summary.md`

## EDA Outputs (`artifacts/eda/`)

`scripts/eda_soiling_signals.py` produces a quantitative signal report and
19 diagnostic plots. See `docs/eda_output_interpretation.md` for how to read
each output.

### Report

| File | Description |
|---|---|
| `eda_signal_report.md` | Go/no-go verdicts for three soiling signals with quantitative metrics |

### Signal 1 Plots (Sawtooth Detection)

| File | Description |
|---|---|
| `plots/s1_loss_proxy_timeseries.png` | Loss proxy & domain soiling index time-series with rain/cleaning overlays |
| `plots/s1_per_inverter_output.png` | Per-inverter normalised output (6 panels) |
| `plots/s1_cycle_deviation.png` | Cycle-aware deviation time-series with cycle boundaries |
| `plots/s1_dryspell_slopes.png` | Linear soiling rates fitted within dry spells |

### Signal 2 Plots (PM/Dust Correlation)

| File | Description |
|---|---|
| `plots/s2_pm10_scatter_panels.png` | PM10 vs loss rate, raw and clear-sky deconfounded |
| `plots/s2_top_predictors_vs_deviation.png` | Top 3 predictors (days since rain, cumul. PM2.5, cumul. PM10) vs cycle deviation |
| `plots/s2_feature_heatmap.png` | Feature correlation matrix (environmental, engineered, targets) |

### Signal 3 Plots (Rain Recovery)

| File | Description |
|---|---|
| `plots/s3_rain_event_study.png` | Mean loss trajectory around significant rain events |
| `plots/s3_dryspell_start_end.png` | Paired comparison: dry-spell start vs end loss |
| `plots/s3_recovery_vs_precipitation.png` | Recovery magnitude vs rainfall amount |
| `plots/s3_rain_event_study_seasonal.png` | Seasonal split of rain event study |

### Supporting Analysis Plots

| File | Description |
|---|---|
| `plots/s4_univariate_distributions.png` | Histograms of loss proxy, precipitation, PM10, cycle deviation, DSPI daily rate, loss rate (2x3 grid) |
| `plots/s4_pvlib_vs_observed.png` | Physics-based estimates (pvlib + DSPI) vs observed proxy (2x2 grid) |
| `plots/s4_sensor_dirt_check.png` | Solcast/ground sensor ratio trend over time |
| `plots/s4_tier_validation.png` | T1 vs T2 loss proxy overlay |
| `plots/s4_seasonal_boxplots.png` | Monthly loss distribution box plots |
| `plots/s4_quality_gating.png` | Quality score distribution and tier counts |

### Domain Soiling Pressure Index Plots

| File | Description |
|---|---|
| `plots/s5_domain_soiling_index.png` | DSPI cumulative index vs cycle deviation time-series with rain/cleaning overlays |
| `plots/s5_dspi_correlation_profile.png` | Horizontal bar chart of DSPI correlation with each environmental and performance feature |
