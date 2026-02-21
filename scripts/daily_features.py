"""Shared daily feature building for PV telemetry.

Provides reusable functions for aggregating inverter, irradiance, and generation
data into daily features used by both the audit and preprocessing pipelines.

Constants and thresholds are defined here so both consumers stay in sync.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

INVERTER_INTERVAL_S = 600       # 10-minute records
IRRADIANCE_INTERVAL_S = 900     # 15-minute records
EXPECTED_INV_RECORDS_PER_DAY = 144   # 24h * 6
EXPECTED_IRR_RECORDS_PER_DAY = 96    # 24h * 4

# Minimum daily irradiance sum (W·s/m²) for a day to qualify for baseline.
# Days below this are treated as sensor outage and excluded from the rolling
# clean baseline.  50 000 W·s/m² ≈ ~14 W/m² average over 1 hour.
MIN_IRRADIANCE_FOR_BASELINE = 50_000.0

# Sanity cap for normalized_output (energy_J / irradiance_sum).  Values above
# this are clipped to prevent single-day spikes from corrupting the baseline.
MAX_NORMALIZED_OUTPUT = 500_000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_block_power_cols(
    columns,
) -> Tuple[List[str], List[str]]:
    """Partition power columns into B1 and B2 lists by inverter name prefix."""
    b1, b2 = [], []
    for col in columns:
        if not col.endswith("Active Power (W)"):
            continue
        lo = col.lower()
        if "b1" in lo or "block1" in lo or "block 1" in lo:
            b1.append(col)
        elif "b2" in lo or "block2" in lo or "block 2" in lo:
            b2.append(col)
    return b1, b2


def detect_irradiance_cols(
    columns,
) -> Tuple[str, str]:
    """Return (tilted_col, horizontal_col) from irradiance column names."""
    irr_cols = [c for c in columns if "Irradiance" in c]
    if not irr_cols:
        raise ValueError("No irradiance columns found.")
    tilted = [c for c in irr_cols if "Tilted" in c]
    horiz = [c for c in irr_cols if "Horizontal" in c]
    tilted_col = tilted[0] if tilted else irr_cols[0]
    horiz_col = horiz[0] if horiz else irr_cols[0]
    return tilted_col, horiz_col


# ---------------------------------------------------------------------------
# Solcast data aggregation
# ---------------------------------------------------------------------------

SOLCAST_INTERVAL_S = 600  # 10-minute Solcast records

# Columns from soiling CSV and their roles
_SOLCAST_SOILING_COLS = {
    "pm2.5_micro_g_m3": "pm25",
    "pm10_micro_g_m3": "pm10",
    "precipitation_rate_mm_h": "precipitation",
    "relative_humidity_percentage": "humidity",
    "wind_speed_10m_m_s": "wind_speed_10m",
    "wind_speed_100m_m_s": "wind_speed_100m",
    "dewpoint_temp_celcius": "dewpoint",
    "air_temp_celcius": "air_temp",
    "min_air_temp_celcius": "air_temp_min_raw",
    "max_air_temp_celcius": "air_temp_max_raw",
    "cloud_opacity_percentage": "cloud_opacity",
    "surface_pressure_hpa": "pressure",
    "weather_type_str": "weather_type",
}


def _load_solcast_csv(path) -> pd.DataFrame:
    """Load a Solcast CSV, parse period_end as datetime, extract day."""
    df = pd.read_csv(path)
    if "period_end" not in df.columns:
        raise ValueError(f"Solcast CSV missing 'period_end' column: {path}")
    df["period_end"] = pd.to_datetime(df["period_end"], utc=True, errors="coerce")
    df["day"] = df["period_end"].dt.tz_convert("Asia/Kolkata").dt.floor("D").dt.tz_localize(None)
    return df


def aggregate_solcast_daily(
    soiling_path,
    irradiance_path=None,
) -> pd.DataFrame:
    """Aggregate 10-minute Solcast data to daily summaries.

    Parameters
    ----------
    soiling_path : Path
        Path to the Solcast soiling/environmental CSV.
    irradiance_path : Path, optional
        Path to the Solcast irradiance CSV.  If None, only environmental
        features are returned.

    Returns
    -------
    pd.DataFrame
        Daily aggregated features with ``day`` as datetime column.
    """
    from pathlib import Path
    soiling_path = Path(soiling_path)
    sol = _load_solcast_csv(soiling_path)

    # Cast numeric columns
    for raw_col in _SOLCAST_SOILING_COLS:
        if raw_col in sol.columns and raw_col != "weather_type_str":
            sol[raw_col] = pd.to_numeric(sol[raw_col], errors="coerce")

    # --- Numeric aggregations ---
    agg_dict = {}

    # PM2.5
    if "pm2.5_micro_g_m3" in sol.columns:
        agg_dict["pm25_mean"] = ("pm2.5_micro_g_m3", "mean")
        agg_dict["pm25_max"] = ("pm2.5_micro_g_m3", "max")

    # PM10
    if "pm10_micro_g_m3" in sol.columns:
        agg_dict["pm10_mean"] = ("pm10_micro_g_m3", "mean")
        agg_dict["pm10_max"] = ("pm10_micro_g_m3", "max")

    # Precipitation: rate (mm/h) × (10 min / 60 min) = mm per interval
    if "precipitation_rate_mm_h" in sol.columns:
        sol["_precip_mm_step"] = sol["precipitation_rate_mm_h"] * (10.0 / 60.0)
        agg_dict["precipitation_total_mm"] = ("_precip_mm_step", "sum")
        sol["_has_rain"] = sol["precipitation_rate_mm_h"] > 0.1
        agg_dict["rain_day"] = ("_has_rain", "any")

    # Humidity
    if "relative_humidity_percentage" in sol.columns:
        agg_dict["humidity_mean"] = ("relative_humidity_percentage", "mean")
        agg_dict["humidity_max"] = ("relative_humidity_percentage", "max")

    # Wind
    if "wind_speed_10m_m_s" in sol.columns:
        agg_dict["wind_speed_10m_mean"] = ("wind_speed_10m_m_s", "mean")
        agg_dict["wind_speed_10m_max"] = ("wind_speed_10m_m_s", "max")
    if "wind_speed_100m_m_s" in sol.columns:
        agg_dict["wind_speed_100m_mean"] = ("wind_speed_100m_m_s", "mean")

    # Dewpoint
    if "dewpoint_temp_celcius" in sol.columns:
        agg_dict["dewpoint_mean"] = ("dewpoint_temp_celcius", "mean")

    # Temperature
    if "air_temp_celcius" in sol.columns:
        agg_dict["air_temp_mean"] = ("air_temp_celcius", "mean")
    if "min_air_temp_celcius" in sol.columns:
        agg_dict["air_temp_min"] = ("min_air_temp_celcius", "min")
    if "max_air_temp_celcius" in sol.columns:
        agg_dict["air_temp_max"] = ("max_air_temp_celcius", "max")

    # Cloud & pressure
    if "cloud_opacity_percentage" in sol.columns:
        agg_dict["cloud_opacity_mean"] = ("cloud_opacity_percentage", "mean")
    if "surface_pressure_hpa" in sol.columns:
        agg_dict["pressure_mean"] = ("surface_pressure_hpa", "mean")

    sol_daily = sol.groupby("day").agg(**agg_dict).reset_index()

    # --- Weather type mode (dominant weather) ---
    if "weather_type_str" in sol.columns:
        weather_mode = (
            sol.groupby("day")["weather_type_str"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
            .reset_index()
            .rename(columns={"weather_type_str": "dominant_weather"})
        )
        sol_daily = sol_daily.merge(weather_mode, on="day", how="left")

    # --- Solcast irradiance (optional) ---
    if irradiance_path is not None:
        irradiance_path = Path(irradiance_path)
        if irradiance_path.exists():
            irr = _load_solcast_csv(irradiance_path)
            for c in ["ghi_w_m2", "gti_w_m2", "dni_w_m2", "dhi_w_m2"]:
                if c in irr.columns:
                    irr[c] = pd.to_numeric(irr[c], errors="coerce")

            irr_agg = {}
            if "ghi_w_m2" in irr.columns:
                irr["_ghi_ws"] = irr["ghi_w_m2"] * SOLCAST_INTERVAL_S
                irr_agg["solcast_ghi_sum"] = ("_ghi_ws", "sum")
            if "gti_w_m2" in irr.columns:
                irr["_gti_ws"] = irr["gti_w_m2"] * SOLCAST_INTERVAL_S
                irr_agg["solcast_gti_sum"] = ("_gti_ws", "sum")
            if "dni_w_m2" in irr.columns:
                irr_agg["solcast_dni_mean"] = ("dni_w_m2", "mean")

            if irr_agg:
                irr_daily = irr.groupby("day").agg(**irr_agg).reset_index()
                sol_daily = sol_daily.merge(irr_daily, on="day", how="outer")

    return sol_daily


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_inverter_daily(inv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build daily inverter aggregates from cleaned 10-min data.

    Expects ``inv`` to have a ``day`` column and ``subset_power_w``.

    Returns ``(inv_daily, power_cols)`` where ``power_cols`` is the list of
    Active Power column names found.
    """
    power_cols = [c for c in inv.columns if c.endswith("Active Power (W)")]
    if not power_cols:
        raise ValueError("No inverter Active Power columns found.")

    if "subset_power_w" not in inv.columns:
        inv["subset_power_w"] = inv[power_cols].sum(axis=1, min_count=1)

    if "subset_energy_j_step" not in inv.columns:
        inv["subset_energy_j_step"] = inv["subset_power_w"] * INVERTER_INTERVAL_S

    if "row_power_completeness" not in inv.columns:
        inv["row_power_completeness"] = inv[power_cols].notna().mean(axis=1)

    inv_daily = (
        inv.groupby("day")
        .agg(
            subset_energy_j=("subset_energy_j_step", "sum"),
            subset_power_w_p95=("subset_power_w", lambda s: s.quantile(0.95)),
            subset_data_availability_mean=("row_power_completeness", "mean"),
            subset_data_availability_p10=("row_power_completeness", lambda s: s.quantile(0.10)),
            inverter_records=("Timestamp", "size"),
        )
        .reset_index()
    )
    inv_daily["inverter_coverage_ratio"] = (
        inv_daily["inverter_records"] / EXPECTED_INV_RECORDS_PER_DAY
    ).clip(upper=1.0)

    return inv_daily, power_cols


def aggregate_block_daily(inv: pd.DataFrame, power_cols: List[str]) -> pd.DataFrame:
    """Build per-block (B1 vs B2) daily energy aggregates.

    Returns a DataFrame with columns: day, b1_energy_j, b2_energy_j,
    b1_data_availability, b2_data_availability, block_mismatch_ratio,
    block_mismatch_ratio_rolling_median.

    Returns empty DataFrame if blocks can't be detected.
    """
    b1_cols, b2_cols = detect_block_power_cols(power_cols)
    if not b1_cols or not b2_cols:
        return pd.DataFrame()

    inv["b1_power_w"] = inv[b1_cols].sum(axis=1, min_count=1)
    inv["b2_power_w"] = inv[b2_cols].sum(axis=1, min_count=1)
    inv["b1_energy_j_step"] = inv["b1_power_w"] * INVERTER_INTERVAL_S
    inv["b2_energy_j_step"] = inv["b2_power_w"] * INVERTER_INTERVAL_S

    block_daily = (
        inv.groupby("day")
        .agg(
            b1_energy_j=("b1_energy_j_step", "sum"),
            b2_energy_j=("b2_energy_j_step", "sum"),
            b1_data_availability=("b1_power_w", lambda s: s.notna().mean()),
            b2_data_availability=("b2_power_w", lambda s: s.notna().mean()),
        )
        .reset_index()
    )
    block_daily["block_mismatch_ratio"] = (
        block_daily["b1_energy_j"] / block_daily["b2_energy_j"].replace(0, np.nan)
    )
    block_daily["block_mismatch_ratio_rolling_median"] = (
        block_daily["block_mismatch_ratio"].rolling(14, min_periods=5).median()
    )
    return block_daily


def aggregate_irradiance_daily(irr: pd.DataFrame) -> pd.DataFrame:
    """Build daily irradiance sums from cleaned 15-min data.

    Expects ``irr`` to have a ``day`` column.
    """
    tilted_col, horiz_col = detect_irradiance_cols(irr.columns)

    irr_daily = (
        irr.groupby("day")
        .agg(
            irradiance_horizontal_sum=(horiz_col, "sum"),
            irradiance_tilted_sum=(tilted_col, "sum"),
            irradiance_records=("Timestamp", "size"),
        )
        .reset_index()
    )
    irr_daily["irradiance_coverage_ratio"] = (
        irr_daily["irradiance_records"] / EXPECTED_IRR_RECORDS_PER_DAY
    ).clip(upper=1.0)
    return irr_daily


# ---------------------------------------------------------------------------
# Performance loss features
# ---------------------------------------------------------------------------

def compute_performance_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add normalized output, rolling baseline, and performance loss proxy.

    Expects columns: ``subset_energy_j``, ``irradiance_tilted_sum``.
    Modifies ``daily`` in place and returns it.
    """
    # Normalized output with irradiance sanity guard
    irradiance_valid = daily["irradiance_tilted_sum"] > MIN_IRRADIANCE_FOR_BASELINE
    daily["normalized_output"] = np.where(
        irradiance_valid,
        daily["subset_energy_j"] / daily["irradiance_tilted_sum"],
        np.nan,
    )
    daily["normalized_output"] = daily["normalized_output"].clip(upper=MAX_NORMALIZED_OUTPUT)

    daily["normalized_output_14d_median"] = (
        daily["normalized_output"].rolling(14, min_periods=5).median()
    )

    # Rolling clean baseline: 95th percentile on clear days
    clear_day_threshold = daily["irradiance_tilted_sum"].quantile(0.60)
    clear_day_mask = (
        (daily["irradiance_tilted_sum"] >= clear_day_threshold)
        & (daily["irradiance_tilted_sum"] > MIN_IRRADIANCE_FOR_BASELINE)
    )
    baseline_src = daily["normalized_output"].where(clear_day_mask)
    daily["rolling_clean_baseline"] = (
        baseline_src.rolling(30, min_periods=7).quantile(0.95).ffill()
    )

    # Performance loss proxy (all-cause deficit vs rolling clean baseline)
    daily["performance_loss_pct_proxy"] = (
        100.0 * (1.0 - (daily["normalized_output"] / daily["rolling_clean_baseline"]))
    ).clip(lower=0, upper=100)

    # Performance loss rate (trend feature for cleaning predictions)
    daily["perf_loss_rate_14d_pct_per_day"] = (
        (daily["performance_loss_pct_proxy"] - daily["performance_loss_pct_proxy"].shift(14)) / 14.0
    )

    return daily


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def compute_quality_flags(daily: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags to the daily table.

    Modifies ``daily`` in place and returns it.
    """
    # Sensor-suspect irradiance: low irradiance but non-trivial inverter output
    daily["flag_sensor_suspect_irradiance"] = (
        (daily["irradiance_tilted_sum"] < MIN_IRRADIANCE_FOR_BASELINE)
        & (daily["subset_energy_j"] > 0)
        & (daily["subset_data_availability_mean"] > 0.30)
    )

    # Coverage gap: less than 30% of expected records (lenient to avoid overcounting)
    daily["flag_coverage_gap"] = False
    if "inverter_coverage_ratio" in daily.columns:
        daily["flag_coverage_gap"] = daily["flag_coverage_gap"] | (
            daily["inverter_coverage_ratio"] < 0.30
        )
    if "irradiance_coverage_ratio" in daily.columns:
        daily["flag_coverage_gap"] = daily["flag_coverage_gap"] | (
            daily["irradiance_coverage_ratio"] < 0.30
        )

    # Block mismatch: B1/B2 ratio deviates >15% from rolling median
    if "block_mismatch_ratio" in daily.columns and "block_mismatch_ratio_rolling_median" in daily.columns:
        daily["flag_block_mismatch"] = (
            (daily["block_mismatch_ratio"] - daily["block_mismatch_ratio_rolling_median"]).abs()
            > 0.15 * daily["block_mismatch_ratio_rolling_median"].abs()
        ) & daily["block_mismatch_ratio"].notna()
    else:
        daily["flag_block_mismatch"] = False

    # Low output under high irradiance
    if "normalized_output_14d_median" in daily.columns:
        irr_high = daily["irradiance_tilted_sum"].quantile(0.60)
        daily["flag_low_output_high_irr"] = (
            (daily["irradiance_tilted_sum"] >= irr_high)
            & (daily["normalized_output"] < 0.70 * daily["normalized_output_14d_median"])
        )
    else:
        daily["flag_low_output_high_irr"] = False

    return daily


# ---------------------------------------------------------------------------
# Transfer readiness
# ---------------------------------------------------------------------------

def compute_transfer_readiness(daily: pd.DataFrame) -> pd.DataFrame:
    """Add transfer quality score, tier, and cross-plant readiness flag.

    Scoring: start at 100, subtract penalties for quality issues.
    Modifies ``daily`` in place and returns it.
    """
    score = np.full(len(daily), 100.0)

    # Penalty: coverage gap (severe — data too sparse to trust)
    score -= np.where(daily.get("flag_coverage_gap", False), 35.0, 0.0)

    # Penalty: sensor suspect (irradiance unreliable)
    score -= np.where(daily.get("flag_sensor_suspect_irradiance", False), 30.0, 0.0)

    # Penalty: low data availability
    if "subset_data_availability_mean" in daily.columns:
        score -= np.where(daily["subset_data_availability_mean"] < 0.50, 20.0, 0.0)

    # Penalty: block mismatch
    score -= np.where(daily.get("flag_block_mismatch", False), 10.0, 0.0)

    # Penalty: missing performance loss proxy
    if "performance_loss_pct_proxy" in daily.columns:
        score -= np.where(daily["performance_loss_pct_proxy"].isna(), 15.0, 0.0)

    daily["transfer_quality_score"] = np.clip(score, 0.0, 100.0)

    daily["transfer_quality_tier"] = np.select(
        [daily["transfer_quality_score"] >= 80, daily["transfer_quality_score"] >= 60],
        ["high", "medium"],
        default="low",
    )
    daily["cross_plant_inference_ready"] = daily["transfer_quality_tier"].isin(
        ["high", "medium"]
    )

    return daily
