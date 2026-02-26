"""Shared daily feature building for PV telemetry.

Provides reusable functions for aggregating inverter, irradiance, and generation
data into daily features used by both the audit and preprocessing pipelines.

Constants and thresholds are defined here so both consumers stay in sync.
"""

from typing import Dict, List, Optional, Tuple

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

# Minimum peak-hour Solcast GTI sum (J/m²) for baseline qualification.
# 1 MJ/m² ≈ 69 W/m² average over 4 h  → excludes only heavily overcast days.
MIN_SOLCAST_GTI_PEAK_FOR_BASELINE = 1_000_000.0

# Sanity cap for normalized_output (energy_J / irradiance_sum).  Values above
# this are clipped to prevent single-day spikes from corrupting the baseline.
MAX_NORMALIZED_OUTPUT = 500_000.0

# ---------------------------------------------------------------------------
# Site configuration (placeholders until confirmed by asset owner)
# ---------------------------------------------------------------------------

P_NOM_KWP = 75.0                # Nameplate DC capacity per inverter [kWp]
SURFACE_TILT_DEG = 10.0          # Panel tilt angle [degrees]
SITE_LAT = 8.561510736689941     # Site latitude [degrees north]
SITE_LON = 80.65921384406597     # Site longitude [degrees east]
G_STC = 1000.0                   # Standard Test Conditions irradiance [W/m^2]
N_INVERTERS_PER_TIER = 3         # Inverters in each tier

PEAK_HOUR_START = 10  # Local time hour, inclusive
PEAK_HOUR_END = 14    # Local time hour, exclusive

# Expected records per day under peak-hour filtering
_PEAK_HOURS = PEAK_HOUR_END - PEAK_HOUR_START
PEAK_INV_RECORDS_PER_DAY = _PEAK_HOURS * (3600 // INVERTER_INTERVAL_S)   # 4h * 6 = 24
PEAK_IRR_RECORDS_PER_DAY = _PEAK_HOURS * (3600 // IRRADIANCE_INTERVAL_S) # 4h * 4 = 16

# Cleaning campaign windows (start_date, end_date) used by pvlib Kimber model
CLEANING_CAMPAIGN_DATES = [
    ("2025-09-20", "2025-09-30"),
    ("2025-10-20", "2025-10-30"),
    ("2025-11-20", "2025-11-30"),
]

SIGNIFICANT_RAIN_MM = 5.0  # Precipitation threshold for "significant rain" [mm]

# Clear-Sky Analyzable (CSA) thresholds
CLOUD_OPACITY_BASELINE_MAX = 40.0   # Max cloud opacity for baseline-eligible days [%]
CLOUD_OPACITY_CSA_MAX = 35.0        # Max cloud opacity for CSA filter [%]
RAIN_CSA_MAX_MM = 1.0               # Max precipitation for CSA filter [mm]
DAYS_SINCE_RAIN_CSA_MIN = 1         # Min days since last rain for CSA filter

# Sri Lankan climate seasons for this tropical site (~8.5 deg N)
_DRY_MONTHS = {1, 2, 3, 6, 7, 8, 9}       # Jan-Mar, Jun-Sep
_WET_MONTHS = {4, 5, 10, 11, 12}           # Apr-May, Oct-Dec (monsoon)


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
# Pre-aggregation filtering
# ---------------------------------------------------------------------------

def filter_peak_hours(
    df: pd.DataFrame,
    hour_start: int = PEAK_HOUR_START,
    hour_end: int = PEAK_HOUR_END,
) -> Tuple[pd.DataFrame, int]:
    """Filter sub-daily records to peak sun hours only.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``Date`` column (datetime).
    hour_start : int
        Start hour (inclusive, local time).
    hour_end : int
        End hour (exclusive, local time).

    Returns
    -------
    (filtered_df, n_removed)
    """
    before = len(df)
    mask = df["Date"].dt.hour.between(hour_start, hour_end - 1)
    out = df[mask].copy()
    return out, before - len(out)


def filter_irradiance_threshold(
    df: pd.DataFrame,
    irr_col: str,
    threshold: Optional[float] = None,
    percentile: float = 0.10,
) -> Tuple[pd.DataFrame, int, float]:
    """Reject sub-daily irradiance records below a threshold.

    If *threshold* is None it is derived from the data: the *percentile*-th
    value of daytime (non-zero) irradiance values is used as the floor.  This
    auto-tunes for the sensor's unit conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *irr_col*.
    irr_col : str
        Name of the irradiance column.
    threshold : float or None
        Explicit threshold.  If None, derived from data.
    percentile : float
        Quantile of positive irradiance values used when *threshold* is None.

    Returns
    -------
    (filtered_df, n_removed, threshold_used)
    """
    positive = df[irr_col] > 0
    if threshold is None:
        threshold = float(df.loc[positive, irr_col].quantile(percentile))

    before = len(df)
    mask = df[irr_col] >= threshold
    out = df[mask].copy()
    return out, before - len(out), threshold


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

            # Peak-hour Solcast GTI (10am-2pm) — consistent satellite irradiance
            # for normalisation, replacing the inconsistent on-site sensor sum.
            if "gti_w_m2" in irr.columns:
                _local_hour = irr["period_end"].dt.tz_convert("Asia/Kolkata").dt.hour
                _peak = irr.loc[
                    (_local_hour >= PEAK_HOUR_START) & (_local_hour < PEAK_HOUR_END)
                ].copy()
                _peak["_gti_peak_ws"] = _peak["gti_w_m2"] * SOLCAST_INTERVAL_S
                _peak_daily = (
                    _peak.groupby("day")
                    .agg(
                        solcast_gti_peak_sum=("_gti_peak_ws", "sum"),
                        solcast_gti_peak_records=("gti_w_m2", "size"),
                        solcast_gti_peak_mean_wm2=("gti_w_m2", "mean"),
                    )
                    .reset_index()
                )
                sol_daily = sol_daily.merge(_peak_daily, on="day", how="outer")

    return sol_daily


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def aggregate_inverter_daily(
    inv: pd.DataFrame,
    expected_records: int = EXPECTED_INV_RECORDS_PER_DAY,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build daily inverter aggregates from cleaned 10-min data.

    Expects ``inv`` to have a ``day`` column and ``subset_power_w``.

    Parameters
    ----------
    inv : pd.DataFrame
        Cleaned sub-daily inverter data.
    expected_records : int
        Expected records per day for computing coverage ratio.  Use
        ``PEAK_INV_RECORDS_PER_DAY`` when peak-hour filtering is active.

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
        inv_daily["inverter_records"] / expected_records
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


def aggregate_tier_daily(
    inv: pd.DataFrame, power_cols: List[str],
) -> pd.DataFrame:
    """Build separate Tier-1 (B2) and Tier-2 (B1) daily energy aggregates.

    Tier-1 (B2-08/13/17): high-availability training signal.
    Tier-2 (B1-08/01/13): cross-block validation signal.

    Returns a DataFrame with day + t1_*/t2_* columns.
    Returns empty DataFrame if either tier has no columns.
    """
    b1_cols, b2_cols = detect_block_power_cols(power_cols)
    if not b1_cols and not b2_cols:
        return pd.DataFrame()

    result_frames = []

    for tier_label, tier_cols in [("t1", b2_cols), ("t2", b1_cols)]:
        if not tier_cols:
            continue
        col_power = f"{tier_label}_power_w"
        col_energy_step = f"{tier_label}_energy_j_step"
        col_completeness = f"{tier_label}_completeness"

        inv[col_power] = inv[tier_cols].sum(axis=1, min_count=1)
        inv[col_energy_step] = inv[col_power] * INVERTER_INTERVAL_S
        inv[col_completeness] = inv[tier_cols].notna().mean(axis=1)

        tier_daily = (
            inv.groupby("day")
            .agg(**{
                f"{tier_label}_energy_j": (col_energy_step, "sum"),
                f"{tier_label}_power_w_p95": (col_power, lambda s: s.quantile(0.95)),
                f"{tier_label}_data_availability": (col_completeness, "mean"),
            })
            .reset_index()
        )
        result_frames.append(tier_daily)

    if not result_frames:
        return pd.DataFrame()

    merged = result_frames[0]
    for extra in result_frames[1:]:
        merged = merged.merge(extra, on="day", how="outer")
    return merged


def aggregate_irradiance_daily(
    irr: pd.DataFrame,
    expected_records: int = EXPECTED_IRR_RECORDS_PER_DAY,
) -> pd.DataFrame:
    """Build daily irradiance sums from cleaned 15-min data.

    Expects ``irr`` to have a ``day`` column.

    Parameters
    ----------
    expected_records : int
        Expected records per day for computing coverage ratio.  Use
        ``PEAK_IRR_RECORDS_PER_DAY`` when peak-hour filtering is active.
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
        irr_daily["irradiance_records"] / expected_records
    ).clip(upper=1.0)

    return irr_daily


# ---------------------------------------------------------------------------
# Per-inverter daily metrics and Performance Ratio
# ---------------------------------------------------------------------------

def _inverter_names_from_power_cols(power_cols: List[str]) -> List[str]:
    """Extract inverter name prefixes from Active Power column names."""
    return [c.replace(" Active Power (W)", "") for c in power_cols]


def _safe_col_label(name: str) -> str:
    """Convert inverter name like 'B2-08' to a column-safe prefix 'b2_08'."""
    return name.lower().replace("-", "_")


def compute_performance_ratio(
    energy_j: pd.Series,
    irradiance_tilted_sum: pd.Series,
    p_nom_kwp: float = P_NOM_KWP,
    g_stc: float = G_STC,
    n_inverters: int = 1,
) -> pd.Series:
    """Compute dimensionless Performance Ratio.

    PR = E_actual / (P_nom_total * H_POA / G_STC)

    Parameters
    ----------
    energy_j : pd.Series
        Daily energy output [J].
    irradiance_tilted_sum : pd.Series
        Daily plane-of-array irradiance sum [W-s/m^2].
    p_nom_kwp : float
        Nameplate DC capacity per inverter [kWp].
    g_stc : float
        Standard Test Conditions irradiance [W/m^2].
    n_inverters : int
        Number of inverters contributing to *energy_j*.

    Returns
    -------
    pd.Series
        Dimensionless PR values, typically in [0, 1].
    """
    energy_kwh = energy_j / 3.6e6                         # J -> kWh
    h_poa_kwh_m2 = irradiance_tilted_sum / (g_stc * 3600) # W-s/m^2 -> kWh/m^2
    p_nom_total = p_nom_kwp * n_inverters                  # kWp

    pr = energy_kwh / (p_nom_total * h_poa_kwh_m2)
    return pr.clip(lower=0)


def aggregate_per_inverter_daily(
    inv: pd.DataFrame,
    irr_daily: pd.DataFrame,
    p_nom_kwp: float = P_NOM_KWP,
) -> pd.DataFrame:
    """Compute per-inverter daily energy, PR, and normalized output.

    Parameters
    ----------
    inv : pd.DataFrame
        Cleaned sub-daily inverter data with ``day`` column.
    irr_daily : pd.DataFrame
        Daily irradiance table with ``day`` and ``irradiance_tilted_sum``.
    p_nom_kwp : float
        Nameplate DC capacity per single inverter [kWp].

    Returns
    -------
    pd.DataFrame
        One row per day, with ``{inv_label}_energy_j``,
        ``{inv_label}_pr``, and ``{inv_label}_normalized_output`` for
        each inverter found.
    """
    power_cols = [c for c in inv.columns if c.endswith("Active Power (W)")]
    if not power_cols:
        return pd.DataFrame(columns=["day"])

    inv_names = _inverter_names_from_power_cols(power_cols)
    per_inv_frames: List[pd.DataFrame] = []

    for pcol, name in zip(power_cols, inv_names):
        label = _safe_col_label(name)
        energy_step_col = f"__{label}_energy_step"
        inv[energy_step_col] = inv[pcol] * INVERTER_INTERVAL_S

        agg = (
            inv.groupby("day")
            .agg(**{f"{label}_energy_j": (energy_step_col, "sum")})
            .reset_index()
        )
        per_inv_frames.append(agg)

    result = per_inv_frames[0]
    for extra in per_inv_frames[1:]:
        result = result.merge(extra, on="day", how="outer")

    result = result.merge(
        irr_daily[["day", "irradiance_tilted_sum"]], on="day", how="left",
    )

    for name in inv_names:
        label = _safe_col_label(name)
        e_col = f"{label}_energy_j"
        irr_valid = result["irradiance_tilted_sum"] > MIN_IRRADIANCE_FOR_BASELINE

        result[f"{label}_pr"] = np.where(
            irr_valid,
            compute_performance_ratio(
                result[e_col], result["irradiance_tilted_sum"],
                p_nom_kwp=p_nom_kwp, n_inverters=1,
            ),
            np.nan,
        )
        result[f"{label}_normalized_output"] = np.where(
            irr_valid,
            result[e_col] / result["irradiance_tilted_sum"],
            np.nan,
        )

    result = result.drop(columns=["irradiance_tilted_sum"])
    return result


# ---------------------------------------------------------------------------
# Performance loss features
# ---------------------------------------------------------------------------

def compute_performance_features(
    daily: pd.DataFrame,
    energy_col: str = "subset_energy_j",
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """Add normalized output, rolling baseline, and performance loss proxy.

    Parameters
    ----------
    daily : pd.DataFrame
        Must contain ``irradiance_tilted_sum`` and the column named by *energy_col*.
    energy_col : str
        Column containing daily energy in Joules.
    prefix : str or None
        If set (e.g. ``"t1"``), all output columns are prefixed: ``t1_normalized_output``,
        ``t1_performance_loss_pct_proxy``, etc.  When None, uses the original
        unprefixed column names for backward compatibility.

    Notes
    -----
    - The irradiance guard is applied on DAILY irradiance sum. It does not reject
      normal morning/evening low irradiance records; it protects against days where
      the daily total is implausibly low for baseline building.
    - ``performance_loss_pct_proxy`` is an all-cause proxy deficit against a rolling
      clean-like baseline. It is not a pure-soiling ground truth label.
    """
    def _col(name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    # Skip if energy column is missing or entirely null
    if energy_col not in daily.columns or daily[energy_col].isna().all():
        return daily

    # Prefer Solcast peak-hour GTI (consistent satellite irradiance in J/m²)
    # over on-site sensor sum (inconsistent ThingsBoard SUM aggregation).
    use_solcast = (
        "solcast_gti_peak_sum" in daily.columns
        and daily["solcast_gti_peak_sum"].notna().sum() > 0
    )
    if use_solcast:
        irr_col = "solcast_gti_peak_sum"
        irr_threshold = MIN_SOLCAST_GTI_PEAK_FOR_BASELINE
    else:
        irr_col = "irradiance_tilted_sum"
        irr_threshold = MIN_IRRADIANCE_FOR_BASELINE

    # Normalized output with irradiance sanity guard
    irradiance_valid = daily[irr_col] > irr_threshold
    daily[_col("normalized_output")] = np.where(
        irradiance_valid,
        daily[energy_col] / daily[irr_col],
        np.nan,
    )
    daily[_col("normalized_output")] = daily[_col("normalized_output")].clip(
        upper=MAX_NORMALIZED_OUTPUT
    )

    # 14-day rolling median provides a robust short-term "typical output" reference.
    # Median is less sensitive than mean to spikes/dropouts from telemetry issues.
    daily[_col("normalized_output_14d_median")] = (
        daily[_col("normalized_output")].rolling(14, min_periods=5).median()
    )

    # Rolling clean baseline: 95th percentile on clear days.
    # Cloud-opacity guard prevents cloudy-but-high-irradiance days from
    # inflating the baseline via diffuse-radiation artefacts.
    clear_day_threshold = daily[irr_col].quantile(0.60)
    clear_day_mask = (
        (daily[irr_col] >= clear_day_threshold)
        & (daily[irr_col] > irr_threshold)
    )
    if "cloud_opacity_mean" in daily.columns:
        clear_day_mask = clear_day_mask & (
            daily["cloud_opacity_mean"] <= CLOUD_OPACITY_BASELINE_MAX
        )
    baseline_src = daily[_col("normalized_output")].where(clear_day_mask)
    daily[_col("rolling_clean_baseline")] = (
        baseline_src.rolling(30, min_periods=7).quantile(0.95).ffill()
    )

    # Performance loss proxy (all-cause deficit vs rolling clean baseline)
    daily[_col("performance_loss_pct_proxy")] = (
        100.0 * (1.0 - (
            daily[_col("normalized_output")] / daily[_col("rolling_clean_baseline")]
        ))
    ).clip(lower=0, upper=100)

    # Performance loss rate (trend feature for cleaning predictions)
    daily[_col("perf_loss_rate_14d_pct_per_day")] = (
        (
            daily[_col("performance_loss_pct_proxy")]
            - daily[_col("performance_loss_pct_proxy")].shift(14)
        )
        / 14.0
    )

    return daily


def compute_cross_block_correlation(daily: pd.DataFrame) -> pd.DataFrame:
    """Add cross-block correlation between Tier-1 and Tier-2 performance loss.

    Requires ``t1_performance_loss_pct_proxy`` and ``t2_performance_loss_pct_proxy``
    to be present (computed via prefixed ``compute_performance_features`` calls).

    Modifies ``daily`` in place and returns it.
    """
    t1_col = "t1_performance_loss_pct_proxy"
    t2_col = "t2_performance_loss_pct_proxy"

    if t1_col not in daily.columns or t2_col not in daily.columns:
        return daily

    # Rolling 30-day Pearson correlation (requires both tiers to have data)
    daily["tier_loss_correlation"] = (
        daily[t1_col].rolling(30, min_periods=10).corr(daily[t2_col])
    )

    # Delta: positive means B2 (Tier-1) shows lower loss (cleaner / better signal)
    daily["tier_loss_delta"] = daily[t1_col] - daily[t2_col]

    # Directional agreement: both tiers trending same direction over 7 days
    t1_trend = daily[t1_col].diff(7)
    t2_trend = daily[t2_col].diff(7)
    daily["tier_agreement_flag"] = (
        (t1_trend > 0) & (t2_trend > 0)
    ) | (
        (t1_trend < 0) & (t2_trend < 0)
    )

    return daily


def compute_common_overlap(daily: pd.DataFrame) -> pd.DataFrame:
    """Mark days that have non-null data from all three core sources.

    Uses ``notna()`` — a day with zero energy/irradiance is still structurally
    present data; the ``> 0`` guard belonged in quality flags, not overlap.

    Adds ``in_common_overlap`` boolean column.
    Modifies ``daily`` in place and returns it.
    """
    has_inverter = daily["subset_energy_j"].notna()
    has_irradiance = daily["irradiance_tilted_sum"].notna()
    has_generation = False
    if "daily_generation_j" in daily.columns:
        has_generation = daily["daily_generation_j"].notna()

    daily["in_common_overlap"] = has_inverter & has_irradiance & has_generation
    return daily


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def compute_quality_flags(daily: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags to the daily table.

    Training-facing flags preferentially use Tier-1 (B2) fields so that
    B1 gaps don't inflate flag counts on training-ready days.

    Modifies ``daily`` in place and returns it.
    """
    # Select best-available energy/availability columns (Tier-1 preferred)
    energy_col = "t1_energy_j" if "t1_energy_j" in daily.columns else "subset_energy_j"
    avail_col = (
        "t1_data_availability" if "t1_data_availability" in daily.columns
        else "subset_data_availability_mean"
    )

    # Sensor-suspect irradiance: low irradiance but non-trivial inverter output
    daily["flag_sensor_suspect_irradiance"] = (
        (daily["irradiance_tilted_sum"] < MIN_IRRADIANCE_FOR_BASELINE)
        & (daily[energy_col] > 0)
        & (daily[avail_col] > 0.30 if avail_col in daily.columns else True)
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

    # Low output under high irradiance — use Tier-1 normalized output
    norm_col = "t1_normalized_output" if "t1_normalized_output" in daily.columns else "normalized_output"
    norm_14d_col = (
        "t1_normalized_output_14d_median" if "t1_normalized_output_14d_median" in daily.columns
        else "normalized_output_14d_median"
    )
    _qf_irr_col = (
        "solcast_gti_peak_sum"
        if "solcast_gti_peak_sum" in daily.columns
           and daily["solcast_gti_peak_sum"].notna().sum() > 0
        else "irradiance_tilted_sum"
    )
    _qf_irr_thr = (
        MIN_SOLCAST_GTI_PEAK_FOR_BASELINE
        if _qf_irr_col == "solcast_gti_peak_sum"
        else MIN_IRRADIANCE_FOR_BASELINE
    )
    if norm_14d_col in daily.columns:
        irr_high = daily[_qf_irr_col].quantile(0.60)
        daily["flag_low_output_high_irr"] = (
            (daily[_qf_irr_col] >= irr_high)
            & (daily[norm_col] < 0.70 * daily[norm_14d_col])
        )
    else:
        daily["flag_low_output_high_irr"] = False

    # Zero output on sunny days — equipment shutdown / data gap
    daily["flag_zero_output"] = (
        (daily[norm_col] <= 0) | daily[norm_col].isna()
    ) & (daily[_qf_irr_col] > _qf_irr_thr)

    return daily


# ---------------------------------------------------------------------------
# Transfer readiness
# ---------------------------------------------------------------------------

def compute_transfer_readiness(daily: pd.DataFrame) -> pd.DataFrame:
    """Add transfer quality score, tier, and cross-plant readiness flag.

    Uses Tier-1 (B2) fields for scoring when available so that B1 data gaps
    don't artificially depress training-data quality scores.

    Scoring: start at 100, subtract penalties for quality issues.
    Modifies ``daily`` in place and returns it.
    """
    score = np.full(len(daily), 100.0)

    # Penalty: coverage gap (severe — data too sparse to trust)
    score -= np.where(daily.get("flag_coverage_gap", False), 35.0, 0.0)

    # Penalty: sensor suspect (irradiance unreliable)
    score -= np.where(daily.get("flag_sensor_suspect_irradiance", False), 30.0, 0.0)

    # Penalty: low data availability (prefer Tier-1 availability)
    avail_col = (
        "t1_data_availability" if "t1_data_availability" in daily.columns
        else "subset_data_availability_mean"
    )
    if avail_col in daily.columns:
        score -= np.where(daily[avail_col] < 0.50, 20.0, 0.0)

    # Penalty: block mismatch
    score -= np.where(daily.get("flag_block_mismatch", False), 10.0, 0.0)

    # Penalty: missing performance loss proxy (prefer Tier-1 proxy)
    loss_col = (
        "t1_performance_loss_pct_proxy" if "t1_performance_loss_pct_proxy" in daily.columns
        else "performance_loss_pct_proxy"
    )
    if loss_col in daily.columns:
        score -= np.where(daily[loss_col].isna(), 15.0, 0.0)

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


# ---------------------------------------------------------------------------
# Clear-Sky Analyzable filter
# ---------------------------------------------------------------------------

def flag_clear_sky_analyzable(daily: pd.DataFrame) -> pd.DataFrame:
    """Mark days suitable for soiling analysis with minimal weather contamination.

    Adds a boolean column ``is_clear_sky_analyzable``.  Thresholds are set via
    module-level constants (``CLOUD_OPACITY_CSA_MAX``, ``RAIN_CSA_MAX_MM``,
    ``DAYS_SINCE_RAIN_CSA_MIN``) so they can be tuned from one place.

    Criteria
    --------
    1. ``transfer_quality_tier == "high"`` and ``flag_count == 0``
    2. ``cloud_opacity_mean < CLOUD_OPACITY_CSA_MAX``
    3. ``precipitation_total_mm < RAIN_CSA_MAX_MM``
    4. ``t1_normalized_output > 0`` (equipment operating)
    5. ``days_since_last_rain >= DAYS_SINCE_RAIN_CSA_MIN`` (no carry-over cloud)

    Modifies *daily* in place and returns it.
    """
    norm_col = (
        "t1_normalized_output"
        if "t1_normalized_output" in daily.columns
        else "normalized_output"
    )

    mask = pd.Series(True, index=daily.index)

    if "transfer_quality_tier" in daily.columns:
        mask &= daily["transfer_quality_tier"] == "high"
    if "flag_count" in daily.columns:
        mask &= daily["flag_count"] == 0
    if "cloud_opacity_mean" in daily.columns:
        mask &= daily["cloud_opacity_mean"] < CLOUD_OPACITY_CSA_MAX
    if "precipitation_total_mm" in daily.columns:
        mask &= daily["precipitation_total_mm"] < RAIN_CSA_MAX_MM
    if norm_col in daily.columns:
        mask &= daily[norm_col] > 0
    if "days_since_last_rain" in daily.columns:
        mask &= daily["days_since_last_rain"] >= DAYS_SINCE_RAIN_CSA_MIN

    daily["is_clear_sky_analyzable"] = mask
    return daily


# ---------------------------------------------------------------------------
# Soiling feature engineering
# ---------------------------------------------------------------------------

def compute_soiling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add soiling-relevant engineered features to the daily table.

    Requires Solcast-derived columns (``rain_day``, ``precipitation_total_mm``,
    ``pm10_mean``, ``pm25_mean``, ``humidity_mean``, ``wind_speed_10m_mean``).
    If any are absent the corresponding features are skipped gracefully.

    Modifies *daily* in place and returns it.
    """
    has = set(daily.columns)

    def _rain_bool(series: pd.Series) -> pd.Series:
        """Convert rain_day to a clean boolean Series without deprecation warnings."""
        return pd.array(series, dtype=pd.BooleanDtype()).fillna(False).astype(bool)

    # --- days_since_last_rain ---
    if "rain_day" in has:
        rain = _rain_bool(daily["rain_day"])
        counter = []
        n = 0
        for is_rain in rain:
            n = 0 if is_rain else n + 1
            counter.append(n)
        daily["days_since_last_rain"] = counter

    # --- days_since_significant_rain ---
    if "precipitation_total_mm" in has:
        sig_rain = daily["precipitation_total_mm"].fillna(0) >= SIGNIFICANT_RAIN_MM
        counter = []
        n = 0
        for is_sig in sig_rain:
            n = 0 if is_sig else n + 1
            counter.append(n)
        daily["days_since_significant_rain"] = counter

    # --- cumulative PM since rain ---
    if "pm10_mean" in has and "rain_day" in has:
        rain = _rain_bool(daily["rain_day"])
        pm10 = daily["pm10_mean"].fillna(0)
        acc = []
        total = 0.0
        for is_rain, val in zip(rain, pm10):
            total = val if is_rain else total + val
            acc.append(total)
        daily["cumulative_pm10_since_rain"] = acc

    if "pm25_mean" in has and "rain_day" in has:
        rain = _rain_bool(daily["rain_day"])
        pm25 = daily["pm25_mean"].fillna(0)
        acc = []
        total = 0.0
        for is_rain, val in zip(rain, pm25):
            total = val if is_rain else total + val
            acc.append(total)
        daily["cumulative_pm25_since_rain"] = acc

    # --- interaction: humidity x PM10 (cementation proxy) ---
    if "humidity_mean" in has and "pm10_mean" in has:
        daily["humidity_x_pm10"] = daily["humidity_mean"] * daily["pm10_mean"]

    # --- rolling wind ---
    if "wind_speed_10m_mean" in has:
        daily["wind_speed_10m_rolling_7d"] = (
            daily["wind_speed_10m_mean"].rolling(7, min_periods=3).mean()
        )

    # --- calendar features ---
    day_dt = pd.to_datetime(daily["day"], errors="coerce")
    daily["month"] = day_dt.dt.month
    daily["season"] = daily["month"].map(
        lambda m: "dry" if m in _DRY_MONTHS else "wet"
    )

    return daily


# ---------------------------------------------------------------------------
# Domain Soiling Pressure Index (DSPI)
# ---------------------------------------------------------------------------

# Default domain-knowledge weights for the DSPI formula.
# PM2.5 weighted 2x vs PM10: finer particles fill interparticle gaps more
# completely and resist wind/rain removal (Appels et al.; confirmed by our
# data where cumul_pm25 outperforms cumul_pm10 in predicting cycle_deviation).
_DSPI_DEFAULTS = {
    "w_pm25": 2.0,
    "w_pm10": 1.0,
    "rh_scale": 1.0,    # divisor for humidity normalisation
    "dew_scale": 0.5,   # dew proximity amplification
    "cement_boost": 0.3, # light-rain cementation boost
}

# Rain threshold (mm) below which precipitation cements dust rather than
# cleaning it.  Literature range 0.5-10 mm (Mejia et al.).  1 mm is
# conservative and aligns with our existing rain_day definition.
_DSPI_CLEANING_RAIN_MM = 1.0


def _build_dspi_daily(
    pm25: np.ndarray,
    pm10: np.ndarray,
    rh: np.ndarray,
    delta_t: np.ndarray,
    light_rain: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Compute the daily soiling pressure rate from the DSPI formula."""
    base = params["w_pm25"] * pm25 + params["w_pm10"] * pm10
    hum_factor = 1.0 + np.clip((rh - 40.0) / (40.0 * params["rh_scale"]), 0, 1)
    dew_factor = 1.0 + params["dew_scale"] * np.clip(1.0 - delta_t / 10.0, 0, 1)
    cement_factor = 1.0 + params["cement_boost"] * light_rain
    return base * hum_factor * dew_factor * cement_factor


def _optimise_dspi_weights(
    pm25: np.ndarray,
    pm10: np.ndarray,
    rh: np.ndarray,
    delta_t: np.ndarray,
    light_rain: np.ndarray,
    precip: np.ndarray,
    cloud: np.ndarray,
    temp: np.ndarray,
) -> dict:
    """Find DSPI weights that maximise the desired correlation profile.

    Objective: maximise positive correlation with PM10/PM25, negative
    correlation with rainfall, while penalising correlation with
    cloud opacity and temperature (non-soiling factors).
    Uses no plant performance data — only environmental features.
    """
    from scipy.optimize import minimize

    mask = np.isfinite(pm25) & np.isfinite(pm10) & np.isfinite(rh)
    mask &= np.isfinite(delta_t) & np.isfinite(precip)
    mask &= np.isfinite(cloud) & np.isfinite(temp)
    if mask.sum() < 30:
        return dict(_DSPI_DEFAULTS)

    pm25_m, pm10_m = pm25[mask], pm10[mask]
    rh_m, dt_m, lr_m = rh[mask], delta_t[mask], light_rain[mask]
    precip_m, cloud_m, temp_m = precip[mask], cloud[mask], temp[mask]

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def objective(x):
        p = {
            "w_pm25": x[0], "w_pm10": x[1], "rh_scale": x[2],
            "dew_scale": x[3], "cement_boost": x[4],
        }
        idx = _build_dspi_daily(pm25_m, pm10_m, rh_m, dt_m, lr_m, p)
        r_pm25 = _corr(idx, pm25_m)
        r_pm10 = _corr(idx, pm10_m)
        r_rain = _corr(idx, precip_m)
        r_cloud = _corr(idx, cloud_m)
        r_temp = _corr(idx, temp_m)

        reward = r_pm25 + r_pm10 - r_rain
        penalty = r_cloud ** 2 + r_temp ** 2
        return -(reward - 5.0 * penalty)

    bounds = [
        (0.5, 5.0),   # w_pm25
        (0.1, 3.0),   # w_pm10
        (0.5, 2.0),   # rh_scale
        (0.1, 1.0),   # dew_scale
        (0.05, 0.5),  # cement_boost
    ]
    x0 = [
        _DSPI_DEFAULTS["w_pm25"],
        _DSPI_DEFAULTS["w_pm10"],
        _DSPI_DEFAULTS["rh_scale"],
        _DSPI_DEFAULTS["dew_scale"],
        _DSPI_DEFAULTS["cement_boost"],
    ]
    try:
        res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.success:
            return {
                "w_pm25": float(res.x[0]),
                "w_pm10": float(res.x[1]),
                "rh_scale": float(res.x[2]),
                "dew_scale": float(res.x[3]),
                "cement_boost": float(res.x[4]),
            }
    except Exception:
        pass
    return dict(_DSPI_DEFAULTS)


def compute_domain_soiling_index(daily: pd.DataFrame) -> pd.DataFrame:
    """Add the Domain Soiling Pressure Index (DSPI) to the daily table.

    The DSPI is a physics-grounded soiling estimate built entirely from
    environmental satellite data (PM2.5, PM10, humidity, dewpoint,
    precipitation).  No plant performance data is used, so the index
    is free of data-leakage concerns.

    **Formula** (literature-grounded, nonlinear):

        daily_rate = (w_pm25 * PM2.5 + w_pm10 * PM10)
                     * humidity_factor * dew_factor * cementation_factor

    - *Base deposition*: PM2.5 weighted 2x vs PM10 (Appels et al.)
    - *Humidity adhesion*: factor 1.0→2.0 over 40→80% RH (Said et al.)
    - *Dew proximity*: amplification when air-dewpoint spread < 10 C
    - *Light-rain cementation*: rain 0–1 mm wets dust without cleaning
      (Mejia et al.)
    - *Cumulative index*: accumulates daily_rate, resets on cleaning rain
      (>= 1 mm) or known cleaning campaigns.

    Relative weights are calibrated via constrained optimisation that
    maximises positive correlation with PM and negative with rainfall
    while penalising correlation with non-soiling factors (cloud opacity,
    temperature).  Falls back to literature defaults if optimisation fails.

    Produces columns: ``domain_soiling_daily``, ``domain_soiling_index``.
    Modifies *daily* in place and returns it.
    """
    required = {"pm25_mean", "pm10_mean", "humidity_mean", "precipitation_total_mm"}
    if not required.issubset(daily.columns):
        return daily

    pm25 = daily["pm25_mean"].fillna(0).values.astype(float)
    pm10 = daily["pm10_mean"].fillna(0).values.astype(float)
    rh = daily["humidity_mean"].fillna(60).values.astype(float)

    if "air_temp_mean" in daily.columns and "dewpoint_mean" in daily.columns:
        delta_t = (
            daily["air_temp_mean"].fillna(25) - daily["dewpoint_mean"].fillna(20)
        ).values.astype(float)
    else:
        delta_t = np.full(len(daily), 10.0)

    precip = daily["precipitation_total_mm"].fillna(0).values.astype(float)
    light_rain = ((precip > 0) & (precip < _DSPI_CLEANING_RAIN_MM)).astype(float)

    cloud = (
        daily["cloud_opacity_mean"].fillna(0).values.astype(float)
        if "cloud_opacity_mean" in daily.columns
        else np.zeros(len(daily))
    )
    temp = (
        daily["air_temp_mean"].fillna(25).values.astype(float)
        if "air_temp_mean" in daily.columns
        else np.full(len(daily), 25.0)
    )

    # --- Optimise weights from data distributions ---
    opt_weights = _optimise_dspi_weights(
        pm25, pm10, rh, delta_t, light_rain, precip, cloud, temp,
    )
    print(f"  DSPI weights (optimised): {opt_weights}")

    daily_rate = _build_dspi_daily(pm25, pm10, rh, delta_t, light_rain, opt_weights)
    daily["domain_soiling_daily"] = daily_rate

    # --- Cumulative index with resets ---
    cleaning_rain = precip >= _DSPI_CLEANING_RAIN_MM
    day_dt = pd.to_datetime(daily["day"], errors="coerce")
    is_clean = pd.Series(False, index=daily.index)
    for start_s, end_s in CLEANING_CAMPAIGN_DATES:
        s, e = pd.Timestamp(start_s), pd.Timestamp(end_s)
        is_clean = is_clean | ((day_dt >= s) & (day_dt <= e))

    is_reset = cleaning_rain | is_clean.values
    acc = []
    total = 0.0
    for rate, reset in zip(daily_rate, is_reset):
        if reset:
            total = float(rate)
        else:
            total += float(rate)
        acc.append(total)
    daily["domain_soiling_index"] = acc

    return daily


# ---------------------------------------------------------------------------
# pvlib soiling integration
# ---------------------------------------------------------------------------

def compute_pvlib_soiling_ratio(
    solcast_soiling_path,
    solcast_irradiance_path=None,
) -> pd.DataFrame:
    """Run pvlib HSU and Kimber soiling models on 10-min Solcast data.

    Returns a daily DataFrame with ``pvlib_soiling_ratio_hsu`` and
    ``pvlib_soiling_loss_kimber`` columns.  If pvlib is not importable or
    the Solcast data is incompatible, returns an empty DataFrame.

    Parameters
    ----------
    solcast_soiling_path : path-like
        Solcast environmental CSV (10-min, has precipitation, PM columns).
    solcast_irradiance_path : path-like, optional
        Not used directly by soiling models but reserved for future use.
    """
    from pathlib import Path

    try:
        from pvlib.soiling import hsu, kimber
    except ImportError:
        print("  [pvlib] pvlib not available -- skipping soiling models")
        return pd.DataFrame()

    sol_path = Path(solcast_soiling_path)
    if not sol_path.exists():
        return pd.DataFrame()

    sol = _load_solcast_csv(sol_path)
    for c in ["precipitation_rate_mm_h", "pm2.5_micro_g_m3", "pm10_micro_g_m3"]:
        if c in sol.columns:
            sol[c] = pd.to_numeric(sol[c], errors="coerce")

    required = {"precipitation_rate_mm_h", "pm2.5_micro_g_m3", "pm10_micro_g_m3"}
    if not required.issubset(sol.columns):
        print(f"  [pvlib] Missing columns for soiling models: {required - set(sol.columns)}")
        return pd.DataFrame()

    sol = sol.dropna(subset=list(required))
    sol = sol.set_index("period_end").sort_index()

    # --- HSU model ---
    # pvlib expects rainfall as accumulated mm per step, PM in g/m^3
    rainfall_mm = sol["precipitation_rate_mm_h"] * (SOLCAST_INTERVAL_S / 3600.0)  # mm per 10-min
    pm25_g = sol["pm2.5_micro_g_m3"] * 1e-6   # ug/m^3 -> g/m^3
    pm10_g = sol["pm10_micro_g_m3"] * 1e-6     # ug/m^3 -> g/m^3

    try:
        hsu_ratio = hsu(
            rainfall=rainfall_mm,
            cleaning_threshold=1.0,     # mm accumulated in rain_accum_period to trigger clean
            surface_tilt=SURFACE_TILT_DEG,
            pm2_5=pm25_g,
            pm10=pm10_g,
            rain_accum_period=pd.Timedelta("1h"),
        )
    except Exception as exc:
        print(f"  [pvlib] HSU model failed: {exc}")
        hsu_ratio = pd.Series(dtype=float)

    # --- Kimber model ---
    # Kimber uses daily accumulated rainfall and manual wash dates
    sol_local = sol.copy()
    sol_local.index = sol_local.index.tz_convert("Asia/Kolkata")
    daily_rain = rainfall_mm.resample("D").sum()

    wash_dates = []
    for start_s, end_s in CLEANING_CAMPAIGN_DATES:
        start_d = pd.Timestamp(start_s).date()
        end_d = pd.Timestamp(end_s).date()
        for d in pd.date_range(start_d, end_d):
            wash_dates.append(d.date())

    try:
        kimber_loss = kimber(
            rainfall=daily_rain,
            cleaning_threshold=6.0,
            soiling_loss_rate=0.0015,
            grace_period=14,
            max_soiling=0.3,
            manual_wash_dates=wash_dates,
        )
    except Exception as exc:
        print(f"  [pvlib] Kimber model failed: {exc}")
        kimber_loss = pd.Series(dtype=float)

    # Aggregate HSU to daily mean ratio
    results = pd.DataFrame()
    if not hsu_ratio.empty:
        hsu_daily = hsu_ratio.resample("D").mean()
        hsu_df = hsu_daily.reset_index()
        hsu_df.columns = ["day", "pvlib_soiling_ratio_hsu"]
        hsu_df["day"] = hsu_df["day"].dt.tz_localize(None)
        results = hsu_df

    if not kimber_loss.empty:
        kim_df = kimber_loss.reset_index()
        kim_df.columns = ["day", "pvlib_soiling_loss_kimber"]
        kim_df["day"] = kim_df["day"].dt.tz_localize(None)
        if results.empty:
            results = kim_df
        else:
            results = results.merge(kim_df, on="day", how="outer")

    return results


def compute_temperature_corrected_pr(daily: pd.DataFrame) -> pd.DataFrame:
    """Add temperature-corrected Performance Ratio to the daily table.

    Uses pvlib SAPM cell temperature model with Solcast air temperature,
    wind speed, and irradiance.  Produces ``pr_temperature_corrected`` by
    adjusting the Tier-1 PR for thermal effects.

    Requires ``air_temp_mean``, ``wind_speed_10m_mean``, and
    ``irradiance_tilted_sum`` columns.  Skips gracefully if unavailable.
    """
    required_env = {"air_temp_mean", "wind_speed_10m_mean"}
    has_solcast_irr = "solcast_gti_peak_sum" in daily.columns
    has_onsite_irr = "irradiance_tilted_sum" in daily.columns
    if not required_env.issubset(daily.columns) or not (has_solcast_irr or has_onsite_irr):
        return daily

    try:
        from pvlib.temperature import sapm_cell, TEMPERATURE_MODEL_PARAMETERS
    except ImportError:
        return daily

    params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]

    # Convert daily irradiance sum (J/m²) to average POA irradiance (W/m²)
    # over peak hours (4 hours = 14400 seconds).
    # Prefer Solcast peak-hour GTI (consistent satellite data).
    peak_seconds = (PEAK_HOUR_END - PEAK_HOUR_START) * 3600.0
    irr_src = "solcast_gti_peak_sum" if has_solcast_irr else "irradiance_tilted_sum"
    poa_avg = daily[irr_src] / peak_seconds

    cell_temp = sapm_cell(
        poa_global=poa_avg,
        temp_air=daily["air_temp_mean"],
        wind_speed=daily["wind_speed_10m_mean"],
        a=params["a"],
        b=params["b"],
        deltaT=params["deltaT"],
    )

    # Temperature coefficient for crystalline silicon ~ -0.004 /degC
    temp_coeff = -0.004  # [1/degC]
    t_stc = 25.0         # [degC]
    temp_correction_factor = 1.0 + temp_coeff * (cell_temp - t_stc)

    pr_col = "t1_performance_loss_pct_proxy" if "t1_performance_loss_pct_proxy" in daily.columns else "performance_loss_pct_proxy"
    norm_col = "t1_normalized_output" if "t1_normalized_output" in daily.columns else "normalized_output"
    baseline_col = "t1_rolling_clean_baseline" if "t1_rolling_clean_baseline" in daily.columns else "rolling_clean_baseline"

    if norm_col in daily.columns and baseline_col in daily.columns:
        corrected_output = daily[norm_col] / temp_correction_factor
        daily["pr_temperature_corrected"] = (
            100.0 * (1.0 - corrected_output / daily[baseline_col])
        ).clip(lower=0, upper=100)

    return daily


# ---------------------------------------------------------------------------
# Cycle-aware deviation feature
# ---------------------------------------------------------------------------

def compute_cycle_deviation(
    daily: pd.DataFrame,
    energy_col: str = "subset_energy_j",
    irr_sum_col: str = "irradiance_tilted_sum",
) -> pd.DataFrame:
    """Add cycle-aware soiling deviation feature.

    X = energy / (irr_sum / tracked_time)  -- normalised by average
    irradiance rate, more robust than raw irr_sum when data coverage varies.

    Cycles are delimited by rain events or cleaning campaigns.  Within each
    cycle the max(X) represents "just cleaned" performance; the deviation
    from that max tracks soiling accumulation.

    Produces ``cycle_id``, ``soiling_index_x``, ``cycle_max_x``, and
    ``cycle_deviation_pct`` columns.

    Modifies *daily* in place and returns it.
    """
    if energy_col not in daily.columns or irr_sum_col not in daily.columns:
        return daily

    # Tracked time in seconds (from inverter record count)
    if "inverter_records" in daily.columns:
        tracked_time = daily["inverter_records"] * INVERTER_INTERVAL_S
    else:
        tracked_time = pd.Series(
            (PEAK_HOUR_END - PEAK_HOUR_START) * 3600.0, index=daily.index,
        )

    mean_irr_rate = daily[irr_sum_col] / tracked_time
    mean_irr_rate = mean_irr_rate.replace(0, np.nan)
    soiling_x = daily[energy_col] / mean_irr_rate

    # Determine cycle boundaries
    day_dt = pd.to_datetime(daily["day"], errors="coerce")
    if "rain_day" in daily.columns:
        is_rain = pd.array(daily["rain_day"], dtype=pd.BooleanDtype()).fillna(False).astype(bool)
        is_rain = pd.Series(is_rain, index=daily.index)
    else:
        is_rain = pd.Series(False, index=daily.index)

    # Mark cleaning campaign days
    is_clean = pd.Series(False, index=daily.index)
    for start_s, end_s in CLEANING_CAMPAIGN_DATES:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        is_clean = is_clean | ((day_dt >= start) & (day_dt <= end))

    is_reset = is_rain | is_clean
    cycle_id = is_reset.cumsum()

    daily["cycle_id"] = cycle_id.values
    daily["soiling_index_x"] = soiling_x.values

    # Max X within each cycle (forward-looking max from cycle start)
    cycle_max = daily.groupby("cycle_id")["soiling_index_x"].transform("max")
    daily["cycle_max_x"] = cycle_max

    daily["cycle_deviation_pct"] = (
        100.0 * (1.0 - daily["soiling_index_x"] / daily["cycle_max_x"])
    ).clip(lower=0, upper=100)

    return daily
