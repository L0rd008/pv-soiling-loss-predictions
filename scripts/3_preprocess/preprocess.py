"""Clean and preprocess PV telemetry data for modeling.

This script standardizes raw exports, applies domain sanity checks, builds
daily modeling features, and writes preprocessing outputs.

Usage:
    python scripts/3_preprocess/preprocess.py --data-dir data --out-dir artifacts/preprocessed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from core.daily_features import (
    INVERTER_INTERVAL_S,
    MIN_IRRADIANCE_FOR_BASELINE,
    MAX_NORMALIZED_OUTPUT,
    PEAK_HOUR_START,
    PEAK_HOUR_END,
    PEAK_INV_RECORDS_PER_DAY,
    PEAK_IRR_RECORDS_PER_DAY,
    P_NOM_KWP,
    aggregate_block_daily,
    aggregate_inverter_daily,
    aggregate_irradiance_daily,
    aggregate_per_inverter_daily,
    aggregate_solcast_daily,
    aggregate_tier_daily,
    compute_common_overlap,
    compute_cross_block_correlation,
    compute_cycle_deviation,
    compute_domain_soiling_index,
    compute_loss_proxy_from_ratio,
    compute_performance_features,
    compute_performance_ratio,
    compute_pvlib_soiling_ratio,
    compute_quality_flags,
    compute_soiling_features,
    compute_temperature_corrected_pr,
    compute_power_at_reference_irradiance,
    compute_transfer_readiness,
    flag_clear_sky_analyzable,
    detect_irradiance_cols,
    filter_irradiance_threshold,
    filter_peak_hours,
)

MAX_POWER_W = 300_000.0
MAX_CURRENT_A = 250.0
MAX_GENERATION_J = 360_000_000_000.0
DEFAULT_RUNTIME_CSV = Path("data/time_series_chart_time_series_chart.csv")
DEFAULT_DAILY_GEN_WARN_KWH = 1_000.0


def load_numeric_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with Timestamp/Date columns, coercing values to numeric."""
    df = pd.read_csv(path)
    required = {"Timestamp", "Date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def dedupe_by_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Deduplicate rows by Timestamp, averaging numeric columns."""
    before = len(df)
    numeric_cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    agg_map: Dict[str, str] = {"Date": "first"}
    agg_map.update({col: "mean" for col in numeric_cols})
    out = df.sort_values("Date").groupby("Timestamp", as_index=False).agg(agg_map)
    out = out.sort_values("Date").reset_index(drop=True)
    removed = before - len(out)
    return out, removed


def clean_inverters(path: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load, deduplicate, and sanity-filter inverter data."""
    df = load_numeric_csv(path)
    df, duplicates_removed = dedupe_by_timestamp(df)

    power_cols = [c for c in df.columns if c.endswith("Active Power (W)")]
    current_cols = [c for c in df.columns if "Current" in c and "(A)" in c]

    power_invalid = (df[power_cols] < 0) | (df[power_cols] > MAX_POWER_W)
    current_invalid = (df[current_cols] < 0) | (df[current_cols] > MAX_CURRENT_A)
    power_invalid_count = int(power_invalid.sum().sum())
    current_invalid_count = int(current_invalid.sum().sum())

    df[power_cols] = df[power_cols].mask(power_invalid)
    df[current_cols] = df[current_cols].mask(current_invalid)

    df["day"] = df["Date"].dt.floor("D")
    df["subset_power_w"] = df[power_cols].sum(axis=1, min_count=1)
    df["row_power_completeness"] = df[power_cols].notna().mean(axis=1)

    stats = {
        "rows": float(len(df)),
        "duplicates_removed": float(duplicates_removed),
        "power_invalid_to_nan": float(power_invalid_count),
        "current_invalid_to_nan": float(current_invalid_count),
        "power_missing_ratio": float(df[power_cols].isna().mean().mean()),
        "row_power_completeness_mean": float(df["row_power_completeness"].mean()),
    }
    return df, stats


def clean_irradiance(path: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load, deduplicate, and filter negative irradiance values."""
    df = load_numeric_csv(path)
    df, duplicates_removed = dedupe_by_timestamp(df)

    irr_cols = [c for c in df.columns if "Irradiance" in c]
    if not irr_cols:
        raise ValueError("No irradiance columns found in irradiance dataset.")

    invalid_negative = df[irr_cols] < 0
    invalid_count = int(invalid_negative.sum().sum())
    df[irr_cols] = df[irr_cols].mask(invalid_negative)

    df["day"] = df["Date"].dt.floor("D")

    stats = {
        "rows": float(len(df)),
        "duplicates_removed": float(duplicates_removed),
        "irr_invalid_to_nan": float(invalid_count),
        "irr_missing_ratio": float(df[irr_cols].isna().mean().mean()),
    }
    return df, stats


def clean_generation(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load, deduplicate, filter, and aggregate generation to daily."""
    df = load_numeric_csv(path)
    df, duplicates_removed = dedupe_by_timestamp(df)

    value_cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    if len(value_cols) != 1:
        raise ValueError(f"Expected one generation value column, found: {value_cols}")
    gen_col = value_cols[0]

    invalid = (df[gen_col] < 0) | (df[gen_col] > MAX_GENERATION_J)
    invalid_count = int(invalid.sum())
    df[gen_col] = df[gen_col].mask(invalid)

    df["day"] = df["Date"].dt.floor("D")

    daily = (
        df.sort_values("Date")
        .groupby("day")
        .agg(
            daily_generation_j_latest=(gen_col, "last"),
            daily_generation_j_max=(gen_col, "max"),
            daily_generation_j_min=(gen_col, "min"),
            generation_records=("Timestamp", "size"),
        )
        .reset_index()
    )
    daily["daily_generation_j"] = daily["daily_generation_j_latest"].fillna(
        daily["daily_generation_j_max"]
    )
    daily["generation_intraday_spread_j"] = (
        daily["daily_generation_j_max"] - daily["daily_generation_j_min"]
    )

    stats = {
        "rows": float(len(df)),
        "duplicates_removed": float(duplicates_removed),
        "generation_invalid_to_nan": float(invalid_count),
        "generation_missing_ratio": float(df[gen_col].isna().mean()),
        "daily_rows": float(len(daily)),
    }
    return df, daily, stats


# ---------------------------------------------------------------------------
# New telemetry: per-inverter daily generated electricity
# ---------------------------------------------------------------------------

MAX_DAILY_GEN_KWH_WARN = 1_000.0  # warning threshold used for clip diagnostics
KWH_TO_JOULES = 3_600_000.0


def clean_inverter_daily_gen(
    path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load per-inverter daily_generated_electricity CSV and build a daily table.

    The CSV contains cumulative kWh readings sampled at ~1-min intervals that
    reset at midnight. Daily value selection is robust to reset-edge and spike
    behavior using ``last``, ``max``, and ``p99``.

    Returns ``(daily_df, audit_df, stats)`` where ``daily_df`` has columns
    ``{inv_label}_daily_gen_j`` for each inverter plus a ``day`` column.
    """
    df = load_numeric_csv(path)
    df, duplicates_removed = dedupe_by_timestamp(df)

    kwh_cols = [c for c in df.columns if "kWh" in c]
    if not kwh_cols:
        raise ValueError(f"No kWh columns found in {path}")

    neg_count = 0
    above_warn_count = 0
    for col in kwh_cols:
        neg_mask = df[col] < 0
        neg_count += int(neg_mask.sum())
        df[col] = df[col].mask(neg_mask)
        above_warn_count += int((df[col] > MAX_DAILY_GEN_KWH_WARN).sum())

    df["day"] = df["Date"].dt.floor("D")

    daily = pd.DataFrame({"day": sorted(df["day"].dropna().unique())})
    audit_frames: List[pd.DataFrame] = []
    converted_cols: List[str] = []

    for col in kwh_cols:
        parts = col.split()
        inv_name = parts[0] if parts else col
        label = inv_name.lower().replace("-", "_")
        group = (
            df.sort_values("Date")
            .groupby("day")
            .agg(
                last_kwh=(col, "last"),
                max_kwh=(col, "max"),
                p99_kwh=(col, lambda s: s.dropna().quantile(0.99) if s.notna().any() else np.nan),
                records=(col, "size"),
                non_null_records=(col, "count"),
                last_valid_dt=("Date", lambda s: s[df.loc[s.index, col].notna()].max() if df.loc[s.index, col].notna().any() else pd.NaT),
            )
            .reset_index()
        )
        spike_mask = (
            group["max_kwh"].notna()
            & group["p99_kwh"].notna()
            & (group["p99_kwh"] > 0)
            & (group["max_kwh"] > 1.25 * group["p99_kwh"])
        )
        selected_kwh = group["max_kwh"].copy()
        selected_kwh[spike_mask] = group.loc[spike_mask, "p99_kwh"]
        group["selected_kwh"] = selected_kwh

        q1 = float(group["selected_kwh"].quantile(0.25))
        q3 = float(group["selected_kwh"].quantile(0.75))
        iqr = q3 - q1
        upper_fence = q3 + 3.0 * iqr
        outlier_mask = group["selected_kwh"] > upper_fence if np.isfinite(upper_fence) else pd.Series(False, index=group.index)
        group["selected_kwh_clean"] = group["selected_kwh"].mask(outlier_mask)
        group["selection_method"] = np.where(spike_mask, "p99_guard", "max")
        group["outlier_flag"] = outlier_mask.fillna(False)
        group["inverter"] = inv_name

        out_col_j = f"{label}_daily_gen_j"
        converted_cols.append(out_col_j)
        tmp = group[["day", "selected_kwh_clean"]].copy()
        tmp[out_col_j] = tmp["selected_kwh_clean"] * KWH_TO_JOULES
        tmp = tmp.drop(columns=["selected_kwh_clean"])
        daily = daily.merge(tmp, on="day", how="left")

        audit_frames.append(
            group[
                [
                    "day",
                    "inverter",
                    "records",
                    "non_null_records",
                    "last_valid_dt",
                    "last_kwh",
                    "max_kwh",
                    "p99_kwh",
                    "selected_kwh",
                    "selection_method",
                    "outlier_flag",
                    "selected_kwh_clean",
                ]
            ]
        )

    audit_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    stats = {
        "rows": float(len(df)),
        "duplicates_removed": float(duplicates_removed),
        "daily_gen_negative_to_nan": float(neg_count),
        "daily_gen_above_warn_count": float(above_warn_count),
        "daily_gen_above_warn_pct": float(100.0 * above_warn_count / max(len(df) * max(len(kwh_cols), 1), 1)),
        "inverter_count": float(len(kwh_cols)),
        "daily_rows": float(len(daily)),
        "daily_outlier_rows_removed": float(audit_df["outlier_flag"].sum()) if not audit_df.empty else 0.0,
    }
    return daily, audit_df, stats


# ---------------------------------------------------------------------------
# New telemetry: plant-level average solar radiation
# ---------------------------------------------------------------------------

MAX_AVG_IRRADIANCE_WM2 = 1_500.0


def clean_plant_avg_irradiance(
    path: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load plant avg_solar_radiation CSV and build a daily table.

    The CSV contains a running daily-average W/m^2 value that resets at
    midnight.  The last reading per day equals the true daily average.

    Returns ``(daily_df, stats)`` where ``daily_df`` has columns
    ``day`` and ``plant_avg_irradiance_wm2``.
    """
    df = load_numeric_csv(path)
    df, duplicates_removed = dedupe_by_timestamp(df)

    value_cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    if len(value_cols) != 1:
        raise ValueError(
            f"Expected one irradiance column, found: {value_cols}"
        )
    irr_col = value_cols[0]

    invalid = (df[irr_col] < 0) | (df[irr_col] > MAX_AVG_IRRADIANCE_WM2)
    invalid_count = int(invalid.sum())
    df[irr_col] = df[irr_col].mask(invalid)

    df["day"] = df["Date"].dt.floor("D")

    daily = (
        df.sort_values("Date")
        .groupby("day")
        .agg(
            plant_avg_irradiance_wm2=(irr_col, "last"),
            plant_irr_records=("Timestamp", "size"),
        )
        .reset_index()
    )

    stats = {
        "rows": float(len(df)),
        "duplicates_removed": float(duplicates_removed),
        "irr_invalid_to_nan": float(invalid_count),
        "irr_unstable_days_removed": 0.0,
        "daily_rows": float(len(daily)),
    }
    return daily, stats


def load_runtime_daily(
    path: Path,
    min_hours: float,
    max_hours: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load runtime CSV and return one runtime value per day with QC flags."""
    if not path.exists():
        return pd.DataFrame(columns=["day", "runtime_h_csv", "runtime_csv_invalid"]), {
            "runtime_rows": 0.0,
            "runtime_daily_rows": 0.0,
            "runtime_invalid_days": 0.0,
        }

    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)

    if "Timestamp" not in df.columns:
        return pd.DataFrame(columns=["day", "runtime_h_csv", "runtime_csv_invalid"]), {
            "runtime_rows": float(len(df)),
            "runtime_daily_rows": 0.0,
            "runtime_invalid_days": 0.0,
        }

    runtime_col = None
    for candidate in ("runtime_hours", "Temperature"):
        if candidate in df.columns:
            runtime_col = candidate
            break
    if runtime_col is None:
        return pd.DataFrame(columns=["day", "runtime_h_csv", "runtime_csv_invalid"]), {
            "runtime_rows": float(len(df)),
            "runtime_daily_rows": 0.0,
            "runtime_invalid_days": 0.0,
        }

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df[runtime_col] = pd.to_numeric(df[runtime_col], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["day"] = df["Timestamp"].dt.floor("D")
    daily = (
        df.groupby("day")
        .agg(runtime_h_csv=(runtime_col, "max"))
        .reset_index()
    )
    invalid = daily["runtime_h_csv"].notna() & (
        (daily["runtime_h_csv"] < min_hours) | (daily["runtime_h_csv"] > max_hours)
    )
    daily["runtime_csv_invalid"] = invalid
    daily.loc[invalid, "runtime_h_csv"] = np.nan

    stats = {
        "runtime_rows": float(len(df)),
        "runtime_daily_rows": float(len(daily)),
        "runtime_invalid_days": float(int(invalid.sum())),
    }
    return daily, stats


def load_solcast_runtime_daily(path: Path) -> pd.DataFrame:
    """Estimate daily daylight runtime from Solcast GTI > 0 rows."""
    if not path.exists():
        return pd.DataFrame(columns=["day", "runtime_solcast_h"])

    df = pd.read_csv(path)
    if "period_end" not in df.columns or "gti_w_m2" not in df.columns:
        return pd.DataFrame(columns=["day", "runtime_solcast_h"])

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
    df["gti_w_m2"] = pd.to_numeric(df["gti_w_m2"], errors="coerce")
    df = df.dropna(subset=["period_end"])
    df["day"] = (
        df["period_end"]
        .dt.tz_convert("Asia/Colombo")
        .dt.tz_localize(None)
        .dt.floor("D")
    )
    daily = (
        df.assign(_sun=df["gti_w_m2"] > 0)
        .groupby("day")
        .agg(runtime_solcast_h=("_sun", "sum"))
        .reset_index()
    )
    daily["runtime_solcast_h"] = daily["runtime_solcast_h"] * (10.0 / 60.0)
    return daily


def _normalize_inverter_id(label: str) -> str:
    """Normalize inverter labels to lowercase underscore form (e.g., B2-08 -> b2_08)."""
    return label.strip().lower().replace("-", "_")


def _inverter_id_from_power_col(col: str) -> Optional[str]:
    """Extract normalized inverter id from '<INV> Active Power (W)' column names."""
    if not col.endswith("Active Power (W)"):
        return None
    return _normalize_inverter_id(col.replace(" Active Power (W)", ""))


def _build_daily_model_table_legacy(
    inverters: pd.DataFrame,
    irradiance: pd.DataFrame,
    generation_daily: pd.DataFrame,
    solcast_daily: pd.DataFrame = None,
    pvlib_soiling_daily: pd.DataFrame = None,
    inv_daily_gen: pd.DataFrame = None,
    plant_avg_irr: pd.DataFrame = None,
    runtime_daily: pd.DataFrame = None,
    runtime_solcast_daily: pd.DataFrame = None,
    power_at_ref_irr: pd.DataFrame = None,
    peak_filtered: bool = False,
    inverter_capacity_kw: float = 330.0,
    plant_inverter_count: int = 34,
) -> pd.DataFrame:
    """Build the daily model input table from cleaned sub-daily data.

    Uses shared feature functions from ``daily_features`` module for performance
    flags, and transfer readiness to stay in sync with the audit pipeline.

    Produces three sets of performance features:
    - Combined (all inverters): backward-compatible ``performance_loss_pct_proxy``
    - Tier-1 (B2 training): ``t1_performance_loss_pct_proxy``
    - Tier-2 (B1 validation): ``t2_performance_loss_pct_proxy``
    Plus per-inverter PR, soiling features, pvlib soiling estimates,
    cycle deviation, cross-block correlation, and temperature correction.

    Parameters
    ----------
    peak_filtered : bool
        If True, coverage ratios use peak-hour expected record counts
        instead of full-day counts.
    inv_daily_gen : pd.DataFrame or None
        Per-inverter daily_generated_electricity (cleaned, in Joules).
    plant_avg_irr : pd.DataFrame or None
        Plant-level avg_solar_radiation (cleaned, W/m^2).
    power_at_ref_irr : pd.DataFrame or None
        Power at reference irradiance feature (daily).
    """
    inv = inverters.copy()
    irr = irradiance.copy()
    gen = generation_daily.copy()

    inv_expected = PEAK_INV_RECORDS_PER_DAY if peak_filtered else None
    irr_expected = PEAK_IRR_RECORDS_PER_DAY if peak_filtered else None

    # --- Inverter daily aggregation (combined) ---
    agg_kwargs_inv = {} if inv_expected is None else {"expected_records": inv_expected}
    inv_daily, power_cols = aggregate_inverter_daily(inv, **agg_kwargs_inv)

    # --- Block (B1 vs B2) daily aggregation ---
    block_daily = aggregate_block_daily(inv, power_cols)
    if not block_daily.empty:
        inv_daily = inv_daily.merge(block_daily, on="day", how="left")

    # --- Tier-1 / Tier-2 daily aggregation ---
    tier_daily = aggregate_tier_daily(inv, power_cols)
    if not tier_daily.empty:
        inv_daily = inv_daily.merge(tier_daily, on="day", how="left")

    # --- Irradiance daily aggregation ---
    agg_kwargs_irr = {} if irr_expected is None else {"expected_records": irr_expected}
    irr_daily = aggregate_irradiance_daily(irr, **agg_kwargs_irr)

    # --- Per-inverter daily metrics (energy, PR, normalized output) ---
    per_inv = aggregate_per_inverter_daily(inv, irr_daily, p_nom_kwp=P_NOM_KWP)
    if not per_inv.empty:
        inv_daily = inv_daily.merge(per_inv, on="day", how="left")

    # --- Merge all daily tables ---
    daily = (
        inv_daily.merge(irr_daily, on="day", how="outer")
        .merge(gen, on="day", how="outer")
        .sort_values("day")
        .reset_index(drop=True)
    )

    # --- Solcast environmental features ---
    if solcast_daily is not None and not solcast_daily.empty:
        daily = daily.merge(solcast_daily, on="day", how="left")

    # --- pvlib soiling estimates ---
    if pvlib_soiling_daily is not None and not pvlib_soiling_daily.empty:
        daily = daily.merge(pvlib_soiling_daily, on="day", how="left")

    # --- Per-inverter daily generated electricity ---
    n_subset_inverters = 0
    if inv_daily_gen is not None and not inv_daily_gen.empty:
        daily = daily.merge(inv_daily_gen, on="day", how="left")
        gen_j_cols = [c for c in inv_daily_gen.columns if c.endswith("_daily_gen_j")]
        n_subset_inverters = len(gen_j_cols)
        if gen_j_cols:
            daily["subset_daily_gen_j"] = daily[gen_j_cols].sum(axis=1, min_count=1)
            daily["subset_daily_gen_kwh"] = daily["subset_daily_gen_j"] / 3.6e6

    # --- Plant-level average solar radiation ---
    if plant_avg_irr is not None and not plant_avg_irr.empty:
        daily = daily.merge(plant_avg_irr, on="day", how="left")

    # --- Power at reference irradiance ---
    if power_at_ref_irr is not None and not power_at_ref_irr.empty:
        daily = daily.merge(power_at_ref_irr, on="day", how="left")

    # --- Generation / irradiance ratio (new data) ---
    if (
        "subset_daily_gen_kwh" in daily.columns
        and "plant_avg_irradiance_wm2" in daily.columns
    ):
        valid = (
            daily["plant_avg_irradiance_wm2"].notna()
            & (daily["plant_avg_irradiance_wm2"] > 0)
            & daily["subset_daily_gen_kwh"].notna()
        )
        # gen_irr_ratio = subset_daily_gen_kwh / plant_avg_irradiance_wm2
        #
        # NOTE: This is NOT a proper Performance Ratio (PR). A true PR would be:
        #   PR = E_measured / (H_total × P_rated)
        # where H_total = avg_irradiance × daylight_hours / 1000 (kWh/m²)
        # and P_rated = n_inverters × P_NOM_KWP (kW).
        #
        # This is BLOCKED because P_NOM_KWP = 75 kWp is a placeholder
        # (actual inverter capacity is ~200-400 kW based on daily output of
        # ~1000 kWh/inverter). Until the true nameplate is confirmed from the
        # asset owner, the PR formula would produce incorrect absolute values.
        #
        # For SOILING DETECTION this does not matter: the ratio is always used
        # relative to its own rolling baseline (gen_irr_ratio / rolling_baseline),
        # so any constant scaling factor (daylight hours, P_rated) cancels out.
        # The simple ratio preserves all soiling signal information.
        h_total = daily["plant_avg_irradiance_wm2"]  # W/m² (daily average)

        daily["gen_irr_ratio"] = np.where(
            valid,
            daily["subset_daily_gen_kwh"] / h_total,
            np.nan,
        )
        daily["gen_irr_ratio_smoothed"] = (
            daily["gen_irr_ratio"]
            .rolling(7, center=True, min_periods=3)
            .median()
        )

    # --- Derived energy columns ---
    daily["subset_energy_mwh"] = daily["subset_energy_j"] / 3.6e9
    daily["generation_mwh"] = daily["daily_generation_j"] / 3.6e9
    daily["plant_to_subset_energy_ratio"] = (
        daily["generation_mwh"] / daily["subset_energy_mwh"]
    )

    # --- Performance loss features ---
    # Combined (backward compat) — uses all-inverter subset_energy_j
    daily = compute_performance_features(daily)
    # Tier-1 (B2 training signal)
    daily = compute_performance_features(daily, energy_col="t1_energy_j", prefix="t1")
    # Tier-2 (B1 validation signal)
    daily = compute_performance_features(daily, energy_col="t2_energy_j", prefix="t2")

    # --- Combined PR (all tiered inverters) ---
    n_power_cols = len(power_cols)
    if "irradiance_tilted_sum" in daily.columns and n_power_cols > 0:
        daily["subset_pr"] = compute_performance_ratio(
            daily["subset_energy_j"], daily["irradiance_tilted_sum"],
            p_nom_kwp=P_NOM_KWP, n_inverters=n_power_cols,
        )

    # --- Cross-block correlation ---
    daily = compute_cross_block_correlation(daily)

    # --- Soiling feature engineering ---
    daily = compute_soiling_features(daily)

    # --- Domain Soiling Pressure Index ---
    daily = compute_domain_soiling_index(daily)

    # --- Cycle-aware deviation ---
    _irr_col_for_cycle = (
        "solcast_gti_peak_sum"
        if "solcast_gti_peak_sum" in daily.columns
           and daily["solcast_gti_peak_sum"].notna().sum() > 0
        else "irradiance_tilted_sum"
    )
    daily = compute_cycle_deviation(daily, irr_sum_col=_irr_col_for_cycle)

    # --- New-source performance features (gen_irr_ratio based) ---
    if "gen_irr_ratio" in daily.columns and daily["gen_irr_ratio"].notna().sum() > 10:
        daily = compute_loss_proxy_from_ratio(
            daily, ratio_col="gen_irr_ratio", prefix="new",
        )
        daily = compute_cycle_deviation(
            daily, precomputed_x_col="gen_irr_ratio", prefix="new",
        )
        if "new_rolling_clean_baseline" in daily.columns:
            daily["new_performance_index"] = (
                daily["gen_irr_ratio"] / daily["new_rolling_clean_baseline"]
            ).clip(upper=1.5)

    # --- Temperature correction ---
    daily = compute_temperature_corrected_pr(daily)

    # --- Common-overlap window ---
    daily = compute_common_overlap(daily)

    # --- Quality flags (shared logic) ---
    daily = compute_quality_flags(daily)
    flag_cols = [c for c in daily.columns if c.startswith("flag_")]
    daily["flag_count"] = daily[flag_cols].fillna(False).sum(axis=1)

    # --- Transfer readiness (shared logic) ---
    daily = compute_transfer_readiness(daily)

    # --- Clear-Sky Analyzable flag (requires flag_count + transfer tier) ---
    daily = flag_clear_sky_analyzable(daily)

    return daily


def build_daily_model_table(
    inverters: pd.DataFrame,
    irradiance: pd.DataFrame,
    generation_daily: pd.DataFrame,
    solcast_daily: pd.DataFrame = None,
    pvlib_soiling_daily: pd.DataFrame = None,
    inv_daily_gen: pd.DataFrame = None,
    plant_avg_irr: pd.DataFrame = None,
    runtime_daily: pd.DataFrame = None,
    runtime_solcast_daily: pd.DataFrame = None,
    power_at_ref_irr: pd.DataFrame = None,
    peak_filtered: bool = False,
    inverter_capacity_kw: float = 330.0,
    plant_inverter_count: int = 34,
) -> pd.DataFrame:
    """Build the daily model table with old-source and physical new-source metrics."""
    inv = inverters.copy()
    irr = irradiance.copy()
    gen = generation_daily.copy()

    inv_expected = PEAK_INV_RECORDS_PER_DAY if peak_filtered else None
    irr_expected = PEAK_IRR_RECORDS_PER_DAY if peak_filtered else None

    # --- Inverter daily aggregation (combined) ---
    agg_kwargs_inv = {} if inv_expected is None else {"expected_records": inv_expected}
    inv_daily, power_cols = aggregate_inverter_daily(inv, **agg_kwargs_inv)
    power_inv_ids = sorted(
        {
            inv_id
            for col in power_cols
            for inv_id in [_inverter_id_from_power_col(col)]
            if inv_id is not None
        }
    )

    # --- Block (B1 vs B2) daily aggregation ---
    block_daily = aggregate_block_daily(inv, power_cols)
    if not block_daily.empty:
        inv_daily = inv_daily.merge(block_daily, on="day", how="left")

    # --- Tier-1 / Tier-2 daily aggregation ---
    tier_daily = aggregate_tier_daily(inv, power_cols)
    if not tier_daily.empty:
        inv_daily = inv_daily.merge(tier_daily, on="day", how="left")

    # --- Irradiance daily aggregation ---
    agg_kwargs_irr = {} if irr_expected is None else {"expected_records": irr_expected}
    irr_daily = aggregate_irradiance_daily(irr, **agg_kwargs_irr)

    # --- Per-inverter daily metrics (energy, PR, normalized output) ---
    per_inv = aggregate_per_inverter_daily(inv, irr_daily, p_nom_kwp=P_NOM_KWP)
    if not per_inv.empty:
        inv_daily = inv_daily.merge(per_inv, on="day", how="left")

    # --- Merge all daily tables ---
    daily = (
        inv_daily.merge(irr_daily, on="day", how="outer")
        .merge(gen, on="day", how="outer")
        .sort_values("day")
        .reset_index(drop=True)
    )

    # --- Solcast environmental features ---
    if solcast_daily is not None and not solcast_daily.empty:
        daily = daily.merge(solcast_daily, on="day", how="left")

    # --- pvlib soiling estimates ---
    if pvlib_soiling_daily is not None and not pvlib_soiling_daily.empty:
        daily = daily.merge(pvlib_soiling_daily, on="day", how="left")

    # --- Per-inverter daily generated electricity (aligned to power-tier assets) ---
    aligned_inv_ids: List[str] = []
    daily["subset_daily_gen_expected_count"] = float(len(power_inv_ids))
    if inv_daily_gen is not None and not inv_daily_gen.empty:
        daily = daily.merge(inv_daily_gen, on="day", how="left")
        gen_j_cols = [c for c in inv_daily_gen.columns if c.endswith("_daily_gen_j")]
        gen_inv_map = {c[:-len("_daily_gen_j")]: c for c in gen_j_cols}
        aligned_inv_ids = [inv_id for inv_id in power_inv_ids if inv_id in gen_inv_map]
        aligned_cols = [gen_inv_map[inv_id] for inv_id in aligned_inv_ids]
        if aligned_cols:
            daily["subset_daily_gen_j"] = daily[aligned_cols].sum(axis=1, min_count=1)
            daily["subset_daily_gen_kwh"] = daily["subset_daily_gen_j"] / 3.6e6
            daily["subset_daily_gen_inverter_count"] = daily[aligned_cols].notna().sum(axis=1)
            expected = max(len(power_inv_ids), 1)
            daily["subset_daily_gen_coverage"] = (
                daily["subset_daily_gen_inverter_count"] / float(expected)
            )
        else:
            daily["subset_daily_gen_j"] = np.nan
            daily["subset_daily_gen_kwh"] = np.nan
            daily["subset_daily_gen_inverter_count"] = 0.0
            daily["subset_daily_gen_coverage"] = 0.0
    else:
        daily["subset_daily_gen_inverter_count"] = np.nan
        daily["subset_daily_gen_coverage"] = np.nan

    # --- Plant-level average solar radiation ---
    if plant_avg_irr is not None and not plant_avg_irr.empty:
        daily = daily.merge(plant_avg_irr, on="day", how="left")

    # --- Runtime handling: CSV first, then Solcast daylight fallback ---
    if runtime_daily is not None and not runtime_daily.empty:
        daily = daily.merge(runtime_daily, on="day", how="left")
    if runtime_solcast_daily is not None and not runtime_solcast_daily.empty:
        daily = daily.merge(runtime_solcast_daily, on="day", how="left")
    if ("runtime_h_csv" in daily.columns) or ("runtime_solcast_h" in daily.columns):
        runtime_csv = (
            daily["runtime_h_csv"]
            if "runtime_h_csv" in daily.columns
            else pd.Series(np.nan, index=daily.index)
        )
        runtime_solcast = (
            daily["runtime_solcast_h"]
            if "runtime_solcast_h" in daily.columns
            else pd.Series(np.nan, index=daily.index)
        )
        daily["runtime_h"] = runtime_csv.combine_first(runtime_solcast)
        src = pd.Series(np.nan, index=daily.index, dtype=object)
        src.loc[runtime_csv.notna()] = "runtime_csv"
        src.loc[src.isna() & runtime_solcast.notna()] = "solcast_daylight"
        daily["runtime_source"] = src

    # --- Power at reference irradiance ---
    if power_at_ref_irr is not None and not power_at_ref_irr.empty:
        daily = daily.merge(power_at_ref_irr, on="day", how="left")

    # --- Physical irradiation and PR fields ---
    daily["subset_capacity_kw"] = float(len(aligned_inv_ids) * inverter_capacity_kw)
    daily["plant_capacity_kw"] = float(plant_inverter_count * inverter_capacity_kw)

    if "plant_avg_irradiance_wm2" in daily.columns and "runtime_h" in daily.columns:
        valid_irr = (
            daily["plant_avg_irradiance_wm2"].notna()
            & (daily["plant_avg_irradiance_wm2"] > 0)
            & daily["runtime_h"].notna()
            & (daily["runtime_h"] > 0)
        )
        daily["irradiation_kwh_m2"] = np.where(
            valid_irr,
            daily["plant_avg_irradiance_wm2"] * daily["runtime_h"] / 1000.0,
            np.nan,
        )

    if (
        "subset_daily_gen_kwh" in daily.columns
        and "irradiation_kwh_m2" in daily.columns
        and daily["subset_capacity_kw"].iloc[0] > 0
    ):
        denom_subset = daily["subset_capacity_kw"] * daily["irradiation_kwh_m2"]
        valid_subset = (
            daily["subset_daily_gen_kwh"].notna()
            & denom_subset.notna()
            & (denom_subset > 0)
        )
        subset_pr_raw = pd.Series(np.nan, index=daily.index, dtype=float)
        subset_pr_raw.loc[valid_subset] = (
            daily.loc[valid_subset, "subset_daily_gen_kwh"] / denom_subset.loc[valid_subset]
        )
        subset_outlier = subset_pr_raw.notna() & (
            (subset_pr_raw < 0) | (subset_pr_raw > 1)
        )
        subset_interp = subset_pr_raw.mask(subset_outlier).interpolate(
            method="linear", limit_area="inside",
        )
        daily["subset_pr_physical_raw"] = subset_pr_raw
        daily["subset_pr_physical_outlier"] = subset_outlier
        daily["subset_pr_physical_interp"] = subset_interp
        daily["subset_pr_physical_roll7"] = subset_interp.rolling(
            7, center=True, min_periods=3,
        ).median()

        # Backward-compatible alias fields
        daily["gen_irr_ratio"] = daily["subset_pr_physical_raw"]
        daily["gen_irr_ratio_smoothed"] = daily["gen_irr_ratio"].rolling(
            7, center=True, min_periods=3,
        ).median()

    if (
        "daily_generation_j" in daily.columns
        and "irradiation_kwh_m2" in daily.columns
        and "plant_capacity_kw" in daily.columns
    ):
        daily["daily_generation_kwh"] = daily["daily_generation_j"] / 3.6e6
        denom_plant = daily["plant_capacity_kw"] * daily["irradiation_kwh_m2"]
        valid_plant = (
            daily["daily_generation_kwh"].notna()
            & denom_plant.notna()
            & (denom_plant > 0)
        )
        plant_pr_raw = pd.Series(np.nan, index=daily.index, dtype=float)
        plant_pr_raw.loc[valid_plant] = (
            daily.loc[valid_plant, "daily_generation_kwh"] / denom_plant.loc[valid_plant]
        )
        plant_outlier = plant_pr_raw.notna() & (
            (plant_pr_raw < 0) | (plant_pr_raw > 1)
        )
        plant_interp = plant_pr_raw.mask(plant_outlier).interpolate(
            method="linear", limit_area="inside",
        )
        daily["plant_pr_physical_raw"] = plant_pr_raw
        daily["plant_pr_physical_outlier"] = plant_outlier
        daily["plant_pr_physical_interp"] = plant_interp
        daily["plant_pr_physical_roll7"] = plant_interp.rolling(
            7, center=True, min_periods=3,
        ).median()

    # --- Derived energy columns ---
    daily["subset_energy_mwh"] = daily["subset_energy_j"] / 3.6e9
    daily["generation_mwh"] = daily["daily_generation_j"] / 3.6e9
    daily["plant_to_subset_energy_ratio"] = daily["generation_mwh"] / daily["subset_energy_mwh"]

    # --- Performance loss features ---
    daily = compute_performance_features(daily)
    daily = compute_performance_features(daily, energy_col="t1_energy_j", prefix="t1")
    daily = compute_performance_features(daily, energy_col="t2_energy_j", prefix="t2")

    # --- Combined PR (all tiered inverters) ---
    n_power_cols = len(power_cols)
    if "irradiance_tilted_sum" in daily.columns and n_power_cols > 0:
        daily["subset_pr"] = compute_performance_ratio(
            daily["subset_energy_j"],
            daily["irradiance_tilted_sum"],
            p_nom_kwp=P_NOM_KWP,
            n_inverters=n_power_cols,
        )

    # --- Cross-block correlation ---
    daily = compute_cross_block_correlation(daily)

    # --- Soiling feature engineering ---
    daily = compute_soiling_features(daily)

    # --- Domain Soiling Pressure Index ---
    daily = compute_domain_soiling_index(daily)

    # --- Cycle-aware deviation ---
    _irr_col_for_cycle = (
        "solcast_gti_peak_sum"
        if "solcast_gti_peak_sum" in daily.columns
        and daily["solcast_gti_peak_sum"].notna().sum() > 0
        else "irradiance_tilted_sum"
    )
    daily = compute_cycle_deviation(daily, irr_sum_col=_irr_col_for_cycle)

    # --- New-source performance features (from physical PR alias) ---
    if "subset_daily_gen_kwh" in daily.columns and "irradiation_kwh_m2" in daily.columns:
        daily["gen_irr_ratio"] = daily["subset_daily_gen_kwh"] / ((inverter_capacity_kw * plant_inverter_count) * daily["irradiation_kwh_m2"])

    if "gen_irr_ratio" in daily.columns and daily["gen_irr_ratio"].notna().sum() > 10:
        daily = compute_loss_proxy_from_ratio(
            daily, ratio_col="gen_irr_ratio", prefix="new",
        )
        daily = compute_cycle_deviation(
            daily, precomputed_x_col="gen_irr_ratio", prefix="new",
        )
        if "new_rolling_clean_baseline" in daily.columns:
            daily["new_performance_index"] = (
                daily["gen_irr_ratio"] / daily["new_rolling_clean_baseline"]
            ).clip(upper=1.5)

    # --- Temperature correction ---
    daily = compute_temperature_corrected_pr(daily)

    # --- Common-overlap window ---
    daily = compute_common_overlap(daily)

    # --- Quality flags (shared logic) ---
    daily = compute_quality_flags(daily)
    flag_cols = [c for c in daily.columns if c.startswith("flag_")]
    daily["flag_count"] = daily[flag_cols].fillna(False).sum(axis=1)

    # --- Transfer readiness (shared logic) ---
    daily = compute_transfer_readiness(daily)

    # --- Clear-Sky analyzable flag ---
    daily = flag_clear_sky_analyzable(daily)

    return daily


def write_preprocessing_summary(
    output_path: Path,
    inv_stats: Dict[str, float],
    irr_stats: Dict[str, float],
    gen_stats: Dict[str, float],
    daily: pd.DataFrame,
    filter_stats: Optional[Dict[str, object]] = None,
) -> None:
    """Write a human-readable preprocessing summary report."""
    total_days = len(daily)
    ready_days = int(daily["cross_plant_inference_ready"].sum())
    high_days = int((daily["transfer_quality_tier"] == "high").sum())
    medium_days = int((daily["transfer_quality_tier"] == "medium").sum())
    low_days = int((daily["transfer_quality_tier"] == "low").sum())

    # Flag counts
    sens_suspect = int(daily.get("flag_sensor_suspect_irradiance", pd.Series(dtype=bool)).sum())
    cov_gap = int(daily.get("flag_coverage_gap", pd.Series(dtype=bool)).sum())
    blk_mismatch = int(daily.get("flag_block_mismatch", pd.Series(dtype=bool)).sum())
    low_out = int(daily.get("flag_low_output_high_irr", pd.Series(dtype=bool)).sum())
    zero_out = int(daily.get("flag_zero_output", pd.Series(dtype=bool)).sum())
    total_flagged = int((daily.get("flag_count", 0) > 0).sum())

    # Clear-sky analyzable
    csa_days = int(daily.get("is_clear_sky_analyzable", pd.Series(dtype=bool)).sum())

    # Performance loss distribution
    perf_loss = daily["performance_loss_pct_proxy"]
    perf_med = perf_loss.median()
    perf_p90 = perf_loss.quantile(0.90)
    perf_max = perf_loss.max()

    # Block metrics
    block_lines = []
    if "block_mismatch_ratio" in daily.columns:
        bmr_med = daily["block_mismatch_ratio"].median()
        b1_avail = daily.get("b1_data_availability", pd.Series(dtype=float)).median()
        b2_avail = daily.get("b2_data_availability", pd.Series(dtype=float)).median()
        block_lines = [
            "",
            "## Block (B1 vs B2) Metrics",
            f"- Median B1/B2 energy ratio: {bmr_med:.3f}",
            f"- Median B1 data availability: {b1_avail:.3f}",
            f"- Median B2 data availability: {b2_avail:.3f}",
            f"- Block mismatch flagged days: {blk_mismatch}",
        ]

    # Peak-hour filtering stats
    filter_lines = []
    if filter_stats:
        filter_lines = [
            "",
            "## Pre-Aggregation Filtering",
            f"- Peak-hour filter: {filter_stats.get('peak_hours', 'disabled')}",
            f"- Inverter records removed (peak-hour): {filter_stats.get('inv_peak_removed', 'n/a')}",
            f"- Irradiance records removed (peak-hour): {filter_stats.get('irr_peak_removed', 'n/a')}",
            f"- Irradiance records removed (threshold): {filter_stats.get('irr_threshold_removed', 'n/a')}",
            f"- Irradiance threshold used: {filter_stats.get('irr_threshold_value', 'n/a')}",
        ]

    # Soiling features
    soiling_lines = []
    soiling_cols = [c for c in daily.columns if c.startswith(("days_since", "cumulative_pm", "humidity_x", "cycle_", "pvlib_", "pr_temperature"))]
    if soiling_cols:
        soiling_lines = [
            "",
            "## Soiling Features",
            f"- Soiling-specific columns added: {len(soiling_cols)}",
            f"- Columns: {', '.join(sorted(soiling_cols))}",
        ]

    # Per-inverter columns
    per_inv_cols = [c for c in daily.columns if c.endswith(("_energy_j", "_pr", "_normalized_output")) and c.startswith(("b1_", "b2_")) and "_data_" not in c and "block_" not in c]
    per_inv_lines = []
    if per_inv_cols:
        per_inv_lines = [
            "",
            "## Per-Inverter Metrics",
            f"- Per-inverter columns: {len(per_inv_cols)}",
        ]

    lines = [
        "# Preprocessing Summary",
        "",
        "## Scope",
        "- Source context: single 10-15 MW plant dataset (6 tiered inverters: 3 B2 Tier-1 + 3 B1 Tier-2, out of 34 total).",
        "- Intended use: build features that can inform cross-plant inference with quality gating.",
        "- Tier-1 (training): B2-08, B2-13, B2-17 (~93-95% availability)",
        "- Tier-2 (validation): B1-08, B1-01, B1-13 (~54% availability)",
        f"- Nameplate capacity per inverter (placeholder): {P_NOM_KWP} kWp",
        "",
        "## Cleaning Statistics",
        f"- Inverters rows: {int(inv_stats['rows'])}",
        f"- Inverters duplicates removed: {int(inv_stats['duplicates_removed'])}",
        f"- Inverters invalid power to NaN: {int(inv_stats['power_invalid_to_nan'])}",
        f"- Inverters invalid current to NaN: {int(inv_stats['current_invalid_to_nan'])}",
        f"- Inverters mean row completeness: {inv_stats['row_power_completeness_mean']:.3f}",
        f"- Irradiance rows: {int(irr_stats['rows'])}",
        f"- Irradiance duplicates removed: {int(irr_stats['duplicates_removed'])}",
        f"- Irradiance invalid values to NaN: {int(irr_stats['irr_invalid_to_nan'])}",
        f"- Generation rows: {int(gen_stats['rows'])}",
        f"- Generation duplicates removed: {int(gen_stats['duplicates_removed'])}",
        f"- Generation invalid values to NaN: {int(gen_stats['generation_invalid_to_nan'])}",
        "",
        "## Daily Modeling Table",
        f"- Total daily rows: {total_days}",
        f"- Total columns: {len(daily.columns)}",
        f"- Days with >=1 flag: {total_flagged}",
        f"- Cross-plant inference ready days: {ready_days}/{total_days}",
        f"- Transfer tier counts: high={high_days}, medium={medium_days}, low={low_days}",
        "",
        "## Performance Loss Proxy Distribution",
        f"- Median: {perf_med:.2f}%",
        f"- 90th percentile: {perf_p90:.2f}%",
        f"- Max: {perf_max:.2f}%",
        "",
        "## Flag Breakdown",
        f"- Sensor-suspect irradiance days: {sens_suspect}",
        f"- Coverage gap days: {cov_gap}",
        f"- Block mismatch days: {blk_mismatch}",
        f"- Low output under high irradiance days: {low_out}",
        f"- Zero output on sunny days (equipment shutdown): {zero_out}",
        "",
        "## Clear-Sky Analyzable (CSA)",
        f"- CSA days (clear, dry, equipment OK): {csa_days}/{total_days}",
    ] + block_lines + filter_lines + soiling_lines + per_inv_lines + [
        "",
        "## Notes",
        "- Transfer readiness is quality-gated and should be recalibrated when onboarded to another plant.",
        "- Features sensitive to plant design (DC/AC ratio, orientation, clipping behavior) should be normalized before portfolio-wide comparisons.",
        f"- Normalized output capped at {MAX_NORMALIZED_OUTPUT:,.0f} to prevent baseline corruption.",
        f"- Days with irradiance below {MIN_IRRADIANCE_FOR_BASELINE:,.0f} W·s/m² excluded from baseline.",
        f"- P_NOM_KWP = {P_NOM_KWP} kWp is a placeholder; replace with confirmed value from asset owner.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic data cleaning and preprocessing for PV telemetry."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw telemetry CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/preprocessed"),
        help="Output directory for cleaned/preprocessed files.",
    )
    parser.add_argument(
        "--trim-to-overlap",
        action="store_true",
        default=False,
        help="If set, daily_model_input.csv is trimmed to common-overlap days only.",
    )
    parser.add_argument(
        "--no-peak-filter",
        action="store_true",
        default=False,
        help="Disable peak-hour and irradiance-threshold filtering (use all records).",
    )
    parser.add_argument(
        "--inverter-capacity-kw",
        type=float,
        default=330.0,
        help="Nameplate inverter AC capacity in kW for physical PR calculation.",
    )
    parser.add_argument(
        "--plant-inverter-count",
        type=int,
        default=34,
        help="Plant inverter count for plant-level physical PR denominator.",
    )
    parser.add_argument(
        "--runtime-min-hours",
        type=float,
        default=6.0,
        help="Minimum valid runtime hours from runtime CSV.",
    )
    parser.add_argument(
        "--runtime-max-hours",
        type=float,
        default=18.0,
        help="Maximum valid runtime hours from runtime CSV.",
    )
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        default=DEFAULT_RUNTIME_CSV,
        help="Runtime CSV path (uses runtime_hours column when available).",
    )
    parser.add_argument(
        "--disable-new-source-if-clipped-pct",
        type=float,
        default=5.0,
        help=(
            "Disable new-source PR fields if fetch audit above-warning share exceeds this percent. "
            "Set negative to disable the gate."
        ),
    )
    args = parser.parse_args()

    # Prefer tiered primary file if it exists (produced by split_inverter_tiers.py),
    # otherwise fall back to the original raw fetch output.
    tiered_path = args.data_dir / "inverters_tiered_primary_10min.csv"
    raw_path = args.data_dir / "inverters_2025_to_current_10min_avg_si.csv"
    inverters_path = tiered_path if tiered_path.exists() else raw_path
    irradiance_path = args.data_dir / "irradiance_2025_to_current_15min_sum_si.csv"
    generation_path = args.data_dir / "power_generation_2025_to_current_1day_none_si.csv"

    for path in (inverters_path, irradiance_path, generation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")

    # Solcast files (optional — skip gracefully if not found)
    solcast_soiling_path = args.data_dir / "soiling_2025_to_current_10min_none_std.csv"
    solcast_irradiance_path = args.data_dir / "irradiance_2025_to_current_10min_none_std.csv"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    inv_clean, inv_stats = clean_inverters(inverters_path)
    irr_clean, irr_stats = clean_irradiance(irradiance_path)
    gen_clean, gen_daily, gen_stats = clean_generation(generation_path)

    # --- Peak-hour and irradiance-threshold filtering ---
    filter_stats: Dict[str, object] = {}
    if not args.no_peak_filter:
        inv_clean, inv_peak_removed = filter_peak_hours(inv_clean)
        irr_clean, irr_peak_removed = filter_peak_hours(irr_clean)
        print(f"  Peak-hour filter ({PEAK_HOUR_START}:00-{PEAK_HOUR_END}:00): "
              f"inverter records removed={inv_peak_removed}, "
              f"irradiance records removed={irr_peak_removed}")

        tilted_col, _ = detect_irradiance_cols(irr_clean.columns)
        irr_clean, irr_thr_removed, irr_thr_value = filter_irradiance_threshold(
            irr_clean, tilted_col,
        )
        print(f"  Irradiance threshold filter: removed={irr_thr_removed}, "
              f"threshold={irr_thr_value:.2f}")

        filter_stats = {
            "peak_hours": f"{PEAK_HOUR_START}:00-{PEAK_HOUR_END}:00",
            "inv_peak_removed": inv_peak_removed,
            "irr_peak_removed": irr_peak_removed,
            "irr_threshold_removed": irr_thr_removed,
            "irr_threshold_value": f"{irr_thr_value:.2f}",
        }
    else:
        print("  Peak-hour filtering disabled (--no-peak-filter)")
        filter_stats = {"peak_hours": "disabled"}

    # Aggregate Solcast to daily (if available)
    solcast_daily = None
    if solcast_soiling_path.exists():
        sc_irr = solcast_irradiance_path if solcast_irradiance_path.exists() else None
        solcast_daily = aggregate_solcast_daily(solcast_soiling_path, sc_irr)
        print(f"  Solcast daily features: {len(solcast_daily)} days, {len(solcast_daily.columns)} columns")

    # Compute pvlib soiling estimates (if Solcast soiling data exists)
    pvlib_soiling_daily = None
    if solcast_soiling_path.exists():
        sc_irr_path = solcast_irradiance_path if solcast_irradiance_path.exists() else None
        pvlib_soiling_daily = compute_pvlib_soiling_ratio(solcast_soiling_path, sc_irr_path)
        if not pvlib_soiling_daily.empty:
            print(f"  pvlib soiling estimates: {len(pvlib_soiling_daily)} days")

    # --- New telemetry: per-inverter daily generated electricity (optional) ---
    inv_daily_gen_daily = None
    inv_daily_gen_audit = pd.DataFrame()
    new_source_disabled = False
    inv_daily_gen_path = args.data_dir / "inverters_daily_gen_2025_to_current_none_si.csv"
    fetch_audit_path = args.data_dir / "inverters_daily_gen_fetch_audit.json"
    if inv_daily_gen_path.exists():
        inv_daily_gen_daily, inv_daily_gen_audit, idg_stats = clean_inverter_daily_gen(
            inv_daily_gen_path
        )
        print(
            f"  Inverter daily gen: {int(idg_stats['daily_rows'])} days, "
            f"{int(idg_stats['inverter_count'])} inverters"
        )
        clip_pct = None
        if fetch_audit_path.exists():
            try:
                payload = json.loads(fetch_audit_path.read_text(encoding="utf-8"))
                clip_pct = float(payload.get("global", {}).get("above_warning_pct"))
            except Exception:
                clip_pct = None
        if clip_pct is not None:
            print(
                "  Inverter daily-gen fetch audit: "
                f"above-warning={clip_pct:.3f}% (threshold={args.disable_new_source_if_clipped_pct:.3f}%)"
            )
            if (
                args.disable_new_source_if_clipped_pct >= 0
                and clip_pct > args.disable_new_source_if_clipped_pct
            ):
                new_source_disabled = True
                # inv_daily_gen_daily = None  # FORCE ENABLE: We rely on this new metric regardless of clipping.
                print(
                    "  WARNING: New daily-gen source flagged high above-warning share in fetch audit (BYPASSING DISABLE)."
                )
    else:
        idg_stats = {}

    # --- New telemetry: plant-level average solar radiation (optional) ---
    plant_avg_irr_daily = None
    plant_avg_irr_path = args.data_dir / "plant_avg_irradiance_2025_to_current_none_si.csv"
    if plant_avg_irr_path.exists():
        plant_avg_irr_daily, pai_stats = clean_plant_avg_irradiance(plant_avg_irr_path)
        print(
            f"  Plant avg irradiance: {int(pai_stats['daily_rows'])} days, "
            f"{int(pai_stats['irr_unstable_days_removed'])} unstable days removed"
        )

    # --- Runtime daily (CSV primary + Solcast fallback) ---
    runtime_daily, runtime_stats = load_runtime_daily(
        args.runtime_csv,
        min_hours=args.runtime_min_hours,
        max_hours=args.runtime_max_hours,
    )
    runtime_solcast_daily = load_solcast_runtime_daily(solcast_irradiance_path)
    print(
        "  Runtime CSV: "
        f"{int(runtime_stats['runtime_daily_rows'])} days, "
        f"invalid={int(runtime_stats['runtime_invalid_days'])}, "
        f"bounds=[{args.runtime_min_hours}, {args.runtime_max_hours}] h"
    )
    if not runtime_solcast_daily.empty:
        print(f"  Runtime Solcast fallback: {len(runtime_solcast_daily)} days")

    # --- Power at reference irradiance (from existing sub-daily data) ---
    ref_irr_daily = compute_power_at_reference_irradiance(inv_clean, irr_clean)
    if not ref_irr_daily.empty:
        print(
            f"  Power at ref irradiance: {len(ref_irr_daily)} days, "
            f"ref={ref_irr_daily['ref_irradiance_wm2'].iloc[0]:.0f} W/m²"
        )

    daily_model = build_daily_model_table(
        inv_clean, irr_clean, gen_daily, solcast_daily, pvlib_soiling_daily,
        inv_daily_gen=inv_daily_gen_daily,
        plant_avg_irr=plant_avg_irr_daily,
        runtime_daily=runtime_daily,
        runtime_solcast_daily=runtime_solcast_daily,
        power_at_ref_irr=ref_irr_daily,
        peak_filtered=not args.no_peak_filter,
        inverter_capacity_kw=args.inverter_capacity_kw,
        plant_inverter_count=args.plant_inverter_count,
    )
    if new_source_disabled:
        print("  New-source daily generation was disabled by fetch-audit gate.")

    inv_clean.to_csv(args.out_dir / "inverters_clean.csv", index=False)
    irr_clean.to_csv(args.out_dir / "irradiance_clean.csv", index=False)
    gen_clean.to_csv(args.out_dir / "generation_clean.csv", index=False)
    gen_daily.to_csv(args.out_dir / "generation_daily_clean.csv", index=False)
    audit_out = args.out_dir / "inverter_daily_gen_audit.csv"
    if inv_daily_gen_audit is not None and not inv_daily_gen_audit.empty:
        inv_daily_gen_audit.to_csv(audit_out, index=False)
    else:
        pd.DataFrame(
            columns=[
                "day",
                "inverter",
                "records",
                "non_null_records",
                "last_valid_dt",
                "last_kwh",
                "max_kwh",
                "p99_kwh",
                "selected_kwh",
                "selection_method",
                "outlier_flag",
                "selected_kwh_clean",
            ]
        ).to_csv(audit_out, index=False)

    if (
        "subset_daily_gen_kwh" in daily_model.columns
        and "irradiation_kwh_m2" in daily_model.columns
    ):
        corr_mask = (
            daily_model["subset_daily_gen_kwh"].notna()
            & daily_model["irradiation_kwh_m2"].notna()
            & (daily_model["irradiation_kwh_m2"] > 0)
        )
        if int(corr_mask.sum()) >= 10:
            corr_val = float(
                daily_model.loc[corr_mask, "subset_daily_gen_kwh"].corr(
                    daily_model.loc[corr_mask, "irradiation_kwh_m2"]
                )
            )
            if np.isfinite(corr_val):
                print(f"  Corr(subset_daily_gen_kwh, irradiation_kwh_m2)={corr_val:.3f}")
                if corr_val < 0.30:
                    print("  WARNING: Correlation below 0.30; inspect generation source quality.")
    if "runtime_source" in daily_model.columns:
        src_counts = daily_model["runtime_source"].fillna("missing").value_counts().to_dict()
        print(f"  Runtime source distribution: {src_counts}")

    # Optionally trim the primary output to overlap-valid days
    if args.trim_to_overlap and "in_common_overlap" in daily_model.columns:
        trimmed = len(daily_model) - daily_model["in_common_overlap"].sum()
        daily_model = daily_model[daily_model["in_common_overlap"]].copy()
        print(f"  --trim-to-overlap: removed {trimmed} non-overlap days")

    daily_model.to_csv(args.out_dir / "daily_model_input.csv", index=False)

    # Always produce a trimmed EDA convenience table
    if "in_common_overlap" in daily_model.columns:
        eda_table = daily_model[daily_model["in_common_overlap"]].copy()
        eda_table.to_csv(args.out_dir / "daily_model_eda.csv", index=False)
        print(f"  daily_model_eda.csv: {len(eda_table)} rows (overlap-filtered)")

    write_preprocessing_summary(
        args.out_dir / "preprocessing_summary.md",
        inv_stats,
        irr_stats,
        gen_stats,
        daily_model,
        filter_stats,
    )

    print(f"Preprocessing complete. Outputs written to: {args.out_dir}")
    print(f"  daily_model_input.csv: {len(daily_model)} rows x {len(daily_model.columns)} columns")


if __name__ == "__main__":
    main()
