"""Clean and preprocess PV telemetry data for modeling.

This script standardizes raw exports, applies domain sanity checks, builds
daily modeling features, and writes preprocessing outputs.

Usage:
    python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure sibling modules (daily_features) are importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from daily_features import (
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
    compute_performance_features,
    compute_performance_ratio,
    compute_pvlib_soiling_ratio,
    compute_quality_flags,
    compute_soiling_features,
    compute_temperature_corrected_pr,
    compute_transfer_readiness,
    detect_irradiance_cols,
    filter_irradiance_threshold,
    filter_peak_hours,
)

MAX_POWER_W = 300_000.0
MAX_CURRENT_A = 250.0
MAX_GENERATION_J = 360_000_000_000.0


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


def build_daily_model_table(
    inverters: pd.DataFrame,
    irradiance: pd.DataFrame,
    generation_daily: pd.DataFrame,
    solcast_daily: pd.DataFrame = None,
    pvlib_soiling_daily: pd.DataFrame = None,
    peak_filtered: bool = False,
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
    daily = compute_cycle_deviation(daily)

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
    total_flagged = int((daily.get("flag_count", 0) > 0).sum())

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

    daily_model = build_daily_model_table(
        inv_clean, irr_clean, gen_daily, solcast_daily, pvlib_soiling_daily,
        peak_filtered=not args.no_peak_filter,
    )

    inv_clean.to_csv(args.out_dir / "inverters_clean.csv", index=False)
    irr_clean.to_csv(args.out_dir / "irradiance_clean.csv", index=False)
    gen_clean.to_csv(args.out_dir / "generation_clean.csv", index=False)
    gen_daily.to_csv(args.out_dir / "generation_daily_clean.csv", index=False)

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
