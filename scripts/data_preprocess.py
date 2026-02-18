"""Clean and preprocess PV telemetry data for modeling.

This script standardizes raw exports, applies domain sanity checks, builds
daily modeling features, and writes preprocessing outputs.

Usage:
    python scripts/data_preprocess.py --data-dir data --out-dir artifacts/preprocessed
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure sibling modules (daily_features) are importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from daily_features import (
    INVERTER_INTERVAL_S,
    MIN_IRRADIANCE_FOR_BASELINE,
    MAX_NORMALIZED_OUTPUT,
    aggregate_block_daily,
    aggregate_inverter_daily,
    aggregate_irradiance_daily,
    compute_quality_flags,
    compute_soiling_features,
    compute_transfer_readiness,
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
) -> pd.DataFrame:
    """Build the daily model input table from cleaned sub-daily data.

    Uses shared feature functions from ``daily_features`` module for soiling,
    flags, and transfer readiness to stay in sync with the audit pipeline.
    """
    inv = inverters.copy()
    irr = irradiance.copy()
    gen = generation_daily.copy()

    # --- Inverter daily aggregation ---
    inv_daily, power_cols = aggregate_inverter_daily(inv)

    # --- Block (B1 vs B2) daily aggregation ---
    block_daily = aggregate_block_daily(inv, power_cols)
    if not block_daily.empty:
        inv_daily = inv_daily.merge(block_daily, on="day", how="left")

    # --- Irradiance daily aggregation ---
    irr_daily = aggregate_irradiance_daily(irr)

    # --- Merge all daily tables ---
    daily = (
        inv_daily.merge(irr_daily, on="day", how="outer")
        .merge(gen, on="day", how="outer")
        .sort_values("day")
        .reset_index(drop=True)
    )

    # --- Derived energy columns ---
    daily["subset_energy_mwh"] = daily["subset_energy_j"] / 3.6e9
    daily["generation_mwh"] = daily["daily_generation_j"] / 3.6e9
    daily["plant_to_subset_energy_ratio"] = (
        daily["generation_mwh"] / daily["subset_energy_mwh"]
    )

    # --- Soiling features (shared logic) ---
    daily = compute_soiling_features(daily)

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

    # Soiling distribution
    soiling = daily["soiling_loss_pct_proxy"]
    soiling_med = soiling.median()
    soiling_p90 = soiling.quantile(0.90)
    soiling_max = soiling.max()

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

    lines = [
        "# Preprocessing Summary",
        "",
        "## Scope",
        "- Source context: single 10-15 MW plant dataset (8 sampled inverters out of 34 total).",
        "- Intended use: build features that can inform cross-plant inference with quality gating.",
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
        f"- Days with >=1 flag: {total_flagged}",
        f"- Cross-plant inference ready days: {ready_days}/{total_days}",
        f"- Transfer tier counts: high={high_days}, medium={medium_days}, low={low_days}",
        "",
        "## Soiling Proxy Distribution",
        f"- Median: {soiling_med:.2f}%",
        f"- 90th percentile: {soiling_p90:.2f}%",
        f"- Max: {soiling_max:.2f}%",
        "",
        "## Flag Breakdown",
        f"- Sensor-suspect irradiance days: {sens_suspect}",
        f"- Coverage gap days: {cov_gap}",
        f"- Block mismatch days: {blk_mismatch}",
        f"- Low output under high irradiance days: {low_out}",
    ] + block_lines + [
        "",
        "## Notes",
        "- Transfer readiness is quality-gated and should be recalibrated when onboarded to another plant.",
        "- Features sensitive to plant design (DC/AC ratio, orientation, clipping behavior) should be normalized before portfolio-wide comparisons.",
        f"- Normalized output capped at {MAX_NORMALIZED_OUTPUT:,.0f} to prevent baseline corruption.",
        f"- Days with irradiance below {MIN_IRRADIANCE_FOR_BASELINE:,.0f} W·s/m² excluded from baseline.",
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
    args = parser.parse_args()

    inverters_path = args.data_dir / "inverters_2025_to_current_10min_avg_si.csv"
    irradiance_path = args.data_dir / "irradiance_2025_to_current_15min_sum_si.csv"
    generation_path = args.data_dir / "power_generation_2025_to_current_1day_none_si.csv"

    for path in (inverters_path, irradiance_path, generation_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    inv_clean, inv_stats = clean_inverters(inverters_path)
    irr_clean, irr_stats = clean_irradiance(irradiance_path)
    gen_clean, gen_daily, gen_stats = clean_generation(generation_path)
    daily_model = build_daily_model_table(inv_clean, irr_clean, gen_daily)

    inv_clean.to_csv(args.out_dir / "inverters_clean.csv", index=False)
    irr_clean.to_csv(args.out_dir / "irradiance_clean.csv", index=False)
    gen_clean.to_csv(args.out_dir / "generation_clean.csv", index=False)
    gen_daily.to_csv(args.out_dir / "generation_daily_clean.csv", index=False)
    daily_model.to_csv(args.out_dir / "daily_model_input.csv", index=False)

    write_preprocessing_summary(
        args.out_dir / "preprocessing_summary.md",
        inv_stats,
        irr_stats,
        gen_stats,
        daily_model,
    )

    print(f"Preprocessing complete. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
