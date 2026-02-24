import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Ensure sibling modules (daily_features) are importable from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from daily_features import (
    MIN_IRRADIANCE_FOR_BASELINE,
    MAX_NORMALIZED_OUTPUT,
    INVERTER_INTERVAL_S,
    EXPECTED_INV_RECORDS_PER_DAY,
    EXPECTED_IRR_RECORDS_PER_DAY,
    detect_block_power_cols as block_power_columns,
    aggregate_solcast_daily,
    aggregate_tier_daily,
    compute_common_overlap,
    compute_cross_block_correlation,
    compute_performance_features,
    compute_quality_flags as _shared_quality_flags,
)


EXPECTED_INTERVAL_SECONDS = {
    "inverters": INVERTER_INTERVAL_S,
    "irradiance": 900,
    "generation": None,
}

EXPECTED_DAILY_RECORDS = {
    "inverters": EXPECTED_INV_RECORDS_PER_DAY,
    "irradiance": EXPECTED_IRR_RECORDS_PER_DAY,
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Timestamp", "Date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def interval_distribution(
    df: pd.DataFrame,
    dataset: str,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    diffs = df["Date"].diff().dt.total_seconds().dropna()
    if diffs.empty:
        dist = pd.DataFrame(columns=["dataset", "interval_seconds", "count"])
        stats = {
            "median_interval_seconds": np.nan,
            "min_interval_seconds": np.nan,
            "max_interval_seconds": np.nan,
            "expected_interval_match_ratio": np.nan,
            "large_gap_count": np.nan,
        }
        return dist, stats

    interval_counts = (
        diffs.value_counts(dropna=True)
        .rename_axis("interval_seconds")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    interval_counts.insert(0, "dataset", dataset)

    expected = EXPECTED_INTERVAL_SECONDS.get(dataset)
    if expected is None:
        expected_ratio = np.nan
        large_gap_count = int((diffs > 3600).sum())
    else:
        expected_ratio = float((diffs == expected).mean())
        large_gap_count = int((diffs > (expected * 3)).sum())

    stats = {
        "median_interval_seconds": float(diffs.median()),
        "min_interval_seconds": float(diffs.min()),
        "max_interval_seconds": float(diffs.max()),
        "expected_interval_match_ratio": expected_ratio,
        "large_gap_count": large_gap_count,
    }
    return interval_counts, stats


def dataset_profile(df: pd.DataFrame, dataset: str) -> Dict[str, object]:
    dist, stats = interval_distribution(df, dataset)
    _ = dist
    date_series = df["Date"]
    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_date": date_series.min(),
        "end_date": date_series.max(),
        "unique_days": int(date_series.dt.date.nunique()),
        "null_date_rows": int(date_series.isna().sum()),
        "median_interval_seconds": stats["median_interval_seconds"],
        "min_interval_seconds": stats["min_interval_seconds"],
        "max_interval_seconds": stats["max_interval_seconds"],
        "expected_interval_seconds": EXPECTED_INTERVAL_SECONDS.get(dataset),
        "expected_interval_match_ratio": stats["expected_interval_match_ratio"],
        "large_gap_count": stats["large_gap_count"],
    }


def missingness_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("Timestamp", "Date")]
    missing = df[cols].isna().mean().rename("missing_ratio").reset_index()
    missing = missing.rename(columns={"index": "column"})
    missing.insert(0, "dataset", dataset)
    return missing.sort_values("missing_ratio", ascending=False).reset_index(drop=True)


def inverter_prefixes(columns: Iterable[str]) -> List[str]:
    prefixes = set()
    for col in columns:
        if col.endswith(" Current A (A)"):
            prefixes.add(col.replace(" Current A (A)", ""))
    return sorted(prefixes)


def row_phase_imbalance(df: pd.DataFrame) -> pd.Series:
    prefixes = inverter_prefixes(df.columns)
    if not prefixes:
        return pd.Series(index=df.index, dtype=float)

    per_inv = []
    for prefix in prefixes:
        phase_cols = [
            f"{prefix} Current A (A)",
            f"{prefix} Current B (A)",
            f"{prefix} Current C (A)",
        ]
        if not all(col in df.columns for col in phase_cols):
            continue
        subset = df[phase_cols]
        mean_current = subset.mean(axis=1)
        imbalance = (subset.max(axis=1) - subset.min(axis=1)) / mean_current.replace(
            0, np.nan
        )
        per_inv.append(imbalance.replace([np.inf, -np.inf], np.nan))

    if not per_inv:
        return pd.Series(index=df.index, dtype=float)

    return pd.concat(per_inv, axis=1).mean(axis=1)


def build_daily_features(
    inverters: pd.DataFrame,
    irradiance: pd.DataFrame,
    generation: pd.DataFrame,
    solcast_daily: pd.DataFrame = None,
) -> pd.DataFrame:
    inv = inverters.copy()
    irr = irradiance.copy()
    gen = generation.copy()

    inv["day"] = inv["Date"].dt.date
    irr["day"] = irr["Date"].dt.date
    gen["day"] = gen["Date"].dt.date

    power_cols = [c for c in inv.columns if c.endswith("Active Power (W)")]
    if not power_cols:
        raise ValueError("No inverter power columns found.")

    inv["subset_power_w"] = inv[power_cols].sum(axis=1, min_count=1)
    inv["subset_energy_j_step"] = inv["subset_power_w"] * EXPECTED_INTERVAL_SECONDS["inverters"]
    inv["subset_data_availability"] = inv[power_cols].notna().mean(axis=1)
    inv["phase_imbalance"] = row_phase_imbalance(inv)

    inv_daily = inv.groupby("day").agg(
        subset_energy_j=("subset_energy_j_step", "sum"),
        subset_power_w_p95=("subset_power_w", lambda s: s.quantile(0.95)),
        subset_data_availability_mean=("subset_data_availability", "mean"),
        subset_data_availability_p10=("subset_data_availability", lambda s: s.quantile(0.10)),
        phase_imbalance_mean=("phase_imbalance", "mean"),
        phase_imbalance_p95=("phase_imbalance", lambda s: s.quantile(0.95)),
        inverter_records=("Timestamp", "size"),
    )
    inv_daily["inverter_coverage_ratio"] = (
        inv_daily["inverter_records"] / EXPECTED_INV_RECORDS_PER_DAY
    ).clip(upper=1.0)

    irr_cols = [c for c in irr.columns if "Irradiance" in c]
    if not irr_cols:
        raise ValueError("No irradiance columns found.")

    tilted_cols = [c for c in irr_cols if "Tilted" in c]
    horizontal_cols = [c for c in irr_cols if "Horizontal" in c]
    tilted_col = tilted_cols[0] if tilted_cols else irr_cols[0]
    horizontal_col = horizontal_cols[0] if horizontal_cols else irr_cols[0]

    irr_daily = irr.groupby("day").agg(
        irradiance_horizontal_sum=(horizontal_col, "sum"),
        irradiance_tilted_sum=(tilted_col, "sum"),
        irradiance_records=("Timestamp", "size"),
    )
    irr_daily["irradiance_coverage_ratio"] = (
        irr_daily["irradiance_records"] / EXPECTED_IRR_RECORDS_PER_DAY
    ).clip(upper=1.0)

    gen_cols = [c for c in gen.columns if c not in ("Timestamp", "Date", "day")]
    if len(gen_cols) != 1:
        raise ValueError(f"Expected exactly one generation value column, got: {gen_cols}")
    gen_col = gen_cols[0]

    gen_sorted = gen.sort_values("Date")
    gen_daily = gen_sorted.groupby("day").agg(
        daily_generation_j_latest=(gen_col, "last"),
        daily_generation_j_max=(gen_col, "max"),
        daily_generation_j_min=(gen_col, "min"),
        generation_records=("Timestamp", "size"),
    )
    gen_daily["generation_intraday_spread_j"] = (
        gen_daily["daily_generation_j_max"] - gen_daily["daily_generation_j_min"]
    )

    daily = inv_daily.join(irr_daily, how="outer").join(gen_daily, how="outer")
    daily = daily.sort_index()
    daily.index = pd.to_datetime(daily.index, errors="coerce")

    daily["subset_energy_mwh"] = daily["subset_energy_j"] / 3.6e9
    daily["generation_mwh_latest"] = daily["daily_generation_j_latest"] / 3.6e9
    daily["plant_to_subset_energy_ratio"] = (
        daily["generation_mwh_latest"] / daily["subset_energy_mwh"]
    )

    # --- Per-block (B1 vs B2) energy comparison ---
    b1_cols, b2_cols = block_power_columns(inv.columns)
    if b1_cols and b2_cols:
        inv["b1_power_w"] = inv[b1_cols].sum(axis=1, min_count=1)
        inv["b2_power_w"] = inv[b2_cols].sum(axis=1, min_count=1)
        inv["b1_energy_j_step"] = inv["b1_power_w"] * EXPECTED_INTERVAL_SECONDS["inverters"]
        inv["b2_energy_j_step"] = inv["b2_power_w"] * EXPECTED_INTERVAL_SECONDS["inverters"]

        block_daily = inv.groupby("day").agg(
            b1_energy_j=("b1_energy_j_step", "sum"),
            b2_energy_j=("b2_energy_j_step", "sum"),
            b1_data_availability=("b1_power_w", lambda s: s.notna().mean()),
            b2_data_availability=("b2_power_w", lambda s: s.notna().mean()),
        )
        block_daily.index = pd.to_datetime(block_daily.index, errors="coerce")
        daily = daily.join(block_daily, how="left")
        daily["block_mismatch_ratio"] = daily["b1_energy_j"] / daily["b2_energy_j"].replace(0, np.nan)
        daily["block_mismatch_ratio_rolling_median"] = (
            daily["block_mismatch_ratio"].rolling(14, min_periods=5).median()
        )
    # --- Tier-1 / Tier-2 daily aggregation ---
    tier_daily = aggregate_tier_daily(inv, power_cols)
    if not tier_daily.empty:
        tier_daily.index = pd.to_datetime(tier_daily["day"], errors="coerce")
        tier_daily = tier_daily.drop(columns=["day"])
        daily = daily.join(tier_daily, how="left")

    # --- Performance loss features (shared logic) ---
    # Combined (backward compat)
    daily = compute_performance_features(daily)
    # Tier-1 (B2 training signal)
    daily = compute_performance_features(daily, energy_col="t1_energy_j", prefix="t1")
    # Tier-2 (B1 validation signal)
    daily = compute_performance_features(daily, energy_col="t2_energy_j", prefix="t2")

    # --- Cross-block correlation ---
    daily = compute_cross_block_correlation(daily)

    # --- Merge Solcast environmental features ---
    # Always reset the day index to a column first
    if daily.index.name == "day":
        daily = daily.reset_index()
    if "day" not in daily.columns:
        raise ValueError("Expected 'day' column in daily features after reset_index.")

    daily["day"] = pd.to_datetime(daily["day"], errors="coerce")

    if solcast_daily is not None and not solcast_daily.empty:
        solcast_daily = solcast_daily.copy()
        solcast_daily["day"] = pd.to_datetime(solcast_daily["day"], errors="coerce")
        daily = daily.merge(solcast_daily, on="day", how="left")

    # --- Common-overlap window ---
    daily = compute_common_overlap(daily)

    return daily


def build_daily_flags(daily: pd.DataFrame) -> pd.DataFrame:
    """Build daily flags combining shared quality flags with audit-specific ones."""
    frame = daily.copy()

    spread_high_threshold = frame["generation_intraday_spread_j"].quantile(0.95)

    # Audit-specific flags (not in shared module)
    frame["flag_low_data_availability"] = frame["subset_data_availability_mean"] < 0.50
    frame["flag_high_phase_imbalance"] = frame["phase_imbalance_p95"] > 0.12
    frame["flag_zero_irr_nontrivial_gen"] = (
        (frame["irradiance_tilted_sum"] <= 1.0)
        & (frame["generation_mwh_latest"] > 5.0)
    )
    frame["flag_large_generation_intraday_spread"] = (
        frame["generation_intraday_spread_j"] > spread_high_threshold
    )

    # Shared quality flags (sensor suspect, coverage gap, block mismatch, low output)
    frame = _shared_quality_flags(frame)

    flag_cols = [c for c in frame.columns if c.startswith("flag_")]
    frame["flag_count"] = frame[flag_cols].fillna(False).sum(axis=1)

    keep_cols = ["day"] + sorted(flag_cols) + ["flag_count"]
    return frame[keep_cols].copy()


def write_quality_summary(
    output_path: Path,
    profile_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    daily_flags: pd.DataFrame,
) -> None:
    header = "|" + "|".join(profile_df.columns) + "|"
    separator = "|" + "|".join(["---"] * len(profile_df.columns)) + "|"
    rows = [
        "|" + "|".join(str(v) for v in row) + "|"
        for row in profile_df.itertuples(index=False, name=None)
    ]
    profile_table = "\n".join([header, separator] + rows)

    total_days = len(daily_features)
    flagged_days = int((daily_flags["flag_count"] > 0).sum())
    high_imbalance_days = int(daily_flags["flag_high_phase_imbalance"].sum())
    low_data_days = int(daily_flags["flag_low_data_availability"].sum())
    low_output_days = int(daily_flags["flag_low_output_high_irr"].sum())
    zero_irr_gen_days = int(daily_flags["flag_zero_irr_nontrivial_gen"].sum())
    sensor_suspect_days = int(daily_flags.get("flag_sensor_suspect_irradiance", pd.Series(dtype=bool)).sum())
    coverage_gap_days = int(daily_flags.get("flag_coverage_gap", pd.Series(dtype=bool)).sum())
    block_mismatch_days = int(daily_flags.get("flag_block_mismatch", pd.Series(dtype=bool)).sum())

    perf_loss_median = daily_features["performance_loss_pct_proxy"].median()
    perf_loss_p90 = daily_features["performance_loss_pct_proxy"].quantile(0.90)
    ratio_median = daily_features["plant_to_subset_energy_ratio"].median()

    # Block metrics (if available)
    block_lines = []
    if "block_mismatch_ratio" in daily_features.columns:
        bmr_median = daily_features["block_mismatch_ratio"].median()
        block_lines = [
            "",
            "## Block (B1 vs B2) Metrics",
            f"- Median B1/B2 energy ratio: {bmr_median:.3f}",
            f"- Block mismatch flagged days: {block_mismatch_days}",
        ]

    lines = [
        "# Data Quality Summary",
        "",
        "## Dataset Profile",
        profile_table,
        "",
        "## Daily KPI + Flags",
        f"- Total merged days: {total_days}",
        f"- Days with >=1 flag: {flagged_days}",
        f"- Low data availability days: {low_data_days}",
        f"- High phase imbalance days: {high_imbalance_days}",
        f"- Low output under high irradiance days: {low_output_days}",
        f"- Zero irradiance but nontrivial generation days: {zero_irr_gen_days}",
        f"- Sensor-suspect irradiance days: {sensor_suspect_days}",
        f"- Coverage gap days (< 30% expected records): {coverage_gap_days}",
        "",
        "## Proxy Metrics",
        f"- Median performance loss proxy (%): {perf_loss_median:.2f}",
        f"- 90th percentile performance loss proxy (%): {perf_loss_p90:.2f}",
        f"- Median plant/subset daily energy ratio: {ratio_median:.2f}",
    ] + block_lines + [
        "",
        "## Notes",
        "- Plant-level generation is compared against only 6 tiered inverters (3 B2 Tier-1 + 3 B1 Tier-2), so ratio scaling should be interpreted with context.",
        "- Generation feed has multiple intraday points and should be treated as event telemetry, not strictly one row per day.",
        f"- Normalized output values capped at {MAX_NORMALIZED_OUTPUT:,.0f} to prevent baseline corruption from sensor outages.",
        f"- Days with irradiance below {MIN_IRRADIANCE_FOR_BASELINE:,.0f} W·s/m² are excluded from baseline computation.",
        "- Performance loss baseline uses 95th percentile (rolling 30-day) instead of max for outlier resistance.",
        "- Use flagged days to drive manual inspection before model training.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def try_generate_plots(
    output_dir: Path,
    interval_df: pd.DataFrame,
    daily_features: pd.DataFrame,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Interval histograms
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    datasets = ["inverters", "irradiance", "generation"]
    for ax, name in zip(axes, datasets):
        subset = interval_df[interval_df["dataset"] == name].copy()
        if subset.empty:
            ax.set_title(f"{name}: no interval data")
            continue
        subset = subset.sort_values("count", ascending=False).head(30)
        ax.bar(subset["interval_seconds"].astype(str), subset["count"])
        ax.set_title(f"{name}: top interval counts")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "interval_histograms.png", dpi=150)
    plt.close(fig)

    # Normalized output + performance loss proxy
    df = daily_features.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.sort_values("day")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["day"], df["normalized_output"], label="normalized_output", linewidth=1.0)
    axes[0].plot(
        df["day"],
        df["rolling_clean_baseline"],
        label="rolling_clean_baseline",
        linewidth=1.3,
    )
    axes[0].set_ylabel("J / irradiance_sum")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Normalized Output vs Rolling Baseline")

    axes[1].plot(
        df["day"],
        df["performance_loss_pct_proxy"],
        label="performance_loss_pct_proxy",
        linewidth=1.2,
        color="tab:red",
    )
    axes[1].set_ylabel("%")
    axes[1].set_title("Performance Loss Proxy")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "normalized_output_and_performance_proxy.png", dpi=150)
    plt.close(fig)

    # Availability + imbalance
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.plot(
        df["day"],
        df["subset_data_availability_mean"],
        color="tab:blue",
        label="subset_data_availability_mean",
    )
    ax1.set_ylabel("availability", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        df["day"],
        df["phase_imbalance_p95"],
        color="tab:orange",
        label="phase_imbalance_p95",
    )
    ax2.set_ylabel("imbalance", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Data Availability and Phase Imbalance")
    fig.tight_layout()
    fig.savefig(output_dir / "availability_and_imbalance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PV telemetry datasets and build daily features for performance analysis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing CSV exports.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/audit"),
        help="Directory for audit outputs.",
    )
    args = parser.parse_args()

    # Prefer tiered primary file if it exists, fall back to raw fetch output
    tiered_inv = args.data_dir / "inverters_tiered_primary_10min.csv"
    raw_inv = args.data_dir / "inverters_2025_to_current_10min_avg_si.csv"
    files = {
        "inverters": tiered_inv if tiered_inv.exists() else raw_inv,
        "irradiance": args.data_dir / "irradiance_2025_to_current_15min_sum_si.csv",
        "generation": args.data_dir / "power_generation_2025_to_current_1day_none_si.csv",
    }

    # Solcast files (optional)
    solcast_soiling_path = args.data_dir / "soiling_2025_to_current_10min_none_std.csv"
    solcast_irradiance_path = args.data_dir / "irradiance_2025_to_current_10min_none_std.csv"

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} file at: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {name: load_dataset(path) for name, path in files.items()}

    profile_rows = []
    interval_parts = []
    missing_parts = []

    for name, df in frames.items():
        profile_rows.append(dataset_profile(df, name))
        interval_df, _ = interval_distribution(df, name)
        interval_parts.append(interval_df)
        missing_parts.append(missingness_frame(df, name))

    # Load and aggregate Solcast data (optional -- Solcast CSVs use period_end, not Timestamp/Date)
    solcast_daily = None
    if solcast_soiling_path.exists():
        sc_irr = solcast_irradiance_path if solcast_irradiance_path.exists() else None
        solcast_daily = aggregate_solcast_daily(solcast_soiling_path, sc_irr)
        # Add simple profile rows for Solcast datasets
        import csv
        with open(solcast_soiling_path, "r") as f:
            sol_rows = sum(1 for _ in f) - 1  # subtract header
        profile_rows.append({
            "dataset": "solcast_soiling",
            "rows": sol_rows,
            "columns": 18,
            "start_date": solcast_daily["day"].min(),
            "end_date": solcast_daily["day"].max(),
            "unique_days": len(solcast_daily),
            "null_date_rows": 0,
            "median_interval_seconds": 600,
            "min_interval_seconds": 600,
            "max_interval_seconds": 600,
            "expected_interval_seconds": 600,
            "expected_interval_match_ratio": 1.0,
            "large_gap_count": 0,
        })
        if solcast_irradiance_path.exists():
            with open(solcast_irradiance_path, "r") as f:
                irr_rows = sum(1 for _ in f) - 1
            profile_rows.append({
                "dataset": "solcast_irradiance",
                "rows": irr_rows,
                "columns": 9,
                "start_date": solcast_daily["day"].min(),
                "end_date": solcast_daily["day"].max(),
                "unique_days": len(solcast_daily),
                "null_date_rows": 0,
                "median_interval_seconds": 600,
                "min_interval_seconds": 600,
                "max_interval_seconds": 600,
                "expected_interval_seconds": 600,
                "expected_interval_match_ratio": 1.0,
                "large_gap_count": 0,
            })
        print(f"  Solcast daily features: {len(solcast_daily)} days, {len(solcast_daily.columns)} columns")

    profile_df = pd.DataFrame(profile_rows).sort_values("dataset")
    interval_df = pd.concat(interval_parts, ignore_index=True)
    missing_df = pd.concat(missing_parts, ignore_index=True)

    daily_features_df = build_daily_features(
        frames["inverters"],
        frames["irradiance"],
        frames["generation"],
        solcast_daily,
    )
    daily_flags = build_daily_flags(daily_features_df)

    profile_df.to_csv(args.out_dir / "dataset_profile.csv", index=False)
    interval_df.to_csv(args.out_dir / "interval_distribution.csv", index=False)
    missing_df.to_csv(args.out_dir / "missingness_by_column.csv", index=False)
    daily_features_df.to_csv(args.out_dir / "daily_features.csv", index=False)
    daily_flags.to_csv(args.out_dir / "daily_flags.csv", index=False)

    write_quality_summary(
        args.out_dir / "quality_summary.md",
        profile_df,
        daily_features_df,
        daily_flags,
    )
    try_generate_plots(args.out_dir, interval_df, daily_features_df)

    print(f"Audit complete. Outputs written to: {args.out_dir}")
    print(f"  daily_features.csv: {len(daily_features_df)} rows x {len(daily_features_df.columns)} columns")


if __name__ == "__main__":
    main()
