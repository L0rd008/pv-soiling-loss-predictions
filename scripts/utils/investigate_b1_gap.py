"""Investigate B1 data availability gaps.

Diagnostic script to analyze per-inverter and per-block data availability,
identify shared missing windows between B1 and B2-04, and report findings.

Usage:
    python scripts/investigate_b1_gap.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_inverters(path: Path) -> pd.DataFrame:
    """Load inverter CSV and parse dates."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["day"] = df["Date"].dt.floor("D")
    return df


def per_inverter_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-inverter daily data availability."""
    power_cols = [c for c in df.columns if c.endswith("Active Power (W)")]
    if not power_cols:
        raise ValueError("No Active Power columns found.")

    records = []
    for col in power_cols:
        # Extract inverter name from column
        inv_name = col.replace(" Active Power (W)", "")
        daily = df.groupby("day")[col].agg(
            total_records="size",
            non_null="count",
        ).reset_index()
        daily["availability"] = daily["non_null"] / daily["total_records"]
        daily["inverter"] = inv_name

        # Detect block
        lo = inv_name.lower()
        if "b1" in lo or "block1" in lo or "block 1" in lo:
            daily["block"] = "B1"
        elif "b2" in lo or "block2" in lo or "block 2" in lo:
            daily["block"] = "B2"
        else:
            daily["block"] = "unknown"

        records.append(daily[["day", "inverter", "block", "total_records", "non_null", "availability"]])

    return pd.concat(records, ignore_index=True)


def find_shared_missing_windows(avail_df: pd.DataFrame) -> pd.DataFrame:
    """Identify days where B1 inverters AND B2-04 are simultaneously unavailable."""
    b1_inv = avail_df[avail_df["block"] == "B1"]
    b2_04 = avail_df[avail_df["inverter"].str.contains("B2-04|B2 04|Block2-04|Block 2-04", case=False, regex=True)]

    if b1_inv.empty:
        print("  WARNING: No B1 inverters found.")
        return pd.DataFrame()

    # Daily mean availability per block
    b1_daily = b1_inv.groupby("day")["availability"].mean().reset_index().rename(columns={"availability": "b1_mean_avail"})

    if not b2_04.empty:
        b2_04_daily = b2_04.groupby("day")["availability"].mean().reset_index().rename(columns={"availability": "b2_04_avail"})
        merged = b1_daily.merge(b2_04_daily, on="day", how="outer")
    else:
        print("  WARNING: No B2-04 inverter found in data.")
        merged = b1_daily
        merged["b2_04_avail"] = np.nan

    # Flag days where both are below 50% availability
    merged["both_low"] = (merged["b1_mean_avail"] < 0.5) & (merged["b2_04_avail"] < 0.5)
    merged["b1_only_low"] = (merged["b1_mean_avail"] < 0.5) & (merged["b2_04_avail"] >= 0.5)
    merged["b2_04_only_low"] = (merged["b1_mean_avail"] >= 0.5) & (merged["b2_04_avail"] < 0.5)

    return merged


def write_report(
    output_path: Path,
    avail_df: pd.DataFrame,
    shared_df: pd.DataFrame,
) -> None:
    """Write investigation report."""
    lines = [
        "# B1 Data Gap Investigation",
        "",
        "## Per-Inverter Summary",
    ]

    # Overall availability per inverter
    inv_summary = (
        avail_df.groupby(["inverter", "block"])["availability"]
        .agg(["mean", "median", "min", "count"])
        .reset_index()
        .sort_values(["block", "inverter"])
    )
    lines.append("")
    lines.append("| Inverter | Block | Mean Avail | Median Avail | Min Avail | Days |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in inv_summary.iterrows():
        lines.append(
            f"| {row['inverter']} | {row['block']} | {row['mean']:.3f} | "
            f"{row['median']:.3f} | {row['min']:.3f} | {int(row['count'])} |"
        )

    # Block-level summary
    lines += [
        "",
        "## Block-Level Summary",
    ]
    block_summary = avail_df.groupby("block")["availability"].agg(["mean", "median"]).reset_index()
    for _, row in block_summary.iterrows():
        lines.append(f"- **{row['block']}**: mean={row['mean']:.3f}, median={row['median']:.3f}")

    # Shared missing windows
    if not shared_df.empty:
        both_low_days = int(shared_df["both_low"].sum())
        b1_only_days = int(shared_df["b1_only_low"].sum())
        b2_04_only_days = int(shared_df["b2_04_only_low"].sum())
        total_days = len(shared_df)

        lines += [
            "",
            "## B1 vs B2-04 Missing Window Analysis",
            f"- Total days analyzed: {total_days}",
            f"- Days where BOTH B1 and B2-04 < 50% availability: **{both_low_days}**",
            f"- Days where only B1 < 50%: {b1_only_days}",
            f"- Days where only B2-04 < 50%: {b2_04_only_days}",
        ]

        if both_low_days > 0:
            shared_days = shared_df[shared_df["both_low"]].sort_values("day")
            lines.append("")
            lines.append("### Shared low-availability days (first 20):")
            lines.append("| Day | B1 Mean Avail | B2-04 Avail |")
            lines.append("|---|---|---|")
            for _, row in shared_days.head(20).iterrows():
                lines.append(
                    f"| {row['day'].strftime('%Y-%m-%d')} | {row['b1_mean_avail']:.3f} | {row['b2_04_avail']:.3f} |"
                )

        # Monthly pattern
        shared_df["month"] = shared_df["day"].dt.to_period("M")
        monthly = shared_df.groupby("month").agg(
            b1_mean=("b1_mean_avail", "mean"),
            b2_04_mean=("b2_04_avail", "mean"),
            both_low_count=("both_low", "sum"),
        ).reset_index()

        lines += [
            "",
            "### Monthly Availability Pattern",
            "| Month | B1 Mean | B2-04 Mean | Shared Low Days |",
            "|---|---|---|---|",
        ]
        for _, row in monthly.iterrows():
            b2_val = f"{row['b2_04_mean']:.3f}" if not np.isnan(row["b2_04_mean"]) else "N/A"
            lines.append(
                f"| {row['month']} | {row['b1_mean']:.3f} | {b2_val} | {int(row['both_low_count'])} |"
            )

    # Diagnostic conclusions
    lines += [
        "",
        "## Diagnostic Conclusions",
        "",
    ]

    if not shared_df.empty:
        both_low_days = int(shared_df["both_low"].sum())
        correlation = shared_df[["b1_mean_avail", "b2_04_avail"]].corr().iloc[0, 1]

        if both_low_days > 10 and correlation > 0.7:
            lines.append(
                "**FINDING**: B1 and B2-04 share highly correlated missing windows "
                f"(r={correlation:.2f}, {both_low_days} shared low days). "
                "This suggests a **shared data path** (e.g., same communication bus, "
                "gateway, or network segment) despite being in different blocks."
            )
        elif both_low_days > 5:
            lines.append(
                f"**FINDING**: Moderate overlap ({both_low_days} shared low days). "
                "Partial data path sharing is possible."
            )
        else:
            lines.append(
                "**FINDING**: Minimal overlap in missing windows. "
                "B1 and B2-04 data gaps are likely independent."
            )

    lines += [
        "",
        "## Recommended Next Steps",
        "1. Check ThingsBoard UI for B1-04 device — verify if telemetry exists for known-missing periods",
        "2. If B1 data is genuinely absent, consider fetching alternate inverters:",
        "   - B1-01 (`783e5900-45af-11ef-b4ce-d5aee9e495ad`)",
        "   - B1-05 (`7886fac0-45af-11ef-b4ce-d5aee9e495ad`)",
        "   - B1-16 (`793fc370-45af-11ef-b4ce-d5aee9e495ad`)",
        "   - B1-12 (`790b9410-45af-11ef-b4ce-d5aee9e495ad`)",
        "3. Maintain exactly 4 inverters per block (not more, not less)",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate B1 data availability gaps.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing inverter CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/b1_investigation"),
        help="Output directory for investigation results.",
    )
    args = parser.parse_args()

    inverters_path = args.data_dir / "inverters_2025_to_current_10min_avg_si.csv"
    if not inverters_path.exists():
        raise FileNotFoundError(f"Missing inverter data: {inverters_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inverter data...")
    inv = load_inverters(inverters_path)
    print(f"  {len(inv)} rows loaded")

    print("Computing per-inverter availability...")
    avail_df = per_inverter_availability(inv)
    avail_df.to_csv(args.out_dir / "per_inverter_daily_availability.csv", index=False)

    print("Analyzing shared missing windows...")
    shared_df = find_shared_missing_windows(avail_df)
    if not shared_df.empty:
        shared_df.to_csv(args.out_dir / "b1_vs_b2_04_analysis.csv", index=False)

    print("Writing report...")
    write_report(args.out_dir / "b1_investigation_report.md", avail_df, shared_df)

    print(f"Investigation complete. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
