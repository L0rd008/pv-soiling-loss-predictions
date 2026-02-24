"""Compare daily data availability for all 8 B1 inverters vs B2.

Reads the existing inverter CSV (8 inverters) and the candidate CSV (4 B1
candidates fetched separately), computes per-inverter daily availability,
and produces a ranked report.

Usage::

    python scripts/b1_availability_comparison.py

Outputs (artifacts/b1_availability/):
    b1_all_availability.csv     – per-inverter daily availability
    b1_availability_report.md   – summary report with rankings
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Inverter sets
CURRENT_B1_SET = {"B1-04", "B1-08", "B1-13", "B1-17"}
CANDIDATE_B1_SET = {"B1-01", "B1-05", "B1-12", "B1-16"}

# Expected 10-min records per day (24h × 6 = 144)
EXPECTED_RECORDS_PER_DAY = 144


def load_and_compute_availability(csv_path: Path) -> pd.DataFrame:
    """Load an inverter CSV and compute per-inverter daily availability."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["day"] = df["Date"].dt.floor("D")

    power_cols = [c for c in df.columns if c.endswith("Active Power (W)")]
    if not power_cols:
        raise ValueError(f"No Active Power columns in {csv_path}")

    records = []
    for col in power_cols:
        inv_name = col.replace(" Active Power (W)", "")
        daily = df.groupby("day")[col].agg(
            total_records="size",
            valid_records="count",
        ).reset_index()
        daily["availability"] = daily["valid_records"] / EXPECTED_RECORDS_PER_DAY
        daily["availability"] = daily["availability"].clip(upper=1.0)
        daily["inverter"] = inv_name

        # Detect block
        lo = inv_name.lower()
        if "b1" in lo:
            daily["block"] = "B1"
        elif "b2" in lo:
            daily["block"] = "B2"
        else:
            daily["block"] = "unknown"

        records.append(daily[["day", "inverter", "block", "total_records",
                              "valid_records", "availability"]])

    return pd.concat(records, ignore_index=True)


def label_set(inv_name: str) -> str:
    if inv_name in CURRENT_B1_SET:
        return "current-B1"
    elif inv_name in CANDIDATE_B1_SET:
        return "candidate-B1"
    else:
        return "B2-baseline"


def write_report(output_path: Path, avail_df: pd.DataFrame) -> None:
    """Write availability comparison report."""
    avail_df = avail_df.copy()
    avail_df["set"] = avail_df["inverter"].apply(label_set)

    lines = [
        "# B1 Inverter Availability Comparison",
        "",
        "Comparison of all 8 B1 inverters plus 4 B2 inverters (baseline).",
        "",
        "- **Current B1 set**: B1-04, B1-08, B1-13, B1-17",
        "- **Candidate B1 alternates**: B1-01, B1-05, B1-12, B1-16",
        "- **B2 baseline**: B2-04, B2-08, B2-13, B2-17",
        "",
    ]

    # --- Per-inverter summary ranked by mean availability ---
    inv_summary = (
        avail_df.groupby(["inverter", "set"])["availability"]
        .agg(["mean", "median", "min", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    lines += [
        "## Per-Inverter Availability (ranked by mean)",
        "",
        "| Rank | Inverter | Set | Mean | Median | Min | Days |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, (_, row) in enumerate(inv_summary.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['inverter']} | {row['set']} | "
            f"{row['mean']:.3f} | {row['median']:.3f} | "
            f"{row['min']:.3f} | {int(row['count'])} |"
        )

    # --- Group comparison ---
    set_summary = (
        avail_df.groupby("set")["availability"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    lines += [
        "",
        "## Group Comparison",
        "",
        "| Group | Mean | Median | Day-Records |",
        "|---|---|---|---|",
    ]
    for _, row in set_summary.iterrows():
        lines.append(
            f"| {row['set']} | {row['mean']:.3f} | "
            f"{row['median']:.3f} | {int(row['count'])} |"
        )

    # --- Monthly breakdown ---
    avail_df["month"] = avail_df["day"].dt.to_period("M")
    monthly = (
        avail_df.groupby(["inverter", "month"])["availability"]
        .mean()
        .unstack("inverter")
        .sort_index()
    )

    inv_cols = sorted(monthly.columns)
    lines += [
        "",
        "## Monthly Availability by Inverter",
        "",
    ]
    header = "| Month | " + " | ".join(inv_cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(inv_cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for month, row in monthly.iterrows():
        vals = " | ".join(
            f"{row[c]:.2f}" if not np.isnan(row[c]) else "—" for c in inv_cols
        )
        lines.append(f"| {month} | {vals} |")

    # --- Low-availability days (<50%) ---
    low_days = (
        avail_df[avail_df["availability"] < 0.50]
        .groupby("inverter")
        .size()
        .reset_index(name="low_days")
        .sort_values("low_days", ascending=False)
    )
    lines += [
        "",
        "## Low-Availability Days (< 50% records)",
        "",
        "| Inverter | Low Days |",
        "|---|---|",
    ]
    for _, row in low_days.iterrows():
        lines.append(f"| {row['inverter']} | {int(row['low_days'])} |")

    # No low days inverters
    all_invs = set(avail_df["inverter"].unique())
    low_invs = set(low_days["inverter"])
    perfect = all_invs - low_invs
    if perfect:
        lines.append(f"\nInverters with zero low-availability days: {', '.join(sorted(perfect))}")

    # --- Best-4 recommendation ---
    b1_ranked = inv_summary[inv_summary["set"].isin(["current-B1", "candidate-B1"])].copy()
    best_4 = b1_ranked.head(4)["inverter"].tolist()

    lines += [
        "",
        "## Recommendation: Best 4 B1 Inverters by Availability",
        "",
        f"Top 4: **{', '.join(best_4)}**",
        "",
    ]

    if set(best_4) == CURRENT_B1_SET:
        lines.append("**Verdict**: Current B1 set is already optimal. No swap needed.")
    elif set(best_4) == CANDIDATE_B1_SET:
        lines.append("**Verdict**: Full swap — all candidates have better availability.")
    else:
        swaps_out = CURRENT_B1_SET - set(best_4)
        swaps_in = set(best_4) - CURRENT_B1_SET
        lines.append("**Verdict**: Partial swap recommended.")
        lines.append(f"- Remove: {', '.join(sorted(swaps_out))}")
        lines.append(f"- Add: {', '.join(sorted(swaps_in))}")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare B1 inverter availability across current and candidate sets."
    )
    parser.add_argument(
        "--existing-csv",
        type=Path,
        default=Path("data/inverters_2025_to_current_10min_avg_si.csv"),
        help="Existing inverter CSV (current 4 B1 + 4 B2).",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("data/b1_candidates/inverters_2025_to_current_10min_avg_si.csv"),
        help="Candidate B1 inverter CSV (B1-01/05/12/16).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/b1_availability"),
        help="Output directory for results.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading existing inverter data...")
    existing = load_and_compute_availability(args.existing_csv)
    print(f"  {len(existing)} day-records from {existing['inverter'].nunique()} inverters")

    print("Loading candidate B1 data...")
    candidates = load_and_compute_availability(args.candidate_csv)
    print(f"  {len(candidates)} day-records from {candidates['inverter'].nunique()} inverters")

    # Combine
    all_avail = pd.concat([existing, candidates], ignore_index=True)
    all_avail = all_avail.sort_values(["inverter", "day"]).reset_index(drop=True)

    # Save
    all_avail.to_csv(args.out_dir / "b1_all_availability.csv", index=False)
    print(f"  Combined: {len(all_avail)} day-records, {all_avail['inverter'].nunique()} inverters")

    # Report
    write_report(args.out_dir / "b1_availability_report.md", all_avail)
    print(f"\nReport: {args.out_dir / 'b1_availability_report.md'}")


if __name__ == "__main__":
    main()
