"""Split inverter data into tiered files for the soiling analysis pipeline.

Reads the existing inverter CSV and the candidate B1 CSV, then produces:

1. Primary file (Tier 1+2): B2-08, B2-13, B2-17, B1-08, B1-01, B1-13
   → data/inverters_tiered_primary_10min.csv
   Adds a 'tier' column: 1 for B2 training set, 2 for B1 validation set.

2. Secondary file: B1-04, B1-05, B1-12, B1-16, B1-17, B2-04
   → data/inverters_secondary_10min_avg_si.csv

Usage::

    python scripts/split_inverter_tiers.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
EXISTING_CSV = DATA_DIR / "inverters_2025_to_current_10min_avg_si.csv"
CANDIDATES_CSV = DATA_DIR / "b1_candidates" / "inverters_2025_to_current_10min_avg_si.csv"

# Tier 1 (training): best B2 inverters
TIER1_INVERTERS = ["B2-08", "B2-13", "B2-17"]

# Tier 2 (validation): best B1 inverters
TIER2_INVERTERS = ["B1-08", "B1-01", "B1-13"]

# Secondary (reserve / augmentation)
SECONDARY_FROM_EXISTING = ["B1-04", "B1-17", "B2-04"]
SECONDARY_FROM_CANDIDATES = ["B1-05", "B1-12", "B1-16"]

PRIMARY_INVERTERS = TIER1_INVERTERS + TIER2_INVERTERS

# Columns per inverter
SUFFIXES = [
    " Active Power (W)",
    " Current A (A)",
    " Current B (A)",
    " Current C (A)",
]


def inv_columns(inv_name: str) -> list[str]:
    return [f"{inv_name}{s}" for s in SUFFIXES]


def main() -> None:
    print("Loading existing inverter CSV ...")
    existing = pd.read_csv(EXISTING_CSV)
    print(f"  {len(existing)} rows, {len(existing.columns)} cols")

    print("Loading candidate B1 CSV ...")
    candidates = pd.read_csv(CANDIDATES_CSV)
    print(f"  {len(candidates)} rows, {len(candidates.columns)} cols")

    # Merge on Timestamp (epoch ms) for a full outer join
    merged = existing.merge(candidates, on=["Timestamp", "Date"], how="outer")
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    print(f"  Merged: {len(merged)} rows")

    # ---- Build primary file ----
    primary_cols = ["Timestamp", "Date"]
    for inv in PRIMARY_INVERTERS:
        cols = inv_columns(inv)
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            print(f"  WARNING: missing columns for {inv}: {missing}")
        primary_cols.extend([c for c in cols if c in merged.columns])

    primary = merged[primary_cols].copy()

    # Add tier column
    tier1_power_cols = [f"{inv} Active Power (W)" for inv in TIER1_INVERTERS]
    tier2_power_cols = [f"{inv} Active Power (W)" for inv in TIER2_INVERTERS]

    # Tier flag per row: 1 if any Tier-1 data present, 2 if only Tier-2, NaN if neither
    has_t1 = primary[tier1_power_cols].notna().any(axis=1)
    has_t2 = primary[tier2_power_cols].notna().any(axis=1)
    primary.insert(2, "tier", 0)
    primary.loc[has_t1, "tier"] = 1
    primary.loc[~has_t1 & has_t2, "tier"] = 2

    out_primary = DATA_DIR / "inverters_tiered_primary_10min.csv"
    primary.to_csv(out_primary, index=False)
    print(f"\n  PRIMARY saved: {out_primary}")
    print(f"    {len(primary)} rows, {len(primary.columns)} cols")
    print(f"    Inverters: {', '.join(PRIMARY_INVERTERS)}")
    print(f"    Tier 1 (training) rows: {(primary['tier'] == 1).sum()}")
    print(f"    Tier 2 (validation) rows: {(primary['tier'] == 2).sum()}")

    # ---- Build secondary file ----
    secondary_inverters = SECONDARY_FROM_EXISTING + SECONDARY_FROM_CANDIDATES
    secondary_cols = ["Timestamp", "Date"]
    for inv in secondary_inverters:
        cols = inv_columns(inv)
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            print(f"  WARNING: missing columns for {inv}: {missing}")
        secondary_cols.extend([c for c in cols if c in merged.columns])

    secondary = merged[secondary_cols].copy()

    out_secondary = DATA_DIR / "inverters_secondary_10min_avg_si.csv"
    secondary.to_csv(out_secondary, index=False)
    print(f"\n  SECONDARY saved: {out_secondary}")
    print(f"    {len(secondary)} rows, {len(secondary.columns)} cols")
    print(f"    Inverters: {', '.join(secondary_inverters)}")

    # Summary
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"  Primary (Tier 1+2) : {out_primary}")
    print(f"    Tier 1 (training): {', '.join(TIER1_INVERTERS)}")
    print(f"    Tier 2 (validation): {', '.join(TIER2_INVERTERS)}")
    print(f"  Secondary (reserve): {out_secondary}")
    print(f"    Reserve: {', '.join(secondary_inverters)}")


if __name__ == "__main__":
    main()
