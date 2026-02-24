"""One-shot recovery: reconstruct the original 8-inverter raw CSV from split outputs.

Merges the current overwritten primary (6 inv) and secondary (6 inv) files,
keeps only the original 8 inverters' columns, drops the 'tier' column,
and writes a clean raw file.  Then re-runs the split to produce the proper
tiered primary file separately.

Usage::

    python scripts/recover_raw_inverters.py
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

PRIMARY = DATA_DIR / "inverters_2025_to_current_10min_avg_si.csv"
SECONDARY = DATA_DIR / "inverters_secondary_10min_avg_si.csv"

# The original 8 inverters that were in the raw fetch
ORIGINAL_8 = ["B1-04", "B1-08", "B1-13", "B1-17", "B2-04", "B2-08", "B2-13", "B2-17"]

SUFFIXES = [
    " Active Power (W)",
    " Current A (A)",
    " Current B (A)",
    " Current C (A)",
]


def main() -> None:
    print("Loading current primary (overwritten) ...")
    primary = pd.read_csv(PRIMARY)
    print(f"  {len(primary)} rows, {len(primary.columns)} cols")

    print("Loading secondary ...")
    secondary = pd.read_csv(SECONDARY)
    print(f"  {len(secondary)} rows, {len(secondary.columns)} cols")

    # Merge on Timestamp + Date (outer join to keep all rows)
    merged = primary.merge(secondary, on=["Timestamp", "Date"], how="outer")
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    print(f"  Merged: {len(merged)} rows")

    # Keep only Timestamp, Date, and the original 8 inverters' columns
    keep_cols = ["Timestamp", "Date"]
    for inv in ORIGINAL_8:
        for suffix in SUFFIXES:
            col = f"{inv}{suffix}"
            if col in merged.columns:
                keep_cols.append(col)
            else:
                print(f"  WARNING: missing {col}")

    raw = merged[keep_cols].copy()

    # Back up the current overwritten file
    backup = DATA_DIR / "inverters_2025_to_current_10min_avg_si.csv.bak"
    PRIMARY.rename(backup)
    print(f"  Backed up overwritten file -> {backup}")

    # Write the reconstructed raw file
    raw.to_csv(PRIMARY, index=False)
    print(f"\n  RECOVERED: {PRIMARY}")
    print(f"    {len(raw)} rows, {len(raw.columns)} cols")
    print(f"    Inverters: {', '.join(ORIGINAL_8)}")
    print(f"\n  Now run: python scripts/split_inverter_tiers.py")


if __name__ == "__main__":
    main()
