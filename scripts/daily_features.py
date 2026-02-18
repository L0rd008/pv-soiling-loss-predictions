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
# Soiling features
# ---------------------------------------------------------------------------

def compute_soiling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add normalized output, rolling baseline, and soiling proxy to daily table.

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

    # Soiling loss proxy
    daily["soiling_loss_pct_proxy"] = (
        100.0 * (1.0 - (daily["normalized_output"] / daily["rolling_clean_baseline"]))
    ).clip(lower=0, upper=100)

    # Soiling rate (trend feature for cleaning predictions)
    daily["soiling_rate_14d_pct_per_day"] = (
        (daily["soiling_loss_pct_proxy"] - daily["soiling_loss_pct_proxy"].shift(14)) / 14.0
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

    # Penalty: missing soiling proxy
    if "soiling_loss_pct_proxy" in daily.columns:
        score -= np.where(daily["soiling_loss_pct_proxy"].isna(), 15.0, 0.0)

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
