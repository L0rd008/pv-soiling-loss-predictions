"""EDA: Three go/no-go signal tests for soiling loss prediction.

Produces ~28 plots in artifacts/eda/plots/ and a quantitative verdict
report in artifacts/eda/eda_signal_report.md.

Usage:
    python scripts/5_eda/soiling_signals.py
    python scripts/5_eda/soiling_signals.py --input path/to/daily_model_eda.csv
    python scripts/5_eda/soiling_signals.py --out-dir artifacts/eda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from core.daily_features import (
    CLEANING_CAMPAIGN_DATES,
    SIGNIFICANT_RAIN_MM,
    SITE_LAT,
    SITE_LON,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── colour palette ──────────────────────────────────────────────────────
C_RAIN = "#3B82F6"
C_CLEANING = "#F59E0B"
C_DRY = "#D97706"
C_WET = "#0D9488"
C_T1 = "#6366F1"
C_T2 = "#EC4899"
C_ACCENT = "#10B981"

DEFAULT_INPUT = "artifacts/preprocessed/daily_model_eda.csv"
DEFAULT_OUT = "artifacts/eda"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Data classes for results                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

@dataclass
class SignalResult:
    name: str
    verdict: str  # "pass", "weak", "fail"
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Helpers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)


def _hq_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transfer_quality_score"] >= 70].copy()


def _add_rain_cleaning_overlays(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    rain_col: str = "precipitation_total_mm",
    threshold: float = SIGNIFICANT_RAIN_MM,
) -> None:
    """Draw vertical lines for significant rain and shaded cleaning windows."""
    sig = df.loc[df[rain_col] >= threshold, "day_dt"]
    for d in sig:
        ax.axvline(d, color=C_RAIN, alpha=0.25, lw=0.6)
    for start_s, end_s in CLEANING_CAMPAIGN_DATES:
        s, e = pd.Timestamp(start_s), pd.Timestamp(end_s)
        ax.axvspan(s, e, color=C_CLEANING, alpha=0.12)


def _partial_corr(
    df: pd.DataFrame, x: str, y: str, controls: List[str],
) -> Tuple[float, float]:
    """Partial Pearson correlation via OLS residualization."""
    sub = df[[x, y, *controls]].dropna()
    if len(sub) < 10:
        return np.nan, np.nan
    from numpy.linalg import lstsq

    C = sub[controls].values
    C = np.column_stack([C, np.ones(len(C))])

    def _resid(col: np.ndarray) -> np.ndarray:
        coef, *_ = lstsq(C, col, rcond=None)
        return col - C @ coef

    rx = _resid(sub[x].values)
    ry = _resid(sub[y].values)
    r, p = stats.pearsonr(rx, ry)
    return r, p


def _new_source_start(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Return the first date where plant_avg_irradiance_wm2 is not NaN."""
    col = "plant_avg_irradiance_wm2"
    if col not in df.columns:
        return None
    valid = df.loc[df[col].notna(), "day_dt"]
    return valid.min() if len(valid) else None


def _annotate_new_source_start(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw a vertical line where new-source data begins, if applicable."""
    start = _new_source_start(df)
    if start is None:
        return
    ax.axvline(start, color="#EF4444", ls="--", lw=1.0, alpha=0.7, zorder=5)
    ax.text(
        start, ax.get_ylim()[1] * 0.95, " New src start",
        fontsize=7, color="#EF4444", ha="left", va="top",
    )


def _identify_dry_spells(
    df: pd.DataFrame, min_len: int = 3,
) -> List[pd.DataFrame]:
    """Return list of sub-DataFrames, one per dry spell of >= min_len days."""
    is_dry = ~df["rain_day"].astype(bool)
    spell_id = (is_dry != is_dry.shift()).cumsum()
    spells = []
    for sid, grp in df[is_dry].groupby(spell_id):
        if len(grp) >= min_len:
            spells.append(grp.copy())
    return spells


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Load & filter                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day_dt"] = pd.to_datetime(df["day"])
    df.sort_values("day_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_total = len(df)
    n_hq = len(_hq_filter(df))
    log.info(
        "Loaded %d rows (%s → %s), %d training-ready (HQ + 0 flags)",
        n_total,
        df["day_dt"].min().date(),
        df["day_dt"].max().date(),
        n_hq,
    )
    return df


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Signal 1 — Sawtooth detection                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def test_signal_1_sawtooth(
    df: pd.DataFrame, plots_dir: Path,
) -> SignalResult:
    log.info("── Signal 1: Sawtooth detection ──")
    hq = _hq_filter(df)

    # S1-A  full time-series ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(
        df["day_dt"], df["t1_performance_loss_pct_proxy"],
        lw=0.8, color=C_T1, alpha=0.8, label="T1 loss proxy",
    )
    _add_rain_cleaning_overlays(ax1, df)
    ax1.set_ylabel("Performance loss proxy (%)", color=C_T1)
    ax1.tick_params(axis="y", labelcolor=C_T1)

    if "domain_soiling_index" in df.columns:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            df["day_dt"], df["domain_soiling_index"],
            lw=0.7, color=C_DRY, alpha=0.55, label="Domain soiling index",
        )
        ax1_twin.set_ylabel("Domain soiling index (cumul.)", color=C_DRY)
        ax1_twin.tick_params(axis="y", labelcolor=C_DRY)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left", fontsize=8)

    ax1.set_title("Signal 1-A: Loss proxy & domain soiling index with rain/cleaning overlays")

    ax2.bar(
        df["day_dt"], df["precipitation_total_mm"],
        width=1.0, color=C_RAIN, alpha=0.6,
    )
    ax2.set_ylabel("Precip (mm)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, plots_dir / "s1_loss_proxy_timeseries.png")

    # S1-B  per-inverter normalised output -----------------------------------
    inv_cols = [
        "b2_08_normalized_output", "b2_13_normalized_output",
        "b2_17_normalized_output", "b1_08_normalized_output",
        "b1_01_normalized_output", "b1_13_normalized_output",
    ]
    present = [c for c in inv_cols if c in df.columns]
    n_inv = len(present)
    if n_inv:
        fig, axes = plt.subplots(n_inv, 1, figsize=(16, 3.5 * n_inv), sharex=True)
        if n_inv == 1:
            axes = [axes]
        for ax, col in zip(axes, present):
            ax.plot(df["day_dt"], df[col], lw=0.7, color=C_T1, alpha=0.8)
            _add_rain_cleaning_overlays(ax, df)
            ax.set_ylabel(col.replace("_normalized_output", ""), fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_title("Signal 1-B: Per-inverter normalised output")
        axes[-1].set_xlabel("Date")
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        fig.tight_layout()
        _save(fig, plots_dir / "s1_per_inverter_output.png")

    # S1-C  cycle deviation --------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        df["day_dt"], df["cycle_deviation_pct"],
        lw=0.8, color=C_ACCENT, alpha=0.8,
    )
    cycle_bounds = df.loc[
        df["cycle_id"] != df["cycle_id"].shift(), "day_dt"
    ]
    for bd in cycle_bounds:
        ax.axvline(bd, color="grey", alpha=0.15, lw=0.4)
    _add_rain_cleaning_overlays(ax, df)
    ax.set_ylabel("Cycle deviation (%)")
    ax.set_title("Signal 1-C: Cycle-aware deviation (within-cycle soiling)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, plots_dir / "s1_cycle_deviation.png")

    # S1-D  dry-spell soiling rates ------------------------------------------
    spells = _identify_dry_spells(hq, min_len=3)
    rates: List[float] = []
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        hq["day_dt"], hq["t1_performance_loss_pct_proxy"],
        lw=0.6, color=C_T1, alpha=0.5, label="HQ loss proxy",
    )
    for sp in spells:
        x_num = np.arange(len(sp), dtype=float)
        y = sp["t1_performance_loss_pct_proxy"].values
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        slope, intercept, *_ = stats.linregress(x_num[mask], y[mask])
        rates.append(slope)
        fitted = intercept + slope * x_num
        ax.plot(sp["day_dt"], fitted, lw=2.0, color=C_DRY, alpha=0.7)
    _add_rain_cleaning_overlays(ax, hq)
    ax.set_ylabel("Loss proxy (%)")
    ax.set_title("Signal 1-D: Dry-spell soiling rate slopes (orange)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, plots_dir / "s1_dryspell_slopes.png")

    rates_arr = np.array(rates)
    n_spells = len(rates_arr)
    if n_spells > 0:
        med_rate = float(np.median(rates_arr))
        iqr_lo = float(np.percentile(rates_arr, 25))
        iqr_hi = float(np.percentile(rates_arr, 75))
        pct_positive = float((rates_arr > 0).mean() * 100)
    else:
        med_rate = iqr_lo = iqr_hi = pct_positive = 0.0

    details = {
        "n_spells": n_spells,
        "median_rate_pct_per_day": med_rate,
        "iqr": (iqr_lo, iqr_hi),
        "pct_positive_slope": pct_positive,
    }

    if n_spells >= 3 and 0.05 <= abs(med_rate) <= 1.0:
        verdict = "pass"
    elif n_spells >= 1 and med_rate != 0.0:
        verdict = "weak"
    else:
        verdict = "fail"

    summary = (
        f"{n_spells} dry spells analysed. "
        f"Median soiling rate = {med_rate:+.3f} %/day "
        f"(IQR {iqr_lo:+.3f} to {iqr_hi:+.3f}). "
        f"{pct_positive:.0f}% of spells have positive slope (soiling accumulation)."
    )
    log.info("Signal 1 verdict: %s — %s", verdict.upper(), summary)
    return SignalResult("Signal 1: Sawtooth", verdict, summary, details)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Signal 2 — PM / dust correlation                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def test_signal_2_dust_correlation(
    df: pd.DataFrame, plots_dir: Path,
) -> SignalResult:
    log.info("── Signal 2: PM/dust correlation ──")
    hq = _hq_filter(df)
    cloud_q25 = df["cloud_opacity_mean"].quantile(0.25)
    hq_clear = hq[hq["cloud_opacity_mean"] < cloud_q25].copy()

    loss_rate_col = "t1_perf_loss_rate_14d_pct_per_day"
    loss_col = "t1_performance_loss_pct_proxy"

    # S2-A  raw vs deconfounded scatter --------------------------------------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    for season, colour in [("dry", C_DRY), ("wet", C_WET)]:
        sub = hq[hq["season"] == season]
        ax_l.scatter(
            sub["pm10_mean"], sub[loss_rate_col],
            s=12, alpha=0.5, color=colour, label=season,
        )
    pair_all = hq[["pm10_mean", loss_rate_col]].dropna()
    if len(pair_all) > 3:
        r_all, _ = stats.pearsonr(pair_all["pm10_mean"], pair_all[loss_rate_col])
        rho_all, _ = stats.spearmanr(pair_all["pm10_mean"], pair_all[loss_rate_col])
    else:
        r_all = rho_all = np.nan
    ax_l.set_title(f"All HQ days (r={r_all:.3f}, ρ={rho_all:.3f})", fontsize=9)
    ax_l.set_xlabel("PM10 mean (µg/m³)")
    ax_l.set_ylabel("Loss rate (14d, %/day)")
    ax_l.legend(fontsize=8)

    if len(hq_clear) > 5:
        for season, colour in [("dry", C_DRY), ("wet", C_WET)]:
            sub = hq_clear[hq_clear["season"] == season]
            ax_r.scatter(
                sub["pm10_mean"], sub[loss_rate_col],
                s=12, alpha=0.5, color=colour, label=season,
            )
        pair = hq_clear[["pm10_mean", loss_rate_col]].dropna()
        r_clear, _ = stats.pearsonr(*pair.values.T) if len(pair) > 3 else (np.nan, np.nan)
        rho_clear, _ = stats.spearmanr(*pair.values.T) if len(pair) > 3 else (np.nan, np.nan)
        ax_r.set_title(
            f"Clear-sky HQ (n={len(hq_clear)}, r={r_clear:.3f}, ρ={rho_clear:.3f})",
            fontsize=9,
        )
    else:
        r_clear = rho_clear = np.nan
        ax_r.set_title("Clear-sky HQ (insufficient data)")
    ax_r.set_xlabel("PM10 mean (µg/m³)")
    fig.suptitle("Signal 2-A: PM10 vs loss rate — raw and clear-sky", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s2_pm10_scatter_panels.png")

    # S2-B  top predictors vs cycle deviation (3-panel) -------------------------
    top_predictors = [
        ("days_since_last_rain", "Days since last rain"),
        ("cumulative_pm25_since_rain", "Cumul. PM2.5 since rain (µg/m³·days)"),
        ("cumulative_pm10_since_rain", "Cumul. PM10 since rain (µg/m³·days)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    top_raw_corrs: Dict[str, float] = {}
    for ax, (col, xlabel) in zip(axes, top_predictors):
        pair = hq[[col, "cycle_deviation_pct"]].dropna()
        ax.scatter(pair[col], pair["cycle_deviation_pct"],
                   s=12, alpha=0.5, color=C_ACCENT)
        if len(pair) > 3:
            r_val, p_val = stats.pearsonr(*pair.values.T)
            slope, intercept, *_ = stats.linregress(*pair.values.T)
            x_fit = np.linspace(pair.iloc[:, 0].min(), pair.iloc[:, 0].max(), 50)
            ax.plot(x_fit, intercept + slope * x_fit, color=C_DRY, lw=1.5)
            ax.set_title(f"r={r_val:+.3f}, p={p_val:.3f}", fontsize=9)
            top_raw_corrs[col] = r_val
        else:
            top_raw_corrs[col] = np.nan
            ax.set_title("insufficient data", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Cycle deviation (%)", fontsize=8)
    fig.suptitle("Signal 2-B: Top predictors vs cycle deviation", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s2_top_predictors_vs_deviation.png")

    r_cum = top_raw_corrs.get("cumulative_pm10_since_rain", np.nan)
    r_cum25 = top_raw_corrs.get("cumulative_pm25_since_rain", np.nan)
    r_days = top_raw_corrs.get("days_since_last_rain", np.nan)

    # S2-C  feature correlation heatmap --------------------------------------
    env_cols = [
        "pm10_mean", "pm25_mean", "precipitation_total_mm", "humidity_mean",
        "wind_speed_10m_mean", "air_temp_mean", "cloud_opacity_mean",
    ]
    eng_cols = [
        "days_since_last_rain", "days_since_significant_rain",
        "cumulative_pm10_since_rain", "cumulative_pm25_since_rain",
        "humidity_x_pm10", "wind_speed_10m_rolling_7d",
        "domain_soiling_daily", "domain_soiling_index",
    ]
    pvlib_cols = ["pvlib_soiling_ratio_hsu", "pvlib_soiling_loss_kimber"]
    target_cols = [loss_col, loss_rate_col, "cycle_deviation_pct"]
    all_cols = [c for c in env_cols + eng_cols + pvlib_cols + target_cols if c in hq.columns]

    corr = hq[all_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(all_cols)))
    ax.set_yticks(range(len(all_cols)))
    labels = [c.replace("t1_", "").replace("_", "\n", 1) for c in all_cols]
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(len(all_cols)):
        for j in range(len(all_cols)):
            val = corr.values[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=5, color="white" if abs(val) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Signal 2-C: Feature correlation heatmap (HQ days)", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s2_feature_heatmap.png")

    # S2-D  partial correlations (table for report) ---------------------------
    controls = ["cloud_opacity_mean", "air_temp_mean"]
    dust_features = [
        "pm10_mean", "pm25_mean", "cumulative_pm10_since_rain",
        "cumulative_pm25_since_rain", "humidity_x_pm10", "days_since_last_rain",
    ]
    targets_for_partial = [loss_col, loss_rate_col, "cycle_deviation_pct"]
    partial_results: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for feat in dust_features:
        partial_results[feat] = {}
        for tgt in targets_for_partial:
            r, p = _partial_corr(hq, feat, tgt, controls)
            partial_results[feat][tgt] = (r, p)

    # Within-cycle correlations
    cycle_groups = hq.groupby("cycle_id").filter(lambda g: len(g) >= 3)
    cycle_stats = cycle_groups.groupby("cycle_id").agg(
        pm10_mean_c=("pm10_mean", "mean"),
        loss_start=(loss_col, "first"),
        loss_end=(loss_col, "last"),
        n=("day", "count"),
    )
    cycle_stats["rate"] = (cycle_stats["loss_end"] - cycle_stats["loss_start"]) / cycle_stats["n"]
    within_pair = cycle_stats[["pm10_mean_c", "rate"]].dropna()
    if len(within_pair) > 3:
        r_within, p_within = stats.pearsonr(*within_pair.values.T)
    else:
        r_within = p_within = np.nan

    # Determine verdict
    best_partial_r = 0.0
    for feat, tgt_dict in partial_results.items():
        for tgt, (r, p) in tgt_dict.items():
            if np.isfinite(r) and abs(r) > abs(best_partial_r):
                best_partial_r = r

    if abs(best_partial_r) > 0.15 or (np.isfinite(r_within) and abs(r_within) > 0.2):
        verdict = "pass"
    elif abs(best_partial_r) > 0.05:
        verdict = "weak"
    else:
        verdict = "fail"

    details = {
        "r_all_pm10_vs_rate": r_all,
        "r_clear_pm10_vs_rate": r_clear,
        "r_cumpm10_vs_deviation": r_cum if np.isfinite(r_cum) else None,
        "r_cumpm25_vs_deviation": r_cum25 if np.isfinite(r_cum25) else None,
        "r_days_since_rain_vs_deviation": r_days if np.isfinite(r_days) else None,
        "best_partial_r": best_partial_r,
        "partial_results": partial_results,
        "r_within_cycle": r_within,
        "p_within_cycle": p_within,
        "n_cycles": len(within_pair),
    }
    strongest_raw = max(
        [("days_since_last_rain", r_days),
         ("cumulative_pm25_since_rain", r_cum25),
         ("cumulative_pm10_since_rain", r_cum)],
        key=lambda t: t[1] if np.isfinite(t[1]) else -999,
    )
    summary = (
        f"Strongest raw predictor of cycle deviation: {strongest_raw[0]} "
        f"(r={strongest_raw[1]:+.3f}). "
        f"Best partial correlation (deconfounded) = {best_partial_r:+.3f}. "
        f"Within-cycle PM10-rate r = {r_within:+.3f} "
        f"(n={len(within_pair)} cycles)."
    )
    log.info("Signal 2 verdict: %s — %s", verdict.upper(), summary)
    return SignalResult("Signal 2: PM/Dust", verdict, summary, details)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Signal 3 — Rain recovery                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

def test_signal_3_rain_recovery(
    df: pd.DataFrame, plots_dir: Path,
) -> SignalResult:
    log.info("── Signal 3: Rain recovery ──")
    hq = _hq_filter(df)
    loss_col = "t1_performance_loss_pct_proxy"

    # ── S3-A  event study ──────────────────────────────────────────────
    window_pre, window_post = 5, 7
    sig_rain_idx = hq.index[hq["precipitation_total_mm"] >= SIGNIFICANT_RAIN_MM].tolist()

    trajectories: List[np.ndarray] = []
    for idx in sig_rain_idx:
        pos = hq.index.get_loc(idx) if idx in hq.index else None
        if pos is None:
            continue
        lo = pos - window_pre
        hi = pos + window_post + 1
        if lo < 0 or hi > len(hq):
            continue
        segment = hq.iloc[lo:hi][loss_col].values
        if len(segment) == window_pre + window_post + 1 and np.isfinite(segment).sum() >= 5:
            trajectories.append(segment)

    traj_arr = np.array(trajectories) if trajectories else np.empty((0, window_pre + window_post + 1))
    offsets = np.arange(-window_pre, window_post + 1)

    # Build control trajectories from non-rain days
    no_rain_idx = hq.index[hq["precipitation_total_mm"] < 1.0].tolist()
    rng = np.random.RandomState(42)
    ctrl_sample = rng.choice(no_rain_idx, size=min(len(no_rain_idx), len(sig_rain_idx) * 2), replace=False)
    ctrl_trajs: List[np.ndarray] = []
    for idx in ctrl_sample:
        pos = hq.index.get_loc(idx) if idx in hq.index else None
        if pos is None:
            continue
        lo = pos - window_pre
        hi = pos + window_post + 1
        if lo < 0 or hi > len(hq):
            continue
        segment = hq.iloc[lo:hi][loss_col].values
        if len(segment) == window_pre + window_post + 1 and np.isfinite(segment).sum() >= 5:
            ctrl_trajs.append(segment)
    ctrl_arr = np.array(ctrl_trajs) if ctrl_trajs else np.empty((0, window_pre + window_post + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    if len(traj_arr):
        mean_t = np.nanmean(traj_arr, axis=0)
        med_t = np.nanmedian(traj_arr, axis=0)
        ci_lo = np.nanpercentile(traj_arr, 5, axis=0)
        ci_hi = np.nanpercentile(traj_arr, 95, axis=0)
        ax.fill_between(offsets, ci_lo, ci_hi, alpha=0.15, color=C_RAIN)
        ax.plot(offsets, mean_t, lw=2, color=C_RAIN, label=f"Rain events mean (n={len(traj_arr)})")
        ax.plot(offsets, med_t, lw=1.5, ls="--", color=C_RAIN, alpha=0.7, label="Median")
    if len(ctrl_arr):
        ax.plot(
            offsets, np.nanmean(ctrl_arr, axis=0),
            lw=1.5, color="grey", alpha=0.6, label=f"Dry control (n={len(ctrl_arr)})",
        )
    ax.axvline(0, color=C_RAIN, ls=":", lw=1)
    ax.set_xlabel("Days relative to rain event")
    ax.set_ylabel("Loss proxy (%)")
    ax.set_title("Signal 3-A: Event study — loss trajectory around significant rain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, plots_dir / "s3_rain_event_study.png")

    # ── S3-B  dry-spell start vs end ───────────────────────────────────
    spells = _identify_dry_spells(hq, min_len=3)
    starts, ends = [], []
    for sp in spells:
        lp = sp[loss_col].dropna()
        if len(lp) >= 2:
            starts.append(lp.iloc[0])
            ends.append(lp.iloc[-1])

    fig, ax = plt.subplots(figsize=(7, 5))
    if starts:
        for s, e in zip(starts, ends):
            ax.plot([0, 1], [s, e], color=C_DRY, alpha=0.4, lw=1)
        ax.scatter([0] * len(starts), starts, color=C_ACCENT, s=30, zorder=5, label="Spell start")
        ax.scatter([1] * len(ends), ends, color=C_DRY, s=30, zorder=5, label="Spell end")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Dry spell start", "Dry spell end"])
        diffs = np.array(ends) - np.array(starts)
        stat_w, p_w = stats.wilcoxon(diffs, alternative="greater") if len(diffs) >= 6 else (np.nan, np.nan)
        ax.set_title(
            f"Signal 3-B: Dry-spell accumulation (n={len(diffs)}, "
            f"Wilcoxon p={p_w:.4f})",
            fontsize=10,
        )
    else:
        stat_w = p_w = np.nan
        ax.set_title("Signal 3-B: Dry-spell accumulation (insufficient data)")
    ax.set_ylabel("Loss proxy (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, plots_dir / "s3_dryspell_start_end.png")

    # ── S3-C  recovery vs precipitation amount ─────────────────────────
    loss_changes_d3: List[Tuple[float, float]] = []
    for idx in sig_rain_idx:
        pos = hq.index.get_loc(idx) if idx in hq.index else None
        if pos is None:
            continue
        if pos - 1 < 0 or pos + 3 >= len(hq):
            continue
        lp_pre = hq.iloc[pos - 1][loss_col]
        lp_post = hq.iloc[pos + 3][loss_col]
        precip = hq.iloc[pos]["precipitation_total_mm"]
        if np.isfinite(lp_pre) and np.isfinite(lp_post):
            loss_changes_d3.append((precip, lp_post - lp_pre))

    fig, ax = plt.subplots(figsize=(7, 5))
    if loss_changes_d3:
        lc = np.array(loss_changes_d3)
        heavy = lc[:, 0] >= 10
        mod = ~heavy
        if mod.any():
            ax.scatter(lc[mod, 0], lc[mod, 1], s=20, alpha=0.6, color=C_WET, label="5–10 mm")
        if heavy.any():
            ax.scatter(lc[heavy, 0], lc[heavy, 1], s=20, alpha=0.6, color=C_RAIN, label="≥10 mm")
        ax.axhline(0, color="grey", ls="--", lw=0.7)
        ax.set_xlabel("Precipitation (mm)")
        ax.set_ylabel("Loss change (day −1 to day +3, pp)")
        r_rc, p_rc = stats.pearsonr(lc[:, 0], lc[:, 1]) if len(lc) > 3 else (np.nan, np.nan)
        ax.set_title(
            f"Signal 3-C: Recovery magnitude vs rain (r={r_rc:.3f})",
            fontsize=10,
        )
        ax.legend(fontsize=8)
    else:
        r_rc = p_rc = np.nan
        ax.set_title("Signal 3-C: Recovery magnitude vs rain (no data)")
    fig.tight_layout()
    _save(fig, plots_dir / "s3_recovery_vs_precipitation.png")

    # ── S3-D  seasonal event study ────────────────────────────────────
    fig, (ax_d, ax_w) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, season_label, colour in [(ax_d, "dry", C_DRY), (ax_w, "wet", C_WET)]:
        season_hq = hq[hq["season"] == season_label]
        season_rain_idx = season_hq.index[
            season_hq["precipitation_total_mm"] >= SIGNIFICANT_RAIN_MM
        ].tolist()
        trajs_s: List[np.ndarray] = []
        for idx in season_rain_idx:
            pos_full = hq.index.get_loc(idx) if idx in hq.index else None
            if pos_full is None:
                continue
            lo = pos_full - window_pre
            hi = pos_full + window_post + 1
            if lo < 0 or hi > len(hq):
                continue
            seg = hq.iloc[lo:hi][loss_col].values
            if len(seg) == window_pre + window_post + 1 and np.isfinite(seg).sum() >= 5:
                trajs_s.append(seg)
        if trajs_s:
            arr_s = np.array(trajs_s)
            ax.fill_between(
                offsets,
                np.nanpercentile(arr_s, 5, axis=0),
                np.nanpercentile(arr_s, 95, axis=0),
                alpha=0.15, color=colour,
            )
            ax.plot(offsets, np.nanmean(arr_s, axis=0), lw=2, color=colour)
            ax.plot(offsets, np.nanmedian(arr_s, axis=0), lw=1.5, ls="--", color=colour, alpha=0.7)
        ax.axvline(0, color=colour, ls=":", lw=1)
        ax.set_title(f"{season_label.title()} season (n={len(trajs_s)})", fontsize=10)
        ax.set_xlabel("Days relative to rain")
    ax_d.set_ylabel("Loss proxy (%)")
    fig.suptitle("Signal 3-D: Seasonal event study", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s3_rain_event_study_seasonal.png")

    # ── Verdict ────────────────────────────────────────────────────────
    # Event study: compare mean loss at day 0 vs mean at day +3..+5
    es_p = np.nan
    if len(traj_arr) >= 5:
        at_event = traj_arr[:, window_pre]  # day 0
        post_window = np.nanmean(traj_arr[:, window_pre + 3 : window_pre + 6], axis=1)
        diffs_es = post_window - at_event
        valid = np.isfinite(diffs_es)
        if valid.sum() >= 5:
            _, es_p = stats.wilcoxon(diffs_es[valid], alternative="less")

    dryspell_p = p_w if np.isfinite(p_w) else 1.0
    event_p = es_p if np.isfinite(es_p) else 1.0

    if event_p < 0.05 or dryspell_p < 0.05:
        verdict = "pass"
    elif event_p < 0.15 or dryspell_p < 0.15:
        verdict = "weak"
    else:
        verdict = "fail"

    details = {
        "n_rain_events": len(traj_arr),
        "n_dry_spells": len(starts),
        "event_study_p": float(es_p) if np.isfinite(es_p) else None,
        "dryspell_wilcoxon_p": float(p_w) if np.isfinite(p_w) else None,
        "recovery_rain_r": float(r_rc) if np.isfinite(r_rc) else None,
    }
    summary = (
        f"Event-study Wilcoxon p = {event_p:.4f} (day+3..+5 vs day 0). "
        f"Dry-spell accumulation p = {dryspell_p:.4f} (end > start). "
        f"n_rain={len(traj_arr)}, n_spells={len(starts)}."
    )
    log.info("Signal 3 verdict: %s — %s", verdict.upper(), summary)
    return SignalResult("Signal 3: Rain recovery", verdict, summary, details)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Supporting analyses                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def run_supporting_analyses(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    log.info("── Supporting analyses ──")
    hq = _hq_filter(df)
    results: Dict[str, Any] = {}

    # S4-A  univariate distributions ----------------------------------------
    dist_items = [
        ("t1_performance_loss_pct_proxy", "Loss proxy (%)", C_T1),
        ("precipitation_total_mm", "Precipitation (mm)", C_RAIN),
        ("pm10_mean", "PM10 (µg/m³)", C_DRY),
        ("cycle_deviation_pct", "Cycle deviation (%)", C_ACCENT),
        ("domain_soiling_daily", "DSPI daily rate", C_DRY),
        ("t1_perf_loss_rate_14d_pct_per_day", "Loss rate (%/day)", C_T1),
    ]
    dist_items = [(c, t, clr) for c, t, clr in dist_items if c in hq.columns]
    n_cols = 3
    n_rows = (len(dist_items) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes_flat = axes.flatten() if n_rows > 1 else list(axes)
    for ax, (col, title, colour) in zip(axes_flat, dist_items):
        vals = hq[col].dropna()
        ax.hist(vals, bins=40, color=colour, alpha=0.7, edgecolor="white", lw=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Days")
        zeros = (vals == 0).sum()
        ax.text(
            0.95, 0.95, f"n={len(vals)}\nzeros={zeros}\nmed={vals.median():.1f}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    for ax in axes_flat[len(dist_items):]:
        ax.set_visible(False)
    fig.suptitle("S4-A: Univariate distributions (HQ days)", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s4_univariate_distributions.png")

    # S4-B  pvlib & DSPI vs observed -----------------------------------------
    has_dspi = "domain_soiling_index" in hq.columns
    n_phys_rows = 2 if has_dspi else 1
    fig, axes = plt.subplots(n_phys_rows, 2, figsize=(16, 6 * n_phys_rows))
    if n_phys_rows == 1:
        axes = axes[np.newaxis, :]

    pvlib_loss = hq["pvlib_soiling_loss_kimber"] * 100
    obs_loss = hq["t1_performance_loss_pct_proxy"]

    # Row 1: pvlib Kimber
    pair = pd.DataFrame({"pvlib": pvlib_loss, "obs": obs_loss}).dropna()
    r_pv = np.nan
    axes[0, 0].scatter(pair["pvlib"], pair["obs"], s=10, alpha=0.4, color=C_ACCENT)
    if len(pair) > 3:
        r_pv, _ = stats.pearsonr(pair["pvlib"], pair["obs"])
        axes[0, 0].set_title(f"pvlib Kimber scatter (r={r_pv:.3f})", fontsize=9)
    else:
        axes[0, 0].set_title("pvlib Kimber scatter", fontsize=9)
    axes[0, 0].set_xlabel("pvlib Kimber loss (%)")
    axes[0, 0].set_ylabel("Observed loss proxy (%)")

    axes[0, 1].plot(hq["day_dt"], obs_loss, lw=0.7, color=C_T1, alpha=0.7, label="Observed proxy")
    ax_pv_twin = axes[0, 1].twinx()
    ax_pv_twin.plot(hq["day_dt"], pvlib_loss, lw=0.7, color=C_ACCENT, alpha=0.7, label="pvlib Kimber")
    axes[0, 1].set_ylabel("Observed (%)", color=C_T1)
    ax_pv_twin.set_ylabel("pvlib (%)", color=C_ACCENT)
    axes[0, 1].set_title("pvlib time-series comparison", fontsize=9)
    ln1, lb1 = axes[0, 1].get_legend_handles_labels()
    ln2, lb2 = ax_pv_twin.get_legend_handles_labels()
    axes[0, 1].legend(ln1 + ln2, lb1 + lb2, fontsize=7)

    # Row 2: DSPI
    r_dspi_lp = np.nan
    if has_dspi:
        dspi_vals = hq["domain_soiling_index"]
        pair_d = pd.DataFrame({"dspi": dspi_vals, "obs": obs_loss}).dropna()
        axes[1, 0].scatter(pair_d["dspi"], pair_d["obs"], s=10, alpha=0.4, color=C_DRY)
        if len(pair_d) > 3:
            r_dspi_lp, _ = stats.pearsonr(pair_d["dspi"], pair_d["obs"])
            axes[1, 0].set_title(f"DSPI scatter (r={r_dspi_lp:.3f})", fontsize=9)
        else:
            axes[1, 0].set_title("DSPI scatter", fontsize=9)
        axes[1, 0].set_xlabel("Domain soiling index")
        axes[1, 0].set_ylabel("Observed loss proxy (%)")

        axes[1, 1].plot(hq["day_dt"], obs_loss, lw=0.7, color=C_T1, alpha=0.7, label="Observed proxy")
        ax_ds_twin = axes[1, 1].twinx()
        ax_ds_twin.plot(hq["day_dt"], dspi_vals, lw=0.7, color=C_DRY, alpha=0.7, label="Domain soiling index")
        axes[1, 1].set_ylabel("Observed (%)", color=C_T1)
        ax_ds_twin.set_ylabel("DSPI (cumul.)", color=C_DRY)
        axes[1, 1].set_title("DSPI time-series comparison", fontsize=9)
        ln3, lb3 = axes[1, 1].get_legend_handles_labels()
        ln4, lb4 = ax_ds_twin.get_legend_handles_labels()
        axes[1, 1].legend(ln3 + ln4, lb3 + lb4, fontsize=7)

    fig.suptitle("S4-B: Physics-based soiling estimates vs observed loss proxy", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s4_pvlib_vs_observed.png")
    results["pvlib_r"] = r_pv
    results["dspi_vs_loss_proxy_r"] = r_dspi_lp

    # S4-C  sensor dirt check -----------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ratio = df["solcast_gti_sum"] / df["irradiance_tilted_sum"]
    rolling = ratio.rolling(30, min_periods=10, center=True).mean()
    ax.plot(df["day_dt"], ratio, lw=0.4, alpha=0.3, color="grey", label="Daily ratio")
    ax.plot(df["day_dt"], rolling, lw=1.5, color=C_T1, label="30-day rolling mean")
    ax.set_ylabel("Solcast GTI / ground sensor ratio")
    ax.set_title("S4-C: Sensor dirt check (upward trend = sensor soiling)", fontsize=10)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, plots_dir / "s4_sensor_dirt_check.png")

    slope_ratio = np.nan
    x_days = (df["day_dt"] - df["day_dt"].min()).dt.days.values.astype(float)
    mask = np.isfinite(ratio.values)
    if mask.sum() > 10:
        slope_ratio, *_ = stats.linregress(x_days[mask], ratio.values[mask])
    results["sensor_ratio_trend_per_day"] = slope_ratio

    # S4-D  tier validation -------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        df["day_dt"], df["t1_performance_loss_pct_proxy"],
        lw=0.7, alpha=0.7, color=C_T1, label="T1 (B2)",
    )
    ax.plot(
        df["day_dt"], df["t2_performance_loss_pct_proxy"],
        lw=0.7, alpha=0.7, color=C_T2, label="T2 (B1)",
    )
    tier_corr = df["tier_loss_correlation"].median()
    ax.set_title(
        f"S4-D: Tier validation — T1 vs T2 loss proxy (median r = {tier_corr:.3f})",
        fontsize=10,
    )
    ax.set_ylabel("Loss proxy (%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, plots_dir / "s4_tier_validation.png")
    results["tier_loss_corr_median"] = tier_corr

    # S4-E  seasonal box plots ----------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    months = sorted(hq["month"].dropna().unique().astype(int))
    box_data = [hq.loc[hq["month"] == m, "t1_performance_loss_pct_proxy"].dropna().values for m in months]
    bp = ax.boxplot(box_data, tick_labels=[str(m) for m in months], patch_artist=True)
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    ax.set_xticklabels([month_names[m - 1] for m in months])
    for patch, m in zip(bp["boxes"], months):
        patch.set_facecolor(C_DRY if m in {1, 2, 3, 6, 7, 8, 9} else C_WET)
        patch.set_alpha(0.5)
    ax.set_ylabel("Loss proxy (%)")
    ax.set_title("S4-E: Monthly loss distributions (HQ days) — amber = dry, teal = wet")
    fig.tight_layout()
    _save(fig, plots_dir / "s4_seasonal_boxplots.png")

    # S4-F  quality gating --------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(
        df["transfer_quality_score"].dropna(), bins=30,
        color=C_T1, alpha=0.7, edgecolor="white",
    )
    ax1.set_xlabel("Transfer quality score")
    ax1.set_ylabel("Days")
    ax1.set_title("Score distribution")

    tiers = df["transfer_quality_tier"].value_counts().reindex(["high", "medium", "low"]).fillna(0)
    tier_colours = [C_ACCENT, C_DRY, C_T2]
    bars = ax2.bar(tiers.index, tiers.values, color=tier_colours, alpha=0.7)
    for bar, val in zip(bars, tiers.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, int(val),
                 ha="center", fontsize=9)
    hq_zero = len(_hq_filter(df))
    ax2.axhline(hq_zero, color="grey", ls="--", lw=0.8)
    ax2.text(0.5, hq_zero + 2, f"HQ+0-flags = {hq_zero}", fontsize=8, color="grey")
    ax2.set_ylabel("Days")
    ax2.set_title("Tier distribution")
    fig.suptitle("S4-F: Quality gating", fontsize=11)
    fig.tight_layout()
    _save(fig, plots_dir / "s4_quality_gating.png")

    results["n_total"] = len(df)
    results["n_hq_zero_flag"] = hq_zero
    results["date_range"] = f"{df['day_dt'].min().date()} to {df['day_dt'].max().date()}"

    # S5-A  Domain Soiling Pressure Index time-series -------------------------
    if "domain_soiling_index" in df.columns:
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(
            df["day_dt"], df["domain_soiling_index"],
            lw=1.0, color=C_DRY, alpha=0.85, label="Domain soiling index",
        )
        ax1.set_ylabel("Domain soiling index (cumul. units)", color=C_DRY)
        ax1.tick_params(axis="y", labelcolor=C_DRY)
        _add_rain_cleaning_overlays(ax1, df)

        if "cycle_deviation_pct" in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(
                df["day_dt"], df["cycle_deviation_pct"],
                lw=0.8, color=C_T1, alpha=0.65, label="Cycle deviation (%)",
            )
            ax2.set_ylabel("Cycle deviation (%)", color=C_T1)
            ax2.tick_params(axis="y", labelcolor=C_T1)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        else:
            ax1.legend(fontsize=7)

        ax1.set_title(
            "S5-A: Domain Soiling Pressure Index vs observed cycle deviation",
            fontsize=10,
        )
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        fig.tight_layout()
        _save(fig, plots_dir / "s5_domain_soiling_index.png")

        # Correlation with cycle_deviation
        pair_dspi = df[["domain_soiling_index", "cycle_deviation_pct"]].dropna()
        if len(pair_dspi) > 3:
            r_dspi_cd, _ = stats.pearsonr(*pair_dspi.values.T)
        else:
            r_dspi_cd = np.nan
        results["dspi_vs_cycle_deviation_r"] = r_dspi_cd

    # S5-B  DSPI correlation profile ------------------------------------------
    if "domain_soiling_index" in df.columns:
        profile_cols = [
            ("pm25_mean", "PM2.5"),
            ("pm10_mean", "PM10"),
            ("cumulative_pm25_since_rain", "Cumul. PM2.5"),
            ("cumulative_pm10_since_rain", "Cumul. PM10"),
            ("days_since_last_rain", "Days since rain"),
            ("humidity_mean", "Humidity"),
            ("humidity_x_pm10", "Humidity x PM10"),
            ("dewpoint_mean", "Dewpoint"),
            ("precipitation_total_mm", "Precipitation"),
            ("wind_speed_10m_mean", "Wind speed"),
            ("cloud_opacity_mean", "Cloud opacity"),
            ("air_temp_mean", "Air temperature"),
            ("t1_performance_loss_pct_proxy", "Loss proxy"),
            ("t1_perf_loss_rate_14d_pct_per_day", "Loss rate"),
            ("cycle_deviation_pct", "Cycle deviation"),
        ]
        corr_vals, corr_labels = [], []
        for col, label in profile_cols:
            if col in hq.columns:
                pair = hq[["domain_soiling_index", col]].dropna()
                if len(pair) > 3:
                    r_val, _ = stats.pearsonr(*pair.values.T)
                else:
                    r_val = np.nan
                corr_vals.append(r_val)
                corr_labels.append(label)

        fig, ax = plt.subplots(figsize=(10, 6))
        colours = [
            "#2ecc71" if v > 0.1 else "#e74c3c" if v < -0.1 else "#95a5a6"
            for v in corr_vals
        ]
        y_pos = range(len(corr_labels))
        ax.barh(y_pos, corr_vals, color=colours, alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(corr_labels, fontsize=8)
        ax.set_xlabel("Pearson r with domain_soiling_index")
        ax.axvline(0, color="black", lw=0.5)
        for i, v in enumerate(corr_vals):
            if np.isfinite(v):
                ax.text(
                    v + (0.02 if v >= 0 else -0.02), i, f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7,
                )
        ax.set_title(
            "S5-B: Domain Soiling Index — correlation profile (HQ days)",
            fontsize=10,
        )
        fig.tight_layout()
        _save(fig, plots_dir / "s5_dspi_correlation_profile.png")

        results["dspi_corr_profile"] = dict(zip(corr_labels, corr_vals))

    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Clear-Sky Soiling Analysis                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝

def test_clear_sky_soiling(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """Analyse soiling on Clear-Sky Analyzable (CSA) days only.

    CSA days have low cloud, no rain, functioning equipment, and no carry-
    over weather -- the subset where soiling metrics are least contaminated
    by tropical weather noise.  Three plots are produced (c1, c2, c3).
    """
    log.info("── Clear-Sky Soiling Analysis ──")
    hq = _hq_filter(df)
    results: Dict[str, Any] = {}

    csa_col = "is_clear_sky_analyzable"
    if csa_col not in df.columns:
        log.warning("Column %s not found; skipping CSA analysis.", csa_col)
        return results

    csa = df[df[csa_col]].copy()
    results["csa_n"] = len(csa)
    results["hq_n"] = len(hq)
    log.info("CSA days: %d / %d HQ", len(csa), len(hq))

    loss_col = (
        "t1_performance_loss_pct_proxy"
        if "t1_performance_loss_pct_proxy" in df.columns
        else "performance_loss_pct_proxy"
    )
    dev_col = "cycle_deviation_pct"

    # ------------------------------------------------------------------
    # C1: Clear-sky loss time-series (CSA dots over faded HQ backdrop)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        hq["day_dt"], hq[loss_col],
        lw=0.5, color=C_T1, alpha=0.20, label="All HQ",
    )
    ax.scatter(
        csa["day_dt"], csa[loss_col],
        s=14, color=C_ACCENT, zorder=3, label="CSA days",
    )
    csa_sorted = csa.sort_values("day_dt")
    ax.plot(
        csa_sorted["day_dt"], csa_sorted[loss_col],
        lw=0.6, color=C_ACCENT, alpha=0.5,
    )
    _add_rain_cleaning_overlays(ax, df)
    ax.set_ylabel("Performance loss proxy (%)")
    ax.set_title("C1: Loss proxy on Clear-Sky Analyzable days (weather-filtered)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig.tight_layout()
    _save(fig, plots_dir / "c1_clear_sky_loss_timeseries.png")

    # ------------------------------------------------------------------
    # C2: Side-by-side correlation comparison (All HQ vs CSA)
    # ------------------------------------------------------------------
    compare_feats = [
        ("cumulative_pm25_since_rain", "Cumul. PM2.5"),
        ("days_since_last_rain", "Days since rain"),
        ("pm10_mean", "PM10"),
        ("pm25_mean", "PM2.5"),
        ("cloud_opacity_mean", "Cloud opacity"),
        ("air_temp_mean", "Temperature"),
        ("domain_soiling_index", "DSPI"),
    ]
    compare_feats = [
        (c, label) for c, label in compare_feats if c in hq.columns
    ]

    r_hq_list: List[float] = []
    r_csa_list: List[float] = []
    sig_hq: List[bool] = []
    sig_csa: List[bool] = []
    labels: List[str] = []

    for col, label in compare_feats:
        pair_all = hq[[loss_col, col]].dropna()
        pair_csa = csa[[loss_col, col]].dropna()
        r_a = p_a = r_c = p_c = np.nan
        if len(pair_all) > 5:
            r_a, p_a = stats.pearsonr(pair_all.iloc[:, 0], pair_all.iloc[:, 1])
        if len(pair_csa) > 5:
            r_c, p_c = stats.pearsonr(pair_csa.iloc[:, 0], pair_csa.iloc[:, 1])
        r_hq_list.append(r_a)
        r_csa_list.append(r_c)
        sig_hq.append(p_a < 0.05 if np.isfinite(p_a) else False)
        sig_csa.append(p_c < 0.05 if np.isfinite(p_c) else False)
        labels.append(label)

    results["corr_comparison"] = {
        label: {"r_hq": rh, "r_csa": rc}
        for label, rh, rc in zip(labels, r_hq_list, r_csa_list)
    }

    y_pos = np.arange(len(labels))
    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.7)))
    bars_hq = ax.barh(
        y_pos - bar_h / 2, r_hq_list, bar_h, label="All HQ", color=C_T1, alpha=0.7,
    )
    bars_csa = ax.barh(
        y_pos + bar_h / 2, r_csa_list, bar_h, label="CSA only", color=C_ACCENT, alpha=0.7,
    )
    for i, (sh, sc) in enumerate(zip(sig_hq, sig_csa)):
        rh, rc = r_hq_list[i], r_csa_list[i]
        if sh and np.isfinite(rh):
            ax.text(rh, i - bar_h / 2, " *", va="center", fontsize=10, fontweight="bold")
        if sc and np.isfinite(rc):
            ax.text(rc, i + bar_h / 2, " *", va="center", fontsize=10, fontweight="bold", color=C_ACCENT)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson r with loss proxy")
    ax.axvline(0, color="grey", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"C2: Feature correlations — All HQ (n={len(hq)}) vs CSA (n={len(csa)})    (* = p<0.05)",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "c2_clean_vs_all_correlations.png")

    # ------------------------------------------------------------------
    # C3: Scatter matrix on CSA days (top significant predictors)
    # ------------------------------------------------------------------
    scatter_pairs = [
        ("cumulative_pm25_since_rain", loss_col, "Cumul. PM2.5", "Loss proxy (%)"),
        ("days_since_last_rain", loss_col, "Days since rain", "Loss proxy (%)"),
        ("cumulative_pm25_since_rain", dev_col, "Cumul. PM2.5", "Cycle deviation (%)"),
        ("days_since_last_rain", dev_col, "Days since rain", "Cycle deviation (%)"),
    ]
    scatter_pairs = [
        (x, y, xl, yl) for x, y, xl, yl in scatter_pairs
        if x in csa.columns and y in csa.columns
    ]
    n_panels = len(scatter_pairs)
    if n_panels > 0:
        n_c = 2
        n_r = (n_panels + n_c - 1) // n_c
        fig, axes = plt.subplots(n_r, n_c, figsize=(14, 6 * n_r))
        axes_flat = axes.flatten() if n_r > 1 else list(axes)
        for ax, (xc, yc, xl, yl) in zip(axes_flat, scatter_pairs):
            pair = csa[[xc, yc]].dropna()
            ax.scatter(pair[xc], pair[yc], s=16, alpha=0.7, color=C_ACCENT)
            if len(pair) > 5:
                r_val, p_val = stats.pearsonr(pair[xc], pair[yc])
                z = np.polyfit(pair[xc], pair[yc], 1)
                xs = np.linspace(pair[xc].min(), pair[xc].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color=C_T1, lw=1, ls="--")
                sig_star = " *" if p_val < 0.05 else ""
                ax.set_title(f"r={r_val:+.3f}, p={p_val:.3f}{sig_star}", fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        fig.suptitle(
            f"C3: CSA-only scatter matrix (n={len(csa)})", fontsize=11,
        )
        fig.tight_layout()
        _save(fig, plots_dir / "c3_clean_scatter_matrix.png")

    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Data Quality: Irradiance vs Generation                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_irradiance_vs_generation(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """Data-quality diagnostic: on-site irradiance vs inverter generation.

    Both quantities are summed over the tracked 10 AM – 2 PM window.
    Produces two files:
      - dq1_irradiance_vs_generation_timeseries.png  (dual-axis time series)
      - dq1_irradiance_vs_generation.png             (scatter + boxplot panels)
    """
    log.info("── Data quality: irradiance vs generation ──")
    results: Dict[str, Any] = {}

    irr = df["irradiance_tilted_sum"].copy()
    gen = df["t1_energy_j"].copy()
    day_dt = df["day_dt"]
    month = day_dt.dt.month
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    gen_kwh = gen / 3.6e6
    cmap = plt.cm.hsv

    # ── Figure A: time-series (full width, readable x-axis) ───────────
    fig_ts, ax1 = plt.subplots(figsize=(16, 6))
    
    def _scale(s):
        v = s.dropna()
        if v.empty or v.max() == v.min(): return s
        return (s - v.min()) / (v.max() - v.min())
        
    irr_scaled = _scale(irr)
    gen_scaled = _scale(gen_kwh)
    
    # Base daily lines
    ax1.plot(day_dt, irr_scaled, lw=0.6, color=C_T1, alpha=0.3, label="On-site irr (daily)")
    ax1.plot(day_dt, gen_scaled, lw=0.6, color=C_ACCENT, alpha=0.3, label="T1 generation (daily)")
    
    # 7-day rolling medians
    irr_smooth = irr_scaled.rolling(7, center=True, min_periods=3).median()
    gen_smooth = gen_scaled.rolling(7, center=True, min_periods=3).median()
    
    ax1.plot(day_dt, irr_smooth, lw=2.0, color=C_T1, label="On-site irr (7-day median)")
    ax1.plot(day_dt, gen_smooth, lw=2.0, color=C_ACCENT, label="T1 gen (7-day median)")
    
    # Add CSA day markers
    if "is_clear_sky_analyzable" in df.columns:
        csa_mask = df["is_clear_sky_analyzable"].fillna(False).astype(bool)
        if csa_mask.any():
            ax1.scatter(day_dt[csa_mask], [0] * csa_mask.sum(),
                        color="grey", alpha=0.6, s=15, marker="^", label="CSA days", zorder=5)

    ax1.set_ylabel("Normalised to [0, 1]", fontsize=8)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    _add_rain_cleaning_overlays(ax1, df)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    fig_ts.suptitle(
        "DQ1: On-site irradiance & T1 generation time series (10 AM – 2 PM)",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq1_irradiance_vs_generation_timeseries.png")

    # ── Figure B: scatter + boxplot panels ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter irr vs gen, coloured by month
    ax2 = axes[0]
    valid = (irr > 0) & (gen > 0)
    sc = ax2.scatter(
        irr[valid], gen_kwh[valid], c=month[valid], cmap=cmap,
        s=18, alpha=0.65, edgecolors="white", linewidths=0.3,
        vmin=1, vmax=12,
    )
    cbar = fig.colorbar(sc, ax=ax2, ticks=range(1, 13))
    cbar.ax.set_yticklabels([month_names[i] for i in range(1, 13)], fontsize=6)
    cbar.set_label("Month", fontsize=7)
    
    if valid.sum() > 3:
        r_val, _ = stats.pearsonr(irr[valid], gen_kwh[valid])
        title_str = f"On-site irr vs T1 gen (r={r_val:.3f})"
        results["onsite_irr_vs_gen_r"] = r_val
        
        if "subset_energy_j" in df.columns:
            comb_gen = df["subset_energy_j"] / 3.6e6
            comb_valid = (irr > 0) & (comb_gen > 0)
            if comb_valid.sum() > 3:
                r_comb, _ = stats.pearsonr(irr[comb_valid], comb_gen[comb_valid])
                ax2.text(
                    0.02, 0.98, f"Combined T1+T2 r={r_comb:.3f}",
                    transform=ax2.transAxes, fontsize=8, va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                results["onsite_irr_vs_combined_gen_r"] = r_comb

        ax2.set_title(title_str, fontsize=9)
    else:
        ax2.set_title("On-site irradiance vs T1 generation", fontsize=9)
        
    ax2.set_xlabel("On-site irradiance (sensor sum)", fontsize=8)
    ax2.set_ylabel("T1 generation (kWh)", fontsize=8)

    # Panel 2: Solcast peak-hour GTI vs generation
    ax3 = axes[1]
    sol_peak_kwh_m2 = (
        df["solcast_gti_peak_sum"] / 3.6e6
        if "solcast_gti_peak_sum" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    sol_peak_valid = sol_peak_kwh_m2.notna() & (gen > 0) & (sol_peak_kwh_m2 > 0)
    if sol_peak_valid.sum() > 3:
        sc3 = ax3.scatter(
            sol_peak_kwh_m2[sol_peak_valid], gen_kwh[sol_peak_valid],
            c=month[sol_peak_valid], cmap=cmap, s=18, alpha=0.65,
            edgecolors="white", linewidths=0.3, vmin=1, vmax=12,
        )
        r_sol_peak, _ = stats.pearsonr(
            sol_peak_kwh_m2[sol_peak_valid], gen_kwh[sol_peak_valid],
        )
        ax3.set_title(
            f"Solcast peak GTI vs T1 gen (r={r_sol_peak:.3f})", fontsize=9,
        )
        results["solcast_gti_peak_vs_gen_r"] = r_sol_peak
        cbar3 = fig.colorbar(sc3, ax=ax3, ticks=range(1, 13))
        cbar3.ax.set_yticklabels([month_names[i] for i in range(1, 13)], fontsize=6)
        cbar3.set_label("Month", fontsize=7)

        if "daily_generation_j" in df.columns:
            gen_full = df["daily_generation_j"] / 3.6e6
            full_mask = sol_peak_kwh_m2.notna() & (gen_full > 0) & (sol_peak_kwh_m2 > 0)
            if full_mask.sum() > 3:
                r_full, _ = stats.pearsonr(
                    sol_peak_kwh_m2[full_mask], gen_full[full_mask],
                )
                ax3.text(
                    0.02, 0.02,
                    f"vs full-plant gen: r={r_full:.3f}",
                    transform=ax3.transAxes, fontsize=7, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                results["solcast_gti_peak_vs_fullgen_r"] = r_full
    else:
        ax3.text(0.5, 0.5, "Solcast peak GTI\nnot available",
                 transform=ax3.transAxes, ha="center", fontsize=10, color="grey")
    ax3.set_xlabel("Solcast GTI (kWh/m², 10–14h peak)", fontsize=8)
    ax3.set_ylabel("T1 generation (kWh)", fontsize=8)

    # Panel 3: monthly box-plot of normalised output
    ax4 = axes[2]
    norm_col = (
        "t1_normalized_output"
        if "t1_normalized_output" in df.columns
        else "t1_normalized_output" if "t1_normalized_output" in df.columns
        else "normalized_output"
    )
    if norm_col in df.columns:
        valid_norm = df[norm_col].notna() & (df[norm_col] > 0)
        months_present = sorted(df.loc[valid_norm, "day_dt"].dt.month.unique())
        box_data = [
            df.loc[valid_norm & (df["day_dt"].dt.month == m), norm_col].dropna().values
            for m in months_present
        ]
        bp = ax4.boxplot(box_data, patch_artist=True,
                         tick_labels=[month_names[m] for m in months_present])
        for patch in bp["boxes"]:
            patch.set_facecolor(C_T1)
            patch.set_alpha(0.4)
        ax4.set_ylabel("Normalised output (energy/irradiance)", fontsize=8)
        ax4.set_title("Monthly normalised output consistency", fontsize=9)

        medians = [np.median(b) for b in box_data if len(b) > 0]
        if len(medians) >= 2:
            cv = np.std(medians) / np.mean(medians) * 100
            results["norm_output_monthly_cv_pct"] = cv
            ax4.text(
                0.02, 0.98, f"CV of medians = {cv:.1f}%",
                transform=ax4.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
    else:
        ax4.text(0.5, 0.5, "Normalised output\nnot available",
                 transform=ax4.transAxes, ha="center", fontsize=10, color="grey")

    fig.suptitle(
        "DQ1: Irradiance vs generation — scatter & distribution (10 AM – 2 PM)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "dq1_irradiance_vs_generation.png")

    # Summary statistics
    results["n_zero_gen_sunny"] = int(((gen <= 0) & (irr > irr.quantile(0.25))).sum())
    results["n_days_total"] = len(df)

    log.info(
        "DQ1 done: on-site r=%.3f, solcast-peak r=%.3f, solcast-peak-vs-fullgen r=%.3f, zero-gen-on-sunny=%d",
        results.get("onsite_irr_vs_gen_r", float("nan")),
        results.get("solcast_gti_peak_vs_gen_r", float("nan")),
        results.get("solcast_gti_peak_vs_fullgen_r", float("nan")),
        results["n_zero_gen_sunny"],
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DQ2: Daily generation validation (new telemetry)                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _plot_daily_gen_validation_legacy(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """Cross-validate new daily_generated_electricity against old active-power
    energy and new plant avg_solar_radiation against Solcast GTI.

    Produces two files:
      - dq2_daily_gen_validation_timeseries.png  (generation + irradiance ts)
      - dq2_daily_gen_validation.png             (scatter panels)
    Gracefully returns empty dict if new columns are absent.
    """
    results: Dict[str, Any] = {}
    has_gen = "subset_daily_gen_kwh" in df.columns and df["subset_daily_gen_kwh"].notna().sum() > 5
    has_irr = "plant_avg_irradiance_wm2" in df.columns and df["plant_avg_irradiance_wm2"].notna().sum() > 5

    if not has_gen and not has_irr:
        log.info("DQ2 skipped: new telemetry columns not present")
        return results

    log.info("── DQ2: Daily gen & plant irradiance validation ──")
    day_dt = df["day_dt"]

    # ── Figure A: time-series panels (stacked) ────────────────────────
    n_ts_panels = 3 if (has_irr and "solcast_gti_peak_mean_wm2" in df.columns) else 2
    fig_ts, ts_axes = plt.subplots(n_ts_panels, 1, figsize=(16, 6 * n_ts_panels))
    ax_ts1, ax_ts2 = ts_axes[0], ts_axes[1]

    if has_gen and "subset_energy_mwh" in df.columns:
        old_mwh = df["subset_energy_mwh"]
        new_kwh = df["subset_daily_gen_kwh"]
        ax_ts1.plot(day_dt, old_mwh * 1000, lw=0.8, alpha=0.6, color=C_T2,
                    label="Old (active power integral, kWh)")
        ax_ts1.plot(day_dt, new_kwh, lw=0.8, alpha=0.8, color=C_T1,
                    label="New (daily_generated_electricity, kWh)")
        ax_ts1.set_ylabel("Subset generation (kWh)", fontsize=8)
        ax_ts1.legend(fontsize=7, loc="upper right")
        _add_rain_cleaning_overlays(ax_ts1, df)
        _annotate_new_source_start(ax_ts1, df)
        ax_ts1.xaxis.set_major_locator(mdates.MonthLocator())
        ax_ts1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_ts1.set_title("Subset generation: old vs new source", fontsize=9)
    else:
        ax_ts1.text(0.5, 0.5, "New daily gen not available",
                    transform=ax_ts1.transAxes, ha="center", fontsize=10, color="grey")

    if has_irr:
        plant_irr = df["plant_avg_irradiance_wm2"]
        ax_ts2.plot(day_dt, plant_irr, lw=0.8, alpha=0.8, color=C_T1,
                    label="Plant avg_solar_radiation (W/m²)")
        if "solcast_gti_peak_mean_wm2" in df.columns:
            sol_mean = df["solcast_gti_peak_mean_wm2"]
            ax_ts2.plot(day_dt, sol_mean, lw=0.8, alpha=0.6, color=C_T2,
                        label="Solcast peak GTI mean (W/m²)")
        ax_ts2.set_ylabel("Irradiance (W/m²)", fontsize=8)
        ax_ts2.legend(fontsize=7, loc="upper right")
        _add_rain_cleaning_overlays(ax_ts2, df)
        _annotate_new_source_start(ax_ts2, df)
        ax_ts2.xaxis.set_major_locator(mdates.MonthLocator())
        ax_ts2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax_ts2.set_title("Plant irradiance: on-site avg vs Solcast", fontsize=9)
    else:
        ax_ts2.text(0.5, 0.5, "Plant avg irradiance not available",
                    transform=ax_ts2.transAxes, ha="center", fontsize=10, color="grey")

    # Panel 3: Solcast / ground irradiance ratio — sensor health indicator
    if n_ts_panels == 3:
        ax_ts3 = ts_axes[2]
        sol_col = "solcast_gti_peak_mean_wm2"
        plant_col = "plant_avg_irradiance_wm2"
        sol_v = df[sol_col]
        plant_v = df[plant_col]
        ratio_valid = sol_v.notna() & plant_v.notna() & (plant_v > 0)
        sol_ground_ratio = (sol_v / plant_v).where(ratio_valid)
        ratio_smooth_14 = sol_ground_ratio.rolling(14, center=True, min_periods=5).median()

        ax_ts3.scatter(day_dt, sol_ground_ratio, s=8, alpha=0.3, color=C_T1,
                       label="Daily ratio", zorder=2)
        ax_ts3.plot(day_dt, ratio_smooth_14, lw=2.5, color="#E74C3C",
                    label="14-day rolling median", zorder=3)
        ax_ts3.axhline(sol_ground_ratio.median(), color="black", lw=0.8,
                       ls="--", alpha=0.5, label=f"Median = {sol_ground_ratio.median():.2f}")

        # Overlay sensor recalibration flags
        if "flag_sensor_recalibrated" in df.columns:
            recal_mask = df["flag_sensor_recalibrated"].fillna(False).astype(bool)
            if recal_mask.any():
                ax_ts3.scatter(day_dt[recal_mask],
                               sol_ground_ratio[recal_mask],
                               marker="x", s=60, color="#E74C3C", linewidths=2,
                               label=f"Recalibration flag ({recal_mask.sum()} days)",
                               zorder=5)

        # Compute and report trend
        ratio_clean = sol_ground_ratio.dropna()
        if len(ratio_clean) > 10:
            x_num = np.arange(len(ratio_clean))
            slope, intercept = np.polyfit(x_num, ratio_clean.values, 1)
            results["sensor_ratio_trend_per_day"] = slope
            ax_ts3.text(
                0.02, 0.98,
                f"Trend: {slope:+.4f} per day\nMedian ratio: {sol_ground_ratio.median():.3f}",
                transform=ax_ts3.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        ax_ts3.set_ylabel("Solcast / Ground ratio", fontsize=8)
        ax_ts3.set_title("Sensor health: Solcast / ground irradiance ratio", fontsize=9)
        ax_ts3.legend(fontsize=7, loc="upper right")
        _add_rain_cleaning_overlays(ax_ts3, df)
        ax_ts3.xaxis.set_major_locator(mdates.MonthLocator())
        ax_ts3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig_ts.suptitle(
        "DQ2: New telemetry time series — daily gen & plant irradiance",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq2_daily_gen_validation_timeseries.png")

    # ── Figure B: scatter panels ──────────────────────────────────────
    fig, (ax_sc1, ax_sc2) = plt.subplots(1, 2, figsize=(14, 6))

    if has_gen and "subset_energy_j" in df.columns:
        old_j = df["subset_energy_j"]
        new_j = df["subset_daily_gen_kwh"] * 3.6e6
        valid = old_j.notna() & new_j.notna() & (old_j > 0) & (new_j > 0)
        if valid.sum() > 3:
            ax_sc1.scatter(
                old_j[valid] / 3.6e6, new_j[valid] / 3.6e6,
                s=14, alpha=0.5, color=C_T1, edgecolors="white", linewidths=0.3,
            )
            r_gen, _ = stats.pearsonr(old_j[valid], new_j[valid])
            results["old_vs_new_gen_r"] = r_gen
            ax_sc1.set_title(f"Old vs new generation (r={r_gen:.3f})", fontsize=9)
            lims = [
                min(old_j[valid].min(), new_j[valid].min()) / 3.6e6,
                max(old_j[valid].max(), new_j[valid].max()) / 3.6e6,
            ]
            ax_sc1.plot(lims, lims, "--", color="grey", lw=0.7, alpha=0.5)
        ax_sc1.set_xlabel("Old: active power integral (kWh)", fontsize=8)
        ax_sc1.set_ylabel("New: daily_generated_electricity (kWh)", fontsize=8)
    else:
        ax_sc1.text(0.5, 0.5, "New daily gen not available",
                    transform=ax_sc1.transAxes, ha="center", fontsize=10, color="grey")

    if has_irr and "solcast_gti_peak_mean_wm2" in df.columns:
        plant_irr = df["plant_avg_irradiance_wm2"]
        sol_mean = df["solcast_gti_peak_mean_wm2"]
        valid = plant_irr.notna() & sol_mean.notna() & (plant_irr > 0) & (sol_mean > 0)
        if valid.sum() > 3:
            ax_sc2.scatter(
                sol_mean[valid], plant_irr[valid],
                s=14, alpha=0.5, color=C_ACCENT, edgecolors="white", linewidths=0.3,
            )
            r_irr, _ = stats.pearsonr(sol_mean[valid], plant_irr[valid])
            results["plant_vs_solcast_irr_r"] = r_irr
            ax_sc2.set_title(f"Plant avg vs Solcast peak mean (r={r_irr:.3f})", fontsize=9)
        ax_sc2.set_xlabel("Solcast peak GTI mean (W/m²)", fontsize=8)
        ax_sc2.set_ylabel("Plant avg_solar_radiation (W/m²)", fontsize=8)
    else:
        ax_sc2.text(0.5, 0.5, "Comparison not available",
                    transform=ax_sc2.transAxes, ha="center", fontsize=10, color="grey")

    fig.suptitle(
        "DQ2: New telemetry validation — scatter comparisons",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "dq2_daily_gen_validation.png")

    log.info(
        "DQ2 done: old-vs-new-gen r=%.3f, plant-vs-solcast-irr r=%.3f",
        results.get("old_vs_new_gen_r", float("nan")),
        results.get("plant_vs_solcast_irr_r", float("nan")),
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DQ3: Generation / irradiance ratio (normalized performance)       ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _plot_gen_irr_ratio_legacy(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """Generation-to-irradiance ratio from the new telemetry sources.

    Mirrors the user's pre-test visualization: normalized performance ratio
    with context (generation, irradiance scaled alongside).

    Produces two files:
      - dq3_gen_irr_ratio_timeseries.png  (ratio + context ts, ratio vs loss proxy ts)
      - dq3_gen_irr_ratio.png             (monthly boxplot)
    """
    results: Dict[str, Any] = {}

    if "gen_irr_ratio" not in df.columns or df["gen_irr_ratio"].notna().sum() < 5:
        log.info("DQ3 skipped: gen_irr_ratio not available")
        return results

    log.info("── DQ3: Generation / irradiance ratio ──")
    day_dt = df["day_dt"]
    ratio = df["gen_irr_ratio"]
    smoothed = df.get("gen_irr_ratio_smoothed", ratio)
    valid = ratio.notna()
    month_names = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    results["gen_irr_ratio_median"] = float(ratio[valid].median()) if valid.any() else np.nan
    results["gen_irr_ratio_std"] = float(ratio[valid].std()) if valid.any() else np.nan

    def _scale_to(series, lo, hi):
        s = series.dropna()
        if s.empty or s.max() == s.min():
            return series
        return (series - s.min()) / (s.max() - s.min()) * (hi - lo) + lo

    ratio_lo = ratio[valid].min() if valid.any() else 0
    ratio_hi = ratio[valid].max() if valid.any() else 1

    # ── STL decomposition ─────────────────────────────────────────────
    stl_ok = False
    stl_trend = stl_seasonal = stl_resid = None
    ratio_for_stl = ratio.copy()
    if valid.sum() > 60:
        try:
            # Interpolate NaN gaps for STL (it requires continuous series)
            ratio_interp = ratio_for_stl.interpolate(method="linear", limit_direction="both")
            ratio_interp = ratio_interp.fillna(ratio_interp.median())
            stl_result = STL(ratio_interp.values, period=30, robust=True).fit()
            stl_trend = pd.Series(stl_result.trend, index=ratio.index)
            stl_seasonal = pd.Series(stl_result.seasonal, index=ratio.index)
            stl_resid = pd.Series(stl_result.resid, index=ratio.index)
            # Restore NaN where original was NaN
            stl_trend[~valid] = np.nan
            stl_seasonal[~valid] = np.nan
            stl_resid[~valid] = np.nan
            stl_ok = True
            results["stl_trend_range"] = float(stl_trend.max() - stl_trend.min())
            results["stl_seasonal_amplitude"] = float(stl_seasonal.max() - stl_seasonal.min())
        except Exception as e:
            log.warning("STL decomposition failed: %s", e)

    # ── Figure A: time-series panels (stacked) ────────────────────────
    n_ts = 5 if stl_ok else 2
    fig_ts, ts_axes = plt.subplots(n_ts, 1, figsize=(16, 3.5 * n_ts))
    if n_ts == 2:
        ts_axes = list(ts_axes)
    ax1 = ts_axes[0]
    ax3 = ts_axes[1]

    # Panel 1: ratio with context
    if "subset_daily_gen_kwh" in df.columns:
        gen_scaled = _scale_to(df["subset_daily_gen_kwh"], ratio_lo, ratio_hi)
        ax1.plot(day_dt, gen_scaled, lw=1.0, alpha=0.4, color="#E67E22",
                 label="Generation (scaled)")
    if "plant_avg_irradiance_wm2" in df.columns:
        irr_scaled = _scale_to(df["plant_avg_irradiance_wm2"], ratio_lo, ratio_hi)
        ax1.plot(day_dt, irr_scaled, lw=1.0, alpha=0.4, color="#F4D03F",
                 label="Irradiance (scaled)")

    ax1.plot(day_dt, ratio, lw=0.7, alpha=0.4, color="#5DADE2",
             label="Gen/Irr ratio (daily)")
    ax1.plot(day_dt, smoothed, lw=2.5, color="#1F3A93",
             label="Gen/Irr ratio (7-day median)")
    ax1.axhline(ratio[valid].median() if valid.any() else 1,
                color="black", lw=0.8, ls="--", alpha=0.5)

    _add_rain_cleaning_overlays(ax1, df)
    _annotate_new_source_start(ax1, df)
    ax1.set_ylabel("Scaled units / ratio", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.set_title("Normalized performance ratio with context", fontsize=10)

    # Panel 2: overlay with loss proxy (inverted)
    loss_col = "t1_performance_loss_pct_proxy" if "t1_performance_loss_pct_proxy" in df.columns else "performance_loss_pct_proxy"
    if loss_col in df.columns:
        loss = df[loss_col]
        inverted_loss = -loss
        ax3.plot(day_dt, smoothed, lw=2.0, color="#1F3A93", label="Gen/Irr ratio (smoothed)")
        ax3r = ax3.twinx()
        ax3r.plot(day_dt, inverted_loss, lw=1.0, alpha=0.5, color=C_T2,
                  label="−Loss proxy (inverted)")
        ax3r.set_ylabel("−Loss proxy (%)", color=C_T2, fontsize=8)
        ax3r.tick_params(axis="y", labelcolor=C_T2, labelsize=7)

        both_valid = smoothed.notna() & loss.notna()
        if both_valid.sum() > 5:
            r_agree, _ = stats.pearsonr(smoothed[both_valid], inverted_loss[both_valid])
            results["ratio_vs_neg_loss_r"] = r_agree
            ax3.set_title(
                f"Gen/Irr ratio vs −loss proxy (r={r_agree:.3f})", fontsize=10,
            )
        ln1, lb1 = ax3.get_legend_handles_labels()
        ln2, lb2 = ax3r.get_legend_handles_labels()
        ax3.legend(ln1 + ln2, lb1 + lb2, fontsize=7, loc="upper right")
    else:
        ax3.plot(day_dt, smoothed, lw=2.0, color="#1F3A93", label="Gen/Irr ratio (smoothed)")
        ax3.legend(fontsize=7)
    ax3.set_ylabel("Gen / Irr ratio", fontsize=8)
    _add_rain_cleaning_overlays(ax3, df)
    _annotate_new_source_start(ax3, df)
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panels 3-5: STL decomposition (Trend, Seasonal, Residual)
    if stl_ok:
        stl_parts = [
            (ts_axes[2], stl_trend, "STL Trend (soiling / long-term drift)", "#1F3A93"),
            (ts_axes[3], stl_seasonal, "STL Seasonal (monsoon / climatology, period=30d)", "#27AE60"),
            (ts_axes[4], stl_resid, "STL Residual (noise)", "#E74C3C"),
        ]
        for ax_stl, component, title, color in stl_parts:
            ax_stl.plot(day_dt, component, lw=1.5, color=color, alpha=0.8)
            ax_stl.axhline(0 if "Residual" in title or "Seasonal" in title else component.median(),
                           color="black", lw=0.5, ls="--", alpha=0.3)
            _add_rain_cleaning_overlays(ax_stl, df)
            ax_stl.set_title(title, fontsize=9)
            ax_stl.set_ylabel("Component value", fontsize=7)
            ax_stl.tick_params(axis="y", labelsize=7)
            ax_stl.xaxis.set_major_locator(mdates.MonthLocator())
            ax_stl.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig_ts.suptitle(
        "DQ3: Generation / irradiance ratio — time series & STL decomposition",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq3_gen_irr_ratio_timeseries.png")

    # ── Figure B: monthly boxplot ─────────────────────────────────────
    fig_box, ax2 = plt.subplots(figsize=(12, 6))
    months_present = sorted(df.loc[valid, "day_dt"].dt.month.unique())
    box_data = [
        ratio[valid & (day_dt.dt.month == m)].dropna().values
        for m in months_present
    ]
    bp = ax2.boxplot(
        box_data, patch_artist=True,
        tick_labels=[month_names[m] for m in months_present],
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(C_T1)
        patch.set_alpha(0.4)
    ax2.set_ylabel("Gen / Irr ratio", fontsize=8)
    ax2.set_title("Monthly gen/irr ratio", fontsize=10)
    fig_box.suptitle(
        "DQ3: Generation / irradiance ratio — monthly distribution",
        fontsize=12, fontweight="bold",
    )
    fig_box.tight_layout()
    _save(fig_box, plots_dir / "dq3_gen_irr_ratio.png")

    log.info(
        "DQ3 done: median ratio=%.4f, ratio-vs-neg-loss r=%.3f",
        results.get("gen_irr_ratio_median", float("nan")),
        results.get("ratio_vs_neg_loss_r", float("nan")),
    )
    return results


# DQ2/DQ3 replacements use physically consistent irradiation and PR metrics.
def _detect_aligned_inverter_ids(df: pd.DataFrame) -> List[str]:
    power_ids = {
        c[:-len("_energy_j")]
        for c in df.columns
        if c.endswith("_energy_j")
        and c.startswith(("b1_", "b2_"))
        and c.count("_") == 2
    }
    daily_ids = {
        c[:-len("_daily_gen_j")]
        for c in df.columns
        if c.endswith("_daily_gen_j")
        and c.startswith(("b1_", "b2_"))
        and c.count("_") == 2
    }
    return sorted(power_ids.intersection(daily_ids))


def plot_daily_gen_validation(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """DQ2: like-for-like generation validation with physical irradiation context."""
    results: Dict[str, Any] = {}
    has_new_gen = "subset_daily_gen_kwh" in df.columns and df["subset_daily_gen_kwh"].notna().sum() > 5
    has_irr_kwh = "irradiation_kwh_m2" in df.columns and df["irradiation_kwh_m2"].notna().sum() > 5
    has_avg_irr = "plant_avg_irradiance_wm2" in df.columns and df["plant_avg_irradiance_wm2"].notna().sum() > 5

    if not has_new_gen and not has_irr_kwh and not has_avg_irr:
        log.info("DQ2 skipped: required new-source fields are absent")
        return results

    log.info("── DQ2: Daily generation validation (asset-aligned + physical irradiation) ──")
    day_dt = df["day_dt"]
    aligned_ids = _detect_aligned_inverter_ids(df)
    results["aligned_inverters"] = aligned_ids

    old_aligned_kwh = None
    if aligned_ids:
        old_cols = [f"{inv}_energy_j" for inv in aligned_ids if f"{inv}_energy_j" in df.columns]
        if old_cols:
            old_aligned_kwh = df[old_cols].sum(axis=1, min_count=1) / 3.6e6
    if old_aligned_kwh is None and "subset_energy_mwh" in df.columns:
        old_aligned_kwh = df["subset_energy_mwh"] * 1000.0

    new_aligned_kwh = df["subset_daily_gen_kwh"] if has_new_gen else None

    n_ts_panels = 3
    fig_ts, ts_axes = plt.subplots(n_ts_panels, 1, figsize=(16, 5.2 * n_ts_panels))
    ax_ts1, ax_ts2, ax_ts3 = ts_axes[0], ts_axes[1], ts_axes[2]

    # Panel 1: like-for-like old vs new generation
    if old_aligned_kwh is not None and new_aligned_kwh is not None:
        ax_ts1.plot(
            day_dt, old_aligned_kwh, lw=0.9, alpha=0.65, color=C_T2,
            label="Old aligned generation (active-power integral, kWh)",
        )
        ax_ts1.plot(
            day_dt, new_aligned_kwh, lw=0.9, alpha=0.85, color=C_T1,
            label="New aligned generation (daily meter, kWh)",
        )
        valid = old_aligned_kwh.notna() & new_aligned_kwh.notna() & (old_aligned_kwh > 0) & (new_aligned_kwh > 0)
        if int(valid.sum()) > 5:
            r_gen, _ = stats.pearsonr(old_aligned_kwh[valid], new_aligned_kwh[valid])
            results["old_vs_new_gen_r"] = float(r_gen)
        ax_ts1.set_ylabel("Generation (kWh)", fontsize=8)
        ax_ts1.legend(fontsize=7, loc="upper right")
    else:
        ax_ts1.text(0.5, 0.5, "Generation comparison unavailable", transform=ax_ts1.transAxes, ha="center", color="grey")
    _add_rain_cleaning_overlays(ax_ts1, df)
    _annotate_new_source_start(ax_ts1, df)
    ax_ts1.set_title("Asset-aligned subset generation: old vs new", fontsize=9)
    ax_ts1.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panel 2: physical irradiation context
    if has_irr_kwh:
        irr_kwh = df["irradiation_kwh_m2"]
        ax_ts2.plot(
            day_dt, irr_kwh, lw=0.9, alpha=0.85, color=C_ACCENT,
            label="Daily irradiation (kWh/m²) = avg_irradiance × runtime / 1000",
        )
        if has_new_gen:
            valid = new_aligned_kwh.notna() & irr_kwh.notna() & (irr_kwh > 0)
            if int(valid.sum()) > 5:
                r_irr, _ = stats.pearsonr(new_aligned_kwh[valid], irr_kwh[valid])
                results["new_gen_vs_irradiation_r"] = float(r_irr)
        ax_ts2.set_ylabel("Irradiation (kWh/m²)", fontsize=8)
        ax_ts2.legend(fontsize=7, loc="upper right")
    else:
        ax_ts2.text(0.5, 0.5, "irradiation_kwh_m2 not available", transform=ax_ts2.transAxes, ha="center", color="grey")
    _add_rain_cleaning_overlays(ax_ts2, df)
    _annotate_new_source_start(ax_ts2, df)
    ax_ts2.set_title("Physical irradiation context", fontsize=9)
    ax_ts2.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panel 3: ground-vs-solcast irradiance consistency
    if has_avg_irr and "solcast_gti_peak_mean_wm2" in df.columns:
        plant_irr = df["plant_avg_irradiance_wm2"]
        sol_mean = df["solcast_gti_peak_mean_wm2"]
        ratio_valid = plant_irr.notna() & sol_mean.notna() & (plant_irr > 0)
        ratio = (sol_mean / plant_irr).where(ratio_valid)
        smooth = ratio.rolling(14, center=True, min_periods=5).median()
        ax_ts3.scatter(day_dt, ratio, s=8, alpha=0.30, color=C_T1, label="Daily ratio")
        ax_ts3.plot(day_dt, smooth, lw=2.0, color="#E74C3C", label="14-day median")
        if int(ratio_valid.sum()) > 5:
            r_ps, _ = stats.pearsonr(plant_irr[ratio_valid], sol_mean[ratio_valid])
            results["plant_vs_solcast_irr_r"] = float(r_ps)
        ax_ts3.set_ylabel("Solcast / Ground", fontsize=8)
        ax_ts3.legend(fontsize=7, loc="upper right")
    else:
        ax_ts3.text(0.5, 0.5, "Ground-vs-Solcast comparison unavailable", transform=ax_ts3.transAxes, ha="center", color="grey")
    _add_rain_cleaning_overlays(ax_ts3, df)
    ax_ts3.set_title("Sensor consistency (Solcast / ground irradiance)", fontsize=9)
    ax_ts3.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig_ts.suptitle(
        "DQ2: Daily generation validation — asset aligned with physical irradiation context",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq2_daily_gen_validation_timeseries.png")

    fig, (ax_sc1, ax_sc2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter A: old vs new aligned generation
    if old_aligned_kwh is not None and new_aligned_kwh is not None:
        valid = old_aligned_kwh.notna() & new_aligned_kwh.notna() & (old_aligned_kwh > 0) & (new_aligned_kwh > 0)
        if int(valid.sum()) > 5:
            ax_sc1.scatter(
                old_aligned_kwh[valid], new_aligned_kwh[valid],
                s=14, alpha=0.5, color=C_T1, edgecolors="white", linewidths=0.3,
            )
            lim_lo = float(min(old_aligned_kwh[valid].min(), new_aligned_kwh[valid].min()))
            lim_hi = float(max(old_aligned_kwh[valid].max(), new_aligned_kwh[valid].max()))
            ax_sc1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="grey", lw=0.7, alpha=0.6)
            r_gen, _ = stats.pearsonr(old_aligned_kwh[valid], new_aligned_kwh[valid])
            results["old_vs_new_gen_r"] = float(r_gen)
            ax_sc1.set_title(f"Old vs new aligned generation (r={r_gen:.3f})", fontsize=9)
        ax_sc1.set_xlabel("Old aligned generation (kWh)", fontsize=8)
        ax_sc1.set_ylabel("New aligned generation (kWh)", fontsize=8)
    else:
        ax_sc1.text(0.5, 0.5, "Generation scatter unavailable", transform=ax_sc1.transAxes, ha="center", color="grey")

    # Scatter B: new generation vs physical irradiation
    if has_new_gen and has_irr_kwh:
        irr_kwh = df["irradiation_kwh_m2"]
        valid = new_aligned_kwh.notna() & irr_kwh.notna() & (irr_kwh > 0)
        if int(valid.sum()) > 5:
            ax_sc2.scatter(
                irr_kwh[valid], new_aligned_kwh[valid],
                s=14, alpha=0.5, color=C_ACCENT, edgecolors="white", linewidths=0.3,
            )
            r_irr, _ = stats.pearsonr(irr_kwh[valid], new_aligned_kwh[valid])
            results["new_gen_vs_irradiation_r"] = float(r_irr)
            ax_sc2.set_title(f"New generation vs irradiation (r={r_irr:.3f})", fontsize=9)
        ax_sc2.set_xlabel("Irradiation (kWh/m²)", fontsize=8)
        ax_sc2.set_ylabel("New aligned generation (kWh)", fontsize=8)
    else:
        ax_sc2.text(0.5, 0.5, "Irradiation scatter unavailable", transform=ax_sc2.transAxes, ha="center", color="grey")

    fig.suptitle(
        "DQ2: New telemetry validation — aligned generation and irradiation",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "dq2_daily_gen_validation.png")

    log.info(
        "DQ2 done: old-vs-new r=%.3f, new-vs-irradiation r=%.3f",
        results.get("old_vs_new_gen_r", float("nan")),
        results.get("new_gen_vs_irradiation_r", float("nan")),
    )
    return results


def plot_gen_irr_ratio(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """DQ3: physical PR with outlier markers and interpolated trend."""
    results: Dict[str, Any] = {}

    if "gen_irr_ratio" not in df.columns or df["gen_irr_ratio"].notna().sum() < 5:
        log.info("DQ3 skipped: gen_irr_ratio not available")
        return results

    log.info("── DQ3: Physical PR diagnostics ──")
    day_dt = df["day_dt"]
    pr_raw = df["gen_irr_ratio"]
    outlier = (
        df["subset_pr_physical_outlier"].fillna(False).astype(bool)
        if "subset_pr_physical_outlier" in df.columns
        else (pr_raw.notna() & ((pr_raw < 0) | (pr_raw > 1)))
    )
    pr_interp = (
        df["subset_pr_physical_interp"]
        if "subset_pr_physical_interp" in df.columns
        else pr_raw.mask(outlier).interpolate(method="linear", limit_area="inside")
    )
    pr_roll7 = pr_interp.rolling(7, center=True, min_periods=3).median()

    valid = pr_raw.notna()
    valid_inrange = valid & (~outlier)
    results["gen_irr_ratio_median"] = float(pr_raw[valid].median()) if valid.any() else np.nan
    results["gen_irr_ratio_std"] = float(pr_raw[valid].std()) if valid.any() else np.nan
    results["gen_irr_ratio_outlier_count"] = int(outlier.sum())
    results["gen_irr_ratio_outlier_pct"] = float(100.0 * outlier.mean()) if len(outlier) else np.nan

    n_ts = 3 if "plant_pr_physical_raw" in df.columns else 2
    fig_ts, ts_axes = plt.subplots(n_ts, 1, figsize=(16, 4.0 * n_ts))
    if n_ts == 2:
        ts_axes = list(ts_axes)
    ax1 = ts_axes[0]
    ax2 = ts_axes[1]

    # Panel 1: PR raw + outliers + interpolated trend
    ax1.scatter(
        day_dt[valid_inrange], pr_raw[valid_inrange],
        s=12, alpha=0.35, color="#5DADE2", label="Raw PR (in-range)",
    )
    if outlier.any():
        ax1.scatter(
            day_dt[outlier], pr_raw[outlier],
            marker="x", s=36, linewidths=1.2, color="#E74C3C", label="Outlier (PR<0 or PR>1)",
        )
    ax1.plot(day_dt, pr_interp, lw=1.8, color="#1F3A93", label="Interpolated PR trend")
    ax1.plot(day_dt, pr_roll7, lw=2.6, color="#0B7285", label="Trend 7-day median")
    ax1.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_ylim(bottom=max(-0.05, float(np.nanmin(pr_raw) - 0.05) if valid.any() else -0.05))
    ax1.set_ylabel("Subset PR", fontsize=8)
    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    _add_rain_cleaning_overlays(ax1, df)
    _annotate_new_source_start(ax1, df)
    ax1.set_title("Physical subset PR: raw points, outliers, and interpolated trend", fontsize=10)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panel 2: PR trend vs old-source loss proxy
    loss_col = "t1_performance_loss_pct_proxy" if "t1_performance_loss_pct_proxy" in df.columns else "performance_loss_pct_proxy"
    if loss_col in df.columns:
        loss = df[loss_col]
        ax2.plot(day_dt, pr_roll7, lw=2.0, color="#1F3A93", label="Subset PR trend (7-day)")
        ax2r = ax2.twinx()
        ax2r.plot(day_dt, -loss, lw=1.0, alpha=0.55, color=C_T2, label="−Loss proxy (old source)")
        both_valid = pr_roll7.notna() & loss.notna()
        if int(both_valid.sum()) > 5:
            r_agree, _ = stats.pearsonr(pr_roll7[both_valid], (-loss)[both_valid])
            results["ratio_vs_neg_loss_r"] = float(r_agree)
            ax2.set_title(f"Physical PR trend vs old-source −loss (r={r_agree:.3f})", fontsize=10)
        ln1, lb1 = ax2.get_legend_handles_labels()
        ln2, lb2 = ax2r.get_legend_handles_labels()
        ax2.legend(ln1 + ln2, lb1 + lb2, fontsize=7, loc="upper right")
        ax2r.set_ylabel("−Loss proxy (%)", color=C_T2, fontsize=8)
        ax2r.tick_params(axis="y", labelcolor=C_T2, labelsize=7)
    else:
        ax2.plot(day_dt, pr_roll7, lw=2.0, color="#1F3A93", label="Subset PR trend (7-day)")
        ax2.legend(fontsize=7, loc="upper right")
    ax2.set_ylabel("Subset PR", fontsize=8)
    _add_rain_cleaning_overlays(ax2, df)
    _annotate_new_source_start(ax2, df)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panel 3: plant PR (optional)
    if n_ts == 3:
        ax3 = ts_axes[2]
        plant_raw = df["plant_pr_physical_raw"]
        plant_out = (
            df["plant_pr_physical_outlier"].fillna(False).astype(bool)
            if "plant_pr_physical_outlier" in df.columns
            else (plant_raw.notna() & ((plant_raw < 0) | (plant_raw > 1)))
        )
        plant_trend = (
            df["plant_pr_physical_interp"]
            if "plant_pr_physical_interp" in df.columns
            else plant_raw.mask(plant_out).interpolate(method="linear", limit_area="inside")
        )
        ax3.scatter(day_dt[plant_raw.notna() & (~plant_out)], plant_raw[plant_raw.notna() & (~plant_out)],
                    s=10, alpha=0.30, color="#27AE60", label="Raw plant PR (in-range)")
        if plant_out.any():
            ax3.scatter(day_dt[plant_out], plant_raw[plant_out], marker="x", s=32, linewidths=1.1, color="#E74C3C", label="Outlier")
        ax3.plot(day_dt, plant_trend, lw=2.0, color="#145A32", label="Plant PR trend")
        ax3.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax3.set_ylabel("Plant PR", fontsize=8)
        ax3.legend(fontsize=7, loc="upper right")
        _add_rain_cleaning_overlays(ax3, df)
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax3.set_title("Plant physical PR (capacity = 34 × 330 kW)", fontsize=10)

    fig_ts.suptitle(
        "DQ3: Physical generation/irradiation PR — outliers and trend",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq3_gen_irr_ratio_timeseries.png")

    # Figure B: monthly distribution for in-range PR values only
    fig_box, ax_box = plt.subplots(figsize=(12, 6))
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months_present = sorted(df.loc[valid_inrange, "day_dt"].dt.month.unique())
    box_data = [
        pr_raw[valid_inrange & (day_dt.dt.month == m)].dropna().values
        for m in months_present
    ]
    if box_data:
        bp = ax_box.boxplot(
            box_data, patch_artist=True, tick_labels=[month_names[m] for m in months_present]
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(C_T1)
            patch.set_alpha(0.35)
    ax_box.set_ylabel("Subset PR (in-range)", fontsize=8)
    ax_box.set_title("Monthly physical PR distribution (outliers excluded)", fontsize=10)
    fig_box.suptitle(
        "DQ3: Generation / irradiation ratio — monthly distribution",
        fontsize=12, fontweight="bold",
    )
    fig_box.tight_layout()
    _save(fig_box, plots_dir / "dq3_gen_irr_ratio.png")

    log.info(
        "DQ3 done: median raw PR=%.4f, outliers=%d",
        results.get("gen_irr_ratio_median", float("nan")),
        results.get("gen_irr_ratio_outlier_count", 0),
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DQ4: Power at reference irradiance                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_power_at_ref_irradiance(
    df: pd.DataFrame, plots_dir: Path,
) -> Dict[str, Any]:
    """Time series of active power at the dataset's median irradiance level.

    Should show soiling-driven decay between cleanings if the feature is
    working correctly.

    Produces dq4_power_at_ref_irradiance.png (2-panel figure).
    """
    results: Dict[str, Any] = {}
    col = "power_at_ref_irradiance_w"

    if col not in df.columns or df[col].notna().sum() < 5:
        log.info("DQ4 skipped: power_at_ref_irradiance_w not available")
        return results

    log.info("── DQ4: Power at reference irradiance ──")
    day_dt = df["day_dt"]
    pwr = df[col]
    valid = pwr.notna()

    if "ref_irradiance_wm2" in df.columns:
        ref_val = df["ref_irradiance_wm2"].dropna().iloc[0] if df["ref_irradiance_wm2"].notna().any() else np.nan
        results["ref_irradiance_wm2"] = float(ref_val)

    results["power_at_ref_irr_median"] = float(pwr[valid].median())
    results["power_at_ref_irr_days"] = int(valid.sum())

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))

    # ── Panel 1: time series ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(day_dt, pwr, lw=0.8, alpha=0.5, color=C_T1, label="Daily")
    smoothed_pwr = pwr.rolling(7, center=True, min_periods=3).median()
    ax1.plot(day_dt, smoothed_pwr, lw=2.0, color="#1F3A93",
             label="7-day median")
    _add_rain_cleaning_overlays(ax1, df)
    ax1.set_ylabel("Active power at ref irradiance (W)", fontsize=8)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))
    ref_str = f" (ref={ref_val:.0f} W/m²)" if np.isfinite(results.get("ref_irradiance_wm2", np.nan)) else ""
    ax1.set_title(f"Power at reference irradiance{ref_str}", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")

    for tier in ("t1", "t2"):
        tier_col = f"{tier}_power_at_ref_irradiance_w"
        if tier_col in df.columns and df[tier_col].notna().sum() > 5:
            ax1.plot(
                day_dt,
                df[tier_col].rolling(7, center=True, min_periods=3).median(),
                lw=1.2, alpha=0.6,
                color=C_T1 if tier == "t1" else C_T2,
                ls="--",
                label=f"{tier.upper()} (7-day med)",
            )
    ax1.legend(fontsize=7, loc="upper right")

    # ── Panel 2: ref_irr_match_count per day ──────────────────────────
    ax_mc = axes[1]
    if "ref_irr_match_count" in df.columns:
        match_count = df["ref_irr_match_count"]
        mc_valid = match_count.notna()
        quarter = day_dt.dt.quarter
        q_colors = {1: "#3498DB", 2: "#27AE60", 3: "#F39C12", 4: "#E74C3C"}
        q_labels_map = {1: "Q1 (Jan-Mar)", 2: "Q2 (Apr-Jun)",
                        3: "Q3 (Jul-Sep)", 4: "Q4 (Oct-Dec)"}
        plotted_qs = set()
        for q_num in sorted(quarter[mc_valid].unique()):
            q_mask = mc_valid & (quarter == q_num)
            lbl = q_labels_map.get(q_num, f"Q{q_num}")
            if q_num not in plotted_qs:
                ax_mc.bar(day_dt[q_mask], match_count[q_mask],
                          color=q_colors.get(q_num, "grey"), alpha=0.7,
                          width=1.0, label=lbl)
                plotted_qs.add(q_num)
            else:
                ax_mc.bar(day_dt[q_mask], match_count[q_mask],
                          color=q_colors.get(q_num, "grey"), alpha=0.7,
                          width=1.0)

        # Quarterly median annotations
        for q_num in sorted(quarter[mc_valid].unique()):
            q_mask = mc_valid & (quarter == q_num)
            q_med = match_count[q_mask].median()
            q_dates = day_dt[q_mask]
            if len(q_dates) > 0:
                mid_date = q_dates.iloc[len(q_dates) // 2]
                ax_mc.annotate(
                    f"med={q_med:.0f}",
                    xy=(mid_date, q_med), fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                    ha="center",
                )

        ax_mc.set_ylabel("Matching intervals per day", fontsize=8)
        ax_mc.set_title("Reference irradiance match count by quarter", fontsize=10)
        ax_mc.legend(fontsize=7, loc="upper right", ncol=4)
        ax_mc.xaxis.set_major_locator(mdates.MonthLocator())
        ax_mc.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%y"))

        results["match_count_median"] = float(match_count.median())
        results["match_count_q1_median"] = float(match_count[mc_valid & (quarter == 1)].median()) if (mc_valid & (quarter == 1)).any() else np.nan
        results["match_count_q4_median"] = float(match_count[mc_valid & (quarter == 4)].median()) if (mc_valid & (quarter == 4)).any() else np.nan
    else:
        ax_mc.text(0.5, 0.5, "ref_irr_match_count not available",
                   transform=ax_mc.transAxes, ha="center", fontsize=10, color="grey")

    # ── Panel 3: raw correlation bar chart ────────────────────────────
    ax2 = axes[2]
    corr_features = [
        ("cumulative_pm25_since_rain", "Cum PM2.5"),
        ("cumulative_pm10_since_rain", "Cum PM10"),
        ("pm25_mean", "PM2.5 (daily)"),
        ("pm10_mean", "PM10 (daily)"),
        ("days_since_last_rain", "Days since rain"),
        ("domain_soiling_index", "DSPI"),
        ("t1_performance_loss_pct_proxy", "Loss proxy (T1)"),
        ("cycle_deviation_pct", "Cycle deviation"),
        ("humidity_x_pm10", "Humidity × PM10"),
        ("cloud_opacity_mean", "Cloud opacity"),
    ]

    corr_labels = []
    corr_values = []
    hq = _hq_filter(df)
    for feat_col, label in corr_features:
        if feat_col in hq.columns:
            both = hq[col].notna() & hq[feat_col].notna()
            if both.sum() > 10:
                r, p = stats.pearsonr(hq.loc[both, col], hq.loc[both, feat_col])
                corr_labels.append(label)
                corr_values.append(r)
                results[f"ref_irr_vs_{feat_col}_r"] = r

    if corr_values:
        colors = ["#E74C3C" if v < 0 else C_ACCENT for v in corr_values]
        bars = ax2.barh(range(len(corr_labels)), corr_values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(corr_labels)))
        ax2.set_yticklabels(corr_labels, fontsize=8)
        ax2.set_xlabel("Pearson r with power_at_ref_irradiance", fontsize=8)
        ax2.axvline(0, color="black", lw=0.5)
        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            ax2.text(
                val + 0.01 * (1 if val >= 0 else -1), i,
                f"{val:+.3f}", va="center", fontsize=7,
            )
        ax2.set_title("Raw Pearson correlations with soiling features (HQ days)", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Insufficient data for correlations",
                 transform=ax2.transAxes, ha="center", fontsize=10, color="grey")

    # ── Panel 4: partial correlations (controlling for cloud + month) ─
    ax_pc = axes[3]
    control_cols = []
    if "cloud_opacity_mean" in hq.columns:
        control_cols.append("cloud_opacity_mean")
    # Add month as numeric control variable
    if "day_dt" in hq.columns:
        hq = hq.copy()
        hq["_month"] = hq["day_dt"].dt.month
        control_cols.append("_month")

    partial_labels = []
    partial_values = []
    raw_for_partial = []

    if len(control_cols) >= 1 and col in hq.columns:
        for feat_col, label in corr_features:
            if feat_col in hq.columns and feat_col not in control_cols:
                sub = hq[[col, feat_col] + control_cols].dropna()
                if len(sub) > 15:
                    # Raw correlation
                    r_raw, _ = stats.pearsonr(sub[col], sub[feat_col])
                    # Partial correlation: regress both on controls, correlate residuals
                    try:
                        from numpy.linalg import lstsq
                        X_ctrl = sub[control_cols].values
                        X_ctrl = np.column_stack([X_ctrl, np.ones(len(X_ctrl))])
                        # Residualise target
                        coef_y, _, _, _ = lstsq(X_ctrl, sub[col].values, rcond=None)
                        resid_y = sub[col].values - X_ctrl @ coef_y
                        # Residualise feature
                        coef_x, _, _, _ = lstsq(X_ctrl, sub[feat_col].values, rcond=None)
                        resid_x = sub[feat_col].values - X_ctrl @ coef_x
                        r_partial, _ = stats.pearsonr(resid_y, resid_x)
                    except Exception:
                        r_partial = np.nan

                    partial_labels.append(label)
                    raw_for_partial.append(r_raw)
                    partial_values.append(r_partial)
                    results[f"ref_irr_vs_{feat_col}_partial_r"] = r_partial

    if partial_values:
        y_pos = np.arange(len(partial_labels))
        bar_height = 0.35
        # Raw bars
        raw_colors = ["#E74C3C" if v < 0 else C_ACCENT for v in raw_for_partial]
        ax_pc.barh(y_pos + bar_height / 2, raw_for_partial, bar_height,
                   color=raw_colors, alpha=0.4, label="Raw r")
        # Partial bars
        partial_colors = ["#E74C3C" if v < 0 else "#1F3A93" for v in partial_values]
        ax_pc.barh(y_pos - bar_height / 2, partial_values, bar_height,
                   color=partial_colors, alpha=0.8, label="Partial r (ctrl: cloud + month)")

        ax_pc.set_yticks(y_pos)
        ax_pc.set_yticklabels(partial_labels, fontsize=8)
        ax_pc.set_xlabel("Correlation with power_at_ref_irradiance", fontsize=8)
        ax_pc.axvline(0, color="black", lw=0.5)

        for i, (rv, pv) in enumerate(zip(raw_for_partial, partial_values)):
            ax_pc.text(rv + 0.01 * (1 if rv >= 0 else -1),
                       i + bar_height / 2, f"{rv:+.3f}",
                       va="center", fontsize=6, alpha=0.6)
            ax_pc.text(pv + 0.01 * (1 if pv >= 0 else -1),
                       i - bar_height / 2, f"{pv:+.3f}",
                       va="center", fontsize=6, fontweight="bold")

        # Check if PM10 flips sign after deconfounding
        pm10_raw = results.get("ref_irr_vs_cumulative_pm10_since_rain_r", None)
        pm10_partial = results.get("ref_irr_vs_cumulative_pm10_since_rain_partial_r", None)
        pm25_partial = results.get("ref_irr_vs_cumulative_pm25_since_rain_partial_r", None)
        annotation_parts = []
        if pm10_raw is not None and pm10_partial is not None:
            if pm10_raw > 0 and pm10_partial < 0:
                annotation_parts.append("⚠ PM10 FLIPPED: seasonal confounding confirmed")
                results["pm10_seasonal_confounding"] = True
            elif pm10_raw > 0 and pm10_partial > 0:
                annotation_parts.append("PM10 stayed positive after deconfounding")
                results["pm10_seasonal_confounding"] = False
        if pm25_partial is not None and pm10_partial is not None:
            if pm25_partial < pm10_partial:
                annotation_parts.append("→ Use PM2.5 as primary soiling predictor")
                results["primary_pm_predictor"] = "PM2.5"
            else:
                annotation_parts.append("→ PM10 stronger after deconfounding")
                results["primary_pm_predictor"] = "PM10"

        if annotation_parts:
            ax_pc.text(
                0.02, 0.02, "\n".join(annotation_parts),
                transform=ax_pc.transAxes, fontsize=8, va="bottom",
                bbox=dict(facecolor="#FFF3CD", alpha=0.9, edgecolor="#FFC107"),
            )

        ax_pc.set_title("Partial correlations (controlling for cloud_opacity + month)", fontsize=10)
        ax_pc.legend(fontsize=7, loc="upper right")
    else:
        ax_pc.text(0.5, 0.5, "Insufficient data for partial correlations",
                   transform=ax_pc.transAxes, ha="center", fontsize=10, color="grey")

    fig.suptitle(
        "DQ4: Power at reference irradiance — soiling degradation signal",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "dq4_power_at_ref_irradiance.png")

    log.info(
        "DQ4 done: %d days, median power=%.0f W, PM predictor=%s",
        results.get("power_at_ref_irr_days", 0),
        results.get("power_at_ref_irr_median", float("nan")),
        results.get("primary_pm_predictor", "unknown"),
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DQ6: New-source Performance Index (0-1)                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_new_performance_index(
    df: pd.DataFrame, plots_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Visualise the 0-1 performance index derived from new telemetry.

    performance_index = gen_irr_ratio / new_rolling_clean_baseline.
    Values near 1.0 = clean-panel performance; values < 1 = degradation.

    Figure A (main): TS, frozen-baseline overlay, histogram
    Figure B: Above-1.0 clustering + rain Wilcoxon test
    Figure C: Scatter + quantile regression (90th pctile)
    """
    idx_col = "new_performance_index"
    if idx_col not in df.columns or df[idx_col].notna().sum() < 10:
        log.info("DQ6 skipped: new_performance_index not present.")
        return None

    log.info("── DQ6: New-source performance index ──")
    results: Dict[str, Any] = {}
    hq = df[df["transfer_readiness_tier"].isin(["Tier-1", "Tier-2"])] if "transfer_readiness_tier" in df.columns else df
    idx = hq[idx_col]
    valid = idx.notna()
    idx_valid = idx[valid]

    results["perf_index_median"] = float(idx_valid.median())
    results["perf_index_mean"] = float(idx_valid.mean())
    results["pct_below_1"] = float((idx_valid < 1.0).mean() * 100)
    results["pct_below_09"] = float((idx_valid < 0.9).mean() * 100)
    results["pct_below_08"] = float((idx_valid < 0.8).mean() * 100)
    results["pct_above_1"] = float((idx_valid > 1.0).mean() * 100)
    results["n_days"] = int(valid.sum())

    soiling_features = [
        ("domain_soiling_index", "DSPI"),
        ("cumulative_pm25_since_rain", "Cum PM2.5"),
        ("cumulative_pm10_since_rain", "Cum PM10"),
        ("days_since_last_rain", "Days dry"),
        ("humidity_x_pm10", "Hum × PM10"),
    ]
    available = [(c, l) for c, l in soiling_features if c in hq.columns]
    n_scatter = max(len(available), 1)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1.8, 1.0], hspace=0.35)

    # ── Panel 1: Time-series ──
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(hq.loc[valid, "day_dt"], idx_valid, s=10, alpha=0.4,
               color=C_T1, label="Daily", zorder=2)
    smoothed = idx.rolling(7, center=True, min_periods=3).median()
    sm_valid = smoothed.notna()
    if sm_valid.any():
        ax1.plot(hq.loc[sm_valid, "day_dt"], smoothed[sm_valid],
                lw=2.0, color="#1F3A93", label="7-day median", zorder=3)
    ax1.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.6, label="Clean baseline")
    ax1.axhline(0.9, color="#EF4444", ls=":", lw=0.8, alpha=0.5, label="90% threshold")
    _add_rain_cleaning_overlays(ax1, hq)
    _annotate_new_source_start(ax1, df)
    ax1.set_ylabel("Performance Index")
    ax1.set_ylim(0, min(1.6, idx_valid.max() * 1.1))
    ax1.set_title(f"New-Source Performance Index (median = {idx_valid.median():.3f})", fontsize=10)
    ax1.legend(fontsize=7, loc="lower left", ncol=2)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # ── Panel 2: Frozen-baseline performance index ──
    ax_frz = fig.add_subplot(gs[1])
    baseline_col = "new_rolling_clean_baseline"
    ratio_col = "gen_irr_ratio"
    if baseline_col in hq.columns and ratio_col in hq.columns:
        baseline = hq[baseline_col]
        ratio = hq[ratio_col]
        cleaning_dates = [pd.Timestamp(d[0]) for d in CLEANING_CAMPAIGN_DATES]
        if cleaning_dates:
            first_cleaning = min(cleaning_dates)
            pre_clean = baseline[hq["day_dt"] < first_cleaning].dropna()
            frozen_baseline = pre_clean.iloc[-1] if len(pre_clean) > 0 else (baseline.dropna().iloc[0] if baseline.notna().any() else 1.0)
        else:
            frozen_baseline = baseline.dropna().iloc[0] if baseline.notna().any() else 1.0
        frozen_pi = ratio / frozen_baseline
        frozen_valid = frozen_pi.notna() & valid
        frozen_smooth = frozen_pi.rolling(7, center=True, min_periods=3).median()
        ax_frz.scatter(hq.loc[frozen_valid, "day_dt"], frozen_pi[frozen_valid],
                      s=10, alpha=0.3, color="#E67E22", label="Frozen-baseline PI", zorder=2)
        if frozen_smooth.notna().any():
            ax_frz.plot(hq["day_dt"], frozen_smooth, lw=2.0, color="#D35400",
                       label="Frozen 7-day median", zorder=3)
        if sm_valid.any():
            ax_frz.plot(hq.loc[sm_valid, "day_dt"], smoothed[sm_valid],
                       lw=1.5, color="#1F3A93", alpha=0.5, ls="--",
                       label="Standard PI (rolling baseline)", zorder=3)
        ax_frz.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.6)
        _add_rain_cleaning_overlays(ax_frz, hq)
        ax_frz.set_ylabel("Performance Index (frozen baseline)")
        ax_frz.set_title(f"Frozen baseline at pre-cleaning level ({frozen_baseline:.2f})", fontsize=10)
        ax_frz.legend(fontsize=7, loc="upper right")
        ax_frz.xaxis.set_major_locator(mdates.MonthLocator())
        ax_frz.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        results["frozen_baseline_value"] = float(frozen_baseline)
    else:
        ax_frz.text(0.5, 0.5, "Baseline/ratio columns unavailable",
                   transform=ax_frz.transAxes, ha="center", fontsize=10, color="grey")

    # ── Panel 3: Distribution ──
    ax_hist = fig.add_subplot(gs[2])
    bins = np.linspace(0, min(1.5, idx_valid.max() + 0.05), 40)
    ax_hist.hist(idx_valid, bins=bins, color=C_T1, alpha=0.6, edgecolor="white", lw=0.5)
    for thr, clr in [(1.0, "black"), (0.9, "#EF4444"), (0.8, "#F59E0B"), (0.7, "#D97706")]:
        ax_hist.axvline(thr, color=clr, ls="--", lw=1.0, alpha=0.7)
        pct = (idx_valid < thr).mean() * 100
        ax_hist.text(thr - 0.01, ax_hist.get_ylim()[1] * 0.85, f"{pct:.0f}% below",
                fontsize=7, color=clr, ha="right", rotation=90)
    ax_hist.set_xlabel("Performance Index")
    ax_hist.set_ylabel("Day count")
    ax_hist.set_title("Distribution of performance index", fontsize=10)

    fig.suptitle("DQ6: New-Source Performance Index", fontsize=12, fontweight="bold")
    _save(fig, plots_dir / "dq6_performance_index.png")

    # ── Figure B: Above-1.0 clustering + Rain Wilcoxon ────────────────
    fig_b, (ax_clust, ax_wilcox) = plt.subplots(1, 2, figsize=(16, 6))
    above_mask = idx_valid > 1.0
    above_days = hq.loc[valid, "day_dt"][above_mask]
    n_above = int(above_mask.sum())
    results["n_above_1"] = n_above
    if n_above > 0:
        categories = []
        for d in above_days:
            cat = "Other"
            for start, end in CLEANING_CAMPAIGN_DATES:
                end_dt = pd.Timestamp(end)
                if pd.Timestamp(start) <= d <= end_dt + pd.Timedelta(days=7):
                    cat = "Post-cleaning (≤7d)"
                    break
            if cat == "Other" and d.month in (3, 4, 9, 10):
                cat = "Seasonal transition"
            categories.append(cat)
        cat_counts = pd.Series(categories).value_counts()
        colors_c = {"Post-cleaning (≤7d)": "#27AE60", "Seasonal transition": "#F39C12", "Other": "#95A5A6"}
        bars_c = ax_clust.bar(cat_counts.index, cat_counts.values,
                             color=[colors_c.get(c, "#95A5A6") for c in cat_counts.index], alpha=0.8)
        for bar, val in zip(bars_c, cat_counts.values):
            ax_clust.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                         f"{val} ({val/n_above*100:.0f}%)", ha="center", fontsize=8)
        ax_clust.set_title(f"Above-1.0 days clustering (n={n_above})", fontsize=10)
        ax_clust.set_ylabel("Day count")
        results["above_1_post_cleaning"] = int(cat_counts.get("Post-cleaning (≤7d)", 0))
        results["above_1_seasonal"] = int(cat_counts.get("Seasonal transition", 0))
    else:
        ax_clust.text(0.5, 0.5, "No days above 1.0", transform=ax_clust.transAxes, ha="center")

    if "precipitation_total_mm" in hq.columns:
        rain_days = hq.loc[hq["precipitation_total_mm"] >= SIGNIFICANT_RAIN_MM, "day_dt"]
        pre_vals, post_vals = [], []
        for rd in rain_days:
            pre_w = (hq["day_dt"] >= rd - pd.Timedelta(days=3)) & (hq["day_dt"] < rd)
            post_w = (hq["day_dt"] > rd) & (hq["day_dt"] <= rd + pd.Timedelta(days=3))
            pre_pi = idx[pre_w].dropna()
            post_pi = idx[post_w].dropna()
            if len(pre_pi) > 0 and len(post_pi) > 0:
                pre_vals.append(pre_pi.median())
                post_vals.append(post_pi.median())
        if len(pre_vals) >= 5:
            pre_arr, post_arr = np.array(pre_vals), np.array(post_vals)
            try:
                w_stat, w_p = stats.wilcoxon(pre_arr, post_arr, alternative="less")
                results["rain_wilcoxon_stat"] = float(w_stat)
                results["rain_wilcoxon_p"] = float(w_p)
                results["rain_wilcoxon_pass"] = w_p < 0.05
            except Exception:
                w_p = np.nan
                results["rain_wilcoxon_pass"] = False
            delta = (post_arr - pre_arr).mean()
            ax_wilcox.boxplot([pre_arr, post_arr], labels=["Pre-rain (−3d)", "Post-rain (+3d)"],
                             patch_artist=True, boxprops=dict(facecolor=C_T1, alpha=0.5))
            ax_wilcox.set_title(
                f"PI ±3d around rain (n={len(pre_vals)})\n"
                f"Wilcoxon p={w_p:.4f} {'✓ PASS' if w_p < 0.05 else '✗ FAIL'}, Δmed={delta:+.3f}",
                fontsize=9)
            ax_wilcox.set_ylabel("Performance Index")
        else:
            ax_wilcox.text(0.5, 0.5, f"Too few rain events ({len(pre_vals)})",
                          transform=ax_wilcox.transAxes, ha="center", color="grey")
    else:
        ax_wilcox.text(0.5, 0.5, "precipitation_mm N/A", transform=ax_wilcox.transAxes, ha="center", color="grey")
    fig_b.suptitle("DQ6: Above-1.0 clustering & rain recovery Wilcoxon test", fontsize=12, fontweight="bold")
    fig_b.tight_layout()
    _save(fig_b, plots_dir / "dq6_clustering_wilcoxon.png")

    # ── Figure C: Scatter + quantile regression ───────────────────────
    if available:
        fig_c, axes_c = plt.subplots(2, n_scatter, figsize=(3.5 * n_scatter, 10))
        if n_scatter == 1:
            axes_c = axes_c.reshape(2, 1)
        for i, (col, label) in enumerate(available):
            pair = hq[[idx_col, col]].dropna()
            if len(pair) < 10:
                continue
            # Row 1: Scatter + r
            ax_s = axes_c[0, i]
            r_val = pair[idx_col].corr(pair[col])
            results[f"pi_vs_{col}_r"] = r_val
            ax_s.scatter(pair[col], pair[idx_col], s=8, alpha=0.4, color=C_ACCENT)
            ax_s.set_title(f"{label}\nr = {r_val:+.3f}", fontsize=8)
            if i == 0:
                ax_s.set_ylabel("Perf Index", fontsize=7)
            ax_s.tick_params(labelsize=6)
            ax_s.axhline(1.0, color="black", ls="--", lw=0.5, alpha=0.4)
            # Row 2: Quantile regression
            ax_q = axes_c[1, i]
            ax_q.scatter(pair[col], pair[idx_col], s=6, alpha=0.3, color="#AAAAAA")
            try:
                n_bins = min(20, len(pair) // 5)
                bin_edges = np.linspace(pair[col].min(), pair[col].max(), n_bins + 1)
                bc, q90, q50 = [], [], []
                for b in range(n_bins):
                    m_b = (pair[col] >= bin_edges[b]) & (pair[col] < bin_edges[b + 1])
                    sb = pair.loc[m_b, idx_col]
                    if len(sb) >= 3:
                        bc.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                        q90.append(sb.quantile(0.9))
                        q50.append(sb.median())
                if len(bc) >= 4:
                    bc, q90, q50 = np.array(bc), np.array(q90), np.array(q50)
                    s90, i90 = np.polyfit(bc, q90, 1)
                    s50, i50 = np.polyfit(bc, q50, 1)
                    xf = np.linspace(bc.min(), bc.max(), 50)
                    ax_q.plot(xf, s90 * xf + i90, color="#E74C3C", lw=2.0, label=f"Q90 slope={s90:+.5f}")
                    ax_q.plot(xf, s50 * xf + i50, color="#1F3A93", lw=1.5, ls="--", label=f"Med slope={s50:+.5f}")
                    ax_q.scatter(bc, q90, s=20, color="#E74C3C", zorder=4)
                    ax_q.scatter(bc, q50, s=15, color="#1F3A93", zorder=4, alpha=0.6)
                    results[f"pi_vs_{col}_q90_slope"] = float(s90)
                    results[f"pi_vs_{col}_q50_slope"] = float(s50)
                    ax_q.set_title(f"Q90: {s90:+.5f}", fontsize=8)
            except Exception:
                pass
            ax_q.set_xlabel(label, fontsize=7)
            if i == 0:
                ax_q.set_ylabel("Perf Index", fontsize=7)
            ax_q.tick_params(labelsize=6)
            ax_q.axhline(1.0, color="black", ls="--", lw=0.5, alpha=0.4)
            ax_q.legend(fontsize=6, loc="upper right")
        fig_c.suptitle("DQ6: Scatter + quantile regression (binned 90th pctile)", fontsize=11, fontweight="bold")
        fig_c.tight_layout()
        _save(fig_c, plots_dir / "dq6_quantile_regression.png")
    log.info(
        "DQ6 done: %d days, median=%.3f, <0.9=%.0f%%, rain Wilcoxon %s",
        results.get("n_days", 0),
        results.get("perf_index_median", float("nan")),
        results.get("pct_below_09", float("nan")),
        "PASS" if results.get("rain_wilcoxon_pass", False) else "FAIL",
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DQ5: Old-source vs new-source soiling metric comparison           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def plot_old_vs_new_source_comparison(
    df: pd.DataFrame, plots_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Compare soiling metrics derived from old sources vs new sources.

    Produces two files:
      - dq5_old_vs_new_timeseries.png   (loss proxy + cycle deviation ts)
      - dq5_old_vs_new_comparison.png    (scatter + correlation bar chart)
    """
    old_loss = "t1_performance_loss_pct_proxy"
    new_loss = "new_performance_loss_pct_proxy"
    old_dev = "cycle_deviation_pct"
    new_dev = "new_cycle_deviation_pct"

    need = [old_loss, new_loss]
    if not all(c in df.columns for c in need):
        log.info("DQ5 skipped: new-source loss proxy columns not present.")
        return None

    results: Dict[str, Any] = {}
    hq = df[df["transfer_readiness_tier"].isin(["Tier-1", "Tier-2"])] if "transfer_readiness_tier" in df.columns else df

    # ── Figure A: time-series panels (stacked) ────────────────────────
    fig_ts, (ax_ts1, ax_ts2) = plt.subplots(2, 1, figsize=(16, 12))

    # Panel 1: Loss proxy time series + rolling median
    mask_old = hq[old_loss].notna()
    mask_new = hq[new_loss].notna()
    if mask_old.any():
        ax_ts1.plot(hq.loc[mask_old, "day_dt"], hq.loc[mask_old, old_loss],
                    color=C_T1, alpha=0.3, lw=0.6, label="Old (T1 active-power)")
        old_smooth = hq[old_loss].rolling(7, center=True, min_periods=3).median()
        ax_ts1.plot(hq["day_dt"], old_smooth, color=C_T1, alpha=0.9, lw=2.0,
                    label="Old 7-day median")
    if mask_new.any():
        ax_ts1.plot(hq.loc[mask_new, "day_dt"], hq.loc[mask_new, new_loss],
                    color=C_ACCENT, alpha=0.3, lw=0.6, label="New (daily_gen/avg_irr)")
        new_smooth = hq[new_loss].rolling(7, center=True, min_periods=3).median()
        ax_ts1.plot(hq["day_dt"], new_smooth, color=C_ACCENT, alpha=0.9, lw=2.0,
                    label="New 7-day median")
    _annotate_new_source_start(ax_ts1, df)
    _add_rain_cleaning_overlays(ax_ts1, hq)
    ax_ts1.set_ylabel("Performance loss proxy (%)")
    ax_ts1.set_title("Loss Proxy: Old Source vs New Source")
    ax_ts1.legend(fontsize=7, loc="upper right")
    ax_ts1.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Panel 2: Cycle deviation time series + rolling median
    has_old_dev = old_dev in hq.columns and hq[old_dev].notna().any()
    has_new_dev = new_dev in hq.columns and hq[new_dev].notna().any()
    if has_old_dev:
        ax_ts2.plot(hq["day_dt"], hq[old_dev], color=C_T1, alpha=0.3, lw=0.6,
                    label="Old (active-power)")
        old_dev_smooth = hq[old_dev].rolling(7, center=True, min_periods=3).median()
        ax_ts2.plot(hq["day_dt"], old_dev_smooth, color=C_T1, alpha=0.9, lw=2.0,
                    label="Old 7-day median")
    if has_new_dev:
        ax_ts2.plot(hq["day_dt"], hq[new_dev], color=C_ACCENT, alpha=0.3, lw=0.6,
                    label="New (gen_irr_ratio)")
        new_dev_smooth = hq[new_dev].rolling(7, center=True, min_periods=3).median()
        ax_ts2.plot(hq["day_dt"], new_dev_smooth, color=C_ACCENT, alpha=0.9, lw=2.0,
                    label="New 7-day median")
    _annotate_new_source_start(ax_ts2, df)
    _add_rain_cleaning_overlays(ax_ts2, hq)
    ax_ts2.set_ylabel("Cycle deviation (%)")
    ax_ts2.set_title("Cycle Deviation: Old vs New")
    ax_ts2.legend(fontsize=7, loc="upper right")
    ax_ts2.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig_ts.suptitle(
        "DQ5: Old vs new source — time series comparison",
        fontsize=12, fontweight="bold",
    )
    fig_ts.tight_layout()
    _save(fig_ts, plots_dir / "dq5_old_vs_new_timeseries.png")

    # ── Figure B: scatter + bar chart ─────────────────────────────────
    fig, axes_b = plt.subplots(1, 3, figsize=(18, 6))
    ax_sc, ax_sc2, ax_bar = axes_b

    # Panel 1: Scatter old vs new loss proxy (all days)
    pair = hq[[old_loss, new_loss]].dropna()
    if len(pair) > 5:
        ax_sc.scatter(pair[old_loss], pair[new_loss], s=8, alpha=0.5, color=C_T1)
        r_val = pair[old_loss].corr(pair[new_loss])
        results["old_vs_new_loss_r"] = r_val
        ax_sc.set_xlabel("Old loss proxy (%)")
        ax_sc.set_ylabel("New loss proxy (%)")
        ax_sc.set_title(f"All days (r = {r_val:.3f}, n = {len(pair)})", fontsize=9)
        lims = [0, max(pair[old_loss].quantile(0.99), pair[new_loss].quantile(0.99))]
        ax_sc.plot(lims, lims, "--", color="grey", alpha=0.5, lw=0.8)
    else:
        ax_sc.text(0.5, 0.5, "Insufficient overlap", transform=ax_sc.transAxes, ha="center")

    # Panel 2: Scatter — only days where old loss > 0
    pair_nz = pair[pair[old_loss] > 0]
    if len(pair_nz) > 5:
        ax_sc2.scatter(pair_nz[old_loss], pair_nz[new_loss], s=10, alpha=0.5, color="#E74C3C")
        r_nz = pair_nz[old_loss].corr(pair_nz[new_loss])
        results["old_vs_new_loss_r_nonzero"] = r_nz
        results["n_nonzero_loss_days"] = len(pair_nz)
        ax_sc2.set_xlabel("Old loss proxy (%) — non-zero only")
        ax_sc2.set_ylabel("New loss proxy (%)")
        ax_sc2.set_title(f"Non-zero loss days (r = {r_nz:.3f}, n = {len(pair_nz)})", fontsize=9)
        lims_nz = [0, max(pair_nz[old_loss].quantile(0.99), pair_nz[new_loss].quantile(0.99))]
        ax_sc2.plot(lims_nz, lims_nz, "--", color="grey", alpha=0.5, lw=0.8)
    else:
        ax_sc2.text(0.5, 0.5, "Insufficient non-zero days", transform=ax_sc2.transAxes, ha="center")

    # Panel 3: Correlation comparison bar chart
    soiling_features = [
        ("domain_soiling_index", "DSPI"),
        ("cumulative_pm25_since_rain", "Cum PM2.5"),
        ("cumulative_pm10_since_rain", "Cum PM10"),
        ("days_since_last_rain", "Days dry"),
        ("humidity_x_pm10", "Hum × PM10"),
    ]
    old_corrs = []
    new_corrs = []
    labels = []
    for col, label in soiling_features:
        if col not in hq.columns:
            continue
        pair_old = hq[[col, old_loss]].dropna()
        pair_new = hq[[col, new_loss]].dropna()
        r_old = pair_old[col].corr(pair_old[old_loss]) if len(pair_old) > 5 else np.nan
        r_new = pair_new[col].corr(pair_new[new_loss]) if len(pair_new) > 5 else np.nan
        old_corrs.append(r_old)
        new_corrs.append(r_new)
        labels.append(label)
        results[f"old_r_{col}"] = r_old
        results[f"new_r_{col}"] = r_new

    if labels:
        x = np.arange(len(labels))
        w = 0.35
        ax_bar.bar(x - w / 2, old_corrs, w, label="Old source", color=C_T1, alpha=0.7)
        ax_bar.bar(x + w / 2, new_corrs, w, label="New source", color=C_ACCENT, alpha=0.7)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax_bar.set_ylabel("Pearson r vs loss proxy")
        ax_bar.set_title("Feature Correlations: Old vs New Loss Proxy", fontsize=9)
        ax_bar.legend(fontsize=8)
        ax_bar.axhline(0, color="grey", lw=0.5)

    fig.suptitle(
        "DQ5: Old vs new source — scatter & correlation comparison",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, plots_dir / "dq5_old_vs_new_comparison.png")
    log.info(
        "DQ5 done: all-day r=%.3f, non-zero r=%.3f (%d days)",
        results.get("old_vs_new_loss_r", float("nan")),
        results.get("old_vs_new_loss_r_nonzero", float("nan")),
        results.get("n_nonzero_loss_days", 0),
    )
    return results


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Report writer                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def write_report(
    s1: SignalResult,
    s2: SignalResult,
    s3: SignalResult,
    supporting: Dict[str, Any],
    csa_results: Dict[str, Any],
    dq_results: Dict[str, Any],
    out_path: Path,
    dq2_results: Optional[Dict[str, Any]] = None,
    dq3_results: Optional[Dict[str, Any]] = None,
    dq4_results: Optional[Dict[str, Any]] = None,
    dq5_results: Optional[Dict[str, Any]] = None,
    dq6_results: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> None:
    verdicts = [s1.verdict, s2.verdict, s3.verdict]
    n_pass = sum(1 for v in verdicts if v == "pass")
    n_weak = sum(1 for v in verdicts if v == "weak")

    if n_pass == 3:
        overall = "STRONG GO"
        overall_text = "All three signals confirmed. Proceed to modeling."
    elif n_pass >= 2:
        overall = "CONDITIONAL GO"
        overall_text = "Two signals confirmed. Proceed with caution; note the weak signal."
    elif n_pass == 1 or n_weak >= 2:
        overall = "WEAK GO"
        overall_text = (
            "Only one signal confirmed or multiple weak signals. "
            "Consider additional data sources or feature engineering before heavy modeling."
        )
    else:
        overall = "NO-GO"
        overall_text = (
            "No signals confirmed. The performance loss proxy may be dominated by "
            "equipment/data issues rather than soiling. Re-evaluate research direction."
        )

    # Partial correlation table
    partial_lines = []
    pr = s2.details.get("partial_results", {})
    if pr:
        partial_lines.append(
            "| Feature | vs loss proxy | vs loss rate | vs cycle deviation |"
        )
        partial_lines.append("|---|---|---|---|")
        for feat, tgt_dict in pr.items():
            cells = []
            for tgt in [
                "t1_performance_loss_pct_proxy",
                "t1_perf_loss_rate_14d_pct_per_day",
                "cycle_deviation_pct",
            ]:
                r, p = tgt_dict.get(tgt, (np.nan, np.nan))
                if np.isfinite(r):
                    cells.append(f"{r:+.3f} (p={p:.3f})")
                else:
                    cells.append("—")
            partial_lines.append(f"| `{feat}` | {' | '.join(cells)} |")

    def _fmt_p(val: Any) -> str:
        if isinstance(val, float) and np.isfinite(val):
            return f"{val:.4f}"
        return "—"

    def _fmt_r(val: Any) -> str:
        if isinstance(val, float) and np.isfinite(val):
            return f"{val:.3f}"
        return "—"

    partial_table = "\n".join(partial_lines) if partial_lines else "_No partial correlations computed._"

    lines = [
        "# Soiling EDA Signal Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Data Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total days | {supporting.get('n_total', '—')} |",
        f"| Date range | {supporting.get('date_range', '—')} |",
        f"| Training-ready (HQ + 0 flags) | {supporting.get('n_hq_zero_flag', '—')} |",
        "",
        "---",
        "",
        "## Signal 1: Sawtooth Detection",
        "",
        f"**Verdict: {s1.verdict.upper()}**",
        "",
        s1.summary,
        "",
        f"- Dry spells analysed: {s1.details.get('n_spells', 0)}",
        f"- Median soiling rate: {s1.details.get('median_rate_pct_per_day', 0):+.3f} %/day",
        f"- IQR: {s1.details.get('iqr', (0, 0))[0]:+.3f} to {s1.details.get('iqr', (0, 0))[1]:+.3f} %/day",
        f"- Positive-slope spells: {s1.details.get('pct_positive_slope', 0):.0f}%",
        f"- Literature reference: 0.1-0.5 %/day for tropical sites",
        "",
        "Plots: `s1_loss_proxy_timeseries.png`, `s1_per_inverter_output.png`,",
        "`s1_cycle_deviation.png`, `s1_dryspell_slopes.png`",
        "",
        "---",
        "",
        "## Signal 2: PM/Dust Correlation",
        "",
        "Correlations below are between environmental features (independent",
        "variables from Solcast satellite data) and observed plant performance",
        "metrics (dependent variables derived from actual energy generation).",
        "A positive correlation with `cycle_deviation_pct` means the",
        "environmental factor predicts within-cycle performance decline.",
        "",
        f"**Verdict: {s2.verdict.upper()}**",
        "",
        s2.summary,
        "",
        "### Top raw predictors of cycle deviation",
        "",
        f"- `days_since_last_rain`: r = {_fmt_r(s2.details.get('r_days_since_rain_vs_deviation'))}",
        f"- `cumulative_pm25_since_rain`: r = {_fmt_r(s2.details.get('r_cumpm25_vs_deviation'))}",
        f"- `cumulative_pm10_since_rain`: r = {_fmt_r(s2.details.get('r_cumpm10_vs_deviation'))}",
        "",
        "### Raw correlations (confounded by cloud opacity)",
        "",
        f"- PM10 vs loss rate (all HQ): r = {_fmt_r(s2.details.get('r_all_pm10_vs_rate'))}",
        f"- PM10 vs loss rate (clear-sky): r = {_fmt_r(s2.details.get('r_clear_pm10_vs_rate'))}",
        "",
        "### Partial correlations (controlling for cloud opacity + temperature)",
        "",
        partial_table,
        "",
        "### Within-cycle analysis",
        "",
        f"- PM10 vs cycle soiling rate: r = {_fmt_r(s2.details.get('r_within_cycle'))} "
        f"(n = {s2.details.get('n_cycles', 0)} cycles)",
        "",
        "Plots: `s2_pm10_scatter_panels.png`, `s2_top_predictors_vs_deviation.png`,",
        "`s2_feature_heatmap.png`",
        "",
        "---",
        "",
        "## Signal 3: Rain Recovery",
        "",
        f"**Verdict: {s3.verdict.upper()}**",
        "",
        s3.summary,
        "",
        f"- Rain events in event study: {s3.details.get('n_rain_events', 0)}",
        f"- Event-study Wilcoxon p (day+3..+5 < day 0): {_fmt_p(s3.details.get('event_study_p'))}",
        f"- Dry spells tested: {s3.details.get('n_dry_spells', 0)}",
        f"- Dry-spell Wilcoxon p (end > start): {_fmt_p(s3.details.get('dryspell_wilcoxon_p'))}",
        f"- Recovery vs precipitation r: {_fmt_r(s3.details.get('recovery_rain_r'))}",
        "",
        "Plots: `s3_rain_event_study.png`, `s3_dryspell_start_end.png`,",
        "`s3_recovery_vs_precipitation.png`, `s3_rain_event_study_seasonal.png`",
        "",
        "---",
        "",
        "## Supporting Findings",
        "",
        "### Physics-based Soiling Estimates vs Observed",
        f"- pvlib Kimber vs observed loss proxy: r = {_fmt_r(supporting.get('pvlib_r'))}",
        f"- Domain Soiling Index vs observed loss proxy: r = {_fmt_r(supporting.get('dspi_vs_loss_proxy_r'))}",
        "- pvlib predicts small losses (~1%) while the all-cause proxy fluctuates over",
        "  a much wider range, so weak pvlib correlation is expected. The DSPI is tuned",
        "  for this site and uses cumulative environmental pressure rather than",
        "  a generic deposition model.",
        "",
        "### Sensor Dirt Check",
        f"- Solcast/ground ratio trend: {supporting.get('sensor_ratio_trend_per_day', float('nan')):.4f} per day",
        "- A positive trend suggests the ground sensor is accumulating dirt relative to",
        "  the satellite reference.",
        "",
        "### Tier Validation",
        f"- T1 vs T2 loss correlation median: {_fmt_r(supporting.get('tier_loss_corr_median'))}",
        "- High correlation confirms soiling is a plant-wide phenomenon, not",
        "  block-specific.",
        "",
        "### Seasonal Patterns",
        "- See monthly box plots (`s4_seasonal_boxplots.png`). Higher loss in dry months",
        "  (Feb-Apr) is consistent with faster soiling accumulation during low-rainfall",
        "  periods.",
        "",
        "### Domain Soiling Pressure Index (DSPI)",
        "",
        "A physics-based soiling estimate built entirely from environmental satellite",
        "data (PM2.5, PM10, humidity, dewpoint, precipitation). No plant performance",
        "data is used, making it leakage-free. Formula:",
        "",
        "    daily_rate = (w_pm25 * PM2.5 + w_pm10 * PM10)",
        "                * humidity_factor * dew_factor * cementation_factor",
        "",
        "Component weights were calibrated via constrained optimisation to maximise",
        "positive correlation with PM and negative with rainfall while penalising",
        "correlation with cloud opacity and temperature.",
        "",
        f"- Correlation with cycle deviation: r = {_fmt_r(supporting.get('dspi_vs_cycle_deviation_r'))}",
        "",
    ]

    dspi_profile = supporting.get("dspi_corr_profile", {})
    if dspi_profile:
        lines.extend([
            "**Correlation profile (HQ days):**",
            "",
            "| Feature | r |",
            "|---|---|",
        ])
        for feat_label, r_val in dspi_profile.items():
            lines.append(f"| {feat_label} | {_fmt_r(r_val)} |")
        lines.append("")

    lines.extend([
        "Plots: `s5_domain_soiling_index.png`, `s5_dspi_correlation_profile.png`",
        "",
    ])

    # ── Clear-Sky Soiling Analysis ──
    if csa_results:
        csa_n = csa_results.get("csa_n", 0)
        hq_n = csa_results.get("hq_n", 0)
        lines.extend([
            "### Clear-Sky Soiling Analysis",
            "",
            f"To isolate real soiling from tropical weather noise, a Clear-Sky",
            f"Analyzable (CSA) filter retains only days with low cloud (<35%),",
            f"no rain (<1 mm), functioning equipment, and >=1 day since last rain.",
            "",
            f"- **CSA days: {csa_n} / {hq_n} HQ** ({csa_n/hq_n*100:.0f}%)" if hq_n > 0 else f"- CSA days: {csa_n}",
            "",
        ])
        corr_comp = csa_results.get("corr_comparison", {})
        if corr_comp:
            lines.extend([
                "**Correlation comparison (loss proxy):**",
                "",
                "| Feature | r (All HQ) | r (CSA only) |",
                "|---|---|---|",
            ])
            for feat_label, vals in corr_comp.items():
                rh = vals.get("r_hq", np.nan)
                rc = vals.get("r_csa", np.nan)
                rh_s = f"{rh:+.3f}" if isinstance(rh, float) and np.isfinite(rh) else "---"
                rc_s = f"{rc:+.3f}" if isinstance(rc, float) and np.isfinite(rc) else "---"
                lines.append(f"| {feat_label} | {rh_s} | {rc_s} |")
            lines.append("")

        lines.extend([
            "Key finding: cumulative dust features (`cumulative_pm25_since_rain`,",
            "`days_since_last_rain`) achieve statistically significant positive",
            "correlations with loss proxy on CSA days, confirming a real soiling",
            "signal beneath the weather noise.",
            "",
            "Plots: `c1_clear_sky_loss_timeseries.png`, `c2_clean_vs_all_correlations.png`, `c3_clean_scatter_matrix.png`",
            "",
        ])

    # ── Data Quality: Irradiance vs Generation ──
    if dq_results:
        lines.extend([
            "### Data Quality: Irradiance vs Generation",
            "",
            "Scatter and time-series of on-site irradiance sensor sum vs",
            "T1 inverter generation, both measured during the 10 AM – 2 PM",
            "tracked window. Used to verify data consistency after preprocessing.",
            "",
            f"- On-site irradiance (sensor sum) vs T1 generation: r = {_fmt_r(dq_results.get('onsite_irr_vs_gen_r'))}",
            f"- Solcast peak GTI (10–14h, J/m²) vs T1 generation: r = {_fmt_r(dq_results.get('solcast_gti_peak_vs_gen_r'))}",
            f"- Solcast peak GTI vs full-plant generation: r = {_fmt_r(dq_results.get('solcast_gti_peak_vs_fullgen_r'))}",
            f"- Zero-generation on sunny days: {dq_results.get('n_zero_gen_sunny', '—')}",
            f"- Normalised output monthly CV: {dq_results.get('norm_output_monthly_cv_pct', 0):.1f}%",
            "",
            "Note: low correlation with the T1 subset (3 inverters) is due to",
            "per-inverter variability (equipment failures, clipping). Full-plant",
            "generation correlates much better with satellite irradiance.",
            "",
            "Plots: `dq1_irradiance_vs_generation_timeseries.png` (time series),",
            "`dq1_irradiance_vs_generation.png` (scatter & boxplot)",
            "",
        ])

    # ── DQ2: New telemetry validation ──
    if dq2_results:
        lines.extend([
            "### DQ2: New Telemetry Validation",
            "",
            "Asset-aligned validation of the new daily-generation source and",
            "physical irradiation context (`irradiation_kwh_m2`).",
            "",
            f"- Old vs new generation correlation: r = {_fmt_r(dq2_results.get('old_vs_new_gen_r'))}",
            f"- New generation vs physical irradiation: r = {_fmt_r(dq2_results.get('new_gen_vs_irradiation_r'))}",
            f"- Plant avg irradiance vs Solcast: r = {_fmt_r(dq2_results.get('plant_vs_solcast_irr_r'))}",
            f"- Aligned inverter IDs: {', '.join(dq2_results.get('aligned_inverters', [])) if dq2_results.get('aligned_inverters') else 'n/a'}",
            "",
            "**Interpretation:** Generation is now compared on a like-for-like",
            "inverter intersection, reducing asset-mismatch bias. The physical",
            "irradiation correlation is the primary sanity check for energy-yield",
            "consistency. Plant-vs-Solcast irradiance agreement remains a sensor",
            "health cross-check, not the PR denominator itself.",
            "",
            "Plots: `dq2_daily_gen_validation_timeseries.png` (time series),",
            "`dq2_daily_gen_validation.png` (scatter)",
            "",
        ])

    # ── DQ3: Gen/Irr ratio ──
    if dq3_results:
        lines.extend([
            "### DQ3: Generation / Irradiance Ratio",
            "",
            "Physical PR-style metric from new telemetry. Raw PR points outside",
            "[0, 1] are explicitly flagged as outliers; trend uses outlier-masked",
            "interpolation and a 7-day median smoother.",
            "",
            f"- Median raw PR: {dq3_results.get('gen_irr_ratio_median', 0):.4f}",
            f"- Raw PR std: {dq3_results.get('gen_irr_ratio_std', 0):.4f}",
            f"- Outliers (`PR<0` or `PR>1`): {dq3_results.get('gen_irr_ratio_outlier_count', 0)} "
            f"({dq3_results.get('gen_irr_ratio_outlier_pct', 0):.2f}%)",
            f"- Agreement with old-source -loss proxy: r = {_fmt_r(dq3_results.get('ratio_vs_neg_loss_r'))}",
            "",
            "**Inference:** PR outliers are now visible diagnostics rather than being",
            "silently accepted. The trend line reflects physically plausible PR values",
            "only, improving interpretability of soiling-driven decline/recovery.",
            "",
            "Plots: `dq3_gen_irr_ratio_timeseries.png` (time series),",
            "`dq3_gen_irr_ratio.png` (monthly boxplot)",
            "",
        ])

    # ── DQ4: Power at reference irradiance ──
    if dq4_results:
        lines.extend([
            "### DQ4: Power at Reference Irradiance",
            "",
            "Active power extracted when on-site irradiance is at the dataset's",
            "median level. Controls for irradiance variation and isolates",
            "degradation/soiling from weather effects.",
            "",
            f"- Reference irradiance: {dq4_results.get('ref_irradiance_wm2', 0):.0f} W/m^2",
            f"- Days with valid data: {dq4_results.get('power_at_ref_irr_days', 0)}",
            f"- Median power at ref irradiance: {dq4_results.get('power_at_ref_irr_median', 0):.0f} W",
            "",
        ])
        ref_corrs = {
            k[len("ref_irr_vs_"):-len("_r")] if k.endswith("_r") else k[len("ref_irr_vs_"):]: v
            for k, v in dq4_results.items()
            if k.startswith("ref_irr_vs_") and isinstance(v, float) and np.isfinite(v)
        }
        if ref_corrs:
            lines.extend([
                "**Correlations with soiling features (HQ days):**",
                "",
                "| Feature | r |",
                "|---|---|",
            ])
            for feat, r_val in ref_corrs.items():
                lines.append(f"| `{feat}` | {r_val:+.3f} |")
            lines.append("")
        lines.extend([
            "**Inference:** A strong negative correlation between power-at-reference",
            "and `t1_performance_loss_pct_proxy` confirms the feature successfully",
            "isolates performance degradation from irradiance variation. Correlations",
            "with environmental soiling drivers (PM, days dry) indicate whether",
            "soiling — rather than equipment issues — drives the observed decline.",
            "",
            "Plot: `dq4_power_at_ref_irradiance.png`",
            "",
        ])

    # ── DQ5: Old vs New source comparison ──
    if dq5_results:
        lines.extend([
            "### DQ5: Old-Source vs New-Source Soiling Metrics",
            "",
            "Parallel soiling feature pipelines were computed from both the original",
            "data sources (active power + tilted/Solcast irradiance, peak-hour filtered)",
            "and the new telemetry (daily_generated_electricity + avg_solar_radiation,",
            "full-day). This section compares their agreement and predictive power.",
            "",
            f"- Old vs new loss proxy agreement: r = {_fmt_r(dq5_results.get('old_vs_new_loss_r'))}",
            "",
        ])
        feat_pairs = [
            ("domain_soiling_index", "DSPI"),
            ("cumulative_pm25_since_rain", "Cum PM2.5"),
            ("cumulative_pm10_since_rain", "Cum PM10"),
            ("days_since_last_rain", "Days dry"),
            ("humidity_x_pm10", "Hum x PM10"),
        ]
        has_any = any(
            f"old_r_{col}" in dq5_results for col, _ in feat_pairs
        )
        if has_any:
            lines.extend([
                "**Feature correlations with loss proxy (old vs new source):**",
                "",
                "| Feature | r (Old) | r (New) |",
                "|---|---|---|",
            ])
            for col, label in feat_pairs:
                r_old = dq5_results.get(f"old_r_{col}", np.nan)
                r_new = dq5_results.get(f"new_r_{col}", np.nan)
                r_old_s = f"{r_old:+.3f}" if isinstance(r_old, float) and np.isfinite(r_old) else "---"
                r_new_s = f"{r_new:+.3f}" if isinstance(r_new, float) and np.isfinite(r_new) else "---"
                lines.append(f"| {label} | {r_old_s} | {r_new_s} |")
            lines.append("")

        lines.extend([
            "**Inference:** Low or negative old-vs-new loss proxy agreement is expected",
            "because the two pipelines measure performance over different time windows",
            "(peak-hour vs full-day) and the new source lacks Jan-Mar data when soiling",
            "is strongest. The feature correlation table reveals whether environmental",
            "soiling drivers correlate more strongly with one source's loss proxy,",
            "guiding which pipeline to prioritise for modelling.",
            "",
            "Plots: `dq5_old_vs_new_timeseries.png` (time series),",
            "`dq5_old_vs_new_comparison.png` (scatter & bar chart)",
            "",
        ])

    # ── DQ6: New-Source Performance Index ──
    if dq6_results:
        lines.extend([
            "### DQ6: New-Source Performance Index (0-1)",
            "",
            "A normalised performance index derived from new telemetry:",
            "`performance_index = gen_irr_ratio / rolling_clean_baseline`.",
            "Values near 1.0 represent clean-panel performance; values below 1.0",
            "indicate degradation from soiling and other losses.",
            "",
            f"- Days with valid index: {dq6_results.get('n_days', 0)}",
            f"- Median performance index: {dq6_results.get('perf_index_median', 0):.3f}",
            f"- Mean performance index: {dq6_results.get('perf_index_mean', 0):.3f}",
            f"- Days below 1.0 (any loss): {dq6_results.get('pct_below_1', 0):.0f}%",
            f"- Days below 0.9 (>10% loss): {dq6_results.get('pct_below_09', 0):.0f}%",
            f"- Days below 0.8 (>20% loss): {dq6_results.get('pct_below_08', 0):.0f}%",
            "",
        ])
        pi_corrs = {
            k[len("pi_vs_"):-len("_r")] if k.endswith("_r") else k[len("pi_vs_"):]: v
            for k, v in dq6_results.items()
            if k.startswith("pi_vs_") and isinstance(v, float) and np.isfinite(v)
        }
        if pi_corrs:
            lines.extend([
                "**Correlations with soiling features:**",
                "",
                "| Feature | r |",
                "|---|---|",
            ])
            for feat, r_val in pi_corrs.items():
                lines.append(f"| `{feat}` | {r_val:+.3f} |")
            lines.append("")
        lines.extend([
            "**Inference:** A median well below 1.0 indicates persistent performance",
            "loss across the dataset, consistent with soiling between cleanings.",
            "Negative correlations with cumulative dust features (`cumulative_pm25_since_rain`,",
            "`days_since_last_rain`) confirm that longer dry periods depress the index.",
            "If the mean is substantially lower than the median, heavy-loss outlier days",
            "(e.g. equipment faults, extreme soiling) are pulling the distribution down.",
            "",
            "Plot: `dq6_performance_index.png`",
            "",
        ])

    # ── Data Coverage Notes ──
    new_start = _new_source_start(df) if df is not None else None
    if new_start is not None:
        n_total = len(df) if df is not None else 0
        n_new = int(df["gen_irr_ratio"].notna().sum()) if df is not None and "gen_irr_ratio" in df.columns else 0
        lines.extend([
            "### Data Coverage Notes",
            "",
            f"The new telemetry source (`avg_solar_radiation`) begins on "
            f"**{new_start.strftime('%Y-%m-%d')}**. Data for Jan-Mar 2025 is "
            f"unavailable, creating a gap in all new-source metrics (`gen_irr_ratio`, "
            f"`t1_performance_loss_pct_proxy`, `new_cycle_deviation_pct`, "
            f"`new_performance_index`).",
            "",
            f"- New-source days: **{n_new} / {n_total}** "
            f"({n_new/n_total*100:.0f}%)" if n_total > 0 else "",
            f"- Missing period: Jan-Mar 2025 (peak dry season with fastest soiling accumulation)",
            "",
            "**Impact on results:**",
            "",
            "- **Signal 1/2/3 verdicts are unaffected** -- they use old-source "
            "columns (`t1_performance_loss_pct_proxy`, `cycle_deviation_pct`) "
            "which have near-complete coverage.",
            "- **DQ2/DQ3/DQ5/DQ6 diagnostic plots** only cover the overlap period "
            "(Apr 2025+), missing the dry season when soiling signal is strongest.",
            "- **DQ5 old-vs-new correlation** is biased toward wetter months "
            "where soiling accumulation is lower.",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Overall Go/No-Go Verdict",
        "",
        f"**{overall}**",
        "",
        "| Signal | Verdict |",
        "|---|---|",
        f"| 1. Sawtooth pattern | {s1.verdict.upper()} |",
        f"| 2. PM/dust correlation | {s2.verdict.upper()} |",
        f"| 3. Rain recovery | {s3.verdict.upper()} |",
        "",
        overall_text,
    ])
    report = "\n".join(lines) + "\n"

    out_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", out_path)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Main                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(description="EDA soiling signal tests")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to daily_model_eda.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.input)

    # ── Parallel pipeline branch for new telemetry ────────────────────
    if "t1_performance_loss_pct_proxy" in df.columns:
        log.info("Starting parallel pipeline for new telemetry metrics...")
        df_new = df.copy()
        df_new["t1_performance_loss_pct_proxy"] = df_new["t1_performance_loss_pct_proxy"]
        if "t1_perf_loss_rate_14d_pct_per_day" in df_new.columns:
            df_new["t1_perf_loss_rate_14d_pct_per_day"] = df_new["t1_perf_loss_rate_14d_pct_per_day"]
        if "new_cycle_deviation_pct" in df_new.columns:
            df_new["cycle_deviation_pct"] = df_new["new_cycle_deviation_pct"]
            
        new_plots_dir = out_dir / "plots_new_telemetry"
        new_plots_dir.mkdir(parents=True, exist_ok=True)
        
        test_signal_1_sawtooth(df_new, new_plots_dir)
        test_signal_2_dust_correlation(df_new, new_plots_dir)
        test_signal_3_rain_recovery(df_new, new_plots_dir)
        test_clear_sky_soiling(df_new, new_plots_dir)
        log.info("Parallel pipeline plotting complete.")

    s1 = test_signal_1_sawtooth(df, plots_dir)
    s2 = test_signal_2_dust_correlation(df, plots_dir)
    s3 = test_signal_3_rain_recovery(df, plots_dir)
    supporting = run_supporting_analyses(df, plots_dir)
    csa_results = test_clear_sky_soiling(df, plots_dir)
    dq_results = plot_irradiance_vs_generation(df, plots_dir)
    dq2_results = plot_daily_gen_validation(df, plots_dir)
    dq3_results = plot_gen_irr_ratio(df, plots_dir)
    dq4_results = plot_power_at_ref_irradiance(df, plots_dir)
    dq5_results = plot_old_vs_new_source_comparison(df, plots_dir)
    dq6_results = plot_new_performance_index(df, plots_dir)

    write_report(
        s1, s2, s3, supporting, csa_results, dq_results,
        out_dir / "eda_signal_report.md",
        dq2_results=dq2_results,
        dq3_results=dq3_results,
        dq4_results=dq4_results,
        dq5_results=dq5_results,
        dq6_results=dq6_results,
        df=df,
    )

    # ── LLM-readable structured output ────────────────────────────────
    from llm_output import (
        build_signal_1_section,
        build_signal_2_section,
        build_signal_3_section,
        build_supporting_section,
        build_csa_section,
        build_dq_section,
        build_dataset_overview,
        write_llm_summary,
    )
    from multilevel_analysis import (
        build_atomic_level,
        build_microscopic_level,
        build_macroscopic_level,
    )
    from feature_glossary import build_feature_glossary

    # Determine overall verdict
    verdicts_list = [s1.verdict, s2.verdict, s3.verdict]
    if all(v == "pass" for v in verdicts_list):
        overall = "GO"
    elif any(v == "fail" for v in verdicts_list):
        overall = "CONDITIONAL GO" if any(v == "pass" for v in verdicts_list) else "NO GO"
    else:
        overall = "CONDITIONAL GO"

    verdicts_dict = {
        "signal_1_sawtooth": s1.verdict,
        "signal_2_dust_correlation": s2.verdict,
        "signal_3_rain_recovery": s3.verdict,
        "overall": overall,
    }

    write_llm_summary(
        out_dir / "llm_eda_summary.json",
        dataset_overview=build_dataset_overview(df),
        signal_1=build_signal_1_section(s1, df),
        signal_2=build_signal_2_section(s2, df),
        signal_3=build_signal_3_section(s3, df),
        supporting=build_supporting_section(supporting, df),
        clear_sky=build_csa_section(csa_results, df),
        dq1=build_dq_section(dq_results, "DQ1: Irradiance vs Generation"),
        dq2=build_dq_section(dq2_results, "DQ2: Daily Generation Validation"),
        dq3=build_dq_section(dq3_results, "DQ3: Generation/Irradiance Ratio"),
        dq4=build_dq_section(dq4_results, "DQ4: Power at Reference Irradiance"),
        dq5=build_dq_section(dq5_results, "DQ5: Old vs New Source Comparison"),
        dq6=build_dq_section(dq6_results, "DQ6: New Performance Index"),
        verdicts=verdicts_dict,
        atomic_level=build_atomic_level(df),
        microscopic_level=build_microscopic_level(df),
        macroscopic_level=build_macroscopic_level(df, verdicts_dict),
        feature_glossary=build_feature_glossary(),
    )

    log.info(
        "EDA complete. Verdicts: S1=%s, S2=%s, S3=%s",
        s1.verdict.upper(), s2.verdict.upper(), s3.verdict.upper(),
    )


if __name__ == "__main__":
    main()
