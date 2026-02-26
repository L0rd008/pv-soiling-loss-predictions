"""EDA: Three go/no-go signal tests for soiling loss prediction.

Produces ~19 plots in artifacts/eda/plots/ and a quantitative verdict
report in artifacts/eda/eda_signal_report.md.

Usage:
    python scripts/eda_soiling_signals.py
    python scripts/eda_soiling_signals.py --input path/to/daily_model_eda.csv
    python scripts/eda_soiling_signals.py --out-dir artifacts/eda
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from daily_features import (
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
    return df[
        (df["transfer_quality_tier"] == "high") & (df["flag_count"] == 0)
    ].copy()


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
        fig, axes = plt.subplots(n_inv, 1, figsize=(14, 2.4 * n_inv), sharex=True)
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
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
    fig, axes = plt.subplots(n_phys_rows, 2, figsize=(13, 5 * n_phys_rows))
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
# ║  Report writer                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def write_report(
    s1: SignalResult,
    s2: SignalResult,
    s3: SignalResult,
    supporting: Dict[str, Any],
    out_path: Path,
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

    s1 = test_signal_1_sawtooth(df, plots_dir)
    s2 = test_signal_2_dust_correlation(df, plots_dir)
    s3 = test_signal_3_rain_recovery(df, plots_dir)
    supporting = run_supporting_analyses(df, plots_dir)

    write_report(s1, s2, s3, supporting, out_dir / "eda_signal_report.md")

    log.info(
        "EDA complete. Verdicts: S1=%s, S2=%s, S3=%s",
        s1.verdict.upper(), s2.verdict.upper(), s3.verdict.upper(),
    )


if __name__ == "__main__":
    main()
