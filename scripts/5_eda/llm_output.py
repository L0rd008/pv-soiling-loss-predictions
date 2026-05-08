"""Utilities for writing LLM-readable structured output from EDA analyses.

Provides comprehensive statistical analysis far beyond what plots convey
to humans.  An LLM can digest raw daily series, autocorrelation structures,
cross-lag correlations, stationarity tests, effect sizes, regression
diagnostics, change-point indicators, and distributional shape measures
— all of which are invisible in a quick visual scan.

Output target: a single ``llm_eda_summary.json`` consumed by Claude.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  JSON encoder for numpy / pandas types
# ═══════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, arrays, and pandas types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _clean_array(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if isinstance(obj, pd.Series):
            return _clean_array(obj.values)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def _clean_array(arr: np.ndarray) -> list:
    """Convert ndarray to list, replacing NaN/Inf with None."""
    result = []
    for val in arr.flat:
        if isinstance(val, (float, np.floating)):
            result.append(None if (np.isnan(val) or np.isinf(val)) else round(float(val), 6))
        elif isinstance(val, (int, np.integer)):
            result.append(int(val))
        elif isinstance(val, (bool, np.bool_)):
            result.append(bool(val))
        else:
            result.append(str(val))
    return result


def _sf(val: Any) -> Optional[float]:
    """Safe float: convert to float, None for NaN/Inf."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _clean_dict(d: Any) -> Any:
    """Recursively sanitise a dict tree for JSON serialisation."""
    if isinstance(d, dict):
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean_dict(item) for item in d]
    if isinstance(d, (float, np.floating)):
        return None if (np.isnan(d) or np.isinf(d)) else float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, np.ndarray):
        return _clean_array(d)
    return d


# ═══════════════════════════════════════════════════════════════════════
#  Comprehensive Series Statistics
# ═══════════════════════════════════════════════════════════════════════

def series_stats(s: pd.Series, name: Optional[str] = None) -> Dict[str, Any]:
    """Exhaustive descriptive statistics for a single numeric Series.

    Goes far beyond what a histogram / box-plot conveys:
      - Full percentile ladder (p1 through p99)
      - Variance, CV, SEM, 95 % confidence interval for the mean
      - MAD (mean absolute deviation)
      - Skewness, kurtosis, Jarque–Bera normality test
      - Outlier counts (IQR rule and 3-sigma rule)
      - Data-quality: % zeros, % negative, longest consecutive-NaN gap
    """
    clean = s.dropna()
    if clean.empty:
        return {"name": name, "n": 0, "n_null": int(s.isna().sum())}

    n = len(clean)
    mean_val = float(clean.mean())
    std_val = float(clean.std())
    var_val = float(clean.var())
    sem_val = std_val / np.sqrt(n) if n > 0 else None

    # Confidence interval (95 %)
    if n >= 2 and sem_val and sem_val > 0:
        t_crit = sp_stats.t.ppf(0.975, df=n - 1)
        ci_lo = mean_val - t_crit * sem_val
        ci_hi = mean_val + t_crit * sem_val
    else:
        ci_lo = ci_hi = None

    # IQR and outlier detection
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    iqr_lower_fence = q1 - 1.5 * iqr
    iqr_upper_fence = q3 + 1.5 * iqr
    n_outliers_iqr = int(((clean < iqr_lower_fence) | (clean > iqr_upper_fence)).sum())

    three_sigma_lo = mean_val - 3 * std_val
    three_sigma_hi = mean_val + 3 * std_val
    n_outliers_3sigma = int(((clean < three_sigma_lo) | (clean > three_sigma_hi)).sum())

    # Normality test (Jarque-Bera; works with n >= 8)
    if n >= 8:
        jb_stat, jb_p = sp_stats.jarque_bera(clean.values)
    else:
        jb_stat = jb_p = None

    # Shapiro-Wilk (up to 5000 samples)
    if 3 <= n <= 5000:
        sw_stat, sw_p = sp_stats.shapiro(clean.values)
    else:
        sw_stat = sw_p = None

    # Mean absolute deviation
    mad = float(np.mean(np.abs(clean.values - mean_val)))

    # Longest consecutive NaN gap
    if s.isna().any():
        is_null = s.isna().values
        gaps = []
        current = 0
        for v in is_null:
            if v:
                current += 1
            else:
                if current > 0:
                    gaps.append(current)
                current = 0
        if current > 0:
            gaps.append(current)
        longest_null_gap = max(gaps) if gaps else 0
    else:
        longest_null_gap = 0

    return {
        "name": name or s.name,
        "n": n,
        "n_null": int(s.isna().sum()),
        "pct_null": round(s.isna().mean() * 100, 2),
        "pct_zeros": round((clean == 0).mean() * 100, 2),
        "pct_negative": round((clean < 0).mean() * 100, 2),
        "longest_null_gap_days": longest_null_gap,
        # Central tendency
        "mean": _sf(mean_val),
        "median": _sf(clean.median()),
        "trimmed_mean_10pct": _sf(sp_stats.trim_mean(clean.values, 0.1)),
        # Dispersion
        "std": _sf(std_val),
        "variance": _sf(var_val),
        "sem": _sf(sem_val),
        "ci_95_lower": _sf(ci_lo),
        "ci_95_upper": _sf(ci_hi),
        "mad": _sf(mad),
        "cv_pct": _sf((std_val / abs(mean_val) * 100) if mean_val != 0 else None),
        "iqr": _sf(iqr),
        # Range
        "min": _sf(clean.min()),
        "max": _sf(clean.max()),
        "range": _sf(clean.max() - clean.min()),
        # Full percentile ladder
        "p1": _sf(clean.quantile(0.01)),
        "p5": _sf(clean.quantile(0.05)),
        "p10": _sf(clean.quantile(0.10)),
        "p25": _sf(q1),
        "p50": _sf(clean.median()),
        "p75": _sf(q3),
        "p90": _sf(clean.quantile(0.90)),
        "p95": _sf(clean.quantile(0.95)),
        "p99": _sf(clean.quantile(0.99)),
        # Shape
        "skewness": _sf(clean.skew()),
        "kurtosis": _sf(clean.kurtosis()),  # excess kurtosis
        "jarque_bera_stat": _sf(jb_stat),
        "jarque_bera_p": _sf(jb_p),
        "shapiro_wilk_stat": _sf(sw_stat),
        "shapiro_wilk_p": _sf(sw_p),
        # Outliers
        "iqr_lower_fence": _sf(iqr_lower_fence),
        "iqr_upper_fence": _sf(iqr_upper_fence),
        "n_outliers_iqr": n_outliers_iqr,
        "n_outliers_3sigma": n_outliers_3sigma,
        "pct_outliers_iqr": round(n_outliers_iqr / n * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Time-Series Analysis Helpers
# ═══════════════════════════════════════════════════════════════════════

def _autocorrelation(s: pd.Series, max_lag: int = 30) -> Dict[str, Optional[float]]:
    """Compute autocorrelation at multiple lags."""
    clean = s.dropna()
    if len(clean) < max_lag + 5:
        return {}
    result = {}
    for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
        if lag <= max_lag and lag < len(clean):
            result[f"lag_{lag}"] = _sf(clean.autocorr(lag=lag))
    return result


def _linear_trend(dates: pd.Series, values: pd.Series) -> Dict[str, Any]:
    """Fit a linear trend and return slope, R², p-value."""
    valid = values.notna()
    if valid.sum() < 10:
        return {"slope_per_day": None, "r_squared": None, "p_value": None, "n": 0}
    x = (dates[valid] - dates[valid].min()).dt.days.values.astype(float)
    y = values[valid].values
    slope, intercept, r_val, p_val, se = sp_stats.linregress(x, y)
    return {
        "slope_per_day": _sf(slope),
        "slope_per_month": _sf(slope * 30),
        "intercept": _sf(intercept),
        "r_squared": _sf(r_val ** 2),
        "p_value": _sf(p_val),
        "std_error": _sf(se),
        "n": int(valid.sum()),
    }


def _stationarity_test(s: pd.Series) -> Dict[str, Any]:
    """Augmented Dickey-Fuller test for stationarity."""
    clean = s.dropna()
    if len(clean) < 20:
        return {"adf_statistic": None, "adf_p_value": None, "is_stationary_5pct": None}
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(clean.values, autolag="AIC")
        return {
            "adf_statistic": _sf(result[0]),
            "adf_p_value": _sf(result[1]),
            "lags_used": int(result[2]),
            "n_observations": int(result[3]),
            "is_stationary_5pct": bool(result[1] < 0.05),
            "critical_values": {k: _sf(v) for k, v in result[4].items()},
        }
    except ImportError:
        return {"adf_statistic": None, "note": "statsmodels not installed"}
    except Exception:
        return {"adf_statistic": None, "note": "test failed"}


def _rolling_stats(s: pd.Series, windows: List[int] = None) -> Dict[str, Any]:
    """Rolling mean and std at multiple windows for trend visibility."""
    if windows is None:
        windows = [7, 14, 30]
    result = {}
    clean = s.dropna()
    if len(clean) < 10:
        return result
    for w in windows:
        if w >= len(clean):
            continue
        rm = clean.rolling(w, center=True, min_periods=max(3, w // 2)).mean()
        rs = clean.rolling(w, center=True, min_periods=max(3, w // 2)).std()
        result[f"rolling_{w}d_mean_stats"] = {
            "mean": _sf(rm.mean()),
            "std_of_rolling_mean": _sf(rm.std()),
            "range": _sf(rm.max() - rm.min()),
        }
        result[f"rolling_{w}d_volatility_mean"] = _sf(rs.mean())
    return result


def _cross_lag_correlations(
    x: pd.Series, y: pd.Series, max_lag: int = 7,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Cross-correlation at multiple lags (x leading y)."""
    result = {}
    both_valid = x.notna() & y.notna()
    if both_valid.sum() < 20:
        return result
    x_clean = x[both_valid].values
    y_clean = y[both_valid].values
    for lag in range(-max_lag, max_lag + 1):
        if abs(lag) >= len(x_clean) - 5:
            continue
        if lag >= 0:
            xi = x_clean[:len(x_clean) - lag]
            yi = y_clean[lag:]
        else:
            xi = x_clean[-lag:]
            yi = y_clean[:len(y_clean) + lag]
        if len(xi) > 5:
            r, p = sp_stats.pearsonr(xi, yi)
            result[f"lag_{lag:+d}"] = {"r": _sf(r), "p": _sf(p)}
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Comprehensive Correlation Helpers
# ═══════════════════════════════════════════════════════════════════════

def comprehensive_pairwise(
    df: pd.DataFrame, col_x: str, col_y: str,
) -> Dict[str, Any]:
    """Full correlation suite between two columns."""
    pair = df[[col_x, col_y]].dropna()
    n = len(pair)
    if n < 5:
        return {"n": n, "insufficient_data": True}

    x, y = pair[col_x].values, pair[col_y].values

    # Pearson
    r_p, p_p = sp_stats.pearsonr(x, y)
    # Spearman (rank)
    r_s, p_s = sp_stats.spearmanr(x, y)
    # Kendall tau
    r_k, p_k = sp_stats.kendalltau(x, y)

    # Linear regression
    slope, intercept, _, _, se = sp_stats.linregress(x, y)
    y_pred = intercept + slope * x
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else None
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(residuals))

    return {
        "n": n,
        "pearson_r": _sf(r_p),
        "pearson_p": _sf(p_p),
        "spearman_rho": _sf(r_s),
        "spearman_p": _sf(p_s),
        "kendall_tau": _sf(r_k),
        "kendall_p": _sf(p_k),
        "linear_regression": {
            "slope": _sf(slope),
            "intercept": _sf(intercept),
            "std_error": _sf(se),
            "r_squared": _sf(r_squared),
            "rmse": _sf(rmse),
            "mae": _sf(mae),
        },
    }


def corr_matrix_to_dict(
    df: pd.DataFrame, columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Full correlation matrix (Pearson, Spearman, or Kendall)."""
    cols = [c for c in (columns or df.columns) if c in df.columns]
    corr = df[cols].corr(method=method)
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for row_label in corr.index:
        result[row_label] = {}
        for col_label in corr.columns:
            result[row_label][col_label] = _sf(corr.loc[row_label, col_label])
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Group Comparison Helpers
# ═══════════════════════════════════════════════════════════════════════

def _group_comparison(
    group_a: pd.Series, group_b: pd.Series,
    label_a: str = "A", label_b: str = "B",
) -> Dict[str, Any]:
    """Compare two groups: means, medians, effect size, stat tests."""
    a = group_a.dropna().values
    b = group_b.dropna().values
    if len(a) < 3 or len(b) < 3:
        return {"insufficient_data": True, "n_a": len(a), "n_b": len(b)}

    # Means and medians
    mean_a, mean_b = a.mean(), b.mean()
    med_a, med_b = np.median(a), np.median(b)

    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2)
                         / (len(a) + len(b) - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else None

    # Welch's t-test
    t_stat, t_p = sp_stats.ttest_ind(a, b, equal_var=False)

    # Mann-Whitney U (non-parametric)
    u_stat, u_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")

    # Kolmogorov-Smirnov (distribution comparison)
    ks_stat, ks_p = sp_stats.ks_2samp(a, b)

    return {
        f"n_{label_a}": len(a),
        f"n_{label_b}": len(b),
        f"mean_{label_a}": _sf(mean_a),
        f"mean_{label_b}": _sf(mean_b),
        f"median_{label_a}": _sf(med_a),
        f"median_{label_b}": _sf(med_b),
        f"std_{label_a}": _sf(a.std()),
        f"std_{label_b}": _sf(b.std()),
        "mean_difference": _sf(mean_a - mean_b),
        "cohens_d": _sf(cohens_d),
        "cohens_d_interpretation": (
            "negligible" if abs(cohens_d or 0) < 0.2 else
            "small" if abs(cohens_d or 0) < 0.5 else
            "medium" if abs(cohens_d or 0) < 0.8 else "large"
        ),
        "welch_t_stat": _sf(t_stat),
        "welch_t_p": _sf(t_p),
        "mann_whitney_u": _sf(u_stat),
        "mann_whitney_p": _sf(u_p),
        "ks_statistic": _sf(ks_stat),
        "ks_p": _sf(ks_p),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Raw Daily Data Extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_daily_series(
    df: pd.DataFrame, columns: List[str], date_col: str = "day_dt",
) -> Dict[str, Any]:
    """Extract raw daily values for key columns — an LLM can process
    hundreds of data points that a human cannot parse visually."""
    cols_present = [c for c in columns if c in df.columns]
    daily = {"dates": [str(d.date()) for d in df[date_col]]}
    for col in cols_present:
        vals = df[col].values
        daily[col] = [
            None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else
            round(float(v), 4) if isinstance(v, (float, np.floating)) else
            int(v) if isinstance(v, (int, np.integer)) else
            bool(v) if isinstance(v, (bool, np.bool_)) else str(v)
            for v in vals
        ]
    return daily


# ═══════════════════════════════════════════════════════════════════════
#  Section Builders
# ═══════════════════════════════════════════════════════════════════════

def _hq(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["transfer_quality_tier"] == "high") & (df["flag_count"] == 0)].copy()


def build_signal_1_section(result: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Signal 1: Sawtooth — comprehensive LLM output."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    dev_col = "cycle_deviation_pct"

    section = {
        "signal_name": result.name,
        "verdict": result.verdict,
        "summary": result.summary,
        "details": _clean_dict(result.details),

        # --- Exhaustive per-feature stats ---
        "loss_proxy_stats": series_stats(hq[loss_col], "t1_loss_proxy_pct"),
        "cycle_deviation_stats": series_stats(
            df[dev_col] if dev_col in df.columns else pd.Series(dtype=float), dev_col,
        ),

        # --- Time-series properties ---
        "loss_proxy_autocorrelation": _autocorrelation(hq[loss_col]),
        "loss_proxy_trend": _linear_trend(hq["day_dt"], hq[loss_col]),
        "loss_proxy_stationarity": _stationarity_test(hq[loss_col]),
        "loss_proxy_rolling": _rolling_stats(hq[loss_col]),

        "cycle_deviation_autocorrelation": _autocorrelation(
            df[dev_col] if dev_col in df.columns else pd.Series(dtype=float),
        ),
        "cycle_deviation_trend": _linear_trend(
            df["day_dt"], df[dev_col] if dev_col in df.columns else pd.Series(dtype=float),
        ),

        # --- Per-inverter stats ---
        "per_inverter_normalized_output": {},

        # --- DSPI ---
        "domain_soiling_index_stats": (
            series_stats(df["domain_soiling_index"], "domain_soiling_index")
            if "domain_soiling_index" in df.columns else None
        ),
        "dspi_autocorrelation": (
            _autocorrelation(df["domain_soiling_index"])
            if "domain_soiling_index" in df.columns else None
        ),

        # --- Loss proxy vs DSPI cross-lag ---
        "loss_proxy_vs_dspi_cross_lag": (
            _cross_lag_correlations(
                df["domain_soiling_index"], df[loss_col], max_lag=7,
            )
            if "domain_soiling_index" in df.columns else None
        ),

        # --- Raw daily series (LLM can digest 300+ points) ---
        "daily_series": _extract_daily_series(df, [
            loss_col, dev_col, "domain_soiling_index",
            "precipitation_total_mm", "rain_day",
        ]),
    }

    # Per-inverter
    inv_cols = [c for c in df.columns
                if c.endswith("_normalized_output")
                and not c.startswith("t1_") and not c.startswith("t2_")]
    for col in inv_cols:
        if col in hq.columns:
            section["per_inverter_normalized_output"][col] = series_stats(hq[col], col)

    return section


def build_signal_2_section(result: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Signal 2: PM/Dust Correlation — comprehensive LLM output."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    rate_col = "t1_perf_loss_rate_14d_pct_per_day"
    dev_col = "cycle_deviation_pct"

    # Clean partial_results — convert (r, p) tuples to dicts
    details = dict(result.details)
    if "partial_results" in details:
        cleaned = {}
        for feat, tgt_dict in details["partial_results"].items():
            cleaned[feat] = {}
            for tgt, (r, p) in tgt_dict.items():
                cleaned[feat][tgt] = {"r": _sf(r), "p": _sf(p)}
        details["partial_results"] = cleaned

    # Feature lists
    env_cols = [
        "pm10_mean", "pm25_mean", "precipitation_total_mm", "humidity_mean",
        "wind_speed_10m_mean", "air_temp_mean", "cloud_opacity_mean",
    ]
    eng_cols = [
        "days_since_last_rain", "days_since_significant_rain",
        "cumulative_pm10_since_rain", "cumulative_pm25_since_rain",
        "humidity_x_pm10", "domain_soiling_daily", "domain_soiling_index",
    ]
    target_cols = [loss_col, rate_col, dev_col]
    all_cols = [c for c in env_cols + eng_cols + target_cols if c in hq.columns]

    # Comprehensive pairwise: top dust features vs targets
    dust_features = [
        "pm10_mean", "pm25_mean", "cumulative_pm10_since_rain",
        "cumulative_pm25_since_rain", "humidity_x_pm10", "days_since_last_rain",
    ]
    pairwise_detail = {}
    for feat in dust_features:
        if feat not in hq.columns:
            continue
        pairwise_detail[feat] = {}
        for tgt in target_cols:
            if tgt in hq.columns:
                pairwise_detail[feat][tgt] = comprehensive_pairwise(hq, feat, tgt)

    # Cross-lag: PM10 leading loss proxy
    pm_cross_lag = {}
    for pm_col in ["pm10_mean", "pm25_mean", "cumulative_pm25_since_rain"]:
        if pm_col in hq.columns and loss_col in hq.columns:
            pm_cross_lag[f"{pm_col}_vs_loss_proxy"] = _cross_lag_correlations(
                hq[pm_col], hq[loss_col], max_lag=7,
            )

    # Seasonal split comparison
    seasonal_comparison = {}
    if "season" in hq.columns:
        dry = hq[hq["season"] == "dry"]
        wet = hq[hq["season"] == "wet"]
        if len(dry) > 5 and len(wet) > 5:
            seasonal_comparison["loss_proxy"] = _group_comparison(
                dry[loss_col], wet[loss_col], "dry", "wet",
            )
            seasonal_comparison["cycle_deviation"] = _group_comparison(
                dry[dev_col] if dev_col in dry.columns else pd.Series(dtype=float),
                wet[dev_col] if dev_col in wet.columns else pd.Series(dtype=float),
                "dry", "wet",
            )
            for pm_col in ["pm10_mean", "pm25_mean"]:
                if pm_col in hq.columns:
                    seasonal_comparison[pm_col] = _group_comparison(
                        dry[pm_col], wet[pm_col], "dry", "wet",
                    )

    section = {
        "signal_name": result.name,
        "verdict": result.verdict,
        "summary": result.summary,
        "details": _clean_dict(details),

        # --- Correlation matrices (Pearson + Spearman) ---
        "pearson_correlation_matrix": corr_matrix_to_dict(hq, all_cols, "pearson"),
        "spearman_correlation_matrix": corr_matrix_to_dict(hq, all_cols, "spearman"),

        # --- Comprehensive pairwise regression/correlation ---
        "dust_vs_target_pairwise": pairwise_detail,

        # --- Cross-lag (PM leading loss by 1-7 days) ---
        "cross_lag_correlations": pm_cross_lag,

        # --- Per-feature exhaustive stats ---
        "dust_feature_stats": {
            col: series_stats(hq[col], col) for col in dust_features if col in hq.columns
        },

        # --- Seasonal comparison with effect sizes ---
        "seasonal_comparison": seasonal_comparison,

        # --- Raw daily series for dust features ---
        "daily_series": _extract_daily_series(df, [
            "pm10_mean", "pm25_mean", "cumulative_pm10_since_rain",
            "cumulative_pm25_since_rain", "days_since_last_rain",
        ]),
    }
    return section


def build_signal_3_section(result: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """Signal 3: Rain Recovery — comprehensive LLM output."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"

    # Pre/post rain comparison
    rain_comparison = None
    if "precipitation_total_mm" in hq.columns:
        sig_rain = hq[hq["precipitation_total_mm"] >= 5.0]
        no_rain = hq[hq["precipitation_total_mm"] < 1.0]
        if len(sig_rain) > 3 and len(no_rain) > 3:
            rain_comparison = _group_comparison(
                sig_rain[loss_col], no_rain[loss_col],
                "rain_days", "dry_days",
            )

    # Precipitation stats
    precip_stats = (
        series_stats(hq["precipitation_total_mm"], "precipitation_mm")
        if "precipitation_total_mm" in hq.columns else None
    )

    # Rain-loss cross-lag
    rain_loss_cross_lag = None
    if "precipitation_total_mm" in hq.columns and loss_col in hq.columns:
        rain_loss_cross_lag = _cross_lag_correlations(
            hq["precipitation_total_mm"], hq[loss_col], max_lag=7,
        )

    section = {
        "signal_name": result.name,
        "verdict": result.verdict,
        "summary": result.summary,
        "details": _clean_dict(result.details),

        "rain_stats": {
            "total_rain_days": int(hq["rain_day"].astype(bool).sum()) if "rain_day" in hq.columns else None,
            "significant_rain_events": int((hq["precipitation_total_mm"] >= 5.0).sum()) if "precipitation_total_mm" in hq.columns else None,
            "precipitation_distribution": precip_stats,
        },

        # --- Loss proxy on rain vs dry days ---
        "rain_vs_dry_comparison": rain_comparison,

        # --- Rain-loss cross lag (does rain today predict lower loss tomorrow?) ---
        "rain_loss_cross_lag": rain_loss_cross_lag,

        # --- Raw daily series: rain + loss ---
        "daily_series": _extract_daily_series(df, [
            "precipitation_total_mm", loss_col, "rain_day",
            "days_since_last_rain",
        ]),
    }
    return section


def build_supporting_section(results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Supporting analyses — comprehensive LLM output."""
    hq = _hq(df)

    # Exhaustive distribution stats for key features
    dist_cols = [
        "t1_performance_loss_pct_proxy", "t2_performance_loss_pct_proxy",
        "precipitation_total_mm", "pm10_mean", "pm25_mean",
        "cycle_deviation_pct", "domain_soiling_daily", "domain_soiling_index",
        "t1_perf_loss_rate_14d_pct_per_day", "humidity_mean",
        "wind_speed_10m_mean", "air_temp_mean", "cloud_opacity_mean",
        "pvlib_soiling_ratio_hsu", "pvlib_soiling_loss_kimber",
        "pr_temperature_corrected", "tier_loss_correlation",
    ]

    # Quality gating deep dive
    tier_dist = df["transfer_quality_tier"].value_counts().to_dict() if "transfer_quality_tier" in df.columns else {}
    flag_dist = df["flag_count"].value_counts().sort_index().to_dict() if "flag_count" in df.columns else {}
    readiness_dist = df["transfer_readiness_tier"].value_counts().to_dict() if "transfer_readiness_tier" in df.columns else {}

    # Individual flag prevalence
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    flag_prevalence = {}
    for fc in flag_cols:
        if fc != "flag_count":
            flag_prevalence[fc] = {
                "n_flagged": int(df[fc].astype(bool).sum()),
                "pct_flagged": round(df[fc].astype(bool).mean() * 100, 2),
            }

    # T1 vs T2 agreement
    t1_t2_comparison = None
    if "t1_performance_loss_pct_proxy" in df.columns and "t2_performance_loss_pct_proxy" in df.columns:
        t1_t2_comparison = comprehensive_pairwise(
            df, "t1_performance_loss_pct_proxy", "t2_performance_loss_pct_proxy",
        )

    # Monthly/seasonal breakdown
    monthly_stats = {}
    if "month" in hq.columns and "t1_performance_loss_pct_proxy" in hq.columns:
        for month_num in sorted(hq["month"].dropna().unique()):
            month_data = hq[hq["month"] == month_num]["t1_performance_loss_pct_proxy"]
            monthly_stats[f"month_{int(month_num):02d}"] = series_stats(
                month_data, f"loss_proxy_month_{int(month_num)}",
            )

    section = {
        "results": _clean_dict(results),
        "distributions": {
            col: series_stats(hq[col], col)
            for col in dist_cols if col in hq.columns
        },
        "quality_gating": {
            "total_days": len(df),
            "hq_days": len(hq),
            "hq_ratio_pct": round(len(hq) / len(df) * 100, 1) if len(df) > 0 else 0,
            "csa_days": int(df["is_clear_sky_analyzable"].sum()) if "is_clear_sky_analyzable" in df.columns else None,
            "tier_distribution": tier_dist,
            "flag_count_distribution": flag_dist,
            "readiness_tier_distribution": readiness_dist,
            "individual_flag_prevalence": flag_prevalence,
        },
        "t1_vs_t2_agreement": t1_t2_comparison,
        "seasonal_loss_stats": monthly_stats,

        # --- Time-series properties of key features ---
        "loss_proxy_trend": _linear_trend(hq["day_dt"], hq["t1_performance_loss_pct_proxy"]),
        "loss_proxy_stationarity": _stationarity_test(hq["t1_performance_loss_pct_proxy"]),
    }
    return section


def build_csa_section(csa_results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Clear-Sky Analysis — comprehensive LLM output."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"

    section = {"results": _clean_dict(csa_results)}

    # CSA vs non-CSA comparison
    if "is_clear_sky_analyzable" in df.columns:
        csa = df[df["is_clear_sky_analyzable"]].copy()
        non_csa_hq = hq[~hq.index.isin(csa.index)]

        if loss_col in csa.columns and len(csa) > 3 and len(non_csa_hq) > 3:
            section["csa_vs_noncsa_loss_comparison"] = _group_comparison(
                csa[loss_col], non_csa_hq[loss_col], "csa", "non_csa",
            )

        # Comprehensive correlations on CSA subset
        dust_features = [
            "cumulative_pm25_since_rain", "cumulative_pm10_since_rain",
            "days_since_last_rain", "pm10_mean", "pm25_mean",
            "domain_soiling_index",
        ]
        csa_corrs = {}
        for feat in dust_features:
            if feat in csa.columns and loss_col in csa.columns:
                csa_corrs[feat] = comprehensive_pairwise(csa, feat, loss_col)
        section["csa_dust_correlations"] = csa_corrs

        # CSA feature stats
        section["csa_loss_proxy_stats"] = series_stats(csa[loss_col], "csa_loss_proxy")

    return section


def build_dq_section(
    dq_results: Optional[Dict[str, Any]], name: str,
) -> Dict[str, Any]:
    """Data Quality diagnostic section."""
    if dq_results is None:
        return {"name": name, "status": "skipped"}
    return {"name": name, "results": _clean_dict(dq_results)}


def build_custom_pr_section(
    debug_csv_path: Path,
    plot_png_path: Path,
) -> Dict[str, Any]:
    """Include custom PR plot/debug outputs in the LLM JSON summary."""
    section: Dict[str, Any] = {
        "status": "missing",
        "files": {
            "debug_csv_path": str(debug_csv_path),
            "debug_csv_exists": debug_csv_path.exists(),
            "plot_png_path": str(plot_png_path),
            "plot_png_exists": plot_png_path.exists(),
        },
    }

    if not debug_csv_path.exists():
        section["reason"] = "custom_pr_debug.csv not found"
        return section

    try:
        df = pd.read_csv(debug_csv_path)
    except Exception as exc:  # pragma: no cover - defensive only
        section["status"] = "error"
        section["reason"] = f"failed_to_read_debug_csv: {exc}"
        return section

    required_cols = [
        "day",
        "series",
        "gen_kwh",
        "avg_irr_wm2",
        "runtime_h",
        "X",
        "Y",
        "PR_raw",
        "outlier_flag",
        "PR_interp",
        "PR_roll7",
        "PR_display",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    section["status"] = "ok" if not missing_cols else "partial"
    section["missing_columns"] = missing_cols
    section["row_count"] = int(len(df))
    section["columns"] = list(df.columns)

    if "day" not in df.columns or "series" not in df.columns:
        section["reason"] = "debug csv missing required keys ('day', 'series')"
        return section

    df["day_dt"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day_dt"]).sort_values(["series", "day_dt"]).reset_index(drop=True)

    series_payload: Dict[str, Any] = {}
    for series_name, sub in df.groupby("series", sort=True):
        sub = sub.copy()
        if "outlier_flag" in sub.columns:
            outlier_bool = (
                sub["outlier_flag"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            outlier_count = int(outlier_bool.sum())
        else:
            outlier_count = 0

        per_series_stats = {}
        for col in ["gen_kwh", "avg_irr_wm2", "runtime_h", "X", "Y", "PR_raw", "PR_interp", "PR_roll7", "PR_display"]:
            if col in sub.columns:
                numeric = pd.to_numeric(sub[col], errors="coerce")
                numeric = numeric.replace([np.inf, -np.inf], np.nan)
                per_series_stats[col] = series_stats(numeric, col)

        # Ensure daily series payload does not emit inf tokens.
        for col in ["gen_kwh", "avg_irr_wm2", "runtime_h", "X", "Y", "PR_raw", "PR_interp", "PR_roll7", "PR_display"]:
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        series_payload[str(series_name)] = {
            "n_rows": int(len(sub)),
            "date_range": {
                "start": str(sub["day_dt"].min().date()),
                "end": str(sub["day_dt"].max().date()),
            },
            "outlier_count": outlier_count,
            "outlier_ratio_pct": round(outlier_count / max(1, len(sub)) * 100, 3),
            "stats": per_series_stats,
            "daily_series": _extract_daily_series(
                sub,
                [
                    "gen_kwh",
                    "avg_irr_wm2",
                    "runtime_h",
                    "X",
                    "Y",
                    "PR_raw",
                    "outlier_flag",
                    "PR_interp",
                    "PR_roll7",
                    "PR_display",
                ],
                date_col="day_dt",
            ),
        }

    section["series"] = series_payload
    section["series_names"] = sorted(series_payload.keys())
    return section


# ═══════════════════════════════════════════════════════════════════════
#  Dataset Overview
# ═══════════════════════════════════════════════════════════════════════

def build_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive dataset summary for LLM context."""
    hq = _hq(df)

    # Feature coverage
    all_features = sorted(df.columns.tolist())
    key_features = [
        "t1_performance_loss_pct_proxy", "t2_performance_loss_pct_proxy",
        "t1_energy_j", "irradiance_tilted_sum",
        "daily_generation_j", "solcast_gti_peak_sum", "solcast_gti_sum",
        "pm10_mean", "pm25_mean",
        "precipitation_total_mm", "cloud_opacity_mean",
        "domain_soiling_index", "domain_soiling_daily",
        "cycle_deviation_pct", "cycle_id",
        "runtime_h", "runtime_source", "irradiation_kwh_m2",
        "subset_capacity_kw", "plant_capacity_kw",
        "gen_irr_ratio", "gen_irr_ratio_smoothed",
        "subset_pr_physical_raw", "subset_pr_physical_outlier", "subset_pr_physical_interp",
        "plant_pr_physical_raw", "plant_pr_physical_outlier",
        "subset_daily_gen_inverter_count", "subset_daily_gen_expected_count", "subset_daily_gen_coverage",
        "power_at_ref_irradiance_w", "new_performance_index",
        "pvlib_soiling_ratio_hsu", "pvlib_soiling_loss_kimber",
        "pr_temperature_corrected",
        "transfer_quality_score", "humidity_mean",
        "wind_speed_10m_mean", "air_temp_mean", "dewpoint_mean",
        "is_clear_sky_analyzable",
    ]
    feature_coverage = {}
    for col in key_features:
        if col in df.columns:
            non_null = df[col].notna().sum()
            feature_coverage[col] = {
                "coverage_pct": round(non_null / len(df) * 100, 1),
                "n_valid": int(non_null),
                "n_null": int(df[col].isna().sum()),
            }

    # Date continuity analysis
    if "day_dt" in df.columns:
        dates = pd.to_datetime(df["day_dt"])
        date_diffs = dates.diff().dt.days.dropna()
        n_gaps = int((date_diffs > 1).sum())
        max_gap = int(date_diffs.max()) if len(date_diffs) > 0 else 0
    else:
        n_gaps = max_gap = 0

    overview = {
        "total_days": len(df),
        "date_range": {
            "start": str(df["day_dt"].min().date()),
            "end": str(df["day_dt"].max().date()),
            "span_days": int((df["day_dt"].max() - df["day_dt"].min()).days),
        },
        "date_continuity": {
            "n_gaps": n_gaps,
            "max_gap_days": max_gap,
            "pct_coverage": round(len(df) / max(1, (df["day_dt"].max() - df["day_dt"].min()).days + 1) * 100, 1),
        },
        "hq_days": len(hq),
        "hq_ratio_pct": round(len(hq) / len(df) * 100, 1),
        "csa_days": int(df["is_clear_sky_analyzable"].sum()) if "is_clear_sky_analyzable" in df.columns else None,
        "total_columns": len(df.columns),
        "all_column_names": all_features,
        "key_feature_coverage": feature_coverage,
    }
    return overview


# ═══════════════════════════════════════════════════════════════════════
#  Master Writer
# ═══════════════════════════════════════════════════════════════════════

def write_llm_summary(
    out_path: Path,
    *,
    dataset_overview: Dict[str, Any],
    signal_1: Dict[str, Any],
    signal_2: Dict[str, Any],
    signal_3: Dict[str, Any],
    supporting: Dict[str, Any],
    clear_sky: Dict[str, Any],
    dq1: Dict[str, Any],
    dq2: Dict[str, Any],
    dq3: Dict[str, Any],
    dq4: Dict[str, Any],
    dq5: Dict[str, Any],
    dq6: Dict[str, Any],
    verdicts: Dict[str, str],
    atomic_level: Optional[Dict[str, Any]] = None,
    microscopic_level: Optional[Dict[str, Any]] = None,
    macroscopic_level: Optional[Dict[str, Any]] = None,
    feature_glossary: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the unified LLM-readable EDA summary as a single JSON file."""

    custom_pr_debug = out_path.parent / "plots" / "custom_pr_debug.csv"
    custom_pr_plot = out_path.parent / "plots" / "custom_pr_inverters.png"

    # -- Build the full summary dict first --------------------------------
    summary = {}
    summary["_description"] = (
        "LLM-readable EDA summary for the PV Soiling Loss Predictions project. "
        "START by reading '_contents' for a deep directory of all sections and sub-keys, "
        "then 'feature_glossary' for units/conventions/caveats BEFORE interpreting numbers."
    )
    summary["generated_at"] = datetime.now().isoformat()
    # _contents placeholder — will be populated after all data is assembled
    if feature_glossary:
        summary["feature_glossary"] = feature_glossary
    summary["dataset_overview"] = dataset_overview
    summary["verdicts"] = verdicts
    summary["signal_1_sawtooth"] = signal_1
    summary["signal_2_dust_correlation"] = signal_2
    summary["signal_3_rain_recovery"] = signal_3
    summary["supporting_analyses"] = supporting
    summary["clear_sky_analysis"] = clear_sky
    summary["dq1_irradiance_vs_generation"] = dq1
    summary["dq2_daily_gen_validation"] = dq2
    summary["dq3_gen_irr_ratio"] = dq3
    summary["dq4_power_at_ref_irradiance"] = dq4
    summary["dq5_old_vs_new_comparison"] = dq5
    summary["dq6_performance_index"] = dq6
    summary["custom_pr_plot_data"] = build_custom_pr_section(custom_pr_debug, custom_pr_plot)
    if atomic_level:
        summary["atomic_level"] = atomic_level
    if microscopic_level:
        summary["microscopic_level"] = microscopic_level
    if macroscopic_level:
        summary["macroscopic_level"] = macroscopic_level

    # -- Auto-generate deep table of contents -----------------------------
    summary["_contents"] = _build_deep_contents(summary)

    # -- Reorder so _contents comes right after generated_at ---------------
    ordered = {}
    for k in ["_description", "generated_at", "_contents"]:
        if k in summary:
            ordered[k] = summary.pop(k)
    ordered.update(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compact_json = _serialize_compact(ordered)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(compact_json)

    size_kb = out_path.stat().st_size / 1024
    lines = compact_json.count("\n")
    log.info("LLM summary written to %s (%.1f KB, %d lines)", out_path, size_kb, lines)


def _serialize_compact(data: Any, indent: int = 0) -> str:
    """Custom JSON serializer optimised for minimum line count.

    Rules:
      - Leaf dicts (only scalar values) → one flat line: {"k1": v1, "k2": v2}
      - Structural dicts (contain nested dict/list) → one key per line, minimal indent
      - Scalar arrays → one flat line: [v1, v2, v3]
      - Arrays of leaf dicts → one dict per line
      - Arrays of structural dicts → expand each, one key per line
    """
    encoder = NumpyEncoder()

    # Pre-sanitize: round-trip through NumpyEncoder to convert numpy → Python native
    clean = json.loads(json.dumps(data, cls=NumpyEncoder))

    def _is_scalar(v: Any) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def _is_leaf_dict(d: dict) -> bool:
        return all(_is_scalar(v) for v in d.values())

    def _is_leaf_list(lst: list) -> bool:
        return all(_is_scalar(v) for v in lst)

    def _scalar(v: Any) -> str:
        return encoder.encode(v)

    def _flat_dict(d: dict) -> str:
        parts = [f"{_scalar(k)}: {_scalar(v)}" for k, v in d.items()]
        return "{" + ", ".join(parts) + "}"

    def _flat_list(lst: list) -> str:
        return "[" + ", ".join(_scalar(v) for v in lst) + "]"

    def _ser(obj: Any, lvl: int) -> str:
        if _is_scalar(obj):
            return _scalar(obj)

        if isinstance(obj, list):
            if len(obj) == 0:
                return "[]"
            if _is_leaf_list(obj):
                return _flat_list(obj)
            # Array of dicts or mixed
            items = []
            for item in obj:
                if isinstance(item, dict) and _is_leaf_dict(item):
                    items.append(" " * lvl + _flat_dict(item))
                elif isinstance(item, dict):
                    items.append(_ser(item, lvl))
                elif isinstance(item, list):
                    items.append(_ser(item, lvl))
                else:
                    items.append(" " * lvl + _scalar(item))
            return "[\n" + ",\n".join(items) + "]"

        if isinstance(obj, dict):
            if len(obj) == 0:
                return "{}"
            if _is_leaf_dict(obj):
                return _flat_dict(obj)
            # Structural dict — one key per line
            lines = []
            for k, v in obj.items():
                key_str = _scalar(k)
                if _is_scalar(v):
                    lines.append(" " * lvl + f"{key_str}: {_scalar(v)}")
                elif isinstance(v, list) and _is_leaf_list(v):
                    lines.append(" " * lvl + f"{key_str}: {_flat_list(v)}")
                elif isinstance(v, dict) and _is_leaf_dict(v):
                    lines.append(" " * lvl + f"{key_str}: {_flat_dict(v)}")
                elif isinstance(v, list) and len(v) > 0 and all(isinstance(i, dict) and _is_leaf_dict(i) for i in v):
                    # Array of leaf dicts — each on one line
                    arr_items = [_flat_dict(i) for i in v]
                    lines.append(" " * lvl + f"{key_str}: [\n" + ",\n".join(" " * (lvl + 1) + ai for ai in arr_items) + "]")
                else:
                    child = _ser(v, lvl + 1)
                    lines.append(" " * lvl + f"{key_str}: {child}")
            return "{\n" + ",\n".join(lines) + "}"

        # Fallback for numpy/pandas types
        return encoder.encode(obj)

    return _ser(clean, 0) + "\n"


def _build_deep_contents(
    data: Dict[str, Any], max_depth: int = 4,
) -> Dict[str, Any]:
    """Auto-generate a nested table of contents from the JSON structure.

    Walks 3-4 levels deep but avoids over-expanding large homogeneous dicts
    (like 66 feature glossary entries or 30 feature rankings). For those,
    it shows the key count and a sample instead of expanding every child.
    """

    def _is_private(k) -> bool:
        return isinstance(k, str) and k.startswith("_")

    def _public_keys(d: dict) -> list:
        return [k for k in d if not _is_private(k)]

    def _node_summary(val: Any, depth: int) -> Any:
        if isinstance(val, dict):
            desc = val.get("_description", None)
            pub_keys = _public_keys(val)
            n_keys = len(pub_keys)
            entry: Dict[str, Any] = {}
            if desc:
                entry["_brief"] = desc if len(desc) <= 120 else desc[:117] + "..."
            entry["_type"] = f"dict ({n_keys} keys)"

            if depth >= max_depth:
                # At max depth, just list key names
                if pub_keys:
                    entry["keys"] = pub_keys
                return entry

            # For large homogeneous dicts (>12 similar-structured children),
            # show a sample + list of all keys to avoid explosion
            if n_keys > 12:
                child_types = set()
                for k in pub_keys[:5]:
                    v = val[k]
                    child_types.add(type(v).__name__)
                if len(child_types) == 1:  # homogeneous
                    entry["all_keys"] = pub_keys
                    # Show first 2 expanded as sample
                    sample = {}
                    for k in pub_keys[:2]:
                        sample[str(k)] = _node_summary(val[k], depth + 1)
                    entry["sample_expanded"] = sample
                    return entry

            # Normal expansion
            sub_keys = {}
            for k in pub_keys:
                sub_keys[str(k)] = _node_summary(val[k], depth + 1)
            if sub_keys:
                entry["keys"] = sub_keys
            return entry

        elif isinstance(val, list):
            n = len(val)
            entry: Dict[str, Any] = {"_type": f"list ({n} items)"}
            if n > 0 and isinstance(val[0], dict):
                sample_keys = [str(k) for k in val[0] if not _is_private(k)]
                entry["item_keys"] = sample_keys
            return entry

        else:
            return type(val).__name__

    contents: Dict[str, Any] = {}
    for key, value in data.items():
        if _is_private(key):
            continue
        contents[key] = _node_summary(value, depth=1)
    return contents


