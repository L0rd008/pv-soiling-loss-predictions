"""Multi-level, multi-dimensional analysis for LLM consumption.

Three zoom levels, each revealing different insight types:
  - ATOMIC:       per-event, per-day granular detail
  - MICROSCOPIC:  per-cycle, per-spell, rolling correlations, interactions
  - MACROSCOPIC:  seasonal summaries, feature rankings, decision insights
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp

log = logging.getLogger(__name__)

def _sf(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None

def _hq(df): return df[(df["transfer_quality_tier"]=="high")&(df["flag_count"]==0)].copy()

# ═══════════════════════════════════════════════════════════════════════
#  ATOMIC LEVEL — per-event, per-day detail
# ═══════════════════════════════════════════════════════════════════════

def _atomic_dry_spells(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every individual dry spell with full per-day detail."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    is_dry = ~hq["rain_day"].astype(bool)
    spell_id = (is_dry != is_dry.shift()).cumsum()
    spells = []
    for sid, grp in hq[is_dry].groupby(spell_id):
        if len(grp) < 3: continue
        y = grp[loss_col].values
        x = np.arange(len(grp), dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2: continue
        sl, ic, r, p, se = sp.linregress(x[mask], y[mask])
        env = {}
        for c in ["pm10_mean","pm25_mean","humidity_mean","wind_speed_10m_mean",
                   "cloud_opacity_mean","air_temp_mean","domain_soiling_daily"]:
            if c in grp.columns:
                env[c] = {"mean": _sf(grp[c].mean()), "std": _sf(grp[c].std())}
        spells.append({
            "spell_index": len(spells),
            "start_date": str(grp["day_dt"].iloc[0].date()),
            "end_date": str(grp["day_dt"].iloc[-1].date()),
            "length_days": len(grp),
            "loss_start": _sf(y[0] if np.isfinite(y[0]) else None),
            "loss_end": _sf(y[-1] if np.isfinite(y[-1]) else None),
            "loss_change": _sf(y[-1]-y[0] if np.isfinite(y[0]) and np.isfinite(y[-1]) else None),
            "soiling_slope_pct_per_day": _sf(sl),
            "slope_r_squared": _sf(r**2),
            "slope_p_value": _sf(p),
            "slope_std_error": _sf(se),
            "slope_direction": "accumulating" if sl > 0.05 else "recovering" if sl < -0.05 else "flat",
            "environmental_conditions": env,
            "daily_loss_values": [_sf(v) for v in y],
        })
    return spells


def _atomic_rain_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every significant rain event with pre/post analysis."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    sig_idx = hq.index[hq["precipitation_total_mm"] >= 5.0].tolist()
    events = []
    for idx in sig_idx:
        pos = hq.index.get_loc(idx)
        row = hq.iloc[pos]
        pre_days = hq.iloc[max(0,pos-3):pos]
        post_days = hq.iloc[pos+1:min(len(hq),pos+4)]
        pre_loss = pre_days[loss_col].mean() if len(pre_days)>0 else None
        post_loss = post_days[loss_col].mean() if len(post_days)>0 else None
        events.append({
            "event_index": len(events),
            "date": str(row["day_dt"].date()),
            "precipitation_mm": _sf(row["precipitation_total_mm"]),
            "season": row.get("season","unknown"),
            "loss_on_day": _sf(row[loss_col]),
            "loss_pre_3d_mean": _sf(pre_loss),
            "loss_post_3d_mean": _sf(post_loss),
            "recovery_magnitude": _sf(pre_loss - post_loss if pre_loss and post_loss else None),
            "pm10_on_day": _sf(row.get("pm10_mean")),
            "cloud_on_day": _sf(row.get("cloud_opacity_mean")),
            "humidity_on_day": _sf(row.get("humidity_mean")),
            "days_since_prev_rain": _sf(row.get("days_since_last_rain")),
            "cum_pm25_at_event": _sf(row.get("cumulative_pm25_since_rain")),
        })
    return events


def _atomic_cleaning_cycles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Every cleaning cycle with boundary conditions."""
    if "cycle_id" not in df.columns: return []
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    dev_col = "cycle_deviation_pct"
    cycles = []
    for cid, grp in hq.groupby("cycle_id"):
        if len(grp) < 2: continue
        y = grp[loss_col].dropna()
        sl = r2 = p_val = None
        if len(y) >= 3:
            x = np.arange(len(y), dtype=float)
            sl_v, _, r_v, p_v, _ = sp.linregress(x, y.values)
            sl, r2, p_val = _sf(sl_v), _sf(r_v**2), _sf(p_v)
        cycles.append({
            "cycle_id": int(cid),
            "start_date": str(grp["day_dt"].iloc[0].date()),
            "end_date": str(grp["day_dt"].iloc[-1].date()),
            "length_days": len(grp),
            "loss_at_start": _sf(y.iloc[0]) if len(y)>0 else None,
            "loss_at_end": _sf(y.iloc[-1]) if len(y)>0 else None,
            "loss_accumulation": _sf(y.iloc[-1]-y.iloc[0]) if len(y)>1 else None,
            "soiling_rate_pct_per_day": sl,
            "rate_r_squared": r2,
            "rate_p_value": p_val,
            "mean_pm10": _sf(grp["pm10_mean"].mean()) if "pm10_mean" in grp.columns else None,
            "mean_pm25": _sf(grp["pm25_mean"].mean()) if "pm25_mean" in grp.columns else None,
            "total_rain_mm": _sf(grp["precipitation_total_mm"].sum()) if "precipitation_total_mm" in grp.columns else None,
            "rain_days": int(grp["rain_day"].astype(bool).sum()) if "rain_day" in grp.columns else None,
            "mean_deviation": _sf(grp[dev_col].mean()) if dev_col in grp.columns else None,
            "max_deviation": _sf(grp[dev_col].max()) if dev_col in grp.columns else None,
            "mean_cloud": _sf(grp["cloud_opacity_mean"].mean()) if "cloud_opacity_mean" in grp.columns else None,
            "mean_dspi_daily": _sf(grp["domain_soiling_daily"].mean()) if "domain_soiling_daily" in grp.columns else None,
        })
    return cycles


def build_atomic_level(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "_description": "Per-event, per-day granular detail. Every dry spell, rain event, and cleaning cycle with full environmental context.",
        "dry_spells": _atomic_dry_spells(df),
        "rain_events": _atomic_rain_events(df),
        "cleaning_cycles": _atomic_cleaning_cycles(df),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MICROSCOPIC LEVEL — rolling, interactions, conditional, segmented
# ═══════════════════════════════════════════════════════════════════════

def _rolling_correlations(df: pd.DataFrame, window: int = 60) -> Dict[str, Any]:
    """How correlations change over time (60-day rolling window)."""
    hq = _hq(df)
    if len(hq) < window + 10: return {}
    loss_col = "t1_performance_loss_pct_proxy"
    features = ["pm10_mean","pm25_mean","cumulative_pm25_since_rain",
                 "days_since_last_rain","domain_soiling_index","humidity_mean"]
    result = {}
    for feat in features:
        if feat not in hq.columns: continue
        rolling_r = hq[[loss_col, feat]].rolling(window, min_periods=20).corr()
        # Extract the off-diagonal correlations
        idx = rolling_r.index.get_level_values(1) == feat
        r_series = rolling_r.loc[idx, loss_col].dropna()
        if len(r_series) < 5: continue
        result[f"{feat}_vs_loss"] = {
            "window_days": window,
            "n_windows": len(r_series),
            "mean_r": _sf(r_series.mean()),
            "std_r": _sf(r_series.std()),
            "min_r": _sf(r_series.min()),
            "max_r": _sf(r_series.max()),
            "pct_positive": _sf((r_series > 0).mean() * 100),
            "pct_significant_pos": _sf((r_series > 0.2).mean() * 100),
            "trend_of_r": _sf(np.polyfit(np.arange(len(r_series)), r_series.values, 1)[0]) if len(r_series)>5 else None,
        }
    return result


def _interaction_effects(df: pd.DataFrame) -> Dict[str, Any]:
    """Test if feature combinations predict loss better than individuals."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    dev_col = "cycle_deviation_pct"
    interactions = [
        ("pm10_mean", "humidity_mean", "PM10 × Humidity"),
        ("pm10_mean", "wind_speed_10m_mean", "PM10 × Wind"),
        ("pm25_mean", "days_since_last_rain", "PM2.5 × Days dry"),
        ("cumulative_pm25_since_rain", "humidity_mean", "Cum PM2.5 × Humidity"),
        ("domain_soiling_daily", "cloud_opacity_mean", "DSPI daily × Cloud"),
        ("pm10_mean", "air_temp_mean", "PM10 × Temperature"),
    ]
    result = {}
    for col_a, col_b, label in interactions:
        if col_a not in hq.columns or col_b not in hq.columns: continue
        for tgt, tgt_label in [(loss_col, "loss_proxy"), (dev_col, "cycle_deviation")]:
            if tgt not in hq.columns: continue
            sub = hq[[col_a, col_b, tgt]].dropna()
            if len(sub) < 15: continue
            a, b, y = sub[col_a].values, sub[col_b].values, sub[tgt].values
            # Individual correlations
            r_a, _ = sp.pearsonr(a, y)
            r_b, _ = sp.pearsonr(b, y)
            # Interaction term
            interaction = a * b
            r_int, p_int = sp.pearsonr(interaction, y)
            # Multiple regression: y = c0 + c1*a + c2*b + c3*a*b
            X = np.column_stack([a, b, interaction, np.ones(len(a))])
            try:
                coef, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ coef
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - y.mean())**2)
                r2_full = 1 - ss_res/ss_tot if ss_tot > 0 else None
                # Without interaction
                X2 = np.column_stack([a, b, np.ones(len(a))])
                coef2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
                y_pred2 = X2 @ coef2
                ss_res2 = np.sum((y - y_pred2)**2)
                r2_no_int = 1 - ss_res2/ss_tot if ss_tot > 0 else None
                r2_improvement = (r2_full - r2_no_int) if r2_full and r2_no_int else None
            except:
                r2_full = r2_no_int = r2_improvement = None

            result[f"{label}_vs_{tgt_label}"] = {
                "n": len(sub),
                "r_individual_a": _sf(r_a),
                "r_individual_b": _sf(r_b),
                "r_interaction_term": _sf(r_int),
                "p_interaction_term": _sf(p_int),
                "r2_with_interaction": _sf(r2_full),
                "r2_without_interaction": _sf(r2_no_int),
                "r2_improvement_from_interaction": _sf(r2_improvement),
                "interaction_adds_value": bool(r2_improvement and r2_improvement > 0.01),
            }
    return result


def _conditional_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Loss distribution conditioned on feature quartiles."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    conditioning_features = ["pm10_mean","pm25_mean","cumulative_pm25_since_rain",
                              "days_since_last_rain","humidity_mean","cloud_opacity_mean",
                              "domain_soiling_index"]
    result = {}
    for feat in conditioning_features:
        if feat not in hq.columns: continue
        both = hq[[feat, loss_col]].dropna()
        if len(both) < 20: continue
        q_bounds = both[feat].quantile([0.25, 0.5, 0.75])
        quartile_stats = {}
        labels = [("Q1_low", 0, q_bounds.iloc[0]), ("Q2", q_bounds.iloc[0], q_bounds.iloc[1]),
                  ("Q3", q_bounds.iloc[1], q_bounds.iloc[2]), ("Q4_high", q_bounds.iloc[2], float("inf"))]
        for qlabel, lo, hi in labels:
            mask = (both[feat] >= lo) & (both[feat] < hi) if hi != float("inf") else (both[feat] >= lo)
            q_loss = both.loc[mask, loss_col]
            if len(q_loss) < 3: continue
            quartile_stats[qlabel] = {
                "n": len(q_loss), "mean": _sf(q_loss.mean()),
                "median": _sf(q_loss.median()), "std": _sf(q_loss.std()),
                "feature_range": f"{_sf(lo)} to {_sf(hi)}",
            }
        # Monotonic trend test (Jonckheere-Terpstra approx via Spearman on quartile means)
        means = [quartile_stats[q]["mean"] for q in quartile_stats if quartile_stats[q]["mean"] is not None]
        monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1)) or \
                    all(means[i] >= means[i+1] for i in range(len(means)-1)) if len(means) >= 3 else None
        result[feat] = {
            "quartile_loss_distributions": quartile_stats,
            "is_monotonic_trend": monotonic,
            "q4_minus_q1_mean_diff": _sf(means[-1] - means[0]) if len(means) >= 2 else None,
        }
    return result


def _segmented_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Split dataset into halves/thirds and compare."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    n = len(hq)
    if n < 20: return {}
    mid = n // 2
    first_half = hq.iloc[:mid]
    second_half = hq.iloc[mid:]
    result = {"first_half_vs_second_half": {}}
    for col in [loss_col, "pm10_mean", "pm25_mean", "cycle_deviation_pct",
                "precipitation_total_mm", "cloud_opacity_mean"]:
        if col not in hq.columns: continue
        a, b = first_half[col].dropna(), second_half[col].dropna()
        if len(a) < 5 or len(b) < 5: continue
        t_s, t_p = sp.ttest_ind(a, b, equal_var=False)
        result["first_half_vs_second_half"][col] = {
            "first_half_mean": _sf(a.mean()), "second_half_mean": _sf(b.mean()),
            "first_half_std": _sf(a.std()), "second_half_std": _sf(b.std()),
            "mean_change_pct": _sf((b.mean()-a.mean())/abs(a.mean())*100 if a.mean()!=0 else None),
            "welch_t_p": _sf(t_p),
            "significant_shift": bool(t_p < 0.05) if np.isfinite(t_p) else None,
        }
    return result


def build_microscopic_level(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "_description": "Per-cycle, rolling windows, interaction effects, conditional distributions. Reveals patterns invisible at coarser granularity.",
        "rolling_correlations_60d": _rolling_correlations(df, 60),
        "interaction_effects": _interaction_effects(df),
        "conditional_distributions": _conditional_distributions(df),
        "temporal_segmented_analysis": _segmented_analysis(df),
    }


# ═══════════════════════════════════════════════════════════════════════
#  MACROSCOPIC LEVEL — rankings, summaries, decisions
# ═══════════════════════════════════════════════════════════════════════

def _feature_importance_ranking(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rank all features by correlation strength with soiling targets."""
    hq = _hq(df)
    targets = ["t1_performance_loss_pct_proxy","t1_perf_loss_rate_14d_pct_per_day","cycle_deviation_pct"]
    features = [c for c in hq.columns if c not in targets
                and hq[c].dtype in [np.float64, np.int64, float, int]
                and not c.startswith("flag_") and c not in ["day","month","year","cycle_id"]
                and hq[c].notna().sum() > 20]
    rankings = []
    for feat in features:
        scores = {}
        for tgt in targets:
            if tgt not in hq.columns: continue
            pair = hq[[feat, tgt]].dropna()
            if len(pair) < 10: continue
            r, p = sp.pearsonr(pair[feat], pair[tgt])
            rho, _ = sp.spearmanr(pair[feat], pair[tgt])
            scores[tgt] = {"pearson_r": _sf(r), "pearson_p": _sf(p),
                           "spearman_rho": _sf(rho), "abs_r": _sf(abs(r))}
        if not scores: continue
        best_abs_r = max(s.get("abs_r",0) or 0 for s in scores.values())
        rankings.append({"feature": feat, "best_abs_pearson_r": best_abs_r,
                          "target_correlations": scores})
    rankings.sort(key=lambda x: x["best_abs_pearson_r"] or 0, reverse=True)
    return rankings[:30]  # Top 30


def _data_sufficiency_assessment(df: pd.DataFrame) -> Dict[str, Any]:
    """Assess whether we have enough data for reliable modeling."""
    hq = _hq(df)
    n_total = len(df)
    n_hq = len(hq)
    csa_n = int(df["is_clear_sky_analyzable"].sum()) if "is_clear_sky_analyzable" in df.columns else 0
    n_dry_spells = 0
    if "rain_day" in hq.columns:
        is_dry = ~hq["rain_day"].astype(bool)
        spell_id = (is_dry != is_dry.shift()).cumsum()
        n_dry_spells = sum(1 for _, g in hq[is_dry].groupby(spell_id) if len(g) >= 3)
    n_rain = int((hq["precipitation_total_mm"] >= 5.0).sum()) if "precipitation_total_mm" in hq.columns else 0
    n_cycles = hq["cycle_id"].nunique() if "cycle_id" in hq.columns else 0
    return {
        "total_days": n_total,
        "hq_days": n_hq,
        "csa_days": csa_n,
        "n_dry_spells_ge3": n_dry_spells,
        "n_significant_rain_events": n_rain,
        "n_cleaning_cycles": n_cycles,
        "assessment": {
            "sample_size": "adequate" if n_hq >= 100 else "marginal" if n_hq >= 50 else "insufficient",
            "dry_spells": "adequate" if n_dry_spells >= 8 else "marginal" if n_dry_spells >= 4 else "insufficient",
            "rain_events": "adequate" if n_rain >= 20 else "marginal" if n_rain >= 10 else "insufficient",
            "csa_days": "adequate" if csa_n >= 40 else "marginal" if csa_n >= 20 else "insufficient",
            "cleaning_cycles": "adequate" if n_cycles >= 10 else "marginal" if n_cycles >= 5 else "insufficient",
        },
    }


def _seasonal_deep_dive(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive seasonal breakdown."""
    hq = _hq(df)
    if "season" not in hq.columns: return {}
    loss_col = "t1_performance_loss_pct_proxy"
    result = {}
    for season in ["dry", "wet"]:
        sub = hq[hq["season"] == season]
        if len(sub) < 5: continue
        entry = {"n_days": len(sub)}
        for col in [loss_col, "cycle_deviation_pct", "pm10_mean", "pm25_mean",
                    "precipitation_total_mm", "humidity_mean", "cloud_opacity_mean",
                    "domain_soiling_index", "days_since_last_rain"]:
            if col in sub.columns:
                s = sub[col].dropna()
                entry[col] = {"mean": _sf(s.mean()), "median": _sf(s.median()),
                              "std": _sf(s.std()), "n": len(s)}
        # Top correlations within this season
        top_corr = {}
        for feat in ["pm10_mean","pm25_mean","cumulative_pm25_since_rain",
                      "days_since_last_rain","domain_soiling_index"]:
            if feat in sub.columns and loss_col in sub.columns:
                pair = sub[[feat, loss_col]].dropna()
                if len(pair) > 5:
                    r, p = sp.pearsonr(pair[feat], pair[loss_col])
                    top_corr[feat] = {"r": _sf(r), "p": _sf(p)}
        entry["feature_correlations_with_loss"] = top_corr
        result[season] = entry
    return result


def _decision_summary(df: pd.DataFrame, verdicts: Dict[str, str]) -> Dict[str, Any]:
    """High-level decision-ready summary."""
    hq = _hq(df)
    loss_col = "t1_performance_loss_pct_proxy"
    dev_col = "cycle_deviation_pct"
    # Best predictors
    best_feats = []
    for feat in ["cumulative_pm25_since_rain","cumulative_pm10_since_rain",
                  "days_since_last_rain","domain_soiling_index","pm25_mean","pm10_mean",
                  "humidity_x_pm10"]:
        if feat not in hq.columns: continue
        for tgt in [loss_col, dev_col]:
            if tgt not in hq.columns: continue
            pair = hq[[feat, tgt]].dropna()
            if len(pair) > 10:
                r, p = sp.pearsonr(pair[feat], pair[tgt])
                best_feats.append({"feature": feat, "target": tgt,
                                   "r": _sf(r), "p": _sf(p), "abs_r": _sf(abs(r))})
    best_feats.sort(key=lambda x: x["abs_r"] or 0, reverse=True)
    return {
        "verdicts": verdicts,
        "top_5_predictors": best_feats[:5],
        "modeling_readiness": {
            "has_soiling_signal": verdicts.get("signal_1_sawtooth") == "pass",
            "has_dust_correlation": verdicts.get("signal_2_dust_correlation") == "pass",
            "has_rain_recovery": verdicts.get("signal_3_rain_recovery") == "pass",
            "recommended_target": dev_col if any(f["target"]==dev_col and (f["abs_r"] or 0)>0.3 for f in best_feats) else loss_col,
        },
        "key_concerns": [],  # Will be populated below
    }


def build_macroscopic_level(df: pd.DataFrame, verdicts: Dict[str, str]) -> Dict[str, Any]:
    return {
        "_description": "Overall summaries, feature rankings, seasonal patterns, data sufficiency, and decision-ready insights.",
        "feature_importance_ranking": _feature_importance_ranking(df),
        "data_sufficiency": _data_sufficiency_assessment(df),
        "seasonal_deep_dive": _seasonal_deep_dive(df),
        "decision_summary": _decision_summary(df, verdicts),
    }
