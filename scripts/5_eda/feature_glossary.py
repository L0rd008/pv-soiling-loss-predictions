"""Feature metadata glossary for LLM context.

Provides units, conventions, computation details, interpretation guidelines,
and caveats for every column in daily_model_eda.csv. Without this context,
statistical numbers are meaningless to an LLM making decisions.

Source: docs/pipeline/04_eda_features.md + scripts/core/daily_features.py
"""
from __future__ import annotations
from typing import Any, Dict

# Each entry: unit, dtype, description, soiling_relevance, caveats, value_range, null_reason
FEATURE_GLOSSARY: Dict[str, Dict[str, Any]] = {
    # ── Time ──────────────────────────────────────────────────────────
    "day": {
        "unit": "date (YYYY-MM-DD)", "dtype": "object",
        "description": "Calendar date. One row per day. Index key.",
        "soiling_relevance": "Time axis for all trend analyses. Soiling accumulates between rain events; ordering matters.",
        "null_reason": None,
    },
    "day_dt": {
        "unit": "datetime", "dtype": "datetime64",
        "description": "Parsed datetime version of 'day'. Used for time-series operations.",
        "soiling_relevance": "Same as 'day'.",
        "null_reason": None,
    },

    # ── Combined Inverter Aggregates ──────────────────────────────────
    "subset_energy_j": {
        "unit": "Joules", "dtype": "float64",
        "description": "Total daily energy from all 6 tiered inverters (sum of 10-min active power × 600s intervals).",
        "soiling_relevance": "PRIMARY TARGET SIGNAL. Declining energy relative to irradiance indicates soiling.",
        "value_range": "1.1e9 to 4.5e10 J", "null_reason": None,
        "caveats": "This is a sum of 10-min AVG power readings × 600s, NOT a cumulative energy meter. May undercount during inverter outages.",
    },
    "subset_power_w_p95": {
        "unit": "Watts", "dtype": "float64",
        "description": "95th percentile of 10-min combined active power. Captures near-peak output.",
        "soiling_relevance": "Peak achievable output drops as soiling accumulates.",
        "value_range": "91k to 1.7M W", "null_reason": None,
    },
    "subset_data_availability_mean": {
        "unit": "ratio (0-1)", "dtype": "float64",
        "description": "Mean row-level power completeness (fraction of inverters reporting per 10-min interval).",
        "soiling_relevance": "Gates data quality. Days below 0.5 should be treated with caution.",
        "value_range": "0.46 to 1.0", "null_reason": None,
    },

    # ── Block-Level ───────────────────────────────────────────────────
    "b1_energy_j": {
        "unit": "Joules", "dtype": "float64",
        "description": "Block B1 (Tier-2: B1-08, B1-01, B1-13) total daily energy.",
        "soiling_relevance": "Cross-block comparison reveals differential soiling (if one block cleaned but not other).",
        "value_range": "9.0e7 to 2.3e10 J", "null_reason": None,
    },
    "b2_energy_j": {
        "unit": "Joules", "dtype": "float64",
        "description": "Block B2 (Tier-1: B2-08, B2-13, B2-17) total daily energy.",
        "soiling_relevance": "High-availability training block. Same as t1_energy_j.",
        "value_range": "1.0e9 to 2.3e10 J", "null_reason": None,
    },
    "block_mismatch_ratio": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "b1_energy_j / b2_energy_j. Near 1.0 = balanced blocks.",
        "soiling_relevance": "Sudden deviations flag anomalous events (partial cleaning, shading, equipment faults).",
        "value_range": "0.09 to 1.29", "caveats": "Very low values indicate B1 data gaps, not differential soiling.",
    },

    # ── Tier-1 (B2 Training) ─────────────────────────────────────────
    "t1_energy_j": {
        "unit": "Joules", "dtype": "float64",
        "description": "Tier-1 (B2 block) daily energy. Same as b2_energy_j. Highest data quality.",
        "soiling_relevance": "USE FOR MODEL TRAINING. Data availability never below 0.77.",
        "value_range": "1.0e9 to 2.3e10 J", "null_reason": None,
    },
    "t1_data_availability": {
        "unit": "ratio (0-1)", "dtype": "float64",
        "description": "Tier-1 data availability. Never below 0.77 — always trustworthy.",
        "soiling_relevance": "Confirms T1 data is reliable on all days.",
        "value_range": "0.77 to 1.0", "null_reason": None,
    },

    # ── Tier-2 (B1 Validation) ────────────────────────────────────────
    "t2_energy_j": {
        "unit": "Joules", "dtype": "float64",
        "description": "Tier-2 (B1 block) daily energy. Cross-block validation set.",
        "soiling_relevance": "VALIDATION TARGET. Model trained on T1 should also predict T2 loss patterns.",
        "value_range": "9.0e7 to 2.3e10 J", "null_reason": None,
        "caveats": "Lower availability than T1. Exclude days with t2_data_availability < 0.5.",
    },

    # ── Irradiance ────────────────────────────────────────────────────
    "irradiance_tilted_sum": {
        "unit": "W/m² (daily sum of 15-min readings)", "dtype": "float64",
        "description": "Tilted (plane-of-array) irradiance daily total. PRIMARY irradiance metric.",
        "soiling_relevance": "DENOMINATOR for normalized output — the single most important feature for soiling isolation.",
        "value_range": "0 to 434k", "null_reason": None,
        "caveats": "Sum of 15-min interval readings, NOT true Wh/m²/day. Unit is effectively (W/m²)·(count_of_intervals).",
    },
    "irradiance_horizontal_sum": {
        "unit": "W/m² (daily sum)", "dtype": "float64",
        "description": "Global Horizontal Irradiance daily total.",
        "value_range": "0 to 413k", "null_reason": None,
    },
    "irradiance_coverage_ratio": {
        "unit": "ratio (0-1)", "dtype": "float64",
        "description": "irradiance_records / 96 (expected 96 records = 24h × 4 per hour).",
        "soiling_relevance": "Low coverage makes irradiance sum unreliable; should be flagged.",
        "value_range": "0.04 to 1.0",
    },

    # ── Solcast Weather / Air Quality ─────────────────────────────────
    "pm10_mean": {
        "unit": "µg/m³", "dtype": "float64",
        "description": "Daily mean PM10 (coarse dust) from Solcast satellite data.",
        "soiling_relevance": "CRITICAL PREDICTOR. Direct proxy for airborne particulates causing soiling. High PM → faster soiling.",
        "value_range": "4.4 to 138.5 µg/m³", "null_reason": None,
    },
    "pm25_mean": {
        "unit": "µg/m³", "dtype": "float64",
        "description": "Daily mean PM2.5 (fine particulates) from Solcast satellite data.",
        "soiling_relevance": "CRITICAL PREDICTOR. Fine dust that adheres more strongly to panels.",
        "value_range": "3.2 to 47.7 µg/m³", "null_reason": None,
    },
    "precipitation_total_mm": {
        "unit": "mm", "dtype": "float64",
        "description": "Total daily precipitation from Solcast.",
        "soiling_relevance": "CRITICAL. Rain is the primary NATURAL CLEANING mechanism. Soiling resets after heavy rain (≥5mm = significant).",
        "value_range": "0 to 161.6 mm", "null_reason": None,
        "caveats": "Light rain (<2mm) may redistribute dust rather than clean panels.",
    },
    "rain_day": {
        "unit": "boolean", "dtype": "bool",
        "description": "Whether any precipitation occurred on this day.",
        "soiling_relevance": "Binary cleaning indicator. Used to define dry spells and soiling cycles.",
    },
    "humidity_mean": {
        "unit": "% (relative humidity)", "dtype": "float64",
        "description": "Daily mean relative humidity from Solcast.",
        "soiling_relevance": "CRITICAL INTERACTION. High humidity + dust → cementation (sticky soiling harder to wash off).",
        "value_range": "61.5 to 98.6%",
    },
    "wind_speed_10m_mean": {
        "unit": "m/s", "dtype": "float64",
        "description": "Mean wind speed at 10m height from Solcast.",
        "soiling_relevance": "High wind resuspends dust (reduces soiling) but can also deposit more (direction-dependent).",
        "value_range": "0.8 to 7.9 m/s",
    },
    "dewpoint_mean": {
        "unit": "°C", "dtype": "float64",
        "description": "Daily mean dewpoint temperature from Solcast.",
        "soiling_relevance": "Condensation overnight glues dust to panels → accelerated soiling.",
        "value_range": "19.1 to 25.3°C",
    },
    "air_temp_mean": {
        "unit": "°C", "dtype": "float64",
        "description": "Daily mean air temperature from Solcast.",
        "soiling_relevance": "Temperature affects soiling adhesion and panel efficiency (temperature coefficient).",
        "value_range": "23.6 to 30.7°C",
    },
    "cloud_opacity_mean": {
        "unit": "% (0=clear, 100=overcast)", "dtype": "float64",
        "description": "Mean cloud opacity from Solcast satellite.",
        "soiling_relevance": "Cloudy days have lower irradiance, making soiling impact harder to isolate. Used for CSA filtering.",
        "value_range": "0.5 to 85.1%",
    },

    # ── Solcast Satellite Irradiance ──────────────────────────────────
    "solcast_gti_sum": {
        "unit": "J/m²/day", "dtype": "float64",
        "description": "Global Tilted Irradiance from Solcast satellite (independent of ground sensors).",
        "soiling_relevance": "Cross-validates ground irradiance. If they diverge, ground sensor may be dirty too.",
        "value_range": "2.8M to 28.1M J/m²/day",
    },
    "solcast_gti_peak_sum": {
        "unit": "J/m²/day (peak hours only)", "dtype": "float64",
        "description": "Solcast GTI filtered to peak hours only. Used as denominator for new_normalized_output baseline.",
        "soiling_relevance": "Alternative irradiance denominator for days where ground sensors are unreliable.",
    },

    # ── Performance Metrics (Combined) ────────────────────────────────
    "normalized_output": {
        "unit": "J/J (dimensionless ratio)", "dtype": "float64",
        "description": "subset_energy_j / irradiance_tilted_sum. NaN when irradiance below baseline threshold. Clipped at 500k.",
        "soiling_relevance": "Core performance ratio. Dropping normalized output = soiling or other degradation.",
        "value_range": "27k to 500k", "null_reason": "Low-irradiance days where normalization is undefined (~9% null).",
        "caveats": "Includes ALL-CAUSE losses, not just soiling: temperature, equipment, shading, sensor errors.",
    },
    "rolling_clean_baseline": {
        "unit": "J/J", "dtype": "float64",
        "description": "95th percentile of normalized output on clear days (30-day rolling window). Represents 'best achievable' output.",
        "soiling_relevance": "Defines what output SHOULD be if panels were clean. Seasonal variation is expected.",
        "null_reason": "Requires sufficient clear days in the window (~8% null).",
    },

    # ── Performance Metrics (Tier-1) ──────────────────────────────────
    "t1_performance_loss_pct_proxy": {
        "unit": "% (0=clean, 100=total loss)", "dtype": "float64",
        "description": "100 × (1 - t1_normalized_output / t1_rolling_clean_baseline). Tier-1 all-cause loss. Clipped [0, 100].",
        "soiling_relevance": "BEST TRAINING TARGET. Slow upward trend = soiling accumulation; sudden drop = cleaning event (rain).",
        "value_range": "0 to 80.7%", "null_reason": "Requires both normalized output and baseline (~14% null).",
        "caveats": "ALL-CAUSE deficit, not pure soiling. Includes temperature, equipment, shading. EDA must isolate soiling from confounders.",
    },
    "t1_perf_loss_rate_14d_pct_per_day": {
        "unit": "%/day", "dtype": "float64",
        "description": "14-day rate of change of t1_performance_loss_pct_proxy. Positive = worsening; negative = improving (rain cleaning).",
        "soiling_relevance": "SOILING VELOCITY. Captures how fast panels are getting dirtier.",
        "value_range": "-5.8 to 5.8 %/day", "null_reason": "Requires 14-day diff + baseline (~23% null).",
    },
    "t1_normalized_output": {
        "unit": "J/J (dimensionless)", "dtype": "float64",
        "description": "t1_energy_j / irradiance_tilted_sum. Tier-1 specific normalized output.",
        "value_range": "14k to 297k", "null_reason": "Low-irradiance days (~9% null).",
    },

    # ── Performance Metrics (Tier-2) ──────────────────────────────────
    "t2_performance_loss_pct_proxy": {
        "unit": "% (0=clean, 100=total loss)", "dtype": "float64",
        "description": "Tier-2 all-cause loss. Same computation as T1 but using B1 block data.",
        "soiling_relevance": "VALIDATION TARGET. If model predicts T2 loss from T1-trained weights, it generalizes.",
        "value_range": "0 to 81.1%", "null_reason": "~14% null.",
        "caveats": "T2 and T1 are highly correlated (r≈0.98), confirming soiling is plant-wide.",
    },

    # ── Cross-Tier Correlation ────────────────────────────────────────
    "tier_loss_correlation": {
        "unit": "Pearson r (-1 to 1)", "dtype": "float64",
        "description": "Rolling 30-day correlation between T1 and T2 loss proxies. Median: 0.98.",
        "soiling_relevance": "Near 1.0 CONFIRMS plant-wide soiling (not localized to one block).",
        "value_range": "0.67 to 1.0", "null_reason": "30-day window warm-up (~10% null).",
    },
    "tier_loss_delta": {
        "unit": "percentage points", "dtype": "float64",
        "description": "t1_loss - t2_loss. Positive = T1 cleaner than T2.",
        "soiling_relevance": "Near 0 = both blocks soil at same rate. Validates that detected soiling is real.",
        "value_range": "-61.5 to 17.4 pp",
    },
    "tier_agreement_flag": {
        "unit": "boolean", "dtype": "bool",
        "description": "True if both tiers trending same direction over 7 days.",
        "soiling_relevance": "False days may indicate localized events (partial cleaning, shading, equipment faults).",
    },

    # ── Quality Flags ─────────────────────────────────────────────────
    "flag_sensor_suspect_irradiance": {
        "unit": "boolean", "dtype": "bool",
        "description": "Low irradiance but non-trivial Tier-1 output → possible sensor fault.",
        "soiling_relevance": "Indicates days where normalized output is unreliable.",
    },
    "flag_coverage_gap": {
        "unit": "boolean", "dtype": "bool",
        "description": "Inverter or irradiance coverage below 30%.",
    },
    "flag_block_mismatch": {
        "unit": "boolean", "dtype": "bool",
        "description": "B1/B2 ratio deviates >15% from rolling median.",
    },
    "flag_low_output_high_irr": {
        "unit": "boolean", "dtype": "bool",
        "description": "Tier-1 normalized output < 70% of 14-day median on a high-irradiance day.",
        "soiling_relevance": "Most direct SOILING ALERT — high sun but low output.",
    },
    "flag_count": {
        "unit": "count (0-4)", "dtype": "int64",
        "description": "Sum of all boolean quality flags. Days with flag_count ≥ 2 should likely be excluded from training.",
        "soiling_relevance": "Data quality gate for model training. HQ = flag_count == 0.",
    },

    # ── Transfer Readiness ────────────────────────────────────────────
    "transfer_quality_score": {
        "unit": "points (0-100)", "dtype": "float64",
        "description": "Starts at 100, penalized for flags and gaps. Uses Tier-1 fields. Median: 100.",
        "soiling_relevance": "Filter to score ≥ 80 ('high' tier) for the most reliable training data.",
        "value_range": "20 to 100",
    },
    "transfer_quality_tier": {
        "unit": "category", "dtype": "object",
        "description": "≥80 → 'high', ≥60 → 'medium', <60 → 'low'.",
        "soiling_relevance": "Filter to 'high' for training. 'high' + flag_count=0 = HQ (high quality) subset.",
    },
    "transfer_readiness_tier": {
        "unit": "category", "dtype": "object",
        "description": "Tier-1, Tier-2, or Tier-3 readiness for cross-plant model transfer.",
    },

    # ── Engineered Features ───────────────────────────────────────────
    "days_since_last_rain": {
        "unit": "days (integer)", "dtype": "float64",
        "description": "Count of consecutive days without any precipitation (rain_day=False).",
        "soiling_relevance": "ACCUMULATION PROXY. Longer dry spells → more soiling accumulation.",
        "value_range": "0 to ~30 days",
    },
    "days_since_significant_rain": {
        "unit": "days (integer)", "dtype": "float64",
        "description": "Days since last precipitation ≥ 5mm ('significant rain' threshold).",
        "soiling_relevance": "More meaningful than days_since_last_rain: only significant rain effectively cleans panels.",
    },
    "cumulative_pm10_since_rain": {
        "unit": "µg/m³·days (cumulative sum)", "dtype": "float64",
        "description": "Running sum of pm10_mean since last rain day. Resets to zero after rain.",
        "soiling_relevance": "STRONG PREDICTOR. Captures total dust exposure since last cleaning. Higher = more soiling expected.",
        "caveats": "Assumes linear accumulation. Real soiling may saturate at high dust loads.",
    },
    "cumulative_pm25_since_rain": {
        "unit": "µg/m³·days (cumulative sum)", "dtype": "float64",
        "description": "Running sum of pm25_mean since last rain day. Resets to zero after rain.",
        "soiling_relevance": "STRONGEST PREDICTOR of cycle_deviation_pct (r≈0.43). Fine particles adhere more strongly.",
    },
    "humidity_x_pm10": {
        "unit": "(%RH)·(µg/m³) — interaction term", "dtype": "float64",
        "description": "humidity_mean × pm10_mean. Captures cementation effect.",
        "soiling_relevance": "High humidity + high dust → sticky soiling that is harder to wash off. Literature-supported predictor.",
        "caveats": "This is a multiplicative interaction, not directly physically meaningful. Use for correlation, not absolute values.",
    },
    "domain_soiling_daily": {
        "unit": "composite score (≈ µg/m³ scale)", "dtype": "float64",
        "description": "Daily domain soiling potential index: weighted combination of PM2.5, PM10, humidity, wind, temperature.",
        "soiling_relevance": "Physics-informed daily soiling rate estimate based on environmental conditions.",
    },
    "domain_soiling_index": {
        "unit": "composite score (cumulative)", "dtype": "float64",
        "description": "Cumulative sum of domain_soiling_daily since last significant rain. Resets on rain.",
        "soiling_relevance": "DSPI — Domain Soiling Potential Index. Integrates multi-factor soiling exposure over dry periods.",
        "caveats": "A model-based feature, not a direct measurement. Weights come from soiling literature.",
    },

    # ── Soiling Cycle Features ────────────────────────────────────────
    "cycle_id": {
        "unit": "integer (sequential)", "dtype": "int64",
        "description": "Soiling cycle identifier. Increments each time rain/cleaning resets the cycle.",
        "soiling_relevance": "Groups days into soiling accumulation periods. Use for per-cycle analysis.",
    },
    "cycle_deviation_pct": {
        "unit": "% (0=at cycle peak, 100=total deviation)", "dtype": "float64",
        "description": "100 × (1 - normalized_output / cycle_max_normalized_output). How far current output has deviated from cycle peak.",
        "soiling_relevance": "KEY ALTERNATIVE TARGET. More robust than loss_proxy for within-cycle soiling measurement.",
        "value_range": "0 to 100%",
        "caveats": "Only meaningful within a cycle. Cross-cycle comparison requires normalization.",
    },

    # ── New Telemetry Features ────────────────────────────────────────
    "subset_daily_gen_kwh": {
        "unit": "kWh", "dtype": "float64",
        "description": "Asset-aligned sum of daily_generated_electricity over the power-tier intersection.",
        "soiling_relevance": "Numerator for physical subset PR. Like-for-like with aligned old-source inverter set.",
        "null_reason": "Only available from new telemetry source start date (~30% null).",
    },
    "plant_avg_irradiance_wm2": {
        "unit": "W/m²", "dtype": "float64",
        "description": "Daily average irradiance from on-site plant sensor. Independent of Solcast.",
        "soiling_relevance": "Combined with runtime_h to form daily irradiation_kwh_m2.",
        "null_reason": "Only available from new telemetry source start date (~30% null).",
    },
    "subset_daily_gen_inverter_count": {
        "unit": "count", "dtype": "float64",
        "description": "Count of aligned inverter daily-generation values present for each day.",
        "soiling_relevance": "Data-coverage guard for subset PR reliability.",
    },
    "subset_daily_gen_expected_count": {
        "unit": "count", "dtype": "float64",
        "description": "Expected inverter count from tiered active-power columns for the subset.",
        "soiling_relevance": "Coverage denominator for aligned new-source generation.",
    },
    "subset_daily_gen_coverage": {
        "unit": "ratio (0-1)", "dtype": "float64",
        "description": "subset_daily_gen_inverter_count / subset_daily_gen_expected_count.",
        "soiling_relevance": "Low coverage days can bias PR downward and should be quality-checked.",
    },
    "runtime_h": {
        "unit": "hours", "dtype": "float64",
        "description": "Daily runtime hours used in physical irradiation calculation (CSV primary, Solcast fallback).",
        "soiling_relevance": "Critical denominator term for physically valid PR.",
    },
    "runtime_source": {
        "unit": "category", "dtype": "object",
        "description": "Source tag for runtime_h: runtime_csv or solcast_daylight.",
        "soiling_relevance": "Tracks denominator provenance and quality.",
    },
    "irradiation_kwh_m2": {
        "unit": "kWh/m²", "dtype": "float64",
        "description": "Daily irradiation = plant_avg_irradiance_wm2 * runtime_h / 1000.",
        "soiling_relevance": "Physical irradiance-energy denominator for PR calculations.",
    },
    "subset_capacity_kw": {
        "unit": "kW", "dtype": "float64",
        "description": "Subset capacity basis used for PR denominator (330 kW * aligned inverter count).",
        "soiling_relevance": "Physical scaling term for subset PR.",
    },
    "plant_capacity_kw": {
        "unit": "kW", "dtype": "float64",
        "description": "Plant capacity basis used for PR denominator (330 kW * plant inverter count).",
        "soiling_relevance": "Physical scaling term for plant PR.",
    },
    "subset_pr_physical_raw": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "subset_daily_gen_kwh / (subset_capacity_kw * irradiation_kwh_m2).",
        "soiling_relevance": "Primary new-source physical PR metric.",
    },
    "subset_pr_physical_outlier": {
        "unit": "boolean", "dtype": "bool",
        "description": "True when subset_pr_physical_raw is outside [0,1].",
        "soiling_relevance": "Explicit diagnostics for non-physical PR values.",
    },
    "subset_pr_physical_interp": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "Subset physical PR after masking outliers and interpolating inside gaps.",
        "soiling_relevance": "Trend-friendly PR used for DQ analysis and smoothing.",
    },
    "plant_pr_physical_raw": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "daily_generation_kwh / (plant_capacity_kw * irradiation_kwh_m2).",
        "soiling_relevance": "Plant-level physical PR diagnostic.",
    },
    "plant_pr_physical_outlier": {
        "unit": "boolean", "dtype": "bool",
        "description": "True when plant_pr_physical_raw is outside [0,1].",
        "soiling_relevance": "Plant-level non-physical PR indicator.",
    },
    "gen_irr_ratio": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "Backward-compatible alias of subset_pr_physical_raw.",
        "soiling_relevance": "Legacy name retained; semantics are now physical PR.",
        "null_reason": "Requires both new telemetry sources (~30% null).",
    },
    "gen_irr_ratio_smoothed": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "7-day centered rolling median of gen_irr_ratio.",
    },
    "power_at_ref_irradiance_w": {
        "unit": "Watts", "dtype": "float64",
        "description": "Mean active power when on-site irradiance is within ±15% of dataset median. Controls for daily irradiance variation.",
        "soiling_relevance": "On a clean system, this should be stable. Soiling reduces it over time. Not affected by irradiance unit issues.",
        "null_reason": "Days where no sub-daily readings match the reference band (~20% null).",
    },
    "new_performance_index": {
        "unit": "ratio (0-1+)", "dtype": "float64",
        "description": "gen_irr_ratio / new_rolling_clean_baseline. 1.0 = clean performance, <1.0 = degradation. Capped at 1.5.",
        "soiling_relevance": "Intuitive: 0.85 means operating at 85% of clean baseline. Equivalent to 1 - new_loss_proxy/100.",
        "null_reason": "Requires both new telemetry sources and sufficient baseline history.",
    },
    "new_performance_loss_pct_proxy": {
        "unit": "% (0=clean, 100=total loss)", "dtype": "float64",
        "description": "100 × (1 - new_normalized_output / new_rolling_clean_baseline). Loss proxy from new telemetry.",
        "soiling_relevance": "Parallel loss proxy from full-day source. Agreement with old proxy confirms both track same signal.",
    },
    "new_cycle_deviation_pct": {
        "unit": "%", "dtype": "float64",
        "description": "Within-cycle deviation from peak performance using new-source gen_irr_ratio.",
        "soiling_relevance": "Compare against old cycle_deviation_pct to validate signal consistency.",
    },

    # ── Physics-Based Reference ───────────────────────────────────────
    "pvlib_soiling_ratio_hsu": {
        "unit": "ratio (0-1)", "dtype": "float64",
        "description": "pvlib HSU soiling model output. 1.0 = clean, <1.0 = soiled. Uses PM2.5, rain, tilt angle.",
        "soiling_relevance": "Literature-based physics model. Cross-validates data-driven approach.",
        "caveats": "Model assumes specific deposition physics. May not match local conditions exactly.",
    },
    "pvlib_soiling_loss_kimber": {
        "unit": "% loss", "dtype": "float64",
        "description": "pvlib Kimber soiling loss model output.",
        "soiling_relevance": "Alternative physics model for cross-validation.",
    },
    "pr_temperature_corrected": {
        "unit": "ratio (dimensionless)", "dtype": "float64",
        "description": "Temperature-corrected performance ratio from pvlib. Removes temperature confounding.",
        "soiling_relevance": "Isolates soiling from temperature effects. Declining trend = soiling-specific degradation.",
    },

    # ── Clear-Sky Analysis ────────────────────────────────────────────
    "is_clear_sky_analyzable": {
        "unit": "boolean", "dtype": "bool",
        "description": "Day passes CSA filter: cloud_opacity < 35%, precipitation < 1mm, days_since_rain ≥ 1.",
        "soiling_relevance": "CSA days have minimal weather noise, giving cleanest view of soiling-only effects.",
        "caveats": "Only ~15% of days pass CSA filter. Results have lower N but higher signal-to-noise.",
    },

    # ── Temporal Features ─────────────────────────────────────────────
    "month": {"unit": "integer (1-12)", "dtype": "int64", "description": "Month of year."},
    "year": {"unit": "integer", "dtype": "int64", "description": "Calendar year."},
    "season": {
        "unit": "category (dry/wet)", "dtype": "object",
        "description": "Dry season: Jan-Mar, Jun-Sep. Wet season: Apr-May, Oct-Dec (monsoon).",
        "soiling_relevance": "Dry season has more soiling accumulation; wet season has frequent rain cleaning.",
        "caveats": "Season definition is site-specific (tropical monsoon climate: Lat ~7°N, Sri Lanka).",
    },
}

# ── Conventions & Interpretation Guide ────────────────────────────────

CONVENTIONS = {
    "hq_filter": {
        "description": "High-Quality (HQ) days: transfer_quality_tier == 'high' AND flag_count == 0.",
        "purpose": "Primary analysis subset. Excludes days with data quality issues.",
        "n_total_vs_hq": "Typically ~60% of days pass HQ filter.",
    },
    "csa_filter": {
        "description": "Clear-Sky Analyzable: cloud_opacity < 35%, precipitation < 1mm, days_since_rain ≥ 1.",
        "purpose": "Removes weather noise. Gives cleanest soiling signal but with smaller N.",
    },
    "significant_rain": {
        "description": "Precipitation ≥ 5mm. Threshold for effective panel cleaning.",
        "convention": "Light rain (<2mm) may redistribute dust rather than clean. Only significant rain resets soiling cycles.",
    },
    "dry_spell": {
        "description": "Consecutive days with rain_day = False (no precipitation).",
        "minimum_length": "3 days (spells shorter than 3 days excluded from sawtooth analysis).",
    },
    "soiling_cycle": {
        "description": "Period between two rain/cleaning events. Identified by cycle_id. Loss proxy should show upward trend (soiling accumulation) within each cycle.",
    },
    "loss_proxy_interpretation": {
        "0_pct": "Panel operating at clean baseline (no loss detected).",
        "10_pct": "Mild soiling — typical for 1-2 weeks without rain in moderate dust.",
        "30_pct": "Significant soiling — likely needs cleaning intervention.",
        "50_plus_pct": "Severe — may include non-soiling factors (equipment fault, sensor error).",
        "caveat": "This is ALL-CAUSE loss. Temperature, shading, equipment issues conflated with soiling.",
    },
    "tier_system": {
        "tier_1": "B2 block (B2-08, B2-13, B2-17). Training set. Highest data availability (0.77-1.0).",
        "tier_2": "B1 block (B1-08, B1-01, B1-13). Validation set. Lower availability (0.10-1.0).",
        "rationale": "Separate blocks ensure model generalizes. T1/T2 loss correlation ≈ 0.98 confirms plant-wide soiling.",
    },
    "normalized_output_units": {
        "description": "Expressed as J/J (energy / irradiance_sum). NOT a true efficiency metric.",
        "caveat": "The denominator (irradiance_tilted_sum) is a sum of 15-min interval readings, not integrated energy. Absolute values are not interpretable as efficiency — only TRENDS and RELATIVE changes matter.",
    },
    "site_location": {
        "latitude": "~7°N (Sri Lanka, tropical)",
        "climate": "Tropical monsoon. Two monsoon seasons with heavy rain.",
        "dust_sources": "Agricultural dust, road dust, construction. PM levels moderate-to-high.",
    },
    "peak_hours": {
        "description": "10:00 to 14:00 local time. Used for sub-daily filtering in original pipeline.",
        "rationale": "Peak sun hours give best signal-to-noise for soiling detection. Low-angle morning/evening readings add noise.",
    },
    "known_cleaning_events": {
        "description": "Manual panel cleanings recorded by site operator.",
        "dates": ["2025-08-19 to 2025-08-25", "2025-11-20 to 2025-11-30"],
        "effect": "Loss proxy should drop sharply during these periods. Used to validate soiling signal.",
    },
}

SIGNAL_DESCRIPTIONS = {
    "signal_1_sawtooth": {
        "name": "Sawtooth Pattern Detection",
        "what_it_tests": "Whether the loss proxy shows gradual increase during dry spells (soiling accumulation) followed by drops after rain (cleaning).",
        "pass_criteria": "≥50% of dry spells show positive soiling slope.",
        "key_metrics": ["median_rate_pct_per_day", "pct_positive_slope", "n_spells"],
        "interpretation": "A positive soiling rate means panels are losing output during dry periods. This is the most basic soiling signal.",
    },
    "signal_2_dust_correlation": {
        "name": "PM/Dust Correlation Analysis",
        "what_it_tests": "Whether airborne particulate matter (PM10, PM2.5) correlates with performance loss.",
        "pass_criteria": "Best partial correlation (deconfounded for cloud/temperature) > 0.15.",
        "key_metrics": ["best_partial_r", "r_cumpm25_vs_deviation", "r_within_cycle"],
        "interpretation": "Strong PM-loss correlation means dust deposition is a real driver of performance loss, not just coincidence.",
    },
    "signal_3_rain_recovery": {
        "name": "Rain Recovery Signal",
        "what_it_tests": "Whether rain events lead to measurable performance recovery (lower loss proxy after rain than before).",
        "pass_criteria": "Event-study Wilcoxon p < 0.05 OR dry-spell accumulation p < 0.10.",
        "key_metrics": ["event_study_p", "dryspell_wilcoxon_p", "n_rain_events"],
        "interpretation": "If rain recovers performance, it confirms soiling is the reversible component of loss (not permanent degradation).",
    },
}


def build_feature_glossary() -> Dict[str, Any]:
    """Build the complete feature glossary for inclusion in the LLM summary."""
    return {
        "_description": (
            "Complete metadata for every column, metric, and convention used in this analysis. "
            "READ THIS SECTION FIRST before interpreting any numbers. "
            "Without understanding units, conventions, and caveats, the statistical results are meaningless."
        ),
        "feature_glossary": FEATURE_GLOSSARY,
        "conventions_and_interpretation": CONVENTIONS,
        "signal_test_descriptions": SIGNAL_DESCRIPTIONS,
    }
