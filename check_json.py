import json

d = json.load(open("artifacts/eda/llm_eda_summary.json"))

# DQ3
dq3 = d.get("dq3_gen_irr_ratio", {}).get("results", {})
print("=== DQ3 new keys ===")
for k in ["stl_trend_range", "stl_seasonal_amplitude"]:
    v = dq3.get(k, "MISSING")
    print("  %s: %s" % (k, v))

# DQ4
dq4 = d.get("dq4_power_at_ref_irradiance", {}).get("results", {})
print("")
print("=== DQ4 new keys ===")
for k in ["match_count_median", "match_count_q1_median", "match_count_q4_median",
           "pm10_seasonal_confounding", "primary_pm_predictor"]:
    v = dq4.get(k, "MISSING")
    print("  %s: %s" % (k, v))
partial_keys = [k for k in dq4 if "partial_r" in k]
print("  partial_r keys (%d): %s" % (len(partial_keys), partial_keys))

# DQ5
dq5 = d.get("dq5_old_vs_new_comparison", {}).get("results", {})
print("")
print("=== DQ5 new keys ===")
for k in ["old_vs_new_loss_r_nonzero", "n_nonzero_loss_days"]:
    v = dq5.get(k, "MISSING")
    print("  %s: %s" % (k, v))

# DQ6
dq6 = d.get("dq6_performance_index", {}).get("results", {})
print("")
print("=== DQ6 new keys ===")
for k in ["pct_above_1", "frozen_baseline_value", "n_above_1",
           "above_1_post_cleaning", "above_1_seasonal",
           "rain_wilcoxon_stat", "rain_wilcoxon_p", "rain_wilcoxon_pass"]:
    v = dq6.get(k, "MISSING")
    print("  %s: %s" % (k, v))
q90_keys = [k for k in dq6 if "q90_slope" in k]
q50_keys = [k for k in dq6 if "q50_slope" in k]
print("  q90_slope keys (%d): %s" % (len(q90_keys), q90_keys))
print("  q50_slope keys (%d): %s" % (len(q50_keys), q50_keys))

# Summary
print("")
print("=== Total key counts ===")
print("  DQ3: %d keys" % len(dq3))
print("  DQ4: %d keys" % len(dq4))
print("  DQ5: %d keys" % len(dq5))
print("  DQ6: %d keys" % len(dq6))
