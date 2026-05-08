import pandas as pd
import numpy as np

# Load the daily model EDA dataset
df = pd.read_csv(r"artifacts\preprocessed\daily_model_eda.csv")

# 1. Look at _hq_filter logic if run on this data
hq = df[(df["transfer_quality_tier"] == "high") & (df["flag_count"] == 0)]

print("--- Telemetry Compatibility Check ---")
valid_both = df.dropna(subset=["t1_normalized_output", "new_normalized_output", "t1_performance_loss_pct_proxy", "new_performance_loss_pct_proxy"])

print(f"Days where BOTH Old and New telemetry are entirely valid: {len(valid_both)}")

if len(valid_both) > 0:
    # Compare normalized output (yield per unit irradiance)
    old_norm = valid_both["t1_normalized_output"]
    new_norm = valid_both["new_normalized_output"]
    
    ratio = new_norm / old_norm
    print(f"\nAverage Ratio (New / Old Normalized Output): {ratio.mean():.4f}")
    print(f"Median Ratio (New / Old Normalized Output): {ratio.median():.4f}")
    print(f"Standard Deviation of Ratio: {ratio.std():.4f}")
    
    # Compare raw energy
    if "t1_energy_j" in df.columns and "subset_daily_gen_j" in df.columns:
        valid_energy = df.dropna(subset=["t1_energy_j", "subset_daily_gen_j"])
        # t1_energy_j is for 3 inverters, subset_daily_gen is for 6. To compare, look at scale.
        # Actually, let's just compare the loss proxies directly since they are normalized to themselves.
        
    old_loss = valid_both["t1_performance_loss_pct_proxy"]
    new_loss = valid_both["new_performance_loss_pct_proxy"]
    
    loss_diff = new_loss - old_loss
    print(f"\nAverage Loss Diff (New - Old Proxy %): {loss_diff.mean():.2f}%")
    print(f"Median Loss Diff (New - Old Proxy %): {loss_diff.median():.2f}%")
    print("Correlation between Old and New Loss Proxies:", old_loss.corr(new_loss))
else:
    print("Not enough days to compare.")
