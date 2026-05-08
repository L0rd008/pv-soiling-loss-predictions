import pandas as pd

with open("out_hq.txt", "w", encoding="utf-8") as f:
    df = pd.read_csv(r"artifacts\preprocessed\daily_model_eda.csv")
    hq = df[(df["transfer_quality_tier"] == "high") & (df["flag_count"] == 0)]
    
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"HQ rows: {len(hq)}\n")
    
    loss_col = "t1_performance_loss_pct_proxy"
    eval_cols = ["pm10_mean", "pm25_mean", "precipitation_total_mm", "cloud_opacity_mean"]
    
    f.write("\n--- Total Valid counts in HQ subset ---\n")
    f.write(f"{loss_col}: {hq[loss_col].notna().sum()}\n")
    for c in eval_cols:
        if c in hq.columns:
            f.write(f"{c}: {hq[c].notna().sum()}\n")
    
    f.write("\n--- Joint Valid counts with Loss Proxy in HQ subset ---\n")
    for c in eval_cols:
        if c in hq.columns:
            joint = hq[[loss_col, c]].dropna()
            f.write(f"{c} + {loss_col}: {len(joint)}\n")
    
    f.write(f"\nHQ dates: {hq['day'].tolist()[:5]}...\n")
    
    pm10_valid = df[df["pm10_mean"].notna()]
    f.write(f"Total days with valid pm10_mean: {len(pm10_valid)}\n")
    
    loss_valid = df[df[loss_col].notna()]
    f.write(f"Total days with valid {loss_col}: {len(loss_valid)}\n")
    
    c_dev = "cycle_deviation_pct"
    f.write(f"Total days with valid {c_dev}: {df[c_dev].notna().sum()}\n")
    
    f.write("\n--- Joint Valid counts with Cycle Deviation in HQ subset ---\n")
    for c in eval_cols:
        if c in hq.columns:
            joint = hq[[c_dev, c]].dropna()
            f.write(f"{c} + {c_dev}: {len(joint)}\n")
