import pandas as pd
import numpy as np
import sys

sys.stdout = open("scaling_out.txt", "w", encoding="utf-8")

# 1. Score vs Flag Count Analysis
df = pd.read_csv(r"artifacts\preprocessed\daily_model_eda.csv")

print("=== PART 1: SCORE VS FLAG COUNT BOTTLENECK ===")
print(f"Total dataset days: {len(df)}")

hq_current = df[(df["transfer_quality_tier"] == "high") & (df["flag_count"] == 0)]
hq_lenient_score_strict_flags = df[(df["transfer_quality_score"] >= 0) & (df["flag_count"] == 0)]
hq_strict_score_lenient_flags = df[(df["transfer_quality_tier"] == "high") & (df["flag_count"] <= 1)]

print(f"Current HQ days (Score >= 80 AND flag_count == 0): {len(hq_current)}")
print(f"Lenient Score HQ (Score >= 0 AND flag_count == 0):   {len(hq_lenient_score_strict_flags)}")
print(f"Lenient Flags HQ (Score >= 80 AND flag_count <= 1):  {len(hq_strict_score_lenient_flags)}")

# 2. Telemetry Scaling Analysis
print("\n=== PART 2: TELEMETRY SCALING ANALYSIS ===")

inv_df = pd.read_csv("data/inverters_tiered_primary_10min.csv")
inv_df["Timestamp"] = pd.to_datetime(inv_df["Timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Colombo")
inv_df["day"] = inv_df["Timestamp"].dt.normalize().dt.tz_localize(None)

b2_08_power = "B2-08 Active Power (W)"
b2_08_integrated_j = inv_df.groupby("day")[b2_08_power].sum() * 600
b2_08_integrated_kwh = b2_08_integrated_j / 3_600_000

hw_df = pd.read_csv("data/inverters_daily_gen_2025_to_current_none_si.csv")
hw_df["Timestamp"] = pd.to_datetime(hw_df["Timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Colombo")
hw_df["day"] = hw_df["Timestamp"].dt.normalize().dt.tz_localize(None)

b2_08_hw_col = [c for c in hw_df.columns if "B2-08" in c or "b2_08" in c.lower()][0]

# Hardware register is a lifetime cumulative counter. Daily yield = Max - Min for the day.
b2_08_hw_kwh = hw_df.groupby("day")[b2_08_hw_col].max() - hw_df.groupby("day")[b2_08_hw_col].min()

comp = pd.DataFrame({
    "Integrated_kWh": b2_08_integrated_kwh,
    "Hardware_kWh": b2_08_hw_kwh
}).dropna()

comp = comp[(comp["Integrated_kWh"] > 0) & (comp["Hardware_kWh"] > 0)]

print(f"Found {len(comp)} valid days with both Old Integrated Power and New HW Register for B2-08.")

if len(comp) > 0:
    comp["Ratio (Integrated / HW)"] = comp["Integrated_kWh"] / comp["Hardware_kWh"]
    print(f"Mean Integrated kWh: {comp['Integrated_kWh'].mean():.2f} kWh")
    print(f"Mean Hardware kWh:   {comp['Hardware_kWh'].mean():.2f} kWh")
    print(f"Average Ratio (Integrated/HW):  {comp['Ratio (Integrated / HW)'].mean():.3f}")
    print(f"Correlation:                    {comp['Integrated_kWh'].corr(comp['Hardware_kWh']):.3f}")
    
    records_per_day = inv_df.groupby("day")[b2_08_power].count()
    comp["Records_Logged"] = records_per_day
    print(f"Average 10-min records logged per day (max expected ~144): {comp['Records_Logged'].mean():.1f}")
    
    comp["Missing_Records"] = 144 - comp["Records_Logged"]
    print(f"Correlation between Ratio and Missing Records: {comp['Ratio (Integrated / HW)'].corr(comp['Missing_Records']):.3f}")

sys.stdout.close()
