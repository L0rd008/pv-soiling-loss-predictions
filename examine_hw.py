import pandas as pd
import sys

sys.stdout = open("hw_dump.txt", "w", encoding="utf-8")

hw_df = pd.read_csv("data/inverters_daily_gen_2025_to_current_none_si.csv")
hw_df["Timestamp"] = pd.to_datetime(hw_df["Timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Colombo")

col = [c for c in hw_df.columns if "B2-08" in c][0]

# Print a few continuous days of data to see if it resets, or if it's a lifetime counter, and how many records per day
sample = hw_df.dropna(subset=[col]).sort_values("Timestamp").head(50)

print("--- First 50 non-null records for B2-08 ---")
for _, row in sample.iterrows():
    print(f"{row['Timestamp']}  |  {row[col]:.2f}")

print("\n--- Daily Max/Min/Count for first 10 days ---")
hw_df["day"] = hw_df["Timestamp"].dt.normalize().dt.tz_localize(None)
day_stats = hw_df.dropna(subset=[col]).groupby("day")[col].agg(['min', 'max', 'count']).head(10)
print(day_stats)

sys.stdout.close()
