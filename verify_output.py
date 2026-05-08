import json

with open(r"artifacts\eda\llm_eda_summary.json", "r") as f:
    data = json.load(f)

print("EDA Result Verdicts:", data.get("signal_results_summary"))

print("\nHQ Filter Result:")
num_days = len(data.get("time_series_data", {}).get("t1_performance_loss_pct_proxy", []))
if num_days == 0:
    for k, v in data.get("time_series_data", {}).items():
        if len(v) > 0:
            num_days = len(v)
            break

print(f"Data points plotted after HQ Filter (previously ~117 or near 0 for new telemetry): {num_days}")
