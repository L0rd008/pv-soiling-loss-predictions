import json
from pathlib import Path

json_path_str = r"M:\Documents\Projects\MAGICBIT\Soling Loss Predictions\artifacts\eda\llm_eda_summary.json"
out_path = Path("dq_summary.txt")

with open(json_path_str, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(out_path, "w", encoding="utf-8") as out:
    out.write("--- Quality Gating ---\n")
    qg = data.get("supporting_analyses", {}).get("quality_gating", {})
    out.write(json.dumps(qg, indent=2) + "\n")
    
    out.write("\n--- Feature Data Points (HQ days) ---\n")
    dist = data.get("supporting_analyses", {}).get("distributions", {})
    for feat, stats in dist.items():
        if isinstance(stats, dict):
            out.write(f"{feat}: (n={stats.get('n', 0)}, n_null={stats.get('n_null', 0)})\n")
    
    out.write("\n--- Signal 1 Daily Series (length) ---\n")
    s1_series = data.get("signal_1_sawtooth", {}).get("daily_series", {})
    if s1_series:
        out.write(f"Total Dates in daily_series: {len(s1_series.get('dates', []))}\n")
        for k, v in s1_series.items():
            if k != "dates":
               non_nulls = [x for x in v if x is not None]
               out.write(f"{k}: valid={len(non_nulls)}, nulls={len(v)-len(non_nulls)}\n")
    
    out.write("\n--- Summary of verdicts ---\n")
    out.write(f"Signal 1: {data.get('signal_1_sawtooth', {}).get('verdict')}\n")
    out.write(f"Signal 2: {data.get('signal_2_dust_correlation', {}).get('verdict')}\n")
    out.write(f"Signal 3: {data.get('signal_3_rain_recovery', {}).get('verdict')}\n")
