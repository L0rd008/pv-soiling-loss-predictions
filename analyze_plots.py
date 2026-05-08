import json
from pathlib import Path

json_path_str = r"M:\Documents\Projects\MAGICBIT\Soling Loss Predictions\artifacts\eda\llm_eda_summary.json"
out_path = Path("plot_analysis.txt")

with open(json_path_str, "r", encoding="utf-8") as f:
    data = json.load(f)

# The total days are available in the quality gating
qg = data.get("supporting_analyses", {}).get("quality_gating", {})
total_days = qg.get("total_days", 0)
hq_days = qg.get("hq_days", 0)
csa_days = qg.get("csa_days", 0)

with open(out_path, "w", encoding="utf-8") as out:
    out.write(f"Total Days in Dataset: {total_days}\n")
    out.write(f"High Quality (HQ) Days Available for most plots: {hq_days}\n")
    out.write(f"Clear-Sky Analyzable (CSA) Days: {csa_days}\n")
    out.write("\n=== Plot Analysis ===\n\n")

    # s1_loss_proxy_timeseries.png -> HQ days, plots 't1_performance_loss_pct_proxy'
    s1_series = data.get("signal_1_sawtooth", {}).get("daily_series", {})
    loss_proxy = s1_series.get("t1_performance_loss_pct_proxy", [])
    valid_loss = sum(1 for x in loss_proxy if x is not None)
    out.write(f"Plot: s1_loss_proxy_timeseries.png\n")
    out.write(f"  Available Data Filter: HQ days ({hq_days} available)\n")
    out.write(f"  Valid points plotted: {valid_loss} / {len(loss_proxy)} in series\n")
    out.write(f"  Missing points in subset: {len(loss_proxy) - valid_loss}\n\n")

    # s1_cycle_deviation_timeseries.png -> HQ days, plots 'cycle_deviation_pct'
    c_dev = s1_series.get("cycle_deviation_pct", [])
    valid_c_dev = sum(1 for x in c_dev if x is not None)
    out.write(f"Plot: s1_cycle_deviation_timeseries.png\n")
    out.write(f"  Available Data Filter: HQ days ({hq_days} available)\n")
    out.write(f"  Valid points plotted: {valid_c_dev} / {len(c_dev)} in series\n")
    out.write(f"  Missing points in subset: {len(c_dev) - valid_c_dev}\n\n")

    # s1_dry_spell_slopes.png -> Plots dry spells >= 3 days
    atomic = data.get("multilevel_analysis", {}).get("atomic_level", {})
    dry_spells = atomic.get("dry_spells", [])
    out.write(f"Plot: s1_dry_spell_slopes.png\n")
    out.write(f"  Condition: HQ days, no rain for >= 3 days, sufficient non-null loss values\n")
    out.write(f"  Valid points (spells) plotted: {len(dry_spells)}\n\n")

    # s2 scatter plots
    s2 = data.get("signal_2_dust_correlation", {})
    s2_corr = s2.get("correlations_with_loss", {})
    out.write(f"Plot: s2_pm10_vs_loss_scatter.png\n")
    pm10_stats = s2_corr.get("pm10_mean", {})
    out.write(f"  Condition: HQ days with valid PM10 and Loss\n")
    out.write(f"  Valid points plotted: {pm10_stats.get('n', 0)}\n\n")

    out.write(f"Plot: s2_pm25_vs_loss_scatter.png\n")
    pm25_stats = s2_corr.get("pm25_mean", {})
    out.write(f"  Condition: HQ days with valid PM25 and Loss\n")
    out.write(f"  Valid points plotted: {pm25_stats.get('n', 0)}\n\n")

    out.write(f"Plot: s2_partial_correlation.png\n")
    partial = s2.get("partial_correlations_controlling_for_weather", {}).get("pm10_mean", {})
    out.write(f"  Condition: HQ days, dropping NaNs across Loss, PM, Cloud, Temp, Humidity\n")
    out.write(f"  Valid points plotted: {partial.get('n', 0)}\n\n")

    # s3 rain recovery
    s3 = data.get("signal_3_rain_recovery", {})
    rain_events = atomic.get("rain_events", [])
    out.write(f"Plot: s3_rain_recovery_events.png\n")
    out.write(f"  Condition: HQ days with rain >= 5mm\n")
    out.write(f"  Valid points (events) plotted: {len(rain_events)}\n\n")
    
    # Let's also parse `soiling_signals.py` separately or just infer from this data structure.
