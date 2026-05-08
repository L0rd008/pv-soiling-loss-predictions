import os

file_path = "scripts/5_eda/soiling_signals.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace metrics
new_content = content.replace("new_performance_loss_pct_proxy", "t1_performance_loss_pct_proxy")
new_content = new_content.replace("new_perf_loss_rate_14d_pct_per_day", "t1_perf_loss_rate_14d_pct_per_day")
new_content = new_content.replace("new_normalized_output", "t1_normalized_output")

if new_content != content:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Files successfully updated.")
else:
    print("No changes made. Strings not found.")
