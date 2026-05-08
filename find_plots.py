import re

with open(r"scripts\5_eda\soiling_signals.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

current_func = None
uses_hq = False
plots_in_func = []

print("--- Plots grouped by function and their dependency on _hq_filter ---")

def dump_state():
    if current_func and plots_in_func:
        print(f"Function: {current_func}")
        print(f"  Uses _hq_filter: {uses_hq}")
        print(f"  Plots generated:")
        for p in plots_in_func:
            print(f"    - {p}")
        print("-" * 40)

for line in lines:
    if line.startswith("def "):
        dump_state()
        current_func = line.split("(")[0].split(" ")[1]
        uses_hq = False
        plots_in_func = []
    
    if "_hq_filter(" in line:
        uses_hq = True
        
    if "_save(" in line:
        match = re.search(r'"([^"]+\.png)"', line)
        if match:
            plots_in_func.append(match.group(1))

dump_state()
