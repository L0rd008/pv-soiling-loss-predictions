import requests
import time
import os
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

TB_URL = os.getenv("TB_URL")
TOKEN = os.getenv("TB_TOKEN")
KEYS = os.getenv("TB_INV_KEYS") 
INVERTERS_ENV = os.getenv("TB_INVERTERS")

if not all([TB_URL, TOKEN, KEYS, INVERTERS_ENV]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Output and request defaults
OUTPUT_DIR = Path(os.getenv("TB_OUTPUT_DIR", "data"))
REQUEST_TIMEOUT_S = int(os.getenv("TB_REQUEST_TIMEOUT_S", "30"))

# Parse the Inverters from the .env file into a dictionary
inverters = {}
for item in INVERTERS_ENV.split(','):
    name, dev_id = item.split(':')
    inverters[name.strip()] = dev_id.strip()

# Value sanity thresholds
MAX_POWER_W = 300000.0    # 300 kW max limit 
MAX_CURRENT_A = 250.0      # 250 A max limit 

# Time Configuration (LK Time)
tz_offset = timezone(timedelta(hours=5, minutes=30))
end_ts = int(time.time() * 1000) 
start_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=tz_offset)
start_ts = int(start_dt.timestamp() * 1000)

interval = 10 * 60 * 1000 # 10 minutes
agg = "AVG"
limit = 100000 

headers = {
    "Content-Type": "application/json",
    "X-Authorization": f"Bearer {TOKEN}"
}

# --- Chunking Logic Setup ---
# Fetch data in 3-day chunks to safely bypass the < 500 interval limit
CHUNK_MS = 3 * 24 * 60 * 60 * 1000 
merged_data = {}
keys_list = [k.strip() for k in KEYS.split(',')]
device_names = list(inverters.keys())

print(f"Starting fetch for {len(inverters)} inverters in 3-day chunks from 2025-01-01...")

# Data Rescue wrapper
try:
    # Loop through each inverter
    for inv_name, inv_id in inverters.items():
        print(f"\n--- Fetching data for {inv_name} ---")
        current_start = start_ts
        
        while current_start < end_ts:
            current_end = current_start + CHUNK_MS
            if current_end > end_ts:
                current_end = end_ts
                
            dt_start_str = datetime.fromtimestamp(current_start / 1000.0, tz=tz_offset).strftime('%Y-%m-%d')
            dt_end_str = datetime.fromtimestamp(current_end / 1000.0, tz=tz_offset).strftime('%Y-%m-%d')
            print(f"  Requesting chunk: {dt_start_str} to {dt_end_str}...")

            url = f"{TB_URL}/api/plugins/telemetry/DEVICE/{inv_id}/values/timeseries"
            params = {
                "keys": KEYS,
                "startTs": current_start,
                "endTs": current_end,
                "interval": interval,
                "agg": agg,
                "limit": limit
            }
            
            # --- RETRY LOGIC ---
            max_retries = 3
            chunk_success = False
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT_S,
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for key in keys_list:
                            if key in data:
                                for point in data[key]:
                                    ts = point['ts']
                                    dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz_offset)
                                    hour = dt_local.hour
                                    
                                    try:
                                        val = float(point['value'])
                                    except (ValueError, TypeError):
                                        val = ""
                                    
                                    # Apply Scaling to Currents
                                    if isinstance(val, float) and "current" in key.lower():
                                        val = val / 1000.0
                                        
                                    # --- AI Post-Processing ---
                                    if isinstance(val, float):
                                        if "power" in key.lower() and val > MAX_POWER_W:
                                            val = ""
                                        elif "current" in key.lower() and val > MAX_CURRENT_A:
                                            val = ""
                                        elif val < 0:
                                            if hour >= 19 or hour < 5:
                                                val = 0.0  
                                            else:
                                                val = ""   
                                                
                                    if ts not in merged_data:
                                        merged_data[ts] = {}
                                    
                                    column_id = f"{inv_name}_{key}"
                                    merged_data[ts][column_id] = val
                                    
                        chunk_success = True
                        break # Break retry loop
                        
                    else:
                        print(f"  Server error on chunk: {response.status_code} - {response.text}")
                        break # Don't retry on a hard 400 server error
                        
                except requests.exceptions.RequestException as e:
                    print(f"    Connection dropped (Attempt {attempt + 1}/{max_retries}). Retrying in 3 seconds...")
                    time.sleep(3)
            
            if not chunk_success:
                print(f"  Failed to fetch chunk. Moving to next to save existing data.")
            
            current_start = current_end
            
            # --- PACING ---
            time.sleep(1) 

except KeyboardInterrupt:
    print("\nScript manually stopped. Proceeding to save what we have...")
except Exception as e:
    print(f"\nUnexpected error: {e}. Proceeding to save what we have...")

# --- DATA RESCUE & EXPORT ---
if merged_data:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / 'inverters_2025_to_current_10min_avg_si.csv'
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            formatted_keys = []
            for inv_name in device_names:
                for k in keys_list:
                    readable_key = k.replace('_', ' ').title()
                    
                    if "power" in readable_key.lower():
                        readable_key += " (W)"
                    elif "current" in readable_key.lower():
                        readable_key += " (A)"
                        
                    formatted_keys.append(f"{inv_name} {readable_key}")
            
            headers_row = ["Timestamp", "Date"] + formatted_keys
            writer.writerow(headers_row)
            
            for ts in sorted(merged_data.keys()):
                dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz_offset)
                date_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                row = [ts, date_str]
                
                for inv_name in device_names:
                    for key in keys_list:
                        column_id = f"{inv_name}_{key}"
                        row.append(merged_data[ts].get(column_id, ""))
                        
                writer.writerow(row)
                
        print(f"\nData successfully cleaned, merged, and exported to {filename}")
    except PermissionError:
        print(f"\nERROR: Could not save file. Is {filename} open? Close it and try again.")
else:
    print("No data was processed to export.")
