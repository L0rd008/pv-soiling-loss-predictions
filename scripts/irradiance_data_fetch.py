import requests
import time
import os
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

# 2. Import Config from .env
TB_URL = os.getenv("TB_URL")
TOKEN = os.getenv("TB_TOKEN")
DEVICE_ID = os.getenv("TB_WSTN_ID")
KEYS = os.getenv("TB_IRR_KEYS") 

# Safety Check
if not all([TB_URL, TOKEN, DEVICE_ID, KEYS]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Output and request defaults
OUTPUT_DIR = Path(os.getenv("TB_OUTPUT_DIR", "data"))
REQUEST_TIMEOUT_S = int(os.getenv("TB_REQUEST_TIMEOUT_S", "30"))

print(f"Connecting to: {TB_URL} for Device: {DEVICE_ID}")

# Define Local Timezone Offset (+05:30)
tz_offset = timezone(timedelta(hours=5, minutes=30))

# Time Configuration
end_ts = int(time.time() * 1000) 
start_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=tz_offset)
start_ts = int(start_dt.timestamp() * 1000)

interval = 15 * 60 * 1000 # 15 mins
agg = "SUM"
limit = 100000 

headers = {
    "Content-Type": "application/json",
    "X-Authorization": f"Bearer {TOKEN}"
}

# --- Chunking Logic Setup ---
CHUNK_MS = 5 * 24 * 60 * 60 * 1000 
current_start = start_ts

merged_data = {}
keys_list = [k.strip() for k in KEYS.split(',')]

print("Fetching data from 2025-01-01 to now in 5-day chunks...")

try:
    while current_start < end_ts:
        current_end = current_start + CHUNK_MS
        if current_end > end_ts:
            current_end = end_ts
            
        dt_start_str = datetime.fromtimestamp(current_start / 1000.0, tz=tz_offset).strftime('%Y-%m-%d')
        dt_end_str = datetime.fromtimestamp(current_end / 1000.0, tz=tz_offset).strftime('%Y-%m-%d')
        print(f"Requesting chunk: {dt_start_str} to {dt_end_str}...")

        url = f"{TB_URL}/api/plugins/telemetry/DEVICE/{DEVICE_ID}/values/timeseries"
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
                                
                                if isinstance(val, float) and val < 0:
                                    if hour >= 19 or hour < 5:
                                        val = 0.0
                                    else:
                                        val = "" 
                                
                                if ts not in merged_data:
                                    merged_data[ts] = {}
                                merged_data[ts][key] = val
                                
                    chunk_success = True
                    break # Break out of the retry loop if successful
                    
                else:
                    print(f"Server error on chunk: {response.status_code} - {response.text}")
                    break # Don't retry on a hard 400 server error
                    
            except requests.exceptions.RequestException as e:
                print(f"  Connection dropped (Attempt {attempt + 1}/{max_retries}). Retrying in 3 seconds...")
                time.sleep(3)
        
        # If all retries failed, exit the while loop but keep the data we have
        if not chunk_success:
            print("Failed to fetch this chunk after multiple retries. Halting fetch to save existing data.")
            break

        current_start = current_end
        
        # --- PACING ---
        # Wait 1 second before asking the server for the next chunk to prevent OS socket exhaustion
        time.sleep(1)

except KeyboardInterrupt:
    print("\nScript manually stopped. Proceeding to save what we have...")
except Exception as e:
    print(f"\nUnexpected error: {e}. Proceeding to save what we have...")

# --- DATA RESCUE & EXPORT ---
# This runs no matter what happens in the loop above!
if merged_data:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f'irradiance_2025_to_current_15min_{agg.lower()}_si.csv'
    
    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            formatted_keys = []
            abbreviations = {
                "wstn1": "Weather Station 1",
                "wstn2": "Weather Station 2",
                "wstn": "Weather Station",
                "horiz": "Horizontal",
                "irr": "Irradiance",
                "temp": "Temperature"
            }
            
            for k in keys_list:
                parts = k.split('_')
                processed_parts = []
                for part in parts:
                    if part.lower() in abbreviations:
                        processed_parts.append(abbreviations[part.lower()])
                    else:
                        processed_parts.append(part.title())
                readable_name = " ".join(processed_parts)
                formatted_keys.append(f"{readable_name} ({agg})")
            
            headers_row = ["Timestamp", "Date"] + formatted_keys
            writer.writerow(headers_row)
            
            for ts in sorted(merged_data.keys()):
                dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz_offset)
                date_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                row = [ts, date_str]
                
                for key in keys_list:
                    row.append(merged_data[ts].get(key, ""))
                    
                writer.writerow(row)
                
        print(f"\nData successfully cleaned, merged, and exported to {filename}")
    except PermissionError:
        print(f"\nERROR: Could not save file. Is {filename} currently open in Excel? Close it and try again.")
else:
    print("\nNo data was processed to export.")
