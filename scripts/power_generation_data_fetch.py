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
ASSET_ID = os.getenv("TB_PLNT_ID")
KEYS = os.getenv("TB_GEN_KEYS") 

# Safety Check
if not all([TB_URL, TOKEN, ASSET_ID, KEYS]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Output and request defaults
OUTPUT_DIR = Path(os.getenv("TB_OUTPUT_DIR", "data"))
REQUEST_TIMEOUT_S = int(os.getenv("TB_REQUEST_TIMEOUT_S", "30"))
MAX_GENERATION_J = float(os.getenv("TB_GEN_MAX_J", str(100000000 * 3600)))

print(f"Connecting to: {TB_URL} for Asset: {ASSET_ID}")

# Define Local Timezone Offset (+05:30 for LK Time)
tz_offset = timezone(timedelta(hours=5, minutes=30))

# Time Configuration
# End time: Now
end_ts = int(time.time() * 1000) 

# Start time: 2025-01-01 00:00:00 Local Time
start_dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=tz_offset)
start_ts = int(start_dt.timestamp() * 1000)

# Aggregation set to NONE for raw data
agg = "NONE"
limit = 50000  # More than enough for daily records over a few years

# API Endpoint
url = f"{TB_URL}/api/plugins/telemetry/ASSET/{ASSET_ID}/values/timeseries"

params = {
    "keys": KEYS,
    "startTs": start_ts,
    "endTs": end_ts,
    "agg": agg,
    "limit": limit
}

headers = {
    "Content-Type": "application/json",
    "X-Authorization": f"Bearer {TOKEN}"
}

try:
    print("Fetching raw asset generation data from 2025-01-01 to now...")
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT_S,
    )
    
    if response.status_code == 200:
        data = response.json()
        keys_list = [k.strip() for k in KEYS.split(',')]
        merged_data = {}
        
        for key in keys_list:
            if key in data:
                print(f"Found {len(data[key])} raw points for {key}.")
                for point in data[key]:
                    ts = point['ts']
                    
                    try:
                        raw_val = float(point['value'])
                    except (ValueError, TypeError):
                        raw_val = "" # Broken strings become NaN downstream
                    
                    # --- Custom AI Post-Processing ---
                    if isinstance(raw_val, float):
                        scaled_value = raw_val * 1000 * 3600
                    else:
                        scaled_value = ""

                    if isinstance(scaled_value, float) and scaled_value > MAX_GENERATION_J:
                        final_val = ""
                    elif isinstance(scaled_value, float) and scaled_value < 0:
                        final_val = "" 
                    else:
                        final_val = scaled_value
                    
                    if ts not in merged_data:
                        merged_data[ts] = {}
                    
                    merged_data[ts][key] = final_val
            else:
                print(f"Warning: No data found for {key}.")
                
        # Export merged data to CSV
        if merged_data:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            filename = OUTPUT_DIR / 'power_generation_2025_to_current_1day_none_si.csv'
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # --- Readable Column Names Logic ---
                formatted_keys = []
                
                # Dictionary to fix the CamelCase key names
                abbreviations = {
                    "energymeter": "Energy Meter",
                    "dailygeneration": "Daily Generation"
                }
                
                for k in keys_list:
                    # Split the key by underscores
                    parts = k.split('_')
                    processed_parts = []
                    
                    for part in parts:
                        # If the part is in our dictionary, format it nicely
                        if part.lower() in abbreviations:
                            processed_parts.append(abbreviations[part.lower()])
                        else:
                            processed_parts.append(part.title())
                            
                    readable_name = " ".join(processed_parts)
                    formatted_keys.append(f"{readable_name} (J)")
                
                headers_row = ["Timestamp", "Date"] + formatted_keys
                writer.writerow(headers_row)
                
                for ts in sorted(merged_data.keys()):
                    dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz_offset)
                    date_str = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                    
                    row = [ts, date_str]
                    
                    for key in keys_list:
                        row.append(merged_data[ts].get(key, ""))
                        
                    writer.writerow(row)
                    
            print(f"\nData successfully cleaned, scaled, and exported to {filename}")
        else:
            print("No data was processed to export.")
            
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"Failed to fetch data: {e}")
