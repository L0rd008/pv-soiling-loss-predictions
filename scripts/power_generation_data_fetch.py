"""Fetch daily power generation telemetry from ThingsBoard.

Pulls raw (``agg=NONE``) asset-level generation data.  The ThingsBoard key
``energymeter_dailygeneration`` stores values in **kWh**.  This script converts
them to SI Joules::

    joules = raw_kWh × 1000 (→ Wh) × 3600 (→ J) = raw_kWh × 3,600,000

Values exceeding ``TB_GEN_MAX_J`` or negative values are treated as invalid
and written as empty (NaN downstream).

Usage::

    python scripts/power_generation_data_fetch.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_PLNT_ID, TB_GEN_KEYS
    Optional: TB_OUTPUT_DIR, TB_REQUEST_TIMEOUT_S, TB_GEN_MAX_J,
              TB_TZ_OFFSET, TB_START_DATE
"""

import logging
import sys
from datetime import datetime

from tb_client import (
    auth_headers,
    get_output_dir,
    get_request_timeout,
    get_tz_offset,
    get_time_range,
    load_env,
    require_env,
    write_merged_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Abbreviation map for column naming
ABBREVIATIONS = {
    "energymeter": "Energy Meter",
    "dailygeneration": "Daily Generation",
}

# Unit conversion: kWh → Joules
KWH_TO_JOULES = 1000.0 * 3600.0  # 3,600,000


def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_PLNT_ID", "TB_GEN_KEYS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    keys = env["TB_GEN_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]

    import os
    max_gen_j = float(os.getenv("TB_GEN_MAX_J", str(100_000_000 * 3600)))

    merged_data: dict = {}

    # Generation uses agg=NONE (raw data), fetched in a single request
    import requests

    url = f"{env['TB_URL']}/api/plugins/telemetry/ASSET/{env['TB_PLNT_ID']}/values/timeseries"
    params = {
        "keys": keys,
        "startTs": start_ts,
        "endTs": end_ts,
        "agg": "NONE",
        "limit": 50_000,
    }

    try:
        logger.info("Fetching raw generation data …")
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)

        if resp.status_code != 200:
            logger.error("Server returned %d: %s", resp.status_code, resp.text)
            sys.exit(1)

        data = resp.json()

        for key in keys_list:
            if key not in data:
                logger.warning("No data for key: %s", key)
                continue

            logger.info("Found %d raw points for %s", len(data[key]), key)

            for point in data[key]:
                ts = point["ts"]

                try:
                    raw_kwh = float(point["value"])
                except (ValueError, TypeError):
                    final_val: object = ""
                else:
                    # Convert kWh → Joules
                    joules = raw_kwh * KWH_TO_JOULES

                    if joules > max_gen_j or joules < 0:
                        final_val = ""
                    else:
                        final_val = joules

                if ts not in merged_data:
                    merged_data[ts] = {}
                merged_data[ts][key] = final_val

    except KeyboardInterrupt:
        logger.warning("Interrupted — saving collected data …")
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed: %s", exc)
        # Still try to save whatever we have

    if not merged_data:
        logger.warning("No data collected.")
        sys.exit(1)

    # Readable column headers
    formatted_keys = []
    for k in keys_list:
        parts = k.split("_")
        processed = [ABBREVIATIONS.get(p.lower(), p.title()) for p in parts]
        formatted_keys.append(f"{' '.join(processed)} (J)")

    write_merged_csv(
        filepath=output_dir / "power_generation_2025_to_current_1day_none_si.csv",
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=keys_list,
        tz=tz,
    )


if __name__ == "__main__":
    main()
