"""Fetch plant-level average solar radiation from ThingsBoard.

Pulls raw (``agg=NONE``) ``avg_solar_radiation`` from the KBG Plant asset.
The key stores a running daily average **W/m²** (Total Irradiance / Time)
that resets at midnight.  The end-of-day value equals the true daily average.

No unit conversion is performed here — values are stored as W/m².
Preprocessing takes the last reading per day as the daily average.

Usage::

    python scripts/plant_avg_irradiance_fetch.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_PLNT_ID, TB_PLNT_IRR_KEYS
    Optional: TB_OUTPUT_DIR, TB_REQUEST_TIMEOUT_S, TB_TZ_OFFSET, TB_START_DATE
"""

import logging
import sys

import requests

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

MAX_IRR_WM2 = 1_500.0  # absolute physical ceiling for solar irradiance


def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_PLNT_ID", "TB_PLNT_IRR_KEYS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    keys = env["TB_PLNT_IRR_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]

    merged_data: dict = {}

    url = (
        f"{env['TB_URL']}/api/plugins/telemetry/ASSET/"
        f"{env['TB_PLNT_ID']}/values/timeseries"
    )
    params = {
        "keys": keys,
        "startTs": start_ts,
        "endTs": end_ts,
        "agg": "NONE",
        "limit": 50_000,
    }

    try:
        logger.info("Fetching plant avg_solar_radiation …")
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
                    val: object = float(point["value"])
                except (ValueError, TypeError):
                    val = ""
                else:
                    if val < 0 or val > MAX_IRR_WM2:
                        val = ""

                if ts not in merged_data:
                    merged_data[ts] = {}
                merged_data[ts][key] = val

    except KeyboardInterrupt:
        logger.warning("Interrupted — saving collected data …")
    except requests.exceptions.RequestException as exc:
        logger.error("Request failed: %s", exc)

    if not merged_data:
        logger.warning("No data collected.")
        sys.exit(1)

    formatted_keys = [
        "Avg Solar Radiation (W/m²)" if "solar" in k.lower() else k.replace("_", " ").title()
        for k in keys_list
    ]

    write_merged_csv(
        filepath=output_dir / "plant_avg_irradiance_2025_to_current_none_si.csv",
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=keys_list,
        tz=tz,
    )


if __name__ == "__main__":
    main()
