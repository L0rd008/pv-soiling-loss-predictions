"""Fetch irradiance telemetry from ThingsBoard.

Pulls 15-minute SUM-aggregated irradiance data (horizontal and tilted) from a
weather station device.

Usage::

    python scripts/irradiance_data_fetch.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_WSTN_ID, TB_IRR_KEYS
    Optional: TB_OUTPUT_DIR, TB_REQUEST_TIMEOUT_S, TB_TZ_OFFSET, TB_START_DATE
"""

import logging
import sys
from datetime import datetime

from tb_client import (
    auth_headers,
    fetch_chunked,
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

# Human-readable abbreviation map for column naming
ABBREVIATIONS = {
    "wstn1": "Weather Station 1",
    "wstn2": "Weather Station 2",
    "wstn": "Weather Station",
    "horiz": "Horizontal",
    "irr": "Irradiance",
    "temp": "Temperature",
}


def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_WSTN_ID", "TB_IRR_KEYS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    keys = env["TB_IRR_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]
    agg = "SUM"

    merged_data: dict = {}
    interval_ms = 15 * 60 * 1000  # 15 minutes
    chunk_ms = 5 * 24 * 60 * 60 * 1000  # 5-day chunks

    def handle_point(key: str, point: dict, _chunk_start: int) -> None:
        ts = point["ts"]
        dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz)
        hour = dt_local.hour

        try:
            val: object = float(point["value"])
        except (ValueError, TypeError):
            val = ""

        # Negative irradiance: zero at night, invalid during day
        if isinstance(val, float) and val < 0:
            val = 0.0 if (hour >= 19 or hour < 5) else ""

        if ts not in merged_data:
            merged_data[ts] = {}
        merged_data[ts][key] = val

    logger.info("Fetching irradiance in 5-day chunks …")

    try:
        fetch_chunked(
            base_url=env["TB_URL"],
            entity_type="DEVICE",
            entity_id=env["TB_WSTN_ID"],
            keys=keys,
            start_ts=start_ts,
            end_ts=end_ts,
            interval_ms=interval_ms,
            agg=agg,
            limit=100_000,
            chunk_ms=chunk_ms,
            headers=headers,
            timeout_s=timeout,
            point_handler=handle_point,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — saving collected data …")
    except Exception as exc:
        logger.error("Unexpected error: %s — saving collected data …", exc)

    if not merged_data:
        logger.warning("No data collected.")
        sys.exit(1)

    # Readable column headers
    formatted_keys = []
    for k in keys_list:
        parts = k.split("_")
        processed = [ABBREVIATIONS.get(p.lower(), p.title()) for p in parts]
        formatted_keys.append(f"{' '.join(processed)} ({agg})")

    write_merged_csv(
        filepath=output_dir / f"irradiance_2025_to_current_15min_{agg.lower()}_si.csv",
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=keys_list,
        tz=tz,
    )


if __name__ == "__main__":
    main()
