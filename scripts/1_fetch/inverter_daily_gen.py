"""Fetch per-inverter daily generated electricity from ThingsBoard.

Pulls raw (``agg=NONE``) cumulative ``daily_generated_electricity`` readings
for each configured inverter.  The key stores **kWh** values that accumulate
throughout the day and reset at midnight.  Preprocessing takes the last reading
per day as the daily total and converts to Joules.

Usage::

    python scripts/inverter_daily_gen_fetch.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_INV_DAILY_GEN_KEYS, TB_INVERTERS
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

MAX_DAILY_GEN_KWH = 1_000.0  # single-inverter sanity cap


def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_INV_DAILY_GEN_KEYS", "TB_INVERTERS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    keys = env["TB_INV_DAILY_GEN_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]

    inverters = {}
    for item in env["TB_INVERTERS"].split(","):
        name, dev_id = item.split(":")
        inverters[name.strip()] = dev_id.strip()

    device_names = list(inverters.keys())
    merged_data: dict = {}

    chunk_ms = 30 * 24 * 60 * 60 * 1000  # 30-day chunks (1-min data fits in 100k limit)

    def handle_point(key: str, point: dict, _chunk_start: int) -> None:
        ts = point["ts"]
        try:
            val: object = float(point["value"])
        except (ValueError, TypeError):
            val = ""
            
        if isinstance(val, float):
            if val < 0 or val > MAX_DAILY_GEN_KWH:
                val = ""

        if ts not in merged_data:
            merged_data[ts] = {}
        merged_data[ts][f"{inv_name}_{key}"] = val

    logger.info(
        "Fetching daily_generated_electricity for %d inverters in 3-day chunks …",
        len(inverters),
    )

    try:
        for inv_name, inv_id in inverters.items():
            logger.info("--- %s ---", inv_name)
            fetch_chunked(
                base_url=env["TB_URL"],
                entity_type="DEVICE",
                entity_id=inv_id,
                keys=keys,
                start_ts=start_ts,
                end_ts=end_ts,
                interval_ms=None,
                agg="NONE",
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

    formatted_keys = []
    key_order = []
    for name in device_names:
        for k in keys_list:
            col_id = f"{name}_{k}"
            key_order.append(col_id)
            formatted_keys.append(f"{name} Daily Generated Electricity (kWh)")

    write_merged_csv(
        filepath=output_dir / "inverters_daily_gen_2025_to_current_none_si.csv",
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=key_order,
        tz=tz,
    )


if __name__ == "__main__":
    main()
