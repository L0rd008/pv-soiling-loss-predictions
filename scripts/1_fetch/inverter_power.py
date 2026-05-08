"""Fetch inverter telemetry from ThingsBoard.

Pulls 10-minute AVG-aggregated inverter data (active power and phase currents)
for a configured set of inverters.  The raw current values from ThingsBoard are
in milliamps and are divided by 1000 to produce SI amperes.

Usage::

    python scripts/inverter_data_fetch.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_INV_KEYS, TB_INVERTERS
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

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MAX_POWER_W = 300_000.0   # 300 kW sanity cap
MAX_CURRENT_A = 250.0     # 250 A sanity cap

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_INV_KEYS", "TB_INVERTERS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    keys = env["TB_INV_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]

    # Parse inverter name→id map
    inverters = {}
    for item in env["TB_INVERTERS"].split(","):
        name, dev_id = item.split(":")
        inverters[name.strip()] = dev_id.strip()

    device_names = list(inverters.keys())
    merged_data: dict = {}

    interval_ms = 10 * 60 * 1000  # 10 minutes
    chunk_ms = 3 * 24 * 60 * 60 * 1000  # 3-day chunks

    def handle_point(key: str, point: dict, _chunk_start: int) -> None:
        ts = point["ts"]
        dt_local = datetime.fromtimestamp(ts / 1000.0, tz=tz)
        hour = dt_local.hour

        try:
            val: object = float(point["value"])
        except (ValueError, TypeError):
            val = ""

        # Scale mA → A for current keys
        if isinstance(val, float) and "current" in key.lower():
            val = val / 1000.0

        # Sanity clipping
        if isinstance(val, float):
            if "power" in key.lower() and val > MAX_POWER_W:
                val = ""
            elif "current" in key.lower() and val > MAX_CURRENT_A:
                val = ""
            elif val < 0:
                val = 0.0 if (hour >= 19 or hour < 5) else ""

        if ts not in merged_data:
            merged_data[ts] = {}
        merged_data[ts][f"{inv_name}_{key}"] = val

    logger.info("Fetching data for %d inverters in 3-day chunks …", len(inverters))

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
                interval_ms=interval_ms,
                agg="AVG",
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

    # Build readable column headers
    formatted_keys = []
    key_order = []
    for name in device_names:
        for k in keys_list:
            col_id = f"{name}_{k}"
            key_order.append(col_id)
            readable = k.replace("_", " ").title()
            if "power" in readable.lower():
                readable += " (W)"
            elif "current" in readable.lower():
                readable += " (A)"
            formatted_keys.append(f"{name} {readable}")

    write_merged_csv(
        filepath=output_dir / "inverters_2025_to_current_10min_avg_si.csv",
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=key_order,
        tz=tz,
    )


if __name__ == "__main__":
    main()
