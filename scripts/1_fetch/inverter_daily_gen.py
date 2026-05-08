"""Fetch per-inverter daily generated electricity from ThingsBoard.

Pulls raw (``agg=NONE``) cumulative ``daily_generated_electricity`` readings
for each configured inverter. Values are preserved as-is except negatives,
which are dropped. A warning-only threshold is used for audit counts.

Usage::

    python scripts/1_fetch/inverter_daily_gen.py

Environment variables (see ``.env.example``):
    TB_URL, TB_TOKEN, TB_INV_DAILY_GEN_KEYS, TB_INVERTERS
    Optional: TB_OUTPUT_DIR, TB_REQUEST_TIMEOUT_S, TB_TZ_OFFSET, TB_START_DATE
    Optional: TB_INV_DAILY_GEN_WARN_KWH (default 1000)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tb_client import (
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

DEFAULT_WARN_DAILY_GEN_KWH = 1_000.0  # warning-only threshold


def main() -> None:
    load_env()
    env = require_env("TB_URL", "TB_TOKEN", "TB_INV_DAILY_GEN_KEYS", "TB_INVERTERS")

    tz = get_tz_offset()
    start_ts, end_ts = get_time_range(tz)
    headers = auth_headers(env["TB_TOKEN"])
    timeout = get_request_timeout()
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = env["TB_INV_DAILY_GEN_KEYS"]
    keys_list = [k.strip() for k in keys.split(",")]
    warn_daily_gen_kwh = float(
        os.getenv("TB_INV_DAILY_GEN_WARN_KWH", str(DEFAULT_WARN_DAILY_GEN_KWH))
    )

    inverters: Dict[str, str] = {}
    for item in env["TB_INVERTERS"].split(","):
        name, dev_id = item.split(":")
        inverters[name.strip()] = dev_id.strip()

    device_names = list(inverters.keys())
    merged_data: Dict[int, Dict[str, Any]] = {}
    fetch_audit = {
        name: {
            "total_points": 0,
            "parse_failures": 0,
            "negative_dropped_points": 0,
            "above_warning_points": 0,
            "min_value_kwh": None,
            "max_value_kwh": None,
        }
        for name in device_names
    }

    chunk_ms = 30 * 24 * 60 * 60 * 1000  # 30-day chunks (1-min data fits in 100k limit)

    def make_point_handler(inv_name: str):
        def handle_point(key: str, point: dict, _chunk_start: int) -> None:
            ts = point["ts"]
            audit = fetch_audit[inv_name]
            audit["total_points"] += 1

            try:
                val: object = float(point["value"])
            except (ValueError, TypeError):
                audit["parse_failures"] += 1
                val = ""

            if isinstance(val, float):
                if audit["min_value_kwh"] is None or val < audit["min_value_kwh"]:
                    audit["min_value_kwh"] = val
                if audit["max_value_kwh"] is None or val > audit["max_value_kwh"]:
                    audit["max_value_kwh"] = val
                if val < 0:
                    audit["negative_dropped_points"] += 1
                    val = ""
                elif val > warn_daily_gen_kwh:
                    audit["above_warning_points"] += 1

            if ts not in merged_data:
                merged_data[ts] = {}
            merged_data[ts][f"{inv_name}_{key}"] = val

        return handle_point

    logger.info(
        "Fetching daily_generated_electricity for %d inverters in 30-day chunks...",
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
                point_handler=make_point_handler(inv_name),
            )
    except KeyboardInterrupt:
        logger.warning("Interrupted - saving collected data...")
    except Exception as exc:
        logger.error("Unexpected error: %s - saving collected data...", exc)

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

    out_csv = output_dir / "inverters_daily_gen_2025_to_current_none_si.csv"
    write_merged_csv(
        filepath=out_csv,
        merged_data=merged_data,
        header_columns=formatted_keys,
        key_order=key_order,
        tz=tz,
    )

    total_points = sum(v["total_points"] for v in fetch_audit.values())
    total_parse_failures = sum(v["parse_failures"] for v in fetch_audit.values())
    total_negative_dropped = sum(v["negative_dropped_points"] for v in fetch_audit.values())
    total_above_warning = sum(v["above_warning_points"] for v in fetch_audit.values())
    above_warning_pct = 100.0 * total_above_warning / total_points if total_points > 0 else 0.0

    audit_payload = {
        "generated_at_local": datetime.now(tz).isoformat(),
        "warning_threshold_kwh": warn_daily_gen_kwh,
        "global": {
            "total_points": total_points,
            "parse_failures": total_parse_failures,
            "negative_dropped_points": total_negative_dropped,
            "above_warning_points": total_above_warning,
            "above_warning_pct": round(above_warning_pct, 4),
        },
        "per_inverter": fetch_audit,
    }
    audit_path = output_dir / "inverters_daily_gen_fetch_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    logger.info(
        "Wrote daily-gen fetch audit to %s (above-warning %.2f%%)",
        audit_path,
        above_warning_pct,
    )


if __name__ == "__main__":
    main()
