"""Shared ThingsBoard telemetry client utilities.

Extracts common patterns from the inverter, irradiance, and generation fetch
scripts: environment loading, time configuration, chunked HTTP fetching with
retry/pacing, and CSV export.
"""

import csv
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load ``.env`` from the project root."""
    load_dotenv()


def require_env(*names: str) -> Dict[str, str]:
    """Return a dict of environment values, raising if any are missing."""
    values: Dict[str, str] = {}
    missing: List[str] = []
    for name in names:
        val = os.getenv(name)
        if not val:
            missing.append(name)
        else:
            values[name] = val
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(sorted(missing))}. "
            "Please check your .env file."
        )
    return values


def get_output_dir() -> Path:
    """Return the configured output directory (default ``data``)."""
    return Path(os.getenv("TB_OUTPUT_DIR", "data"))


def get_request_timeout() -> int:
    """Return the HTTP request timeout in seconds (default 30)."""
    return int(os.getenv("TB_REQUEST_TIMEOUT_S", "30"))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def get_tz_offset() -> timezone:
    """Return a fixed ``timezone`` from ``TB_TZ_OFFSET`` (default +05:30)."""
    raw = os.getenv("TB_TZ_OFFSET", "+05:30")
    sign = 1 if raw.startswith("+") else -1
    parts = raw.lstrip("+-").split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))


def get_time_range(tz: timezone) -> Tuple[int, int]:
    """Return ``(start_ts_ms, end_ts_ms)`` from config / defaults.

    Start defaults to 2025-01-01 00:00 local; end defaults to *now*.
    """
    start_str = os.getenv("TB_START_DATE", "2025-01-01")
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=tz)
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    return start_ts, end_ts


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------

def auth_headers(token: str) -> Dict[str, str]:
    """Return ThingsBoard HTTP headers with Bearer token."""
    return {
        "Content-Type": "application/json",
        "X-Authorization": f"Bearer {token}",
    }


# ---------------------------------------------------------------------------
# Chunked fetch with retry + pacing
# ---------------------------------------------------------------------------

def fetch_chunked(
    *,
    base_url: str,
    entity_type: str,
    entity_id: str,
    keys: str,
    start_ts: int,
    end_ts: int,
    interval_ms: Optional[int],
    agg: str,
    limit: int,
    chunk_ms: int,
    headers: Dict[str, str],
    timeout_s: int,
    point_handler: Callable[[str, Dict[str, Any], int], None],
    max_retries: int = 3,
    pace_seconds: float = 1.0,
) -> None:
    """Fetch telemetry from ThingsBoard in time-window chunks.

    Parameters
    ----------
    base_url : str
        ThingsBoard base URL (no trailing slash).
    entity_type : str
        ``"DEVICE"`` or ``"ASSET"``.
    entity_id : str
        UUID of the device or asset.
    keys : str
        Comma-separated telemetry key list.
    start_ts, end_ts : int
        Epoch milliseconds for the query window.
    interval_ms : int or None
        Aggregation interval.  ``None`` omits the param (raw fetch).
    agg : str
        Aggregation function (``"AVG"``, ``"SUM"``, ``"NONE"``).
    limit : int
        Max points per request.
    chunk_ms : int
        Size of each time-window chunk in milliseconds.
    headers : dict
        HTTP headers (use :func:`auth_headers`).
    timeout_s : int
        HTTP request timeout.
    point_handler : callable
        Called as ``point_handler(key, point_dict, chunk_start_ts)``
        for every data point received.
    max_retries : int
        Retry count on connection errors.
    pace_seconds : float
        Sleep between successful chunks.
    """
    url = f"{base_url}/api/plugins/telemetry/{entity_type}/{entity_id}/values/timeseries"
    keys_list = [k.strip() for k in keys.split(",")]
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + chunk_ms, end_ts)

        params: Dict[str, Any] = {
            "keys": keys,
            "startTs": current_start,
            "endTs": current_end,
            "agg": agg,
            "limit": limit,
        }
        if interval_ms is not None:
            params["interval"] = interval_ms

        chunk_ok = False
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)

                if resp.status_code == 200:
                    data = resp.json()
                    for key in keys_list:
                        for point in data.get(key, []):
                            point_handler(key, point, current_start)
                    chunk_ok = True
                    break
                else:
                    logger.warning(
                        "Server returned %d for chunk starting %d: %s",
                        resp.status_code, current_start, resp.text,
                    )
                    break  # don't retry on server errors

            except requests.exceptions.RequestException:
                logger.warning(
                    "Connection error (attempt %d/%d). Retrying in 3s …",
                    attempt + 1, max_retries,
                )
                time.sleep(3)

        if not chunk_ok:
            logger.error(
                "Failed chunk starting at %d. Moving on to preserve collected data.",
                current_start,
            )

        current_start = current_end
        time.sleep(pace_seconds)


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def write_merged_csv(
    filepath: Path,
    merged_data: Dict[int, Dict[str, Any]],
    header_columns: List[str],
    key_order: List[str],
    tz: timezone,
) -> None:
    """Write the merged timestamp→column dict to a CSV.

    Parameters
    ----------
    filepath : Path
        Destination file.
    merged_data : dict
        ``{timestamp_ms: {column_id: value, …}, …}``.
    header_columns : list[str]
        Human-readable column names (excluding Timestamp/Date).
    key_order : list[str]
        Internal column IDs in the order they should appear.
    tz : timezone
        For formatting the Date column.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Date"] + header_columns)
            for ts in sorted(merged_data):
                dt = datetime.fromtimestamp(ts / 1000.0, tz=tz)
                row = [ts, dt.strftime("%Y-%m-%d %H:%M:%S")]
                for col_id in key_order:
                    row.append(merged_data[ts].get(col_id, ""))
                writer.writerow(row)
        logger.info("Exported %d rows to %s", len(merged_data), filepath)
    except PermissionError:
        logger.error("Could not write %s — is the file open?", filepath)
