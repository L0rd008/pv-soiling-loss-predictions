import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Locked input paths
RAW_GEN_CSV = Path("data/inverters_daily_gen_2025_to_current_none_si.csv")
RAW_IRR_CSV = Path("data/plant_avg_irradiance_2025_to_current_none_si.csv")
METER_CSV = Path("data/power_generation_2025_to_current_1day_none_si.csv")
RUNTIME_CSV = Path("data/time_series_chart_time_series_chart.csv")
SOLCAST_IRR_CSV = Path("data/irradiance_2025_to_current_10min_none_std.csv")
EDA_CSV = Path("artifacts/preprocessed/daily_model_eda.csv")

OUT_DIR = Path("artifacts/eda/plots")
OUT_PLOT = OUT_DIR / "custom_pr_inverters.png"
OUT_DEBUG = OUT_DIR / "custom_pr_debug.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from core.daily_features import CLEANING_CAMPAIGN_DATES

LOCKED_INVERTER_ORDER = [
    "B2-08",
    "B2-13",
    "B2-17",
    "B1-08",
    "B1-13",
    "B1-04",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inverter/aggregate/plant PR using daily generation and avg irradiance."
    )
    parser.add_argument("--inverter-capacity-kw", type=float, default=330.0)
    parser.add_argument("--plant-inverter-count", type=int, default=34)
    parser.add_argument("--runtime-min-hours", type=float, default=6.0)
    parser.add_argument("--runtime-max-hours", type=float, default=18.0)
    parser.add_argument(
        "--normalize-plant",
        choices=["minmax", "clip", "none"],
        default="minmax",
    )
    parser.add_argument("--export-debug-csv", type=Path, default=OUT_DEBUG)
    parser.add_argument("--out-plot", type=Path, default=OUT_PLOT)
    return parser.parse_args()


def _require_column(df: pd.DataFrame, col: str, source: Path) -> None:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in {source}")


def _find_single_col(df: pd.DataFrame, pattern: str, source: Path) -> str:
    matches = [c for c in df.columns if pattern in c]
    if not matches:
        raise ValueError(f"No column containing '{pattern}' found in {source}")
    return matches[0]


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_inverter_daily_generation(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    _require_column(df, "Date", path)

    inv_cols = [c for c in df.columns if "Daily Generated Electricity" in c]
    if not inv_cols:
        raise ValueError(f"No inverter daily generation columns found in {path}")

    _coerce_numeric(df, inv_cols)
    df["day_dt"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day_dt"])
    daily = df.groupby("day_dt")[inv_cols].max().reset_index()
    return daily, inv_cols


def load_avg_irradiance_daily(path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    _require_column(df, "Date", path)

    irr_col = _find_single_col(df, "Avg Solar Radiation", path)
    _coerce_numeric(df, [irr_col])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["day_dt"] = df["Date"].dt.normalize()
    df = df.dropna(subset=["day_dt"])

    # Running daily average: use last reading per day.
    daily = (
        df.sort_values("Date")
        .groupby("day_dt")
        .agg(**{irr_col: (irr_col, "last")})
        .reset_index()
    )
    return daily, irr_col


def load_runtime_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    _require_column(df, "Timestamp", path)

    if "runtime_hours" in df.columns:
        runtime_col = "runtime_hours"
    elif "Temperature" in df.columns:
        # Backward compatibility for older ad-hoc exports.
        runtime_col = "Temperature"
    else:
        raise ValueError(
            f"No runtime column found in {path}; expected 'runtime_hours' or 'Temperature'."
        )

    _coerce_numeric(df, [runtime_col])
    df["day_dt"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day_dt"])
    daily = (
        df.groupby("day_dt")
        .agg(running_time_h=(runtime_col, "max"))
        .reset_index()
    )
    return daily


def load_solcast_runtime_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["day_dt", "runtime_solcast_h"])

    df = pd.read_csv(path)
    required = {"period_end", "gti_w_m2"}
    missing = required.difference(df.columns)
    if missing:
        return pd.DataFrame(columns=["day_dt", "runtime_solcast_h"])

    _coerce_numeric(df, ["gti_w_m2"])
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce", utc=True)
    df = df.dropna(subset=["period_end"])

    # Convert to local day key for fallback matching.
    df["day_dt"] = (
        df["period_end"]
        .dt.tz_convert("Asia/Colombo")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily = (
        df.assign(_sun=df["gti_w_m2"] > 0)
        .groupby("day_dt")
        .agg(runtime_solcast_h=("_sun", "sum"))
        .reset_index()
    )
    daily["runtime_solcast_h"] = daily["runtime_solcast_h"] * (10.0 / 60.0)
    return daily


def load_meter_daily(path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    _require_column(df, "Date", path)

    meter_col = _find_single_col(df, "Energy Meter Daily Generation", path)
    _coerce_numeric(df, [meter_col])

    df["day_dt"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day_dt"])
    daily = df.groupby("day_dt")[meter_col].max().reset_index()
    return daily, meter_col


def load_rain_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["day_dt", "precipitation_total_mm"])

    df = pd.read_csv(path)
    if "day" not in df.columns or "precipitation_total_mm" not in df.columns:
        return pd.DataFrame(columns=["day_dt", "precipitation_total_mm"])

    df["day_dt"] = pd.to_datetime(df["day"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day_dt"])
    rain = (
        df.groupby("day_dt")["precipitation_total_mm"]
        .max()
        .reset_index()
    )
    return rain


def select_locked_inverters(inv_cols: List[str]) -> List[str]:
    selected: List[str] = []
    for inv in LOCKED_INVERTER_ORDER:
        matches = [c for c in inv_cols if c.startswith(f"{inv} ")]
        if not matches:
            raise ValueError(
                f"Locked inverter '{inv}' missing from input. Available columns: {inv_cols}"
            )
        selected.append(matches[0])
    return selected


def compute_pr(gen_kwh: pd.Series, x_wh_m2: pd.Series, capacity_kw: float) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (gen_kwh * 1000.0) / (capacity_kw * x_wh_m2)


def prep_pr_series(pr_raw: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    outlier = (pr_raw > 1.0) | (pr_raw < 0.0)
    pr_interp = pr_raw.mask(outlier).interpolate(limit_area="inside")
    pr_roll = pr_interp.rolling(7, min_periods=3, center=True).median()
    return outlier, pr_interp, pr_roll


def minmax_scale(series: pd.Series, ref: pd.Series) -> pd.Series:
    ref_valid = ref.dropna()
    if ref_valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo = float(ref_valid.min())
    hi = float(ref_valid.max())
    if hi <= lo:
        return pd.Series(np.nan, index=series.index)
    return (series - lo) / (hi - lo)


def _add_overlays(ax: plt.Axes, df: pd.DataFrame) -> None:
    if "precipitation_total_mm" in df.columns:
        rainy_days = df.loc[df["precipitation_total_mm"] >= 5.0, "day_dt"]
        for i, day in enumerate(rainy_days):
            ax.axvline(
                day,
                color="#1f77b4",
                alpha=0.3,
                lw=1.0,
                label="Rain >= 5 mm" if i == 0 else None,
            )

    for i, (start_s, end_s) in enumerate(CLEANING_CAMPAIGN_DATES):
        start = pd.Timestamp(start_s)
        # Inclusive day window so short cleaning campaigns remain visible.
        end_inclusive = pd.Timestamp(end_s) + pd.Timedelta(days=1)
        ax.axvspan(
            start,
            end_inclusive,
            color="#ff7f0e",
            alpha=0.24,
            label="Cleaning period" if i == 0 else None,
            zorder=0,
        )
        ax.axvline(start, color="#d95f02", alpha=0.7, lw=1.0, linestyle="--")
        ax.axvline(end_inclusive, color="#d95f02", alpha=0.7, lw=1.0, linestyle="--")


def _append_debug_rows(
    bag: List[pd.DataFrame],
    series_name: str,
    day: pd.Series,
    gen_kwh: pd.Series,
    avg_irr: pd.Series,
    runtime_h: pd.Series,
    x: pd.Series,
    y: pd.Series,
    pr_raw: pd.Series,
    outlier: pd.Series,
    pr_interp: pd.Series,
    pr_roll7: pd.Series,
    pr_display: pd.Series,
) -> None:
    bag.append(
        pd.DataFrame(
            {
                "day": day,
                "series": series_name,
                "gen_kwh": gen_kwh,
                "avg_irr_wm2": avg_irr,
                "runtime_h": runtime_h,
                "X": x,
                "Y": y,
                "PR_raw": pr_raw,
                "outlier_flag": outlier,
                "PR_interp": pr_interp,
                "PR_roll7": pr_roll7,
                "PR_display": pr_display,
            }
        )
    )


def run_formula_sanity_check(capacity_kw: float) -> None:
    # Synthetic check: if E = P * H then PR should be exactly 1.
    avg_irr = pd.Series([500.0])
    runtime_h = pd.Series([10.0])
    x = avg_irr * runtime_h  # Wh/m^2
    h_kwh_m2 = x / 1000.0
    energy_kwh = capacity_kw * h_kwh_m2
    pr = compute_pr(energy_kwh, x, capacity_kw).iloc[0]
    if not np.isfinite(pr) or abs(pr - 1.0) > 1e-9:
        raise RuntimeError(f"Formula sanity check failed. Computed PR={pr}")


def main() -> None:
    args = parse_args()
    run_formula_sanity_check(args.inverter_capacity_kw)

    print("Loading datasets...")
    df_gen, inv_cols = load_inverter_daily_generation(RAW_GEN_CSV)
    df_irr, irr_col = load_avg_irradiance_daily(RAW_IRR_CSV)
    df_runtime = load_runtime_daily(RUNTIME_CSV)
    df_runtime_sol = load_solcast_runtime_daily(SOLCAST_IRR_CSV)
    df_meter, meter_col = load_meter_daily(METER_CSV)
    df_rain = load_rain_daily(EDA_CSV)

    selected_cols = select_locked_inverters(inv_cols)
    print(f"Using locked inverter set: {[c.split(' ')[0] for c in selected_cols]}")

    date_range = pd.date_range(
        start=min(df_gen["day_dt"].min(), df_meter["day_dt"].min()),
        end=max(df_gen["day_dt"].max(), df_meter["day_dt"].max()),
        freq="D",
    )
    df = pd.DataFrame({"day_dt": date_range})
    df = df.merge(df_gen, on="day_dt", how="left")
    df = df.merge(df_irr[["day_dt", irr_col]], on="day_dt", how="left")
    df = df.merge(df_runtime[["day_dt", "running_time_h"]], on="day_dt", how="left")
    df = df.merge(df_runtime_sol, on="day_dt", how="left")
    df = df.merge(df_meter[["day_dt", meter_col]], on="day_dt", how="left")
    if not df_rain.empty:
        df = df.merge(df_rain, on="day_dt", how="left")
    df = df.sort_values("day_dt").reset_index(drop=True)

    # Runtime QC then fallback from Solcast daylight runtime.
    runtime_raw = df["running_time_h"].copy()
    invalid_runtime = (
        runtime_raw.notna()
        & ((runtime_raw < args.runtime_min_hours) | (runtime_raw > args.runtime_max_hours))
    )
    df.loc[invalid_runtime, "running_time_h"] = np.nan
    fallback_mask = df["running_time_h"].isna() & df["runtime_solcast_h"].notna()
    df.loc[fallback_mask, "running_time_h"] = df.loc[fallback_mask, "runtime_solcast_h"]

    runtime_source = pd.Series("runtime_csv", index=df.index)
    runtime_source[df["running_time_h"].isna()] = "missing"
    runtime_source[invalid_runtime] = "runtime_csv_invalid"
    runtime_source[fallback_mask] = "solcast_fallback"
    df["runtime_source"] = runtime_source

    avg_irr = df[irr_col]
    runtime_h = df["running_time_h"]
    x = avg_irr * runtime_h

    n_valid_runtime = int(df["running_time_h"].notna().sum())
    n_fallback = int(fallback_mask.sum())
    print(
        "Runtime availability:",
        f"valid_days={n_valid_runtime}",
        f"solcast_fallback_days={n_fallback}",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True)
    axes = axes.flatten()

    debug_rows: List[pd.DataFrame] = []
    aggregated_gen_kwh = df[selected_cols].sum(axis=1, min_count=1)

    # 1) Individual inverter panels
    for idx, col in enumerate(selected_cols):
        ax = axes[idx]
        inv_name = col.split(" ")[0]
        gen_kwh = df[col]

        y = gen_kwh / x
        pr_raw = compute_pr(gen_kwh, x, args.inverter_capacity_kw)
        outlier, pr_interp, pr_roll = prep_pr_series(pr_raw)
        pr_display = pr_raw

        valid_mask = (~outlier) & pr_raw.notna()
        outlier_mask = outlier & pr_raw.notna()

        ax.plot(
            df.loc[valid_mask, "day_dt"],
            pr_display[valid_mask],
            marker=".",
            linestyle="none",
            alpha=0.3,
            color="steelblue",
            label="Daily PR",
        )
        if outlier_mask.any():
            ax.plot(
                df.loc[outlier_mask, "day_dt"],
                pr_display[outlier_mask],
                marker="X",
                linestyle="none",
                color="red",
                markersize=6,
                label="PR outlier",
            )
        ax.plot(
            df["day_dt"],
            pr_roll,
            linestyle="-",
            alpha=0.9,
            lw=2,
            color="darkorange",
            label="7d median (interp)",
        )
        _add_overlays(ax, df)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Inverter {inv_name} PR")
        ax.set_ylabel("PR")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        _append_debug_rows(
            debug_rows,
            inv_name,
            df["day_dt"],
            gen_kwh,
            avg_irr,
            runtime_h,
            x,
            y,
            pr_raw,
            outlier,
            pr_interp,
            pr_roll,
            pr_display,
        )

    # 2) Aggregated 6-inverter panel
    ax_agg = axes[6]
    agg_capacity_kw = args.inverter_capacity_kw * len(selected_cols)
    y_agg = aggregated_gen_kwh / x
    pr_agg_raw = compute_pr(aggregated_gen_kwh, x, agg_capacity_kw)
    outlier_agg, pr_agg_interp, pr_agg_roll = prep_pr_series(pr_agg_raw)
    pr_agg_display = pr_agg_raw

    valid_agg = (~outlier_agg) & pr_agg_raw.notna()
    out_agg = outlier_agg & pr_agg_raw.notna()

    ax_agg.plot(
        df.loc[valid_agg, "day_dt"],
        pr_agg_display[valid_agg],
        marker=".",
        linestyle="none",
        alpha=0.3,
        color="forestgreen",
        label="Daily PR",
    )
    if out_agg.any():
        ax_agg.plot(
            df.loc[out_agg, "day_dt"],
            pr_agg_display[out_agg],
            marker="X",
            linestyle="none",
            color="red",
            markersize=6,
            label="PR outlier",
        )
    ax_agg.plot(
        df["day_dt"],
        pr_agg_roll,
        linestyle="-",
        alpha=0.9,
        lw=2,
        color="darkorange",
        label="7d median (interp)",
    )
    _add_overlays(ax_agg, df)
    ax_agg.set_ylim(0, 1.05)
    ax_agg.set_title(f"Aggregated ({len(selected_cols)} Inverters) PR")
    ax_agg.set_ylabel("PR")
    ax_agg.legend(loc="best")
    ax_agg.grid(True, alpha=0.3)

    _append_debug_rows(
        debug_rows,
        "aggregate_6",
        df["day_dt"],
        aggregated_gen_kwh,
        avg_irr,
        runtime_h,
        x,
        y_agg,
        pr_agg_raw,
        outlier_agg,
        pr_agg_interp,
        pr_agg_roll,
        pr_agg_display,
    )

    # 3) Plant panel
    ax_plant = axes[7]
    plant_gen_kwh = df[meter_col] / 3.6e6
    plant_capacity_kw = args.inverter_capacity_kw * args.plant_inverter_count
    y_plant = plant_gen_kwh / x
    pr_plant_raw = compute_pr(plant_gen_kwh, x, plant_capacity_kw)
    outlier_plant, pr_plant_interp, pr_plant_roll = prep_pr_series(pr_plant_raw)

    if args.normalize_plant == "minmax":
        pr_plant_display = minmax_scale(pr_plant_raw, pr_plant_interp).clip(lower=0.0, upper=1.0)
        pr_plant_roll_display = minmax_scale(pr_plant_roll, pr_plant_interp).clip(lower=0.0, upper=1.0)
    elif args.normalize_plant == "clip":
        pr_plant_display = pr_plant_raw.clip(lower=0.0, upper=1.0)
        pr_plant_roll_display = pr_plant_roll.clip(lower=0.0, upper=1.0)
    else:
        pr_plant_display = pr_plant_raw
        pr_plant_roll_display = pr_plant_roll

    valid_plant = (~outlier_plant) & pr_plant_raw.notna()
    out_plant = outlier_plant & pr_plant_raw.notna()

    ax_plant.plot(
        df.loc[valid_plant, "day_dt"],
        pr_plant_display[valid_plant],
        marker=".",
        linestyle="none",
        alpha=0.3,
        color="purple",
        label="Daily PR",
    )
    if out_plant.any():
        ax_plant.plot(
            df.loc[out_plant, "day_dt"],
            pr_plant_display[out_plant],
            marker="X",
            linestyle="none",
            color="red",
            markersize=6,
            label="PR outlier",
        )
    ax_plant.plot(
        df["day_dt"],
        pr_plant_roll_display,
        linestyle="-",
        alpha=0.9,
        lw=2,
        color="darkorange",
        label="7d median (interp)",
    )
    _add_overlays(ax_plant, df)
    ax_plant.set_ylim(0, 1.05)
    ax_plant.set_title(
        f"Plant PR ({args.plant_inverter_count} x {args.inverter_capacity_kw:.0f} kW, {args.normalize_plant})"
    )
    ax_plant.set_ylabel("PR display")
    ax_plant.legend(loc="best")
    ax_plant.grid(True, alpha=0.3)

    _append_debug_rows(
        debug_rows,
        "plant",
        df["day_dt"],
        plant_gen_kwh,
        avg_irr,
        runtime_h,
        x,
        y_plant,
        pr_plant_raw,
        outlier_plant,
        pr_plant_interp,
        pr_plant_roll,
        pr_plant_display,
    )

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "Daily Performance Ratio (PR)\n"
        "PR = E_kWh * 1000 / (P_kW * avg_irr_Wm2 * runtime_h)",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out_plot, dpi=150)
    print(f"Saved PR plots to {args.out_plot}")

    debug_df = pd.concat(debug_rows, ignore_index=True).sort_values(["series", "day"])
    debug_df.to_csv(args.export_debug_csv, index=False)
    print(f"Saved debug CSV to {args.export_debug_csv}")

    print(
        "Plant PR diagnostics:",
        f"raw_valid={int(pr_plant_raw.notna().sum())}",
        f"raw_outliers={int(outlier_plant.sum())}",
    )


if __name__ == "__main__":
    main()
