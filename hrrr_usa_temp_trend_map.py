from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from scipy.ndimage import gaussian_filter


NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
DEFAULT_DATE_FORMAT = "%Y%m%d"
DEFAULT_OUTPUT_DIR = "/var/data/output"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_FORECAST_HOUR = 18
DEFAULT_SMOOTHING_SIGMA = 4.0
DEFAULT_BLOCK_SIZE = 8
CONUS_BOUNDS = {
    "leftlon": -125.0,
    "rightlon": -66.5,
    "bottomlat": 24.0,
    "toplat": 49.5,
}
EASTERN_TIMEZONE = ZoneInfo("America/New_York")
FIELD_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, datetime] | None] = {}
TREND_LEVELS = np.arange(-5.0, 5.5, 0.5)
TREND_CMAP = ListedColormap(
    [
        "#1e3a8a",
        "#1d4ed8",
        "#2563eb",
        "#3b82f6",
        "#60a5fa",
        "#93c5fd",
        "#bfdbfe",
        "#dbeafe",
        "#d1d5db",
        "#d1d5db",
        "#fee2e2",
        "#fecaca",
        "#fca5a5",
        "#f87171",
        "#ef4444",
        "#dc2626",
        "#b91c1c",
        "#991b1b",
        "#7f1d1d",
        "#651111",
    ]
)
TREND_NORM = BoundaryNorm(TREND_LEVELS, TREND_CMAP.N)


@dataclass(frozen=True)
class ForecastTrend:
    cycle_hour: int
    comparison_cycle_hours: list[int]
    forecast_hour: int
    valid_time: datetime
    field_f: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HRRR 2 m temperature fields for CONUS and build smoothed USA maps "
            "showing whether each area is warming or cooling compared with the prior run."
        )
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime(DEFAULT_DATE_FORMAT),
        help="UTC date to query in YYYYMMDD format. Defaults to today in UTC.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw GRIBs and map outputs. Defaults to /var/data/output.",
    )
    parser.add_argument(
        "--max-forecast-hour",
        type=int,
        default=DEFAULT_MAX_FORECAST_HOUR,
        help="Largest forecast hour to compare for each run. Default is 18.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=DEFAULT_SMOOTHING_SIGMA,
        help="Gaussian smoothing strength for the trend field. Default is 2.2.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Aggregate HRRR grid cells into NxN blocks before plotting. Default is 8.",
    )
    return parser.parse_args()


def build_url(date_str: str, cycle_hour: int, forecast_hour: int) -> str:
    query = {
        "dir": f"/hrrr.{date_str}/conus",
        "file": f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "subregion": "1",
        **{key: str(value) for key, value in CONUS_BOUNDS.items()},
    }
    prepared = requests.Request("GET", NOMADS_FILTER_URL, params=query).prepare()
    assert prepared.url is not None
    return prepared.url


def get_run_grib_dir(raw_dir: Path, cycle_hour: int) -> Path:
    return raw_dir / f"{cycle_hour:02d}z"


def load_or_download_grib(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    timeout: int,
    raw_dir: Path,
) -> Path | None:
    run_dir = get_run_grib_dir(raw_dir, cycle_hour)
    raw_file = run_dir / f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.tmp2m.conus.grib2"
    if raw_file.exists():
        print(f"  using cached {cycle_hour:02d}z f{forecast_hour:02d}", flush=True)
        return raw_file

    print(f"  downloading {cycle_hour:02d}z f{forecast_hour:02d}", flush=True)
    response = session.get(build_url(date_str, cycle_hour, forecast_hour), timeout=timeout)
    if response.status_code == 404:
        return None

    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" in content_type.lower() or not response.content.startswith(b"GRIB"):
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    raw_file.write_bytes(response.content)
    return raw_file


def load_temperature_field(grib_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime]:
    with xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        lat_name = "latitude" if "latitude" in ds else "lat"
        lon_name = "longitude" if "longitude" in ds else "lon"

        lats = ds[lat_name].values
        lons = ds[lon_name].values
        if np.nanmax(lons) > 180:
            lons = ((lons + 180) % 360) - 180

        temp_f = ((ds["t2m"].values.astype(float) - 273.15) * 9 / 5) + 32
        valid_value = ds["t2m"].coords.get("valid_time", ds["t2m"].coords.get("time"))
        if valid_value is None:
            raise ValueError(f"Missing valid time in {grib_file}")

        valid_time = datetime.fromisoformat(str(np.datetime_as_string(valid_value.values, unit="s"))).replace(tzinfo=timezone.utc)
        return lats, lons, temp_f, valid_time


def get_temperature_field(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    timeout: int,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, datetime] | None:
    cache_key = (date_str, cycle_hour, forecast_hour)
    if cache_key in FIELD_CACHE:
        return FIELD_CACHE[cache_key]

    grib_file = load_or_download_grib(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
    if grib_file is None:
        FIELD_CACHE[cache_key] = None
        return None

    field = load_temperature_field(grib_file)
    FIELD_CACHE[cache_key] = field
    return field


def clear_field_cache() -> None:
    FIELD_CACHE.clear()
    gc.collect()


def iter_prior_run_requests(
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    max_forecast_hour: int,
) -> list[tuple[str, int, int]]:
    run_start = datetime.strptime(date_str, DEFAULT_DATE_FORMAT).replace(tzinfo=timezone.utc) + timedelta(hours=cycle_hour)
    requests: list[tuple[str, int, int]] = []
    for hours_back in range(1, max_forecast_hour + 1):
        prior_forecast_hour = forecast_hour + hours_back
        if prior_forecast_hour > max_forecast_hour:
            break

        prior_run_start = run_start - timedelta(hours=hours_back)
        requests.append(
            (
                prior_run_start.strftime(DEFAULT_DATE_FORMAT),
                prior_run_start.hour,
                prior_forecast_hour,
            )
        )

    return requests


def resolve_latest_cycle(target_date: datetime, now_utc: datetime) -> int:
    if target_date.date() < now_utc.date():
        return 23

    eastern_now = now_utc.astimezone(EASTERN_TIMEZONE)
    if eastern_now.hour in {20, 21}:
        return 23

    return now_utc.hour


def resolve_run_date_and_latest_cycle(requested_date_str: str, now_utc: datetime) -> tuple[str, int]:
    requested_date = datetime.strptime(requested_date_str, DEFAULT_DATE_FORMAT).replace(tzinfo=timezone.utc)
    eastern_now = now_utc.astimezone(EASTERN_TIMEZONE)
    current_utc_date_str = now_utc.strftime(DEFAULT_DATE_FORMAT)

    if requested_date_str == current_utc_date_str and eastern_now.hour in {20, 21}:
        prior_utc_date = (now_utc - timedelta(days=1)).strftime(DEFAULT_DATE_FORMAT)
        return prior_utc_date, 23

    return requested_date_str, resolve_latest_cycle(requested_date, now_utc)


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    mask = np.isfinite(field)
    if not np.any(mask):
        return field

    filled = np.where(mask, field, 0.0)
    weights = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    smoothed_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smoothed_values / weights
    result[weights == 0] = np.nan
    return result


def block_reduce_mean(field: np.ndarray, block_size: int) -> np.ndarray:
    if block_size <= 1:
        return field

    rows = field.shape[0] // block_size
    cols = field.shape[1] // block_size
    if rows == 0 or cols == 0:
        return field

    trimmed = field[: rows * block_size, : cols * block_size]
    reshaped = trimmed.reshape(rows, block_size, cols, block_size)
    return np.nanmean(reshaped, axis=(1, 3))


def aggregate_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    field: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        block_reduce_mean(lats, block_size),
        block_reduce_mean(lons, block_size),
        block_reduce_mean(field, block_size),
    )


def collect_forecast_trend(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    max_forecast_hour: int,
    timeout: int,
    raw_dir: Path,
    smoothing_sigma: float,
    block_size: int,
) -> ForecastTrend | None:
    current_field = get_temperature_field(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
    if current_field is None:
        return None

    current_lats, current_lons, current_temp_f, valid_time = current_field

    prior_fields: list[np.ndarray] = []
    comparison_labels: list[str] = []
    for previous_date_str, previous_cycle_hour, previous_forecast_hour in iter_prior_run_requests(
        date_str,
        cycle_hour,
        forecast_hour,
        max_forecast_hour,
    ):
        previous_field = get_temperature_field(
            session,
            previous_date_str,
            previous_cycle_hour,
            previous_forecast_hour,
            timeout,
            raw_dir,
        )
        if previous_field is None:
            continue

        previous_lats, previous_lons, previous_temp_f, _ = previous_field
        if current_temp_f.shape != previous_temp_f.shape:
            continue
        if current_lats.shape != previous_lats.shape or current_lons.shape != previous_lons.shape:
            continue

        prior_fields.append(previous_temp_f)
        comparison_labels.append(f"{previous_date_str} {previous_cycle_hour:02d}z f{previous_forecast_hour:02d}")

    if not prior_fields:
        return None

    prior_mean_f = np.nanmean(np.stack(prior_fields), axis=0)
    grouped_lats, grouped_lons, grouped_delta = aggregate_grid(
        current_lats,
        current_lons,
        current_temp_f - prior_mean_f,
        block_size,
    )
    smoothed = smooth_field(grouped_delta, smoothing_sigma)
    return ForecastTrend(
        cycle_hour=cycle_hour,
        comparison_cycle_hours=comparison_labels,
        forecast_hour=forecast_hour,
        valid_time=valid_time,
        field_f=smoothed,
        lats=grouped_lats,
        lons=grouped_lons,
        mode="trend",
    )


def prefetch_temperature_fields(
    session: requests.Session,
    date_str: str,
    latest_cycle: int,
    max_forecast_hour: int,
    timeout: int,
    raw_dir: Path,
) -> None:
    print("Downloading HRRR fields before plotting", flush=True)
    for cycle_hour in range(0, latest_cycle + 1):
        print(f"Prefetching run {cycle_hour:02d}z", flush=True)
        for forecast_hour in range(0, max_forecast_hour + 1):
            load_or_download_grib(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
        clear_field_cache()


def draw_trend_map(trend: ForecastTrend, output_path: Path, date_str: str) -> None:
    projection = ccrs.LambertConformal(central_longitude=-96, central_latitude=38)
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=projection)
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f1e8")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d7e8f6")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor="#4b5563")

    levels = TREND_LEVELS
    extend = "both"
    colorbar_ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    mesh = ax.contourf(
        trend.lons,
        trend.lats,
        trend.field_f,
        levels=levels,
        cmap=TREND_CMAP,
        norm=TREND_NORM,
        extend=extend,
        transform=ccrs.PlateCarree(),
        antialiased=True,
    )
    contour_levels = np.linspace(levels[0], levels[-1], 9)
    ax.contour(
        trend.lons,
        trend.lats,
        trend.field_f,
        levels=contour_levels,
        colors="#334155",
        linewidths=0.35,
        alpha=0.55,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(
        "HRRR Smoothed 2 m Temperature Change\n"
        f"{date_str} run {trend.cycle_hour:02d}z forecast f{trend.forecast_hour:02d} | "
        f"valid {trend.valid_time:%m-%d %HZ}",
        fontsize=16,
        pad=16,
    )
    note = "Blue = cooler than the earlier-run average | Gray = near zero change | Red = warmer than the earlier-run average"
    colorbar_label = "Temperature change (F): -5 cooler to 0 neutral to 5 warmer"

    ax.text(
        0.01,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )

    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, shrink=0.82)
    colorbar.set_label(colorbar_label)
    if colorbar_ticks is not None:
        colorbar.set_ticks(colorbar_ticks)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    FIELD_CACHE.clear()
    now_utc = datetime.now(timezone.utc)
    resolved_date_str, latest_cycle = resolve_run_date_and_latest_cycle(args.date, now_utc)
    output_dir = Path(args.output_dir)
    date_dir = output_dir / resolved_date_str
    raw_dir = date_dir / "raw_grib_usa_temp_trend"
    plot_dir = date_dir / "plots" / "usa_temp_trend"

    print(f"Starting CONUS HRRR temperature trend run for {resolved_date_str}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Building run maps from 00z through {latest_cycle:02d}z", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "hrrr-usa-temp-trend/1.0"})

    prefetch_temperature_fields(
        session=session,
        date_str=resolved_date_str,
        latest_cycle=latest_cycle,
        max_forecast_hour=args.max_forecast_hour,
        timeout=args.timeout,
        raw_dir=raw_dir,
    )

    saved_maps = 0
    for cycle_hour in range(0, latest_cycle + 1):
        print(f"Processing run {cycle_hour:02d}z", flush=True)
        for forecast_hour in range(0, args.max_forecast_hour + 1):
            trend = collect_forecast_trend(
                session=session,
                date_str=resolved_date_str,
                cycle_hour=cycle_hour,
                forecast_hour=forecast_hour,
                max_forecast_hour=args.max_forecast_hour,
                timeout=args.timeout,
                raw_dir=raw_dir,
                smoothing_sigma=args.smoothing_sigma,
                block_size=args.block_size,
            )
            if trend is None:
                print(
                    f"Skipping {cycle_hour:02d}z f{forecast_hour:02d} because no comparable fields were available.",
                    flush=True,
                )
                continue

            output_path = plot_dir / f"hrrr_usa_temp_trend_{resolved_date_str}_{cycle_hour:02d}z_f{forecast_hour:02d}.png"
            draw_trend_map(trend, output_path, resolved_date_str)
            print(f"Saved output: {output_path}", flush=True)
            saved_maps += 1
            clear_field_cache()

        clear_field_cache()

    if saved_maps == 0:
        raise SystemExit("No comparable HRRR runs were available for the requested date.")

    print(f"Saved {saved_maps} USA temperature trend maps in: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
