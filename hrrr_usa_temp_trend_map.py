from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
CONUS_BOUNDS = {
    "leftlon": -125.0,
    "rightlon": -66.5,
    "bottomlat": 24.0,
    "toplat": 49.5,
}
FIELD_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, datetime] | None] = {}


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
    cache_key = (cycle_hour, forecast_hour)
    if cache_key in FIELD_CACHE:
        return FIELD_CACHE[cache_key]

    grib_file = load_or_download_grib(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
    if grib_file is None:
        FIELD_CACHE[cache_key] = None
        return None

    field = load_temperature_field(grib_file)
    FIELD_CACHE[cache_key] = field
    return field


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


def collect_forecast_trend(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    timeout: int,
    raw_dir: Path,
    smoothing_sigma: float,
) -> ForecastTrend | None:
    current_field = get_temperature_field(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
    if current_field is None:
        return None

    current_lats, current_lons, current_temp_f, valid_time = current_field

    if cycle_hour == 0:
        smoothed = smooth_field(current_temp_f, smoothing_sigma)
        return ForecastTrend(
            cycle_hour=cycle_hour,
            comparison_cycle_hours=[0],
            forecast_hour=forecast_hour,
            valid_time=valid_time,
            field_f=smoothed,
            lats=current_lats,
            lons=current_lons,
            mode="absolute",
        )

    prior_fields: list[np.ndarray] = []
    comparison_cycle_hours: list[int] = []
    for previous_cycle_hour in range(0, cycle_hour):
        previous_field = get_temperature_field(session, date_str, previous_cycle_hour, forecast_hour, timeout, raw_dir)
        if previous_field is None:
            continue

        previous_lats, previous_lons, previous_temp_f, _ = previous_field
        if current_temp_f.shape != previous_temp_f.shape:
            continue
        if current_lats.shape != previous_lats.shape or current_lons.shape != previous_lons.shape:
            continue

        prior_fields.append(previous_temp_f)
        comparison_cycle_hours.append(previous_cycle_hour)

    if not prior_fields:
        return None

    prior_mean_f = np.nanmean(np.stack(prior_fields), axis=0)
    smoothed = smooth_field(current_temp_f - prior_mean_f, smoothing_sigma)
    return ForecastTrend(
        cycle_hour=cycle_hour,
        comparison_cycle_hours=comparison_cycle_hours,
        forecast_hour=forecast_hour,
        valid_time=valid_time,
        field_f=smoothed,
        lats=current_lats,
        lons=current_lons,
        mode="trend",
    )


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

    if trend.mode == "absolute":
        vmin = float(np.nanpercentile(trend.field_f, 5))
        vmax = float(np.nanpercentile(trend.field_f, 95))
        levels = np.linspace(vmin, vmax, 17)
        cmap = "coolwarm"
        extend = "both"
        colorbar_ticks = None
    else:
        levels = np.arange(-5, 5.5, 0.5)
        cmap = "RdBu_r"
        extend = "both"
        colorbar_ticks = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    mesh = ax.contourf(
        trend.lons,
        trend.lats,
        trend.field_f,
        levels=levels,
        cmap=cmap,
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

    if trend.mode == "absolute":
        ax.set_title(
            "HRRR Smoothed 2 m Temperature\n"
            f"{date_str} run {trend.cycle_hour:02d}z forecast f{trend.forecast_hour:02d} | "
            f"valid {trend.valid_time:%m-%d %HZ}",
            fontsize=16,
            pad=16,
        )
        note = "Run 00 baseline map | Smoothed 2 m temperature field"
        colorbar_label = "Temperature (F)"
    else:
        compared = " ".join(f"{hour:02d}z" for hour in trend.comparison_cycle_hours)
        ax.set_title(
            "HRRR Smoothed 2 m Temperature Change\n"
            f"{date_str} run {trend.cycle_hour:02d}z forecast f{trend.forecast_hour:02d} versus {compared} | "
            f"valid {trend.valid_time:%m-%d %HZ}",
            fontsize=16,
            pad=16,
        )
        note = "Red = warmer than earlier runs | Blue = cooler than earlier runs | 0 is neutral"
        colorbar_label = "Temperature trend scale: -5 cooler to 0 neutral to 5 warmer"

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
    output_dir = Path(args.output_dir)
    date_dir = output_dir / args.date
    raw_dir = date_dir / "raw_grib_usa_temp_trend"
    plot_dir = date_dir / "plots" / "usa_temp_trend"

    target_date = datetime.strptime(args.date, DEFAULT_DATE_FORMAT).replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    latest_cycle = 23 if target_date.date() < now_utc.date() else now_utc.hour

    print(f"Starting CONUS HRRR temperature trend run for {args.date}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Building run maps from 00z through {latest_cycle:02d}z", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "hrrr-usa-temp-trend/1.0"})

    saved_maps = 0
    for cycle_hour in range(0, latest_cycle + 1):
        print(f"Processing run {cycle_hour:02d}z", flush=True)
        for forecast_hour in range(0, args.max_forecast_hour + 1):
            trend = collect_forecast_trend(
                session=session,
                date_str=args.date,
                cycle_hour=cycle_hour,
                forecast_hour=forecast_hour,
                timeout=args.timeout,
                raw_dir=raw_dir,
                smoothing_sigma=args.smoothing_sigma,
            )
            if trend is None:
                print(
                    f"Skipping {cycle_hour:02d}z f{forecast_hour:02d} because no comparable fields were available.",
                    flush=True,
                )
                continue

            output_path = plot_dir / f"hrrr_usa_temp_trend_{args.date}_{cycle_hour:02d}z_f{forecast_hour:02d}.png"
            draw_trend_map(trend, output_path, args.date)
            print(f"Saved output: {output_path}", flush=True)
            saved_maps += 1

    if saved_maps == 0:
        raise SystemExit("No comparable HRRR runs were available for the requested date.")

    print(f"Saved {saved_maps} USA temperature trend maps in: {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
