from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr


NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
DEFAULT_DATE_FORMAT = "%Y%m%d"
DEFAULT_OUTPUT_DIR = "/var/data/output"
LOCATIONS = {
    "dc": {"label": "Washington, DC", "lat": 38.9072, "lon": -77.0369},
    "hyattsville_md": {"label": "Hyattsville, MD", "lat": 38.9559, "lon": -76.9450},
}


@dataclass(frozen=True)
class RunRecord:
    location_key: str
    location_label: str
    run_time: datetime
    forecast_hour: int
    valid_time: datetime
    temperature_c: float
    temperature_f: float
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download HRRR 2 m temperature for Washington, DC from today's 00z run "
            "through the most recent available run, then write a spreadsheet-style grid."
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
        help="Directory for CSV and plot outputs. Defaults to /var/data/output.",
    )
    parser.add_argument(
        "--max-forecast-hour",
        type=int,
        default=18,
        help="Largest forecast hour to try for each run. Default is 18 for each model run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args()


def build_url(date_str: str, cycle_hour: int, forecast_hour: int) -> str:
    lats = [location["lat"] for location in LOCATIONS.values()]
    lons = [location["lon"] for location in LOCATIONS.values()]
    query = {
        "dir": f"/hrrr.{date_str}/conus",
        "file": f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2",
        "var_TMP": "on",
        "lev_2_m_above_ground": "on",
        "subregion": "1",
        "leftlon": str(min(lons) - 0.25),
        "rightlon": str(max(lons) + 0.25),
        "bottomlat": str(min(lats) - 0.25),
        "toplat": str(max(lats) + 0.25),
    }
    prepared = requests.Request("GET", NOMADS_FILTER_URL, params=query).prepare()
    return prepared.url


def save_raw_grib(content: bytes, raw_dir: Path, cycle_hour: int, forecast_hour: int) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.tmp2m.grib2"
    raw_file.write_bytes(content)
    return raw_file


def extract_temperature(ds: xr.Dataset, target_lat: float, target_lon: float) -> tuple[datetime, float]:
    lat_name = "latitude" if "latitude" in ds else "lat"
    lon_name = "longitude" if "longitude" in ds else "lon"
    latitudes = ds[lat_name].values
    longitudes = ds[lon_name].values

    if np.nanmax(longitudes) > 180:
        longitudes = ((longitudes + 180) % 360) - 180

    distance = (latitudes - target_lat) ** 2 + (longitudes - target_lon) ** 2
    nearest_index = np.unravel_index(np.nanargmin(distance), distance.shape)
    point_dims = ds[lat_name].dims
    point_indexers = {dim: idx for dim, idx in zip(point_dims, nearest_index)}
    field = ds["t2m"].isel(point_indexers)

    valid_value = field.coords.get("valid_time", field.coords.get("time"))
    if valid_value is None:
        raise ValueError("Dataset did not contain a valid time coordinate.")

    valid_time = pd.to_datetime(valid_value.item()).to_pydatetime()
    temp_k = float(field.item())
    temp_c = temp_k - 273.15
    return valid_time, temp_c


def load_or_download_grib(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    timeout: int,
    raw_dir: Path,
) -> Path | None:
    raw_file = raw_dir / f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.tmp2m.grib2"
    if raw_file.exists():
        print(f"  using cached f{forecast_hour:02d}", flush=True)
        return raw_file

    print(f"  downloading f{forecast_hour:02d}", flush=True)
    url = build_url(date_str, cycle_hour, forecast_hour)
    response = session.get(url, timeout=timeout)

    if response.status_code == 404:
        return None

    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "html" in content_type.lower() or not response.content.startswith(b"GRIB"):
        return None

    return save_raw_grib(response.content, raw_dir, cycle_hour, forecast_hour)


def try_fetch_record(
    cycle_hour: int,
    forecast_hour: int,
    date_str: str,
    grib_file: Path,
) -> list[RunRecord]:
    run_time = datetime.strptime(f"{date_str}{cycle_hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    records: list[RunRecord] = []
    with xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    ) as ds:
        for location_key, location in LOCATIONS.items():
            valid_time, temp_c = extract_temperature(ds, location["lat"], location["lon"])
            valid_time = valid_time.replace(tzinfo=timezone.utc) if valid_time.tzinfo is None else valid_time.astimezone(timezone.utc)
            temp_f = (temp_c * 9 / 5) + 32
            records.append(
                RunRecord(
                    location_key,
                    location["label"],
                    run_time,
                    forecast_hour,
                    valid_time,
                    temp_c,
                    temp_f,
                    str(grib_file),
                )
            )
    return records


def collect_records(date_str: str, max_forecast_hour: int, timeout: int, raw_dir: Path) -> list[RunRecord]:
    target_date = datetime.strptime(date_str, DEFAULT_DATE_FORMAT).replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    latest_cycle = 23 if target_date.date() < now_utc.date() else now_utc.hour
    session = requests.Session()
    session.headers.update({"User-Agent": "hrrr-dc-temp-grid/1.0"})

    records: list[RunRecord] = []
    for cycle_hour in range(0, latest_cycle + 1):
        print(f"Checking run {cycle_hour:02d}z", flush=True)
        cycle_records: list[RunRecord] = []
        miss_streak = 0
        for forecast_hour in range(0, max_forecast_hour + 1):
            grib_file = load_or_download_grib(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
            if grib_file is None:
                print(f"  no data for {cycle_hour:02d}z f{forecast_hour:02d}", flush=True)
                miss_streak += 1
                if forecast_hour == 0 or miss_streak >= 2:
                    break
                continue

            records_for_hour = try_fetch_record(cycle_hour, forecast_hour, date_str, grib_file)
            cycle_records.extend(records_for_hour)
            print("  saved " + ", ".join(
                f"{record.location_key} {record.valid_time:%HZ} {record.temperature_f:.1f}F"
                for record in records_for_hour
            ), flush=True)
            miss_streak = 0

        records.extend(cycle_records)

    return records


def build_outputs(records: list[RunRecord], output_dir: Path, date_str: str) -> tuple[list[Path], Path]:
    date_dir = output_dir / date_str
    csv_dir = date_dir / "csv"
    plot_dir = date_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tidy = pd.DataFrame(
        {
            "location_key": [record.location_key for record in records],
            "location_label": [record.location_label for record in records],
            "run_time_utc": [record.run_time for record in records],
            "forecast_hour": [record.forecast_hour for record in records],
            "valid_time_utc": [record.valid_time for record in records],
            "temp_c": [record.temperature_c for record in records],
            "temp_f": [record.temperature_f for record in records],
            "source_file": [record.source_file for record in records],
        }
    ).sort_values(["run_time_utc", "valid_time_utc"])

    tidy_csv_path = csv_dir / f"hrrr_temp_tidy_{date_str}.csv"
    tidy.to_csv(tidy_csv_path, index=False)

    output_paths: list[Path] = []
    for location_key, location in LOCATIONS.items():
        location_rows = tidy[tidy["location_key"] == location_key]
        grid = location_rows.pivot(index="run_time_utc", columns="valid_time_utc", values="temp_f").sort_index(axis=0).sort_index(axis=1)
        grid.index = [ts.strftime("%Y-%m-%d %HZ") for ts in pd.to_datetime(grid.index, utc=True)]
        grid.columns = [ts.strftime("%m-%d %HZ") for ts in pd.to_datetime(grid.columns, utc=True)]

        csv_path = csv_dir / f"hrrr_{location_key}_temp_grid_{date_str}.csv"
        png_path = plot_dir / f"hrrr_{location_key}_temp_grid_{date_str}.png"
        grid.to_csv(csv_path)
        write_heatmap(grid, png_path, date_str, location["label"])
        output_paths.extend([csv_path, png_path])

    return output_paths, tidy_csv_path


def write_heatmap(grid: pd.DataFrame, png_path: Path, date_str: str, location_label: str) -> None:
    fig_width = max(12, len(grid.columns) * 0.45)
    fig_height = max(6, len(grid.index) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=90)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("Valid Time (UTC)")
    ax.set_ylabel("Model Run (UTC)")
    ax.set_title(f"HRRR 2 m Temperature for {location_label} ({date_str})")

    for row_idx, row in enumerate(grid.to_numpy(dtype=float)):
        for col_idx, value in enumerate(row):
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Temperature (F)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=175)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / args.date / "raw_grib"

    records = collect_records(args.date, args.max_forecast_hour, args.timeout, raw_dir)
    if not records:
        raise SystemExit("No HRRR runs were available for the requested UTC date.")

    output_paths, tidy_csv_path = build_outputs(records, output_dir, args.date)
    print(f"Saved raw GRIB files in: {raw_dir}")
    print(f"Saved tidy CSV: {tidy_csv_path}")
    for output_path in output_paths:
        print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()
