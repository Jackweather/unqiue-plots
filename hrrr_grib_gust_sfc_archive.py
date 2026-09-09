from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests


NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"
DEFAULT_DATE_FORMAT = "%Y%m%d"
DEFAULT_OUTPUT_DIR = "/var/data/output"
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_FORECAST_HOUR = 18
CONUS_BOUNDS = {
    "leftlon": -125.0,
    "rightlon": -66.5,
    "bottomlat": 24.0,
    "toplat": 49.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HRRR surface wind gust GRIB files and save them in the raw archive layout."
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime(DEFAULT_DATE_FORMAT),
        help="UTC date to query in YYYYMMDD format. Defaults to today in UTC.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where raw GRIB files are archived. Defaults to /var/data/output.",
    )
    parser.add_argument(
        "--max-forecast-hour",
        type=int,
        default=DEFAULT_MAX_FORECAST_HOUR,
        help="Largest forecast hour to try for each run. Default is 18.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args()


def build_url(date_str: str, cycle_hour: int, forecast_hour: int) -> str:
    query = {
        "dir": f"/hrrr.{date_str}/conus",
        "file": f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.grib2",
        "var_GUST": "on",
        "lev_surface": "on",
        "subregion": "1",
        **{key: str(value) for key, value in CONUS_BOUNDS.items()},
    }
    prepared = requests.Request("GET", NOMADS_FILTER_URL, params=query).prepare()
    assert prepared.url is not None
    return prepared.url


def get_run_grib_dir(raw_dir: Path, cycle_hour: int) -> Path:
    return raw_dir / f"{cycle_hour:02d}z"


def get_target_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, DEFAULT_DATE_FORMAT).replace(tzinfo=timezone.utc)


def detect_latest_cycle(date_str: str) -> int:
    target_date = get_target_date(date_str)
    now_utc = datetime.now(timezone.utc)
    return 23 if target_date.date() < now_utc.date() else now_utc.hour


def load_or_download_grib(
    session: requests.Session,
    date_str: str,
    cycle_hour: int,
    forecast_hour: int,
    timeout: int,
    raw_dir: Path,
) -> Path | None:
    run_dir = get_run_grib_dir(raw_dir, cycle_hour)
    raw_file = run_dir / f"hrrr.t{cycle_hour:02d}z.wrfsfcf{forecast_hour:02d}.gust_surface.grib2"
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


def archive_gribs(date_str: str, max_forecast_hour: int, timeout: int, output_dir: Path) -> int:
    raw_dir = output_dir / date_str / "raw_grib_gust"
    latest_cycle = detect_latest_cycle(date_str)
    session = requests.Session()
    session.headers.update({"User-Agent": "hrrr-grib-gust-archive/1.0"})

    saved_files = 0
    print(f"Starting HRRR surface gust GRIB archive run for {date_str}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Archiving runs from 00z through {latest_cycle:02d}z", flush=True)

    for cycle_hour in range(0, latest_cycle + 1):
        print(f"Checking run {cycle_hour:02d}z", flush=True)
        miss_streak = 0
        for forecast_hour in range(0, max_forecast_hour + 1):
            grib_file = load_or_download_grib(session, date_str, cycle_hour, forecast_hour, timeout, raw_dir)
            if grib_file is None:
                print(f"  no data for {cycle_hour:02d}z f{forecast_hour:02d}", flush=True)
                miss_streak += 1
                if forecast_hour == 0 or miss_streak >= 2:
                    break
                continue

            saved_files += 1
            miss_streak = 0

    if saved_files == 0:
        raise SystemExit("No HRRR surface gust GRIB files were available for the requested UTC date.")

    print(f"Saved {saved_files} GRIB files in: {raw_dir}", flush=True)
    return saved_files


def main() -> None:
    args = parse_args()
    archive_gribs(
        date_str=args.date,
        max_forecast_hour=args.max_forecast_hour,
        timeout=args.timeout,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()