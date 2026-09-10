from __future__ import annotations

from datetime import datetime
import gzip
from io import BytesIO
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, abort, redirect, render_template, request, send_file, send_from_directory
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/data")
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
RENDER_BASE_DIR = Path("/opt/render/project/src/")
EASTERN_TIMEZONE = ZoneInfo("America/New_York")

app = Flask(__name__)

PRODUCTS = {
    "surface_full": {
        "label": "Combined Surface Dataset",
        "short_label": "Full Surface",
        "archive_dir": "raw_grib_full_surface",
    },
}
LEGACY_PRODUCTS = {
    "temp_2m": {
        "label": "2 m Temperature",
        "short_label": "Temperature",
        "archive_dir": "raw_grib",
    },
    "total_precip": {
        "label": "Total Precipitation",
        "short_label": "Total Precip",
        "archive_dir": "raw_grib_total_precip",
    },
    "surface_gust": {
        "label": "Surface Wind Gust",
        "short_label": "Wind Gust",
        "archive_dir": "raw_grib_gust",
    },
    "surface_cape": {
        "label": "Surface CAPE",
        "short_label": "CAPE",
        "archive_dir": "raw_grib_cape",
    },
    "surface_cfrzr": {
        "label": "Surface Freezing Rain",
        "short_label": "Freezing Rain",
        "archive_dir": "raw_grib_cfrzr",
    },
    "surface_cicep": {
        "label": "Surface Ice Pellets",
        "short_label": "Ice Pellets",
        "archive_dir": "raw_grib_cicep",
    },
    "surface_csnow": {
        "label": "Surface Snow",
        "short_label": "Snow",
        "archive_dir": "raw_grib_csnow",
    },
    "surface_hpbl_prate_snod_vis_weasd": {
        "label": "Surface HPBL PRATE SNOD VIS WEASD",
        "short_label": "HPBL + PRATE + SNOD + VIS + WEASD",
        "archive_dir": "raw_grib_hpbl_prate_snod_vis_weasd",
    },
}
DEFAULT_PRODUCT_KEY = "surface_full"


def get_date_directories() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [path for path in OUTPUT_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: path.name,
        reverse=True,
    )


def get_product_config(product_key: str | None) -> tuple[str, dict[str, str]]:
    all_products = PRODUCTS | LEGACY_PRODUCTS
    normalized_key = product_key if product_key in all_products else DEFAULT_PRODUCT_KEY
    return normalized_key, all_products[normalized_key]


def list_legacy_products(date_str: str | None) -> list[dict[str, object]]:
    if not date_str:
        return []

    legacy_entries: list[dict[str, object]] = []
    for product_key, product in LEGACY_PRODUCTS.items():
        raw_dir = OUTPUT_DIR / date_str / product["archive_dir"]
        if not raw_dir.exists():
            continue

        run_count = len([path for path in raw_dir.iterdir() if path.is_dir()])
        legacy_entries.append(
            {
                "key": product_key,
                "label": product["label"],
                "short_label": product["short_label"],
                "archive_dir": product["archive_dir"],
                "run_count": run_count,
            }
        )

    return legacy_entries


def list_raw_grib_runs(date_str: str, product_key: str) -> list[dict[str, str | int]]:
    raw_dir = get_raw_grib_dir(date_str, product_key)
    if not raw_dir.exists():
        return []

    runs: list[dict[str, str | int]] = []
    for run_dir in sorted([path for path in raw_dir.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True):
        file_count = len(list_grib_files(run_dir))
        runs.append(
            {
                "name": run_dir.name,
                "label": run_dir.name.upper(),
                "file_count": file_count,
            }
        )
    return runs


def get_run_files(date_str: str, run_name: str, product_key: str) -> list[Path]:
    run_dir = get_raw_grib_dir(date_str, product_key) / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    return list_grib_files(run_dir)


def list_grib_files(path: Path) -> list[Path]:
    return sorted([*path.glob("*.grib2"), *path.glob("*.grib2.gz")])


def open_grib_dataset_from_path(dataset_path: Path):
    backend_options = [
        {"indexpath": ""},
        {"indexpath": "", "filter_by_keys": {"stepType": "instant"}},
        {"indexpath": "", "filter_by_keys": {"stepType": "accum"}},
    ]
    last_error: Exception | None = None

    for backend_kwargs in backend_options:
        try:
            dataset = xr.open_dataset(dataset_path, engine="cfgrib", backend_kwargs=backend_kwargs)
            return dataset, backend_kwargs
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def open_grib_dataset(grib_path: Path):
    if grib_path.suffix != ".gz":
        dataset, backend_kwargs = open_grib_dataset_from_path(grib_path)
        return dataset, None, backend_kwargs

    with gzip.open(grib_path, "rb") as compressed_stream:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as temp_file:
            temp_file.write(compressed_stream.read())
            temp_path = Path(temp_file.name)

    try:
        dataset, backend_kwargs = open_grib_dataset_from_path(temp_path)
        return dataset, temp_path, backend_kwargs
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def summarize_grib_dataset(grib_path: Path) -> dict[str, object]:
    dataset, temp_path, backend_kwargs = open_grib_dataset(grib_path)
    ds = dataset
    try:
        dataset_attrs = {key: str(value) for key, value in ds.attrs.items()}
        source_value = dataset_attrs.get("source")
        if source_value:
            dataset_attrs["source"] = Path(source_value).name
        filter_by_keys = backend_kwargs.get("filter_by_keys")
        if filter_by_keys:
            dataset_attrs["filter_by_keys"] = str(filter_by_keys)
        history_value = dataset_attrs.get("history")
        if history_value:
            sanitized_history = re.sub(
                r'("source"\s*:\s*")([^"]+)(")',
                lambda match: f'{match.group(1)}{Path(match.group(2)).name}{match.group(3)}',
                history_value,
            )
            sanitized_history = re.sub(
                r"\s*GRIB to CDM\+CF via cfgrib-[^\s]+/ecCodes-[^\s]+ with\s*",
                " ",
                sanitized_history,
            ).strip()
            dataset_attrs["history"] = sanitized_history

        return {
            "dimensions": [{"name": name, "size": size} for name, size in ds.sizes.items()],
            "coordinates": [
                {"name": name, "dims": list(coord.dims), "dtype": str(coord.dtype)}
                for name, coord in ds.coords.items()
            ],
            "variables": [
                {
                    "name": name,
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                    "attrs": {key: str(value) for key, value in list(variable.attrs.items())[:8]},
                }
                for name, variable in ds.data_vars.items()
            ],
            "attributes": dataset_attrs,
        }
    finally:
        ds.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def get_raw_grib_dir(date_str: str, product_key: str) -> Path:
    _, product = get_product_config(product_key)
    return OUTPUT_DIR / date_str / product["archive_dir"]


def get_downloadable_dates(date_dirs: list[Path], requested_dates: list[str], include_all: bool) -> list[str]:
    product_key = request.args.get("product")
    normalized_key, _ = get_product_config(product_key)
    available_dates = [path.name for path in date_dirs if get_raw_grib_dir(path.name, normalized_key).exists()]
    if include_all:
        return available_dates

    valid_dates = [date_str for date_str in requested_dates if date_str in available_dates]
    return valid_dates


def create_raw_grib_archive(date_strs: list[str], product_key: str) -> BytesIO:
    if not date_strs:
        abort(404)

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for date_str in date_strs:
            raw_dir = get_raw_grib_dir(date_str, product_key)
            if not raw_dir.exists():
                continue

            for grib_path in sorted([*raw_dir.rglob("*.grib2"), *raw_dir.rglob("*.grib2.gz")]):
                archive_path = Path(date_str) / grib_path.relative_to(raw_dir)
                archive.write(grib_path, arcname=str(archive_path))

    if archive_buffer.getbuffer().nbytes == 0:
        abort(404)

    archive_buffer.seek(0)
    return archive_buffer


@app.route("/")
def index() -> str:
    date_dirs = get_date_directories()
    requested_date = request.args.get("date")
    selected_product, product = get_product_config(request.args.get("product"))
    selected_dir = next((path for path in date_dirs if path.name == requested_date), None)
    if selected_dir is None and date_dirs:
        selected_dir = date_dirs[0]

    selected_date = selected_dir.name if selected_dir else None
    downloadable_dates = [path.name for path in date_dirs if get_raw_grib_dir(path.name, selected_product).exists()]
    selected_download_dates = request.args.getlist("download_date")
    selected_download_dates = [date_str for date_str in selected_download_dates if date_str in downloadable_dates]
    if not selected_download_dates and selected_date in downloadable_dates:
        selected_download_dates = [selected_date]
    run_entries = list_raw_grib_runs(selected_date, selected_product) if selected_date else []
    legacy_products = list_legacy_products(selected_date)

    return render_template(
        "index.html",
        available_dates=[path.name for path in date_dirs],
        selected_date=selected_date,
        selected_product=selected_product,
        selected_product_label=product["label"],
        product_options=[{"key": key, "label": value["label"]} for key, value in PRODUCTS.items()],
        downloadable_dates=downloadable_dates,
        selected_download_dates=selected_download_dates,
        run_entries=run_entries,
        raw_grib_available=bool(selected_date and get_raw_grib_dir(selected_date, selected_product).exists()),
        legacy_products=legacy_products,
    )


@app.route("/plots/<date_str>/<state_key>/<filename>")
def serve_plot(date_str: str, state_key: str, filename: str):
    plot_dir = OUTPUT_DIR / date_str / "plots" / state_key
    if not plot_dir.exists():
        abort(404)
    return send_from_directory(plot_dir, filename)


@app.route("/usa-trend-plots/<date_str>/<filename>")
def serve_usa_trend_plot(date_str: str, filename: str):
    plot_dir = OUTPUT_DIR / date_str / "plots" / "usa_temp_trend"
    if not plot_dir.exists():
        abort(404)
    return send_from_directory(plot_dir, filename)


@app.route("/downloads/raw-grib.zip")
def download_raw_grib_archive():
    date_dirs = get_date_directories()
    selected_product, _ = get_product_config(request.args.get("product"))
    requested_dates = request.args.getlist("date")
    include_all = request.args.get("all") == "1"
    date_strs = get_downloadable_dates(date_dirs, requested_dates, include_all)
    archive_buffer = create_raw_grib_archive(date_strs, selected_product)
    if include_all:
        download_name = f"all_dates_{selected_product}_raw_grib.zip"
    elif len(date_strs) == 1:
        download_name = f"{date_strs[0]}_{selected_product}_raw_grib.zip"
    else:
        download_name = f"selected_dates_{len(date_strs)}_{selected_product}_raw_grib.zip"

    return send_file(
        archive_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
    )


@app.route("/grib-dataset")
def view_grib_dataset() -> str:
    date_str = request.args.get("date", "")
    run_name = request.args.get("run", "")
    selected_product, product = get_product_config(request.args.get("product"))
    run_files = get_run_files(date_str, run_name, selected_product)
    if not run_files:
        abort(404)

    selected_file = run_files[0]
    summary: dict[str, object] | None = None
    load_error: str | None = None
    try:
        summary = summarize_grib_dataset(selected_file)
    except Exception as exc:
        load_error = str(exc)

    return render_template(
        "grib_dataset.html",
        selected_date=date_str,
        selected_product=selected_product,
        selected_product_label=product["label"],
        run_name=run_name,
        selected_file=selected_file.name,
        file_count=len(run_files),
        summary=summary,
        load_error=load_error,
    )

if __name__ == "__main__":
    app.run(debug=True)
