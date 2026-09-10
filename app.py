from __future__ import annotations

from datetime import datetime
import contextlib
import gzip
import lzma
from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, abort, redirect, render_template, request, send_file, send_from_directory
import cfgrib
import eccodes
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/data")
OUTPUT_DIR = DATA_DIR / "output"
RENDER_BASE_DIR = Path("/opt/render/project/src/")
EASTERN_TIMEZONE = ZoneInfo("America/New_York")

app = Flask(__name__)

SURFACE_TRACKED_GRIB_FIELDS = [
    {"request_key": "var_4LFTX", "label": "Best 4-layer lifted index", "aliases": {"4lftx"}, "level_hint": "pressureFromGroundLayer 18000", "expected_absence_reason": "Excluded when the request is limited to lev_surface=on."},
    {"request_key": "var_TMP", "label": "Temperature", "aliases": {"tmp", "t", "2t"}, "level_hint": "2 m above ground"},
    {"request_key": "var_APCP", "label": "Total precipitation", "aliases": {"apcp", "tp"}},
    {"request_key": "var_GUST", "label": "Wind gust", "aliases": {"gust", "i10fg"}},
    {"request_key": "var_CAPE", "label": "Convective available potential energy", "aliases": {"cape"}},
    {"request_key": "var_CFRZR", "label": "Categorical freezing rain", "aliases": {"cfrzr"}},
    {"request_key": "var_CICEP", "label": "Categorical ice pellets", "aliases": {"cicep"}},
    {"request_key": "var_CSNOW", "label": "Categorical snow", "aliases": {"csnow"}},
    {"request_key": "var_FRICV", "label": "Frictional velocity", "aliases": {"fricv"}},
    {"request_key": "var_HGT", "label": "Surface elevation / orography", "aliases": {"hgt", "gh", "orog"}, "level_hint": "surface", "expected_absence_reason": "For surface-only HRRR subsets this request maps to the surface orography record."},
    {"request_key": "var_HPBL", "label": "Boundary layer height", "aliases": {"hpbl", "blh"}},
    {"request_key": "var_PRATE", "label": "Precipitation rate", "aliases": {"prate"}},
    {"request_key": "var_SNOD", "label": "Snow depth", "aliases": {"snod", "sd", "sde"}},
    {"request_key": "var_SNOWC", "label": "Snow cover", "aliases": {"snowc"}},
    {"request_key": "var_VIS", "label": "Visibility", "aliases": {"vis"}},
    {"request_key": "var_WEASD", "label": "Water equivalent of accumulated snow depth", "aliases": {"weasd", "sdwe"}, "expected_absence_reason": "This can appear in both instant and accum slices for the same GRIB file."},
    {"request_key": "lev_surface", "label": "Surface level filter", "request_only": True},
    {"request_key": "subregion", "label": "CONUS subregion crop", "request_only": True},
]

ENTIRE_ATMOSPHERE_TRACKED_GRIB_FIELDS = [
    {"request_key": "var_HAIL", "label": "Maximum hail size", "aliases": {"hail"}},
    {"request_key": "var_LTNG", "label": "Lightning", "aliases": {"ltng"}},
    {"request_key": "var_REFC", "label": "Maximum or composite radar reflectivity", "aliases": {"refc", "refd"}},
    {"request_key": "var_RHPW", "label": "Relative humidity", "aliases": {"rhpw", "param1_242"}, "display_short_name": "rhpw", "level_hint": "entire atmosphere", "expected_absence_reason": "This field is encoded with an unknown short name in the default GRIB tables, so the inspector matches it by GRIB parameter metadata."},
    {"request_key": "var_TCDC", "label": "Total cloud cover", "aliases": {"tcdc", "tcc"}},
    {"request_key": "var_TCOLI", "label": "Total column integrated condensate", "aliases": {"tcoli", "param1_70"}, "display_short_name": "tcoli", "expected_absence_reason": "This field is encoded with an unknown short name in the default GRIB tables, so the inspector matches it by GRIB parameter metadata."},
    {"request_key": "var_VIL", "label": "Vertically integrated liquid", "aliases": {"vil", "veril"}},
    {"request_key": "lev_entire_atmosphere", "label": "Entire atmosphere level filter", "request_only": True},
]

TRACKED_GRIB_FIELDS_BY_PRODUCT = {
    "surface_full": SURFACE_TRACKED_GRIB_FIELDS,
    "entire_atmosphere": ENTIRE_ATMOSPHERE_TRACKED_GRIB_FIELDS,
}

PRODUCTS = {
    "surface_full": {
        "label": "Combined Surface Dataset",
        "short_label": "Full Surface",
        "archive_dir": "raw_grib_full_surface",
        "tracked_fields_key": "surface_full",
    },
    "entire_atmosphere": {
        "label": "Entire Atmosphere Severe Weather Dataset",
        "short_label": "Entire Atmosphere",
        "archive_dir": "raw_grib_entire_atmosphere",
        "tracked_fields_key": "entire_atmosphere",
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


def build_task_run_id() -> str:
    return datetime.now(EASTERN_TIMEZONE).strftime("task1-%Y%m%d-%H%M%S")


def build_task_summary_log_path(task_number: int) -> Path:
    timestamp = datetime.now(EASTERN_TIMEZONE).strftime("%y%m%d_%I%M%S%p").lower()
    return BASE_DIR / f"task{task_number}_summary_{timestamp}.log"


def format_duration_hhmmss(started_at: datetime, finished_at: datetime) -> str:
    total_seconds = max(0, int((finished_at - started_at).total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_script_path(render_path: str, local_name: str) -> Path:
    render_script = Path(render_path)
    if render_script.exists():
        return render_script

    local_script = BASE_DIR / local_name
    if local_script.exists():
        return local_script

    abort(500, description=f"Script not found: {local_name}")


def run_scripts(scripts: list[Path], task_run_id: str, task_number: int) -> None:
    log_path = build_task_summary_log_path(task_number)

    with log_path.open("a", encoding="utf-8") as log_file:
        started_at = datetime.now(EASTERN_TIMEZONE)
        log_file.write(f"Task run id: {task_run_id}\n")
        log_file.write(f"Starting task{task_number} at {started_at.isoformat()}\n")
        log_file.flush()

        for script_path in scripts:
            log_file.write(f"Running {script_path.name}\n")
            log_file.flush()
            subprocess.run(
                [sys.executable, str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        finished_at = datetime.now(EASTERN_TIMEZONE)
        log_file.write(f"Finished task{task_number} at {finished_at.isoformat()}\n")
        log_file.write(f"Total run time: {format_duration_hhmmss(started_at, finished_at)}\n")
        log_file.flush()


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
    return sorted([*path.glob("*.grib2"), *path.glob("*.grib2.gz"), *path.glob("*.grib2.xz")])


def open_grib_datasets_from_path(dataset_path: Path) -> list[tuple[xr.Dataset, dict[str, object]]]:
    backend_options = [
        {"indexpath": ""},
        {"indexpath": "", "filter_by_keys": {"stepType": "instant"}},
        {"indexpath": "", "filter_by_keys": {"stepType": "accum"}},
    ]
    last_error: Exception | None = None

    for backend_kwargs in backend_options:
        try:
            datasets = cfgrib.open_datasets(dataset_path, backend_kwargs=backend_kwargs)
            if datasets:
                return [(dataset, backend_kwargs) for dataset in datasets]
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def open_grib_datasets(grib_path: Path):
    if grib_path.suffix not in {".gz", ".xz"}:
        datasets = open_grib_datasets_from_path(grib_path)
        return datasets, None

    open_compressed = gzip.open if grib_path.suffix == ".gz" else lzma.open
    with open_compressed(grib_path, "rb") as compressed_stream:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as temp_file:
            temp_file.write(compressed_stream.read())
            temp_path = Path(temp_file.name)

    try:
        datasets = open_grib_datasets_from_path(temp_path)
        return datasets, temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def normalize_field_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def safe_grib_get(handle, key: str) -> str:
    try:
        return str(eccodes.codes_get(handle, key))
    except Exception:
        return ""


def get_grib_message_match_label(message: dict[str, str], dataset_label: str) -> str:
    preferred_name = message.get("shortName") or message.get("name") or "unknown"
    type_of_level = message.get("typeOfLevel", "").strip()
    level = message.get("level", "").strip()
    level_suffix = ""
    if type_of_level and level:
        level_suffix = f", {type_of_level} {level}"
    elif type_of_level:
        level_suffix = f", {type_of_level}"
    return f"{preferred_name} ({dataset_label}{level_suffix})"


def extract_grib_message_inventory(grib_path: Path) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    with grib_path.open("rb") as stream:
        while True:
            handle = eccodes.codes_grib_new_from_file(stream)
            if handle is None:
                break
            try:
                messages.append(
                    {
                        "shortName": safe_grib_get(handle, "shortName"),
                        "name": safe_grib_get(handle, "name"),
                        "typeOfLevel": safe_grib_get(handle, "typeOfLevel"),
                        "level": safe_grib_get(handle, "level"),
                        "stepType": safe_grib_get(handle, "stepType"),
                        "paramId": safe_grib_get(handle, "paramId"),
                        "parameterCategory": safe_grib_get(handle, "parameterCategory"),
                        "parameterNumber": safe_grib_get(handle, "parameterNumber"),
                    }
                )
            finally:
                eccodes.codes_release(handle)
    return messages


def get_variable_match_label(name: str, variable: xr.DataArray, dataset_label: str) -> str:
    preferred_name = str(variable.attrs.get("GRIB_shortName") or variable.attrs.get("short_name") or name)
    type_of_level = str(variable.attrs.get("GRIB_typeOfLevel") or "").strip()
    level = str(variable.attrs.get("GRIB_level") or "").strip()
    level_suffix = ""
    if type_of_level and level:
        level_suffix = f", {type_of_level} {level}"
    elif type_of_level:
        level_suffix = f", {type_of_level}"
    return f"{preferred_name} ({dataset_label}{level_suffix})"


def apply_match_display_overrides(field: dict[str, object], matched_names: list[str]) -> list[str]:
    display_short_name = str(field.get("display_short_name", "")).strip()
    if not display_short_name:
        return matched_names

    updated_names: list[str] = []
    for match_name in matched_names:
        if match_name.startswith("unknown ("):
            updated_names.append(f"{display_short_name}{match_name[len('unknown'):]}")
        else:
            updated_names.append(match_name)
    return updated_names


def get_tracked_fields_for_product(product_key: str) -> list[dict[str, object]]:
    normalized_key, product = get_product_config(product_key)
    tracked_fields_key = product.get("tracked_fields_key", normalized_key)
    return TRACKED_GRIB_FIELDS_BY_PRODUCT.get(tracked_fields_key, SURFACE_TRACKED_GRIB_FIELDS)


def build_tracked_field_summary(
    datasets: list[tuple[xr.Dataset, dict[str, object]]], product_key: str, grib_messages: list[dict[str, str]]
) -> list[dict[str, object]]:
    variable_lookup: dict[str, set[str]] = {}
    message_lookup: dict[str, set[str]] = {}
    for dataset_index, (ds, backend_kwargs) in enumerate(datasets, start=1):
        dataset_label = backend_kwargs.get("filter_by_keys", {}).get("stepType", f"dataset-{dataset_index}")
        for name, variable in ds.data_vars.items():
            display_name = get_variable_match_label(name, variable, dataset_label)
            tokens = {
                normalize_field_token(name),
                normalize_field_token(str(variable.attrs.get("GRIB_shortName", ""))),
                normalize_field_token(str(variable.attrs.get("short_name", ""))),
                normalize_field_token(str(variable.attrs.get("standard_name", ""))),
                normalize_field_token(str(variable.attrs.get("long_name", ""))),
                normalize_field_token(str(variable.attrs.get("GRIB_name", ""))),
                normalize_field_token(
                    f"param{variable.attrs.get('GRIB_parameterCategory', '')}_{variable.attrs.get('GRIB_parameterNumber', '')}"
                ),
            }
            for token in [token for token in tokens if token]:
                variable_lookup.setdefault(token, set()).add(display_name)

    for message_index, message in enumerate(grib_messages, start=1):
        dataset_label = f"message-{message_index}"
        display_name = get_grib_message_match_label(message, dataset_label)
        tokens = {
            normalize_field_token(message.get("shortName", "")),
            normalize_field_token(message.get("name", "")),
            normalize_field_token(f"param{message.get('parameterCategory', '')}_{message.get('parameterNumber', '')}"),
            normalize_field_token(f"paramid{message.get('paramId', '')}"),
        }
        for token in [token for token in tokens if token]:
            message_lookup.setdefault(token, set()).add(display_name)

    summary_rows: list[dict[str, object]] = []
    for field in get_tracked_fields_for_product(product_key):
        aliases = {normalize_field_token(alias) for alias in field.get("aliases", set())}
        matched_names = sorted(
            {
                name
                for alias in aliases
                for name in [*variable_lookup.get(alias, set()), *message_lookup.get(alias, set())]
            }
        )
        matched_names = apply_match_display_overrides(field, matched_names)
        if field.get("request_only"):
            status = "request-filter"
        elif matched_names:
            status = "present"
        else:
            status = "missing"

        summary_rows.append(
            {
                "request_key": field["request_key"],
                "label": field["label"],
                "status": status,
                "matches": matched_names,
                "level_hint": field.get("level_hint", ""),
                "expected_absence_reason": field.get("expected_absence_reason", ""),
            }
        )

    return summary_rows


def summarize_grib_dataset(grib_path: Path, product_key: str) -> dict[str, object]:
    opened_datasets, temp_path = open_grib_datasets(grib_path)
    try:
        inventory_path = temp_path if temp_path is not None else grib_path
        grib_messages = extract_grib_message_inventory(inventory_path)
        variable_entries: list[dict[str, object]] = []
        combined_dimensions: dict[str, int] = {}
        coordinates: list[dict[str, object]] = []
        attributes: list[dict[str, object]] = []

        for dataset_index, (ds, backend_kwargs) in enumerate(opened_datasets, start=1):
            dataset_label = backend_kwargs.get("filter_by_keys", {}).get("stepType", f"dataset-{dataset_index}")
            for name, size in ds.sizes.items():
                combined_dimensions[name] = max(combined_dimensions.get(name, 0), size)

            coordinates.extend(
                {
                    "dataset": dataset_label,
                    "name": name,
                    "dims": list(coord.dims),
                    "dtype": str(coord.dtype),
                }
                for name, coord in ds.coords.items()
            )

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
            attributes.append({"dataset": dataset_label, "attrs": dataset_attrs})

            variable_entries.extend(
                {
                    "dataset": dataset_label,
                    "name": name,
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                    "attrs": {key: str(value) for key, value in list(variable.attrs.items())[:8]},
                }
                for name, variable in ds.data_vars.items()
            )

        return {
            "tracked_fields": build_tracked_field_summary(opened_datasets, product_key, grib_messages),
            "dimensions": [{"name": name, "size": size} for name, size in combined_dimensions.items()],
            "coordinates": coordinates,
            "variables": variable_entries,
            "attributes": attributes,
        }
    finally:
        for ds, _backend_kwargs in opened_datasets:
            with contextlib.suppress(Exception):
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

            for grib_path in sorted([*raw_dir.rglob("*.grib2"), *raw_dir.rglob("*.grib2.gz"), *raw_dir.rglob("*.grib2.xz")]):
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


@app.route("/run-task1")
def run_task1():
    task_run_id = build_task_run_id()
    scripts = [
        resolve_script_path(
            "/opt/render/project/src/hrrr_grib_full_surface_archive.py",
            "hrrr_grib_full_surface_archive.py",
        ),
        resolve_script_path(
            "/opt/render/project/src/hrrr_grib_entire_atmosphere_archive.py",
            "hrrr_grib_entire_atmosphere_archive.py",
        ),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, task_run_id, 1), daemon=True).start()
    return f"Task started in background as {task_run_id}.", 200


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
        summary = summarize_grib_dataset(selected_file, selected_product)
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
