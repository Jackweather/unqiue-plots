from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import threading
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, abort, redirect, render_template, request, send_file, send_from_directory
import xarray as xr

from ai_hrrr_learning import load_learning_summary, train_learning_summary


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/data")
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
RENDER_BASE_DIR = Path("/opt/render/project/src/")
EASTERN_TIMEZONE = ZoneInfo("America/New_York")

app = Flask(__name__)


def resolve_script_path(render_script: str, local_script: str) -> tuple[str, str]:
    render_path = Path(render_script)
    if render_path.exists():
        return str(render_path), str(render_path.parent)

    local_path = BASE_DIR / local_script
    return str(local_path), str(local_path.parent)


def get_summary_log_dir() -> Path:
    if RENDER_BASE_DIR.exists():
        return RENDER_BASE_DIR
    return BASE_DIR


def build_task_run_id(now: datetime | None = None) -> str:
    return "task1"


def format_duration(duration_seconds: float) -> str:
    total_seconds = int(round(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def run_scripts(scripts: list[tuple[str, str]], task_run_id: str, max_parallel: int = 1) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = threading.Semaphore(max_parallel)
    threads: list[threading.Thread] = []
    result_lock = threading.Lock()
    run_results: list[tuple[str, float]] = []
    summary_log_path = get_summary_log_dir() / f"{task_run_id}.log"
    log_lock = threading.Lock()

    def run_one(script_path: str, working_dir: str) -> None:
        with semaphore:
            started_at = datetime.now()
            with summary_log_path.open("a", encoding="utf-8") as log_file:
                with log_lock:
                    log_file.write(f"Task run id: {task_run_id}\n")
                    log_file.write(f"Starting {Path(script_path).name}: {started_at.isoformat()}\n")
                    log_file.flush()
                process = subprocess.Popen(
                    ["python", script_path],
                    cwd=working_dir,
                    stderr=subprocess.STDOUT,
                    text=True,
                    stdout=subprocess.PIPE,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    with log_lock:
                        log_file.write(line)
                        log_file.flush()

                process.wait()
                finished_at = datetime.now()
                duration_seconds = (finished_at - started_at).total_seconds()
                with log_lock:
                    log_file.write(f"Finished {Path(script_path).name}: {finished_at.isoformat()}\n")
                    log_file.write(f"Duration {Path(script_path).name}: {format_duration(duration_seconds)}\n")
                    log_file.write(f"Exit code {Path(script_path).name}: {process.returncode}\n\n")
                    log_file.flush()
                with result_lock:
                    run_results.append((Path(script_path).name, duration_seconds))
                print(f"[{Path(script_path).name}] Exit code: {process.returncode}", flush=True)

    for script_path, working_dir in scripts:
        worker = threading.Thread(target=run_one, args=(script_path, working_dir), daemon=True)
        worker.start()
        threads.append(worker)

    for worker in threads:
        worker.join()

    with summary_log_path.open("a", encoding="utf-8") as summary_log:
        summary_log.write("Script durations:\n")
        for script_name, duration_seconds in run_results:
            summary_log.write(f"{script_name}: {format_duration(duration_seconds)}\n")


def get_date_directories() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [path for path in OUTPUT_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: path.name,
        reverse=True,
    )


def list_raw_grib_runs(date_str: str) -> list[dict[str, str | int]]:
    raw_dir = get_raw_grib_dir(date_str)
    if not raw_dir.exists():
        return []

    runs: list[dict[str, str | int]] = []
    for run_dir in sorted([path for path in raw_dir.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True):
        file_count = len(list(run_dir.glob("*.grib2")))
        runs.append(
            {
                "name": run_dir.name,
                "label": run_dir.name.upper(),
                "file_count": file_count,
            }
        )
    return runs


def get_run_files(date_str: str, run_name: str) -> list[Path]:
    run_dir = get_raw_grib_dir(date_str) / run_name
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("*.grib2"))


def summarize_grib_dataset(grib_path: Path) -> dict[str, object]:
    with xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
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
            "attributes": {key: str(value) for key, value in ds.attrs.items()},
        }


def get_raw_grib_dir(date_str: str) -> Path:
    return OUTPUT_DIR / date_str / "raw_grib"


def get_downloadable_dates(date_dirs: list[Path], requested_dates: list[str], include_all: bool) -> list[str]:
    available_dates = [path.name for path in date_dirs if get_raw_grib_dir(path.name).exists()]
    if include_all:
        return available_dates

    valid_dates = [date_str for date_str in requested_dates if date_str in available_dates]
    return valid_dates


def create_raw_grib_archive(date_strs: list[str]) -> BytesIO:
    if not date_strs:
        abort(404)

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for date_str in date_strs:
            raw_dir = get_raw_grib_dir(date_str)
            if not raw_dir.exists():
                continue

            for grib_path in sorted(raw_dir.rglob("*.grib2")):
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
    selected_dir = next((path for path in date_dirs if path.name == requested_date), None)
    if selected_dir is None and date_dirs:
        selected_dir = date_dirs[0]

    selected_date = selected_dir.name if selected_dir else None
    downloadable_dates = [path.name for path in date_dirs if get_raw_grib_dir(path.name).exists()]
    selected_download_dates = request.args.getlist("download_date")
    selected_download_dates = [date_str for date_str in selected_download_dates if date_str in downloadable_dates]
    if not selected_download_dates and selected_date in downloadable_dates:
        selected_download_dates = [selected_date]
    run_entries = list_raw_grib_runs(selected_date) if selected_date else []

    return render_template(
        "index.html",
        available_dates=[path.name for path in date_dirs],
        selected_date=selected_date,
        downloadable_dates=downloadable_dates,
        selected_download_dates=selected_download_dates,
        run_entries=run_entries,
        raw_grib_available=bool(selected_date and get_raw_grib_dir(selected_date).exists()),
    )


@app.route("/ai-learning")
def view_ai_learning() -> str:
    refresh = request.args.get("refresh") == "1"
    summary = None if refresh else load_learning_summary(DATA_DIR)
    if summary is None:
        summary = train_learning_summary(DATA_DIR, OUTPUT_DIR)

    return render_template("ai_learning.html", summary=summary)


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
    requested_dates = request.args.getlist("date")
    include_all = request.args.get("all") == "1"
    date_strs = get_downloadable_dates(date_dirs, requested_dates, include_all)
    archive_buffer = create_raw_grib_archive(date_strs)
    if include_all:
        download_name = "all_dates_raw_grib.zip"
    elif len(date_strs) == 1:
        download_name = f"{date_strs[0]}_raw_grib.zip"
    else:
        download_name = f"selected_dates_{len(date_strs)}_raw_grib.zip"

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
    run_files = get_run_files(date_str, run_name)
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
        run_name=run_name,
        selected_file=selected_file.name,
        file_count=len(run_files),
        summary=summary,
        load_error=load_error,
    )



@app.route("/run-task1")
def run_task1():
    task_run_id = build_task_run_id()
    scripts = [
        resolve_script_path(
            "/opt/render/project/src/hrrr_grib_2m_sfc_archive.py",
            "hrrr_grib_2m_sfc_archive.py",
        ),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, task_run_id, 1), daemon=True).start()
    return f"Task started in background as {task_run_id}.", 200




if __name__ == "__main__":
    app.run(debug=True)