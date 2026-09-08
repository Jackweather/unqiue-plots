from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import subprocess
import sys
import threading
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, abort, render_template, request, send_file, send_from_directory

from location_catalog import get_state_options


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/data")
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
RENDER_BASE_DIR = Path("/opt/render/project/src/")

app = Flask(__name__)

RUN_META_PATTERN = re.compile(r"^(Task run id|Started at|Finished at|Duration seconds):\s*(.+)$")


def resolve_script_path(render_script: str, local_script: str) -> tuple[str, str]:
    render_path = Path(render_script)
    if render_path.exists():
        return str(render_path), str(render_path.parent)

    local_path = BASE_DIR / local_script
    return str(local_path), str(local_path.parent)


def run_scripts(scripts: list[tuple[str, str]], task_run_id: str, max_parallel: int = 1) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = threading.Semaphore(max_parallel)
    threads: list[threading.Thread] = []

    def run_one(script_path: str, working_dir: str) -> None:
        started_at = datetime.now()
        log_stamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        log_path = LOG_DIR / f"{task_run_id}_{Path(script_path).stem}_{log_stamp}.log"
        with semaphore:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"Task run id: {task_run_id}\n")
                log_file.write(f"Started at: {started_at.isoformat()}\n")
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
                    log_file.write(line)
                    log_file.flush()

                process.wait()
                finished_at = datetime.now()
                duration_seconds = (finished_at - started_at).total_seconds()
                log_file.write(f"Finished at: {finished_at.isoformat()}\n")
                log_file.write(f"Duration seconds: {duration_seconds:.2f}\n")
                log_file.write(f"\nExit code: {process.returncode}\n")
                log_file.flush()
                print(f"[{Path(script_path).name}] Exit code: {process.returncode}", flush=True)

    for script_path, working_dir in scripts:
        worker = threading.Thread(target=run_one, args=(script_path, working_dir), daemon=True)
        worker.start()
        threads.append(worker)

    for worker in threads:
        worker.join()


def get_date_directories() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [path for path in OUTPUT_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: path.name,
        reverse=True,
    )


def format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "Running"

    total_seconds = int(round(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_run_history() -> list[dict[str, str | float | int | None | list[dict[str, str | float | int | None]]]]:
    if not LOG_DIR.exists():
        return []

    history_by_task: dict[str, dict[str, str | float | int | None | list[dict[str, str | float | int | None]]]] = {}
    for log_path in sorted(LOG_DIR.glob("*.log"), reverse=True):
        task_run_id: str | None = None
        started_at: str | None = None
        finished_at: str | None = None
        duration_seconds: float | None = None
        exit_code: int | None = None

        with log_path.open("r", encoding="utf-8") as log_file:
            for raw_line in log_file:
                line = raw_line.strip()
                meta_match = RUN_META_PATTERN.match(line)
                if meta_match:
                    key, value = meta_match.groups()
                    if key == "Task run id":
                        task_run_id = value
                    elif key == "Started at":
                        started_at = value
                    elif key == "Finished at":
                        finished_at = value
                    elif key == "Duration seconds":
                        try:
                            duration_seconds = float(value)
                        except ValueError:
                            duration_seconds = None
                    continue

                if line.startswith("Exit code:"):
                    try:
                        exit_code = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        exit_code = None

        if task_run_id is None:
            task_run_id = log_path.stem.rsplit("_", 3)[0]

        script_stem_parts = log_path.stem.rsplit("_", 3)
        script_name = f"{script_stem_parts[-3]}.py" if len(script_stem_parts) >= 4 else f"{log_path.stem.split('_')[0]}.py"
        run_entry = {
            "script_name": script_name,
            "log_name": log_path.name,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "duration_label": format_duration(duration_seconds),
            "exit_code": exit_code,
        }

        if task_run_id not in history_by_task:
            history_by_task[task_run_id] = {
                "task_run_id": task_run_id,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_seconds or 0.0,
                "duration_label": format_duration(duration_seconds),
                "runs": [run_entry],
            }
            continue

        task_entry = history_by_task[task_run_id]
        task_runs = task_entry["runs"]
        assert isinstance(task_runs, list)
        task_runs.append(run_entry)

        first_started_at = task_entry.get("started_at")
        if first_started_at is None or (started_at is not None and started_at < first_started_at):
            task_entry["started_at"] = started_at

        last_finished_at = task_entry.get("finished_at")
        if finished_at is None or last_finished_at is None:
            task_entry["finished_at"] = finished_at if last_finished_at is not None else last_finished_at
        elif finished_at > last_finished_at:
            task_entry["finished_at"] = finished_at

        total_duration_seconds = task_entry.get("duration_seconds")
        if isinstance(total_duration_seconds, (int, float)) and duration_seconds is not None:
            task_entry["duration_seconds"] = float(total_duration_seconds) + duration_seconds
            task_entry["duration_label"] = format_duration(task_entry["duration_seconds"])

    history = list(history_by_task.values())
    history.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
    return history


STATE_OPTIONS = get_state_options()
STATE_LABELS = {str(option["key"]): str(option["label"]) for option in STATE_OPTIONS}
SUPPORTED_STATES = [str(option["key"]) for option in STATE_OPTIONS if bool(option["supported"])]


def get_plot_entries(date_dir: Path, state_key: str | None) -> list[dict[str, str]]:
    plot_root = date_dir / "plots"
    if not plot_root.exists() or not state_key:
        return []

    plot_dir = plot_root / state_key
    if not plot_dir.exists():
        return []

    entries: list[dict[str, str]] = []
    for plot_path in sorted(plot_dir.glob("*.png")):
        city_name = plot_path.stem.replace("hrrr_", "").replace("_temp_grid_", " ").replace("_", " ").title()
        entries.append(
            {
                "name": plot_path.name,
                "label": f"{city_name} | {STATE_LABELS.get(state_key, state_key.title())}",
                "url": f"/plots/{date_dir.name}/{state_key}/{plot_path.name}",
            }
        )
    return entries


def get_usa_trend_entries(date_dir: Path | None) -> list[dict[str, str]]:
    if date_dir is None:
        return []

    plot_dir = date_dir / "plots" / "usa_temp_trend"
    if not plot_dir.exists():
        return []

    entries: list[dict[str, str]] = []
    for plot_path in sorted(plot_dir.glob("*.png")):
        stem_parts = plot_path.stem.split("_")
        if len(stem_parts) >= 7:
            run_token = stem_parts[-2]
            forecast_token = stem_parts[-1]
            label = f"Run {run_token.upper()} | Forecast {forecast_token.upper()}"
        else:
            label = plot_path.stem.replace("_", " ").title()

        entries.append(
            {
                "name": plot_path.name,
                "label": label,
                "url": f"/usa-trend-plots/{date_dir.name}/{plot_path.name}",
            }
        )

    return entries


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
    requested_state = request.args.get("state")
    selected_dir = next((path for path in date_dirs if path.name == requested_date), None)
    if selected_dir is None and date_dirs:
        selected_dir = date_dirs[0]

    selected_date = selected_dir.name if selected_dir else None
    selected_state = requested_state if requested_state in STATE_LABELS else (SUPPORTED_STATES[0] if SUPPORTED_STATES else None)
    plots = get_plot_entries(selected_dir, selected_state) if selected_dir else []
    downloadable_dates = [path.name for path in date_dirs if get_raw_grib_dir(path.name).exists()]

    return render_template(
        "index.html",
        available_dates=[path.name for path in date_dirs],
        selected_date=selected_date,
        selected_state=selected_state,
        selected_state_label=STATE_LABELS.get(selected_state, "Selected State") if selected_state else None,
        state_options=STATE_OPTIONS,
        plots=plots,
        downloadable_dates=downloadable_dates,
        raw_grib_available=bool(selected_date and get_raw_grib_dir(selected_date).exists()),
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


@app.route("/usa-trends")
def usa_trends() -> str:
    date_dirs = get_date_directories()
    requested_date = request.args.get("date")
    selected_dir = next((path for path in date_dirs if path.name == requested_date), None)
    if selected_dir is None and date_dirs:
        selected_dir = date_dirs[0]

    selected_date = selected_dir.name if selected_dir else None
    trend_plots = get_usa_trend_entries(selected_dir)

    return render_template(
        "usa_trends.html",
        available_dates=[path.name for path in date_dirs],
        selected_date=selected_date,
        trend_plots=trend_plots,
    )


@app.route("/run-history")
def run_history() -> str:
    return render_template("run_history.html", run_history=get_run_history())


@app.route("/run-task1")
def run_task1():
    task_run_id = datetime.now().strftime("task1_%Y%m%d_%H%M%S_%f")
    scripts = [
        resolve_script_path(
            "/opt/render/project/src/hrrr_dc_temp_grid.py",
            "hrrr_dc_temp_grid.py",
        ),
        resolve_script_path(
            "/opt/render/project/src/hrrr_usa_temp_trend_map.py",
            "hrrr_usa_temp_trend_map.py",
        ),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, task_run_id, 1), daemon=True).start()
    return f"Task started in background as {task_run_id}. Check /run-history for timing details.", 200




if __name__ == "__main__":
    app.run(debug=True)
