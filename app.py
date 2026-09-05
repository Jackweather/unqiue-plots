from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import subprocess
import threading
from zipfile import ZIP_DEFLATED, ZipFile

from flask import Flask, abort, render_template, request, send_file, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/var/data")
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = DATA_DIR / "logs"
RENDER_BASE_DIR = Path("/opt/render/project/src/NY")

app = Flask(__name__)


def resolve_script_path(render_script: str, local_script: str) -> tuple[str, str]:
    render_path = Path(render_script)
    if render_path.exists():
        return str(render_path), str(render_path.parent)

    local_path = BASE_DIR / local_script
    return str(local_path), str(local_path.parent)


def run_scripts(scripts: list[tuple[str, str]], max_parallel: int = 1) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = threading.Semaphore(max_parallel)
    threads: list[threading.Thread] = []

    def run_one(script_path: str, working_dir: str) -> None:
        log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = LOG_DIR / f"{Path(script_path).stem}_{log_stamp}.log"
        with semaphore:
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    ["python", script_path],
                    cwd=working_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                log_file.write(f"\nExit code: {process.returncode}\n")

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


def get_plot_entries(date_dir: Path) -> list[dict[str, str]]:
    plot_dir = date_dir / "plots"
    if not plot_dir.exists():
        return []

    entries: list[dict[str, str]] = []
    for plot_path in sorted(plot_dir.glob("*.png")):
        label = plot_path.stem.replace("hrrr_", "").replace("_temp_grid_", " ").replace("_", " ").title()
        entries.append(
            {
                "name": plot_path.name,
                "label": label,
                "url": f"/plots/{date_dir.name}/{plot_path.name}",
            }
        )
    return entries


def get_raw_grib_dir(date_str: str) -> Path:
    return OUTPUT_DIR / date_str / "raw_grib"


def create_raw_grib_archive(date_str: str) -> BytesIO:
    raw_dir = get_raw_grib_dir(date_str)
    if not raw_dir.exists():
        abort(404)

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
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
    plots = get_plot_entries(selected_dir) if selected_dir else []

    return render_template(
        "index.html",
        available_dates=[path.name for path in date_dirs],
        selected_date=selected_date,
        plots=plots,
        raw_grib_available=bool(selected_date and get_raw_grib_dir(selected_date).exists()),
    )


@app.route("/plots/<date_str>/<filename>")
def serve_plot(date_str: str, filename: str):
    plot_dir = OUTPUT_DIR / date_str / "plots"
    if not plot_dir.exists():
        abort(404)
    return send_from_directory(plot_dir, filename)


@app.route("/downloads/<date_str>/raw-grib.zip")
def download_raw_grib_archive(date_str: str):
    archive_buffer = create_raw_grib_archive(date_str)
    return send_file(
        archive_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{date_str}_raw_grib.zip",
    )


@app.route("/run-task1")
def run_task1():
    scripts = [
        resolve_script_path(
            "/opt/render/project/src/hrrr_dc_temp_grid.py",
            "hrrr_dc_temp_grid.py",
        ),
    ]
    threading.Thread(target=lambda: run_scripts(scripts, 1), daemon=True).start()
    return f"Task started in background! Check {LOG_DIR} for output.", 200


if __name__ == "__main__":
    app.run(debug=True)
