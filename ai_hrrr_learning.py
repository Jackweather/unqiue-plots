from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import re
from pathlib import Path

import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("/var/data")
LOCAL_OUTPUT_DIR = BASE_DIR / "output"

RUN_DIR_PATTERN = re.compile(r"^(?P<run_hour>\d{2})z$")
FILE_PATTERN = re.compile(r"wrfsfcf(?P<forecast_hour>\d{2})")


@dataclass(frozen=True)
class TrainingSample:
    date_str: str
    source_group: str
    run_hour: int
    forecast_hour: int
    valid_hour: int
    month: int
    day_of_year: int
    weekday: int
    variable_name: str
    file_path: str
    target_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the lightweight HRRR learning summary from archived GRIB files."
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory used for cached learning output. Defaults to /var/data.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory containing dated raw_grib folders. Defaults to /var/data/output, or local ./output if /var/data/output does not exist.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force retraining even if a cached summary exists.",
    )
    return parser.parse_args()


def resolve_output_dir(data_dir: Path, requested_output_dir: str | None) -> Path:
    if requested_output_dir:
        return Path(requested_output_dir)

    default_output_dir = data_dir / "output"
    if default_output_dir.exists():
        return default_output_dir
    return LOCAL_OUTPUT_DIR


def get_learning_cache_path(data_dir: Path) -> Path:
    return data_dir / "ml" / "hrrr_learning_summary.json"


def get_model_artifact_path(data_dir: Path, trained_at: datetime) -> Path:
    return data_dir / "ml" / f"hrrr_linear_model_{trained_at.strftime('%Y%m%d')}.json"


def get_latest_model_artifact_path(data_dir: Path) -> Path:
    return data_dir / "ml" / "hrrr_linear_model_latest.json"


def list_learning_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []

    return sorted(output_dir.glob("*/raw_grib*/*/*.grib2"))


def build_data_snapshot(output_dir: Path) -> dict[str, object]:
    files = list_learning_files(output_dir)
    if not files:
        return {
            "file_count": 0,
            "latest_file": None,
            "latest_mtime": None,
        }

    latest_file = max(files, key=lambda path: path.stat().st_mtime)
    return {
        "file_count": len(files),
        "latest_file": str(latest_file),
        "latest_mtime": latest_file.stat().st_mtime,
    }


def learning_summary_is_stale(data_dir: Path, output_dir: Path) -> bool:
    summary = load_learning_summary(data_dir)
    current_snapshot = build_data_snapshot(output_dir)
    if summary is None:
        return True

    cached_snapshot = summary.get("data_snapshot")
    if not isinstance(cached_snapshot, dict):
        return True

    return cached_snapshot != current_snapshot


def parse_training_metadata(grib_path: Path) -> tuple[str, str, int, int, datetime]:
    date_str = grib_path.parents[2].name
    source_group = grib_path.parents[1].name
    run_dir_name = grib_path.parent.name
    run_match = RUN_DIR_PATTERN.match(run_dir_name)
    file_match = FILE_PATTERN.search(grib_path.name)
    if run_match is None or file_match is None:
        raise ValueError(f"Unable to parse run or forecast hour from {grib_path}")

    run_hour = int(run_match.group("run_hour"))
    forecast_hour = int(file_match.group("forecast_hour"))
    run_time = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(hours=run_hour)
    valid_time = run_time + timedelta(hours=forecast_hour)
    return date_str, source_group, run_hour, forecast_hour, valid_time


def build_prediction_feature_row(
    date_str: str,
    source_group: str,
    run_hour: int,
    forecast_hour: int,
    variable_name: str,
    categories: list[str],
) -> np.ndarray:
    run_time = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(hours=run_hour)
    valid_time = run_time + timedelta(hours=forecast_hour)

    row = [
        1.0,
        float(run_hour),
        float(forecast_hour),
        float(valid_time.hour),
        float(valid_time.month),
        float(valid_time.timetuple().tm_yday),
        float(valid_time.weekday()),
    ]
    category_index = {name: position for position, name in enumerate(categories)}
    one_hot = [0.0] * len(categories)
    if source_group in category_index:
        one_hot[category_index[source_group]] = 1.0
    if variable_name in category_index:
        one_hot[category_index[variable_name]] = 1.0
    row.extend(one_hot)
    return np.asarray([row], dtype=np.float64)


def summarize_grib_target(grib_path: Path) -> tuple[str, float]:
    with xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as dataset:
        if not dataset.data_vars:
            raise ValueError(f"No data variables found in {grib_path}")

        variable_name = next(iter(dataset.data_vars))
        variable = dataset[variable_name].astype("float64")
        mean_value = float(variable.mean(skipna=True).item())
        if math.isnan(mean_value):
            raise ValueError(f"Mean value is NaN for {grib_path}")
        return variable_name, mean_value


def build_training_samples(output_dir: Path) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for grib_path in list_learning_files(output_dir):
        try:
            date_str, source_group, run_hour, forecast_hour, valid_time = parse_training_metadata(grib_path)
            variable_name, target_value = summarize_grib_target(grib_path)
        except Exception:
            continue

        samples.append(
            TrainingSample(
                date_str=date_str,
                source_group=source_group,
                run_hour=run_hour,
                forecast_hour=forecast_hour,
                valid_hour=valid_time.hour,
                month=valid_time.month,
                day_of_year=valid_time.timetuple().tm_yday,
                weekday=valid_time.weekday(),
                variable_name=variable_name,
                file_path=str(grib_path),
                target_value=target_value,
            )
        )
    return samples


def get_feature_categories(samples: list[TrainingSample]) -> list[str]:
    return sorted({sample.source_group for sample in samples} | {sample.variable_name for sample in samples})


def build_feature_matrix(
    samples: list[TrainingSample], categories: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not samples:
        return np.empty((0, 0)), np.empty((0,)), []

    if categories is None:
        categories = get_feature_categories(samples)
    category_index = {name: position for position, name in enumerate(categories)}

    rows: list[list[float]] = []
    targets: list[float] = []
    for sample in samples:
        row = [
            1.0,
            float(sample.run_hour),
            float(sample.forecast_hour),
            float(sample.valid_hour),
            float(sample.month),
            float(sample.day_of_year),
            float(sample.weekday),
        ]
        one_hot = [0.0] * len(categories)
        one_hot[category_index[sample.source_group]] = 1.0
        one_hot[category_index[sample.variable_name]] = 1.0
        row.extend(one_hot)
        rows.append(row)
        targets.append(sample.target_value)

    feature_names = [
        "bias",
        "run_hour",
        "forecast_hour",
        "valid_hour",
        "month",
        "day_of_year",
        "weekday",
    ] + [f"category:{name}" for name in categories]
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64), feature_names


def split_samples(samples: list[TrainingSample]) -> tuple[list[TrainingSample], list[TrainingSample]]:
    ordered = sorted(samples, key=lambda sample: (sample.date_str, sample.run_hour, sample.forecast_hour, sample.file_path))
    if len(ordered) < 4:
        return ordered, []

    split_index = max(1, int(len(ordered) * 0.8))
    split_index = min(split_index, len(ordered) - 1)
    return ordered[:split_index], ordered[split_index:]


def fit_linear_model(feature_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(feature_matrix, targets, rcond=None)
    return coefficients


def predict(coefficients: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.empty((0,), dtype=np.float64)
    return feature_matrix @ coefficients


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0}

    errors = predicted - actual
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
    }


def build_daily_trends(samples: list[TrainingSample], predictions: np.ndarray) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for sample, predicted in zip(samples, predictions):
        key = (sample.date_str, sample.source_group)
        bucket = grouped.setdefault(
            key,
            {
                "actual_total": 0.0,
                "predicted_total": 0.0,
                "count": 0.0,
                "forecast_total": 0.0,
            },
        )
        bucket["actual_total"] += sample.target_value
        bucket["predicted_total"] += float(predicted)
        bucket["forecast_total"] += sample.forecast_hour
        bucket["count"] += 1.0

    trends: list[dict[str, object]] = []
    for (date_str, source_group), bucket in sorted(grouped.items()):
        count = int(bucket["count"])
        actual_mean = bucket["actual_total"] / bucket["count"]
        predicted_mean = bucket["predicted_total"] / bucket["count"]
        trends.append(
            {
                "date": date_str,
                "source_group": source_group,
                "sample_count": count,
                "avg_forecast_hour": round(bucket["forecast_total"] / bucket["count"], 2),
                "actual_mean": round(actual_mean, 3),
                "predicted_mean": round(predicted_mean, 3),
                "mean_error": round(predicted_mean - actual_mean, 3),
            }
        )
    return trends


def build_recent_predictions(samples: list[TrainingSample], predictions: np.ndarray, limit: int = 18) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordered_pairs = sorted(
        zip(samples, predictions),
        key=lambda item: (item[0].date_str, item[0].run_hour, item[0].forecast_hour, item[0].file_path),
        reverse=True,
    )
    for sample, predicted in ordered_pairs[:limit]:
        rows.append(
            {
                "date": sample.date_str,
                "source_group": sample.source_group,
                "run_hour": sample.run_hour,
                "forecast_hour": sample.forecast_hour,
                "valid_hour": sample.valid_hour,
                "variable_name": sample.variable_name,
                "actual_value": round(sample.target_value, 3),
                "predicted_value": round(float(predicted), 3),
                "error": round(float(predicted) - sample.target_value, 3),
                "file_name": Path(sample.file_path).name,
            }
        )
    return rows


def build_model_artifact(
    samples: list[TrainingSample],
    coefficients: np.ndarray,
    categories: list[str],
    feature_names: list[str],
    trained_at: datetime,
    output_dir: Path,
) -> dict[str, object]:
    variable_names = sorted({sample.variable_name for sample in samples})
    source_groups = sorted({sample.source_group for sample in samples})
    available_dates = sorted({sample.date_str for sample in samples})
    return {
        "model_type": "linear_regression_least_squares",
        "trained_at": trained_at.isoformat(),
        "target_name": "mean_grib_value",
        "target_variables": variable_names,
        "source_groups": source_groups,
        "training_date_range": {
            "first_date": available_dates[0],
            "last_date": available_dates[-1],
        },
        "sample_count": len(samples),
        "output_dir": str(output_dir),
        "feature_names": feature_names,
        "categories": categories,
        "coefficients": [float(value) for value in coefficients.tolist()],
    }


def save_model_artifact(data_dir: Path, artifact: dict[str, object], trained_at: datetime) -> Path:
    artifact_path = get_model_artifact_path(data_dir, trained_at)
    latest_artifact_path = get_latest_model_artifact_path(data_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2)
    artifact_path.write_text(payload, encoding="utf-8")
    latest_artifact_path.write_text(payload, encoding="utf-8")
    return artifact_path


def load_model_artifact(data_dir: Path, model_path: str | None = None) -> dict[str, object]:
    artifact_path = Path(model_path) if model_path else get_latest_model_artifact_path(data_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def predict_from_model_artifact(
    artifact: dict[str, object],
    date_str: str,
    source_group: str,
    run_hour: int,
    forecast_hour: int,
    variable_name: str,
) -> float:
    categories = [str(value) for value in artifact["categories"]]
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    feature_row = build_prediction_feature_row(
        date_str=date_str,
        source_group=source_group,
        run_hour=run_hour,
        forecast_hour=forecast_hour,
        variable_name=variable_name,
        categories=categories,
    )
    return float(predict(coefficients, feature_row)[0])


def train_learning_summary(data_dir: Path, output_dir: Path) -> dict[str, object]:
    data_snapshot = build_data_snapshot(output_dir)
    samples = build_training_samples(output_dir)
    if not samples:
        return {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sample_count": 0,
            "train_count": 0,
            "test_count": 0,
            "feature_names": [],
            "metrics": {},
            "train_metrics": {},
            "daily_trends": [],
            "recent_predictions": [],
            "model_artifact": None,
            "data_snapshot": data_snapshot,
            "status": "No GRIB files were available under /var/data/output for training.",
        }

    trained_at = datetime.now(timezone.utc)
    train_samples, test_samples = split_samples(samples)
    categories = get_feature_categories(samples)
    train_x, train_y, feature_names = build_feature_matrix(train_samples, categories)
    coefficients = fit_linear_model(train_x, train_y)
    train_predictions = predict(coefficients, train_x)
    train_metrics = calculate_metrics(train_y, train_predictions)

    test_x, test_y, _ = build_feature_matrix(test_samples, categories)
    test_predictions = predict(coefficients, test_x)
    test_metrics = calculate_metrics(test_y, test_predictions)

    all_x, all_y, _ = build_feature_matrix(samples, categories)
    all_predictions = predict(coefficients, all_x)
    model_artifact = build_model_artifact(
        samples=samples,
        coefficients=coefficients,
        categories=categories,
        feature_names=feature_names,
        trained_at=trained_at,
        output_dir=output_dir,
    )
    model_artifact_path = save_model_artifact(data_dir, model_artifact, trained_at)

    summary = {
        "trained_at": trained_at.isoformat(),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "feature_names": feature_names,
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "daily_trends": build_daily_trends(samples, all_predictions),
        "recent_predictions": build_recent_predictions(samples, all_predictions),
        "data_snapshot": data_snapshot,
        "model_artifact": {
            "path": str(model_artifact_path),
            "model_type": model_artifact["model_type"],
            "target_variables": model_artifact["target_variables"],
            "source_groups": model_artifact["source_groups"],
        },
        "status": "trained",
    }

    cache_path = get_learning_cache_path(data_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_learning_summary(data_dir: Path) -> dict[str, object] | None:
    cache_path = get_learning_cache_path(data_dir)
    if not cache_path.exists():
        return None

    return json.loads(cache_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = resolve_output_dir(data_dir, args.output_dir)

    summary = None if args.refresh else load_learning_summary(data_dir)
    if summary is None:
        summary = train_learning_summary(data_dir, output_dir)

    print(f"Training status: {summary['status']}")
    print(f"Output directory: {output_dir}")
    print(f"Samples: {summary['sample_count']}")
    print(f"Train/Test: {summary['train_count']}/{summary['test_count']}")
    if summary["metrics"]:
        metrics = summary["metrics"]
        print(
            "Holdout metrics: "
            f"MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, Bias={metrics['bias']:.3f}"
        )


if __name__ == "__main__":
    main()
