from pathlib import Path
import subprocess
import zipfile
from typing import Any

import pandas as pd
from kaggle import KaggleApi

SUPPORTED_PRIMARY_METRICS = {
    "root mean squared error": "rmse",
    "root mean squared logarithmic error": "rmsle",
    "mean absolute error": "mae",
    "area under the roc curve": "roc_auc",
    "log loss": "log_loss",
    "accuracy": "accuracy",
}


def normalize_primary_metric(metric_name: str) -> str | None:
    normalized_name = metric_name.strip().lower()
    if normalized_name in SUPPORTED_PRIMARY_METRICS.values():
        return normalized_name
    return SUPPORTED_PRIMARY_METRICS.get(normalized_name)


def is_metric_valid_for_task(task_type: str, primary_metric: str) -> bool:
    regression_metrics = {"rmse", "rmsle", "mae"}
    binary_metrics = {"roc_auc", "log_loss", "accuracy"}
    if task_type == "regression":
        return primary_metric in regression_metrics
    if task_type == "binary":
        return primary_metric in binary_metrics
    return False


def infer_task_type_from_tags(tags: list[Any]) -> str | None:
    task_tags: list[str] = []
    for tag in tags:
        full_path = str(getattr(tag, "full_path", "")).strip().lower()
        if full_path.startswith("task > "):
            task_tags.append(full_path)

    if "task > regression" in task_tags:
        return "regression"
    if "task > binary classification" in task_tags:
        return "binary"
    return None


def fetch_competition_metadata(competition_slug: str) -> dict[str, Any]:
    api = KaggleApi()
    api.authenticate()
    response = api.competitions_list(search=competition_slug, page_size=20)
    competitions = response.competitions if response else []
    slug_suffix = "/" + competition_slug

    exact_match = None
    for competition in competitions:
        url = str(getattr(competition, "url", "")).strip()
        if url.endswith(slug_suffix):
            exact_match = competition
            break

    if exact_match is None:
        raise ValueError(f"Could not find exact competition metadata match for slug: {competition_slug}")

    return {
        "evaluation_metric": str(getattr(exact_match, "evaluation_metric", "")).strip(),
        "tags": list(getattr(exact_match, "tags", [])),
    }


def resolve_task_type_and_primary_metric(
    competition_slug: str,
    configured_task_type: str | None,
    configured_primary_metric: str | None,
) -> tuple[str, str, str]:
    if configured_task_type and configured_primary_metric:
        normalized_primary_metric = normalize_primary_metric(configured_primary_metric)
        if normalized_primary_metric is None:
            raise ValueError(
                "Configured primary_metric is not supported. "
                f"Supported values: {sorted(set(SUPPORTED_PRIMARY_METRICS.values()))}"
            )
        if not is_metric_valid_for_task(configured_task_type, normalized_primary_metric):
            raise ValueError(
                f"Configured primary_metric '{normalized_primary_metric}' is not valid for task_type '{configured_task_type}'."
            )
        return configured_task_type, normalized_primary_metric, "config"

    metadata = fetch_competition_metadata(competition_slug)
    inferred_task_type = infer_task_type_from_tags(metadata["tags"])
    inferred_primary_metric = normalize_primary_metric(metadata["evaluation_metric"])

    final_task_type = configured_task_type or inferred_task_type
    final_primary_metric = (
        normalize_primary_metric(configured_primary_metric)
        if configured_primary_metric
        else inferred_primary_metric
    )

    if configured_primary_metric and final_primary_metric is None:
        raise ValueError(
            "Configured primary_metric is not supported. "
            f"Supported values: {sorted(set(SUPPORTED_PRIMARY_METRICS.values()))}"
        )
    if final_task_type is None and final_primary_metric is None:
        raise ValueError(
            "Could not infer task_type and primary_metric from Kaggle metadata. "
            "Set both values explicitly in config.yaml."
        )
    if final_task_type is None or final_primary_metric is None:
        raise ValueError(
            "Partial competition metadata inference is not allowed. "
            "Set both task_type and primary_metric explicitly in config.yaml."
        )
    if not is_metric_valid_for_task(final_task_type, final_primary_metric):
        raise ValueError(
            f"Resolved primary_metric '{final_primary_metric}' is not valid for task_type '{final_task_type}'."
        )

    if configured_task_type or configured_primary_metric:
        return final_task_type, final_primary_metric, "mixed"
    return final_task_type, final_primary_metric, "kaggle"


def fetch_competition_data(competition_slug: str) -> Path:
    target_dir = Path("data") / competition_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.glob("*.zip")):
        return target_dir

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(target_dir),
        ],
        check=True,
    )
    return target_dir


def find_competition_zip(competition_slug: str) -> Path:
    data_dir = Path("data") / competition_slug
    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        raise ValueError(f"No competition zip found in {data_dir}")
    return zip_files[0]


def read_csv_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise ValueError(f"Missing {member_name} in {zip_path}")
        with archive.open(member_name) as f:
            return pd.read_csv(f)


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    candidate_columns = [col for col in train_df.columns if col not in test_df.columns]
    if len(candidate_columns) != 1:
        raise ValueError(f"Could not infer a single target column. Candidates: {candidate_columns}")
    return candidate_columns[0]
