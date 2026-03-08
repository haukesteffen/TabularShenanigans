from dataclasses import dataclass
from pathlib import Path
import subprocess
import zipfile

import pandas as pd

SUPPORTED_PRIMARY_METRICS = {
    "root mean squared error": "rmse",
    "mean squared error": "mse",
    "root mean squared logarithmic error": "rmsle",
    "mean absolute error": "mae",
    "area under the roc curve": "roc_auc",
    "roc auc score": "roc_auc",
    "log loss": "log_loss",
    "accuracy": "accuracy",
}


def normalize_primary_metric(metric_name: str) -> str | None:
    normalized_name = metric_name.strip().lower()
    if normalized_name in SUPPORTED_PRIMARY_METRICS.values():
        return normalized_name
    return SUPPORTED_PRIMARY_METRICS.get(normalized_name)


def is_metric_valid_for_task(task_type: str, primary_metric: str) -> bool:
    regression_metrics = {"rmse", "mse", "rmsle", "mae"}
    binary_metrics = {"roc_auc", "log_loss", "accuracy"}
    if task_type == "regression":
        return primary_metric in regression_metrics
    if task_type == "binary":
        return primary_metric in binary_metrics
    return False


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


def load_sample_submission_template(competition_slug: str) -> pd.DataFrame:
    zip_path = find_competition_zip(competition_slug)
    return read_csv_from_zip(zip_path, "sample_submission.csv")


def validate_sample_submission_schema(
    sample_submission_df: pd.DataFrame,
    id_column: str,
    label_column: str,
) -> None:
    expected_submission_columns = [id_column, label_column]
    sample_submission_columns = sample_submission_df.columns.tolist()
    if sample_submission_columns != expected_submission_columns:
        raise ValueError(
            "sample_submission.csv must match the resolved schema exactly. "
            f"Expected columns {expected_submission_columns}, got {sample_submission_columns}"
        )


@dataclass(frozen=True)
class CompetitionDatasetContext:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    sample_submission_df: pd.DataFrame
    id_column: str
    label_column: str


def resolve_id_and_label_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    configured_id_column: str | None = None,
    configured_label_column: str | None = None,
) -> tuple[str, str]:
    train_columns = train_df.columns.tolist()
    test_columns = test_df.columns.tolist()
    sample_submission_columns = sample_submission_df.columns.tolist()

    id_candidates = [
        column
        for column in train_columns
        if column in test_columns and column in sample_submission_columns
    ]
    label_candidates = [
        column
        for column in train_columns
        if column not in test_columns and column in sample_submission_columns
    ]

    if configured_id_column is not None:
        if configured_id_column not in id_candidates:
            raise ValueError(
                f"Configured id_column '{configured_id_column}' is invalid. "
                "Expected a column present in train.csv, test.csv, and sample_submission.csv. "
                f"Candidates: {id_candidates}"
            )
        id_column = configured_id_column
    else:
        if len(id_candidates) != 1:
            raise ValueError(
                "Could not infer a single id_column. "
                "Columns present in train.csv, test.csv, and sample_submission.csv: "
                f"{id_candidates}"
            )
        id_column = id_candidates[0]

    if configured_label_column is not None:
        if configured_label_column not in label_candidates:
            raise ValueError(
                f"Configured label_column '{configured_label_column}' is invalid. "
                "Expected a column present in train.csv and sample_submission.csv but not test.csv. "
                f"Candidates: {label_candidates}"
            )
        label_column = configured_label_column
    else:
        if len(label_candidates) != 1:
            raise ValueError(
                "Could not infer a single label_column. "
                "Columns present in train.csv and sample_submission.csv but not test.csv: "
                f"{label_candidates}"
            )
        label_column = label_candidates[0]

    validate_sample_submission_schema(
        sample_submission_df=sample_submission_df,
        id_column=id_column,
        label_column=label_column,
    )

    return id_column, label_column


def load_competition_dataset_context(
    competition_slug: str,
    configured_id_column: str | None = None,
    configured_label_column: str | None = None,
) -> CompetitionDatasetContext:
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    sample_submission_df = load_sample_submission_template(competition_slug)
    id_column, label_column = resolve_id_and_label_columns(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
        configured_id_column=configured_id_column,
        configured_label_column=configured_label_column,
    )
    return CompetitionDatasetContext(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
        id_column=id_column,
        label_column=label_column,
    )
