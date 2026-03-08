import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.data import find_competition_zip, read_csv_from_zip, resolve_id_and_label_columns


def _append_submission_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row])
    if ledger_path.exists():
        ledger_df.to_csv(ledger_path, mode="a", header=False, index=False)
        return
    ledger_df.to_csv(ledger_path, index=False)


def _load_run_metadata(run_dir: Path) -> tuple[str, str, str, float]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["run_id"])

    summary_path = run_dir / "cv_summary.csv"
    if not summary_path.exists():
        raise ValueError(f"Missing CV summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.shape[0] != 1:
        raise ValueError(f"Expected exactly one row in CV summary, got {summary_df.shape[0]}")

    model_name = str(summary_df.loc[0, "model_name"])
    metric_name = str(summary_df.loc[0, "metric_name"])
    metric_mean = float(summary_df.loc[0, "metric_mean"])
    return run_id, model_name, metric_name, metric_mean


def _load_run_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_submission_ids(
    prediction_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    id_column: str,
) -> None:
    prediction_ids = prediction_df[id_column]
    sample_ids = sample_submission_df[id_column]

    if prediction_ids.duplicated().any():
        duplicate_ids = prediction_ids[prediction_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "Submission ID column contains duplicate values in test_predictions.csv. "
            f"Duplicate IDs: {duplicate_ids[:10]}"
        )

    if sample_ids.duplicated().any():
        duplicate_ids = sample_ids[sample_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "sample_submission.csv ID column contains duplicate values. "
            f"Duplicate IDs: {duplicate_ids[:10]}"
        )

    prediction_id_list = prediction_ids.tolist()
    sample_id_list = sample_ids.tolist()
    if prediction_id_list == sample_id_list:
        return

    prediction_id_set = set(prediction_id_list)
    sample_id_set = set(sample_id_list)
    if prediction_id_set == sample_id_set:
        raise ValueError(
            "Submission ID order does not match sample_submission.csv. "
            "IDs contain the same values but appear in a different order."
        )

    missing_ids = sorted(sample_id_set - prediction_id_set)
    extra_ids = sorted(prediction_id_set - sample_id_set)
    raise ValueError(
        "Submission ID values do not match sample_submission.csv. "
        f"Missing IDs: {missing_ids[:10]}; extra IDs: {extra_ids[:10]}"
    )


def prepare_submission_file(
    competition_slug: str,
    run_dir: Path,
    id_column: str | None = None,
    label_column: str | None = None,
) -> Path:
    prediction_path = run_dir / "test_predictions.csv"
    if not prediction_path.exists():
        raise ValueError(f"Missing test predictions file: {prediction_path}")

    prediction_df = pd.read_csv(prediction_path)
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    sample_submission_df = read_csv_from_zip(zip_path, "sample_submission.csv")
    resolved_id_column, resolved_label_column = resolve_id_and_label_columns(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
        configured_id_column=id_column,
        configured_label_column=label_column,
    )

    expected_columns = [resolved_id_column, resolved_label_column]
    actual_columns = prediction_df.columns.tolist()

    if actual_columns != expected_columns:
        raise ValueError(
            "Submission columns do not match sample_submission.csv. "
            f"Expected {expected_columns}, got {actual_columns}"
        )
    if prediction_df.shape[0] != sample_submission_df.shape[0]:
        raise ValueError(
            "Submission row count does not match sample_submission.csv. "
            f"Expected {sample_submission_df.shape[0]}, got {prediction_df.shape[0]}"
        )
    manifest = _load_run_manifest(run_dir)
    task_type = manifest.get("task_type")
    if task_type is None:
        config_snapshot = manifest.get("config_snapshot", {})
        if isinstance(config_snapshot, dict):
            task_type = config_snapshot.get("task_type")
    if task_type == "binary":
        prediction_values = prediction_df[resolved_label_column]
        if not pd.api.types.is_numeric_dtype(prediction_values):
            raise ValueError("Binary submission predictions must be numeric probabilities.")
        if not prediction_values.map(pd.notna).all():
            raise ValueError("Binary submission predictions contain missing values.")
        if not np.isfinite(prediction_values.to_numpy(dtype=float)).all():
            raise ValueError("Binary submission predictions must be finite.")
        if ((prediction_values < 0.0) | (prediction_values > 1.0)).any():
            raise ValueError("Binary submission predictions must be within [0, 1].")
    _validate_submission_ids(
        prediction_df=prediction_df,
        sample_submission_df=sample_submission_df,
        id_column=resolved_id_column,
    )

    submission_path = run_dir / "submission.csv"
    prediction_df.to_csv(submission_path, index=False)
    return submission_path


def build_submission_message(run_dir: Path, submit_message_prefix: str | None = None) -> str:
    run_id, model_name, metric_name, metric_mean = _load_run_metadata(run_dir)
    message_parts = []
    if submit_message_prefix:
        message_parts.append(submit_message_prefix.strip())
    message_parts.append(f"run={run_id}")
    message_parts.append(f"model={model_name}")
    message_parts.append(f"{metric_name}={metric_mean:.6f}")
    return " | ".join(message_parts)


def run_submission(
    competition_slug: str,
    run_dir: Path,
    submit_enabled: bool,
    submit_message_prefix: str | None = None,
    id_column: str | None = None,
    label_column: str | None = None,
) -> tuple[Path, str]:
    submission_path = prepare_submission_file(
        competition_slug=competition_slug,
        run_dir=run_dir,
        id_column=id_column,
        label_column=label_column,
    )
    message = build_submission_message(run_dir=run_dir, submit_message_prefix=submit_message_prefix)
    run_id, model_name, metric_name, metric_mean = _load_run_metadata(run_dir)

    if submit_enabled:
        completed = subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                competition_slug,
                "-f",
                str(submission_path),
                "-m",
                message,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip())
        status = "submitted"
    else:
        print("Submission dry-run mode: validation complete, Kaggle submit skipped.")
        status = "prepared"

    ledger_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": competition_slug,
        "run_id": run_id,
        "submission_path": str(submission_path),
        "submit_enabled": submit_enabled,
        "status": status,
        "message": message,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_mean": metric_mean,
    }
    manifest = _load_run_manifest(run_dir)
    ledger_row["positive_label"] = manifest.get("positive_label")
    ledger_row["negative_label"] = manifest.get("negative_label")
    ledger_path = Path("artifacts") / competition_slug / "train" / "submissions.csv"
    _append_submission_ledger(ledger_path=ledger_path, row=ledger_row)

    return submission_path, status
