import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tabular_shenanigans.data import find_competition_zip, read_csv_from_zip


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


def prepare_submission_file(competition_slug: str, run_dir: Path) -> Path:
    prediction_path = run_dir / "test_predictions.csv"
    if not prediction_path.exists():
        raise ValueError(f"Missing test predictions file: {prediction_path}")

    prediction_df = pd.read_csv(prediction_path)
    zip_path = find_competition_zip(competition_slug)
    sample_submission_df = read_csv_from_zip(zip_path, "sample_submission.csv")

    expected_columns = sample_submission_df.columns.tolist()
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
) -> tuple[Path, str]:
    submission_path = prepare_submission_file(competition_slug=competition_slug, run_dir=run_dir)
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
    ledger_path = Path("artifacts") / competition_slug / "train" / "submissions.csv"
    _append_submission_ledger(ledger_path=ledger_path, row=ledger_row)

    return submission_path, status
