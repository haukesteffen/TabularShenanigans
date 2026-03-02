from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def submit_to_kaggle(
    competition_slug: str,
    submission_path: Path,
    message: str,
) -> dict[str, str]:
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not in PATH. Install it with `uv pip install kaggle`."
        )

    cmd = [
        kaggle_bin,
        "competitions",
        "submit",
        "-c",
        competition_slug,
        "-f",
        str(submission_path),
        "-m",
        message,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
        raise RuntimeError(f"Kaggle submission failed: {stderr}") from exc

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": competition_slug,
        "submission_path": str(submission_path),
        "message": message,
        "stdout": result.stdout.strip(),
    }


def list_submissions(competition_slug: str) -> str:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not in PATH. Install it with `uv pip install kaggle`."
        )

    cmd = [kaggle_bin, "competitions", "submissions", "-c", competition_slug]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
        raise RuntimeError(f"Failed to list Kaggle submissions: {stderr}") from exc
    return result.stdout.strip()


def write_submission_meta(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
