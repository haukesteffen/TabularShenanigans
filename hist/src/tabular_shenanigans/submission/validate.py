from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_submission_file(
    submission_path: Path,
    ids_path: Path,
    id_col: str,
    target_col: str,
    prediction_type: str,
    positive_label: Any,
    negative_label: Any,
) -> None:
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Expected ids file not found: {ids_path}")

    submission = pd.read_csv(submission_path)
    ids = pd.read_csv(ids_path)

    required_cols = {id_col, target_col}
    missing = [c for c in required_cols if c not in submission.columns]
    if missing:
        raise ValueError(f"Submission is missing required columns: {missing}")

    if id_col not in ids.columns:
        raise ValueError(f"IDs file is missing id column '{id_col}'")

    if len(submission) != len(ids):
        raise ValueError(
            f"Submission row count ({len(submission)}) != expected ids row count ({len(ids)})"
        )

    if submission[id_col].isna().any():
        raise ValueError("Submission contains null values in id column")
    if submission[target_col].isna().any():
        raise ValueError("Submission contains null values in target column")
    if submission[id_col].duplicated().any():
        raise ValueError("Submission contains duplicate ids")

    expected_ids = ids[id_col].tolist()
    submitted_ids = submission[id_col].tolist()
    if submitted_ids != expected_ids:
        raise ValueError(
            "Submission ids do not match expected ids/order from processed test set."
        )

    prediction_mode = prediction_type.lower()
    if prediction_mode == "label":
        allowed = {str(positive_label), str(negative_label)}
        observed = {str(v) for v in submission[target_col].dropna().unique()}
        unexpected = sorted(observed - allowed)
        if unexpected:
            raise ValueError(
                f"Submission contains labels outside allowed set {sorted(allowed)}: {unexpected}"
            )
    elif prediction_mode == "raw":
        numeric = pd.to_numeric(submission[target_col], errors="coerce")
        if numeric.isna().any():
            raise ValueError("Raw submission predictions must be numeric.")
    else:
        raise ValueError("Unsupported submission.prediction_type. Use one of: raw, label.")
