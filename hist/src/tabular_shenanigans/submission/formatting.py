from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _convert_predictions(
    pred_values: pd.Series,
    prediction_type: str,
    threshold: float,
    positive_label: str | int | float,
    negative_label: str | int | float,
) -> pd.Series:
    if prediction_type == "raw":
        return pred_values

    if prediction_type == "label":
        numeric_preds = pd.to_numeric(pred_values, errors="coerce")
        if numeric_preds.isna().any():
            raise ValueError(
                "prediction_type='label' requires numeric predictions for thresholding."
            )
        return pd.Series(
            np.where(numeric_preds >= threshold, positive_label, negative_label),
            index=pred_values.index,
        )

    raise ValueError("Unsupported prediction_type. Use one of: raw, label.")


def make_submission(
    ids_path: Path,
    preds_path: Path,
    output_path: Path,
    id_col: str,
    target_col: str,
    prediction_type: str = "raw",
    classification_threshold: float = 0.5,
    positive_label: str | int | float = 1,
    negative_label: str | int | float = 0,
) -> Path:
    ids = pd.read_csv(ids_path)
    preds = pd.read_csv(preds_path)

    if id_col not in ids.columns:
        raise ValueError(f"Missing id column: {id_col}")
    pred_col = "prediction"
    if pred_col not in preds.columns:
        raise ValueError(f"Predictions must include '{pred_col}' column")

    if len(ids) != len(preds):
        raise ValueError("IDs and predictions row count mismatch")

    submission_values = _convert_predictions(
        pred_values=preds[pred_col],
        prediction_type=prediction_type,
        threshold=classification_threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    submission = pd.DataFrame({id_col: ids[id_col], target_col: submission_values})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path
