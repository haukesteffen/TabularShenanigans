from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from tabular_shenanigans.core.metrics import compute_metric, resolve_metric


def _load_column(path: Path, expected_col: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if expected_col not in df.columns:
        raise ValueError(f"File {path} must contain '{expected_col}' column")
    return df[expected_col]


def _task_type_from_cfg(runtime_cfg: dict[str, Any], y: pd.Series) -> str:
    task_type = str(runtime_cfg.get("schema", {}).get("task_type", "auto")).lower()
    if task_type in {"classification", "regression"}:
        return task_type
    if y.dtype == "object" or y.nunique(dropna=False) <= 20:
        return "classification"
    return "regression"


def build_stack_artifacts(
    runtime_cfg: dict[str, Any],
    artifacts_dir: Path,
    data_dir: Path,
    competition: str,
    run_ids: list[str],
    output_dir: Path,
    method: str = "linear",
    meta_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(run_ids) < 2:
        raise ValueError("Provide at least two --run-id values for stacking.")

    y_path = data_dir / competition / "processed" / "y_train.csv"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing {y_path}. Run prepare-data/train first.")

    target_col = str(
        runtime_cfg.get("schema", {}).get("target")
        or runtime_cfg.get("submission", {}).get("target_column", "target")
    )
    y_df = pd.read_csv(y_path)
    if target_col not in y_df.columns:
        raise ValueError(f"Expected target column '{target_col}' in {y_path}")
    y = y_df[target_col]

    oof_cols: dict[str, pd.Series] = {}
    pred_cols: dict[str, pd.Series] = {}
    for run_id in run_ids:
        run_dir = artifacts_dir / competition / "runs" / run_id
        oof_cols[run_id] = _load_column(run_dir / "oof_predictions.csv", "oof_prediction")
        pred_cols[run_id] = _load_column(run_dir / "predictions.csv", "prediction")

    oof_matrix = pd.DataFrame(oof_cols)
    pred_matrix = pd.DataFrame(pred_cols)

    if len(oof_matrix) != len(y):
        raise ValueError("OOF row count mismatch with y_train.")

    task_type = _task_type_from_cfg(runtime_cfg, y)
    method = method.lower()
    metric_name, metric_direction = resolve_metric(runtime_cfg, task_type)
    meta_params = meta_params or {}

    if method == "mean":
        stacked_oof = oof_matrix.mean(axis=1).to_numpy()
        stacked_pred = pred_matrix.mean(axis=1).to_numpy()
        model_info = {"method": "mean"}
    elif method == "linear":
        if task_type == "classification":
            c = float(meta_params.get("C", 1.0))
            meta_model = LogisticRegression(max_iter=2000, C=c)
            meta_model.fit(oof_matrix, y)
            stacked_oof = meta_model.predict_proba(oof_matrix)[:, 1]
            stacked_pred = meta_model.predict_proba(pred_matrix)[:, 1]
            model_info = {
                "method": "linear",
                "model": "logistic_regression",
                "params": {"C": c},
                "intercept": float(meta_model.intercept_[0]),
                "coefficients": {
                    col: float(coef)
                    for col, coef in zip(oof_matrix.columns.tolist(), meta_model.coef_[0].tolist())
                },
            }
        else:
            alpha = float(meta_params.get("alpha", 1.0))
            meta_model = Ridge(alpha=alpha)
            meta_model.fit(oof_matrix, y)
            stacked_oof = meta_model.predict(oof_matrix)
            stacked_pred = meta_model.predict(pred_matrix)
            model_info = {
                "method": "linear",
                "model": "ridge",
                "params": {"alpha": alpha},
                "intercept": float(meta_model.intercept_),
                "coefficients": {
                    col: float(coef)
                    for col, coef in zip(oof_matrix.columns.tolist(), meta_model.coef_.tolist())
                },
            }
    else:
        raise ValueError("Unsupported stack method. Use one of: mean, linear.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"oof_prediction": stacked_oof}).to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame({"prediction": stacked_pred}).to_csv(output_dir / "predictions.csv", index=False)
    oof_matrix.to_csv(output_dir / "stack_oof_matrix.csv", index=False)
    pred_matrix.to_csv(output_dir / "stack_test_matrix.csv", index=False)

    stack_cv_score = compute_metric(
        metric_name=metric_name,
        task_type=task_type,
        y_true=y.to_numpy(),
        y_pred=stacked_oof,
    )

    info = {
        "status": "ok",
        "stack_method": method,
        "base_run_ids": run_ids,
        "task_type": task_type,
        "metric": metric_name,
        "metric_direction": metric_direction,
        "cv_score": float(stack_cv_score),
        "meta": model_info,
    }
    (output_dir / "stack_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
