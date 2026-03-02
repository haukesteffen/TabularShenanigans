from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def metric_direction(metric_name: str) -> str:
    metric = metric_name.lower()
    if metric in {"rmse", "mae", "logloss"}:
        return "minimize"
    return "maximize"


def resolve_metric(runtime_cfg: dict[str, Any], task_type: str) -> tuple[str, str]:
    eval_cfg = runtime_cfg.get("evaluation", {})
    configured_metric = eval_cfg.get("metric")

    if configured_metric:
        metric = str(configured_metric).lower()
    else:
        metric = "accuracy" if task_type == "classification" else "rmse"

    direction = str(eval_cfg.get("direction") or metric_direction(metric)).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("evaluation.direction must be 'maximize' or 'minimize'.")
    return metric, direction


def validate_metric_compatibility(metric_name: str, task_type: str, n_classes: int | None) -> None:
    metric = metric_name.lower()
    if task_type == "classification":
        if metric in {"f1", "roc_auc", "logloss"} and n_classes != 2:
            raise ValueError(
                f"evaluation.metric='{metric}' currently supports binary classification only; "
                f"found n_classes={n_classes}. Use accuracy or implement multiclass metric handling."
            )
        return

    if task_type == "regression":
        if metric in {"accuracy", "f1", "roc_auc"}:
            raise ValueError(
                f"evaluation.metric='{metric}' is incompatible with regression task."
            )
        return

    raise ValueError("task_type must be 'classification' or 'regression'.")


def _binary_labels_from_scores(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_score >= threshold).astype(int)


def compute_metric(
    metric_name: str,
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    metric = metric_name.lower()
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if task_type == "classification":
        if metric == "accuracy":
            if y_pred_arr.dtype.kind in {"f", "c"}:
                labels = _binary_labels_from_scores(y_pred_arr)
            else:
                labels = y_pred_arr
            return float(accuracy_score(y_true_arr, labels))

        if metric == "f1":
            if y_pred_arr.dtype.kind in {"f", "c"}:
                labels = _binary_labels_from_scores(y_pred_arr)
            else:
                labels = y_pred_arr
            return float(f1_score(y_true_arr, labels, average="binary"))

        if metric == "roc_auc":
            return float(roc_auc_score(y_true_arr, y_pred_arr))

        if metric == "logloss":
            eps = 1e-15
            probs = np.clip(y_pred_arr, eps, 1 - eps)
            return float(log_loss(y_true_arr, probs))

        raise ValueError(
            "Unsupported classification metric. Use one of: accuracy, f1, roc_auc, logloss."
        )

    if task_type == "regression":
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        if metric == "mae":
            return float(mean_absolute_error(y_true_arr, y_pred_arr))
        raise ValueError("Unsupported regression metric. Use one of: rmse, mae.")

    raise ValueError("task_type must be 'classification' or 'regression'.")


def available_metrics(task_type: str, n_classes: int | None, has_proba: bool) -> list[str]:
    if task_type == "regression":
        return ["rmse", "mae"]

    if task_type == "classification":
        if n_classes == 2:
            metrics = ["accuracy", "f1"]
            if has_proba:
                metrics.extend(["roc_auc", "logloss"])
            return metrics
        return ["accuracy"]

    raise ValueError("task_type must be 'classification' or 'regression'.")


def compute_metrics_bundle(
    task_type: str,
    n_classes: int | None,
    y_true: np.ndarray,
    y_pred_labels: np.ndarray | None,
    y_pred_scores: np.ndarray | None,
) -> dict[str, float]:
    has_proba = y_pred_scores is not None
    metrics = available_metrics(task_type, n_classes, has_proba=has_proba)
    out: dict[str, float] = {}
    for metric in metrics:
        if metric in {"roc_auc", "logloss"}:
            if y_pred_scores is None:
                continue
            out[metric] = compute_metric(metric, task_type, y_true, y_pred_scores)
        elif metric in {"accuracy", "f1"} and task_type == "classification":
            if y_pred_labels is None:
                if y_pred_scores is None:
                    continue
                labels = (np.asarray(y_pred_scores) >= 0.5).astype(int)
            else:
                labels = np.asarray(y_pred_labels)
            out[metric] = compute_metric(metric, task_type, y_true, labels)
        else:
            # Regression metrics use the prediction values directly.
            preds = y_pred_scores if y_pred_scores is not None else y_pred_labels
            if preds is None:
                continue
            out[metric] = compute_metric(metric, task_type, y_true, np.asarray(preds))
    return out
