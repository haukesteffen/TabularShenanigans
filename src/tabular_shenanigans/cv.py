import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


def build_splitter(task_type: str, n_splits: int, shuffle: bool, random_state: int) -> KFold | StratifiedKFold:
    splitter_kwargs = {
        "n_splits": n_splits,
        "shuffle": shuffle,
    }
    if shuffle:
        splitter_kwargs["random_state"] = random_state

    if task_type == "regression":
        return KFold(**splitter_kwargs)
    if task_type == "binary":
        return StratifiedKFold(**splitter_kwargs)
    raise ValueError(f"Unsupported task_type for CV splitter: {task_type}")


def resolve_binary_labels(y_values: pd.Series) -> tuple[object, object]:
    unique_labels = pd.unique(y_values)
    if len(unique_labels) != 2:
        raise ValueError(f"Binary tasks require exactly two unique labels, got {len(unique_labels)}: {list(unique_labels)}")

    ordered_labels = sorted(unique_labels.tolist())
    negative_label = ordered_labels[0]
    positive_label = ordered_labels[1]
    return negative_label, positive_label


def score_predictions(
    task_type: str,
    primary_metric: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    positive_label: object | None = None,
) -> float:
    if task_type == "regression":
        y_true_array = y_true.to_numpy()
        y_pred_array = np.asarray(y_pred)
        if primary_metric == "rmse":
            return float(math.sqrt(mean_squared_error(y_true_array, y_pred_array)))
        if primary_metric == "mse":
            return float(mean_squared_error(y_true_array, y_pred_array))
        if primary_metric == "rmsle":
            clipped_predictions = np.clip(y_pred_array, a_min=0.0, a_max=None)
            return float(math.sqrt(mean_squared_log_error(y_true_array, clipped_predictions)))
        if primary_metric == "mae":
            return float(mean_absolute_error(y_true_array, y_pred_array))
        raise ValueError(f"Unsupported regression metric: {primary_metric}")

    if task_type == "binary":
        y_true_array = y_true.to_numpy()
        y_pred_array = np.asarray(y_pred)
        if positive_label is None:
            raise ValueError("Binary scoring requires positive_label.")
        negative_label, _ = resolve_binary_labels(y_true)
        if primary_metric == "roc_auc":
            y_true_binary = (y_true == positive_label).astype(int).to_numpy()
            return float(roc_auc_score(y_true_binary, y_pred_array))
        if primary_metric == "log_loss":
            y_true_binary = (y_true == positive_label).astype(int).to_numpy()
            return float(log_loss(y_true_binary, y_pred_array))
        if primary_metric == "accuracy":
            predicted_labels = np.where(y_pred_array >= 0.5, positive_label, negative_label)
            return float(accuracy_score(y_true_array, predicted_labels))
        raise ValueError(f"Unsupported binary metric: {primary_metric}")

    raise ValueError(f"Unsupported task_type for scoring: {task_type}")


def is_higher_better(primary_metric: str) -> bool:
    return primary_metric in {"roc_auc", "accuracy"}
