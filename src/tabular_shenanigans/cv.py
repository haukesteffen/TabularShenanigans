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


def score_predictions(task_type: str, primary_metric: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    if task_type == "regression":
        y_true_array = y_true.to_numpy()
        y_pred_array = np.asarray(y_pred)
        if primary_metric == "rmse":
            return float(math.sqrt(mean_squared_error(y_true_array, y_pred_array)))
        if primary_metric == "rmsle":
            clipped_predictions = np.clip(y_pred_array, a_min=0.0, a_max=None)
            return float(math.sqrt(mean_squared_log_error(y_true_array, clipped_predictions)))
        if primary_metric == "mae":
            return float(mean_absolute_error(y_true_array, y_pred_array))
        raise ValueError(f"Unsupported regression metric: {primary_metric}")

    if task_type == "binary":
        y_true_array = y_true.to_numpy()
        y_pred_array = np.asarray(y_pred)
        if primary_metric == "roc_auc":
            return float(roc_auc_score(y_true_array, y_pred_array))
        if primary_metric == "log_loss":
            return float(log_loss(y_true_array, y_pred_array))
        if primary_metric == "accuracy":
            labels = (y_pred_array >= 0.5).astype(int)
            return float(accuracy_score(y_true_array, labels))
        raise ValueError(f"Unsupported binary metric: {primary_metric}")

    raise ValueError(f"Unsupported task_type for scoring: {task_type}")


def is_higher_better(primary_metric: str) -> bool:
    return primary_metric in {"roc_auc", "accuracy"}
