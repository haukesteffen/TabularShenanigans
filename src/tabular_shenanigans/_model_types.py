from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

ModelBuilder = Callable[[int, dict[str, object] | None], tuple[object, dict[str, object]]]
FitKwargsBuilder = Callable[[object, list[str], list[str]], dict[str, object]]
TuningSpaceBuilder = Callable[[object], dict[str, object]]


@dataclass(frozen=True)
class GpuRoutingRule:
    gpu_backends: tuple[str, ...]
    requires_native_categorical: bool = False
    rejects_sparse: bool = False


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    model_name: str
    builder: ModelBuilder
    fit_kwargs_builder: FitKwargsBuilder | None = None
    tuning_space_builder: TuningSpaceBuilder | None = None
    supports_native_categorical_preprocessing: bool = False
    supports_sparse_preprocessed_input: bool = False
    supports_gpu_native_dense_onehot_input: bool = False
    gpu_routing_rules: tuple[GpuRoutingRule, ...] = ()
    is_cpu_only: bool = False


class BinaryLabelEncodingClassifier:
    def __init__(self, estimator: object) -> None:
        self.estimator = estimator
        self.classes_: np.ndarray | None = None
        self._class_to_encoded_value: dict[object, int] = {}

    def fit(self, x_train: object, y_train: pd.Series, **fit_kwargs: object) -> "BinaryLabelEncodingClassifier":
        observed_classes = pd.unique(y_train)
        if len(observed_classes) != 2:
            raise ValueError(
                "BinaryLabelEncodingClassifier requires exactly two unique labels. "
                f"Observed labels: {list(observed_classes)!r}"
            )
        self.classes_ = np.asarray(observed_classes, dtype=object)
        self._class_to_encoded_value = {
            class_value: encoded_value for encoded_value, class_value in enumerate(self.classes_)
        }
        encoded_labels = y_train.map(self._class_to_encoded_value).astype(int)
        self.estimator.fit(x_train, encoded_labels, **fit_kwargs)
        return self

    def predict(self, x_values: object) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("BinaryLabelEncodingClassifier must be fit before predict.")
        predicted_labels = self.estimator.predict(x_values)
        predicted_indices = np.asarray(predicted_labels, dtype=int)
        return self.classes_[predicted_indices]

    def predict_proba(self, x_values: object) -> np.ndarray:
        return self.estimator.predict_proba(x_values)


class SingleTargetRegressionAdapter:
    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def fit(self, x_train: object, y_train: pd.Series, **fit_kwargs: object) -> "SingleTargetRegressionAdapter":
        self.estimator.fit(x_train, y_train, **fit_kwargs)
        return self

    def predict(self, x_values: object) -> object:
        return normalize_single_target_regression_predictions(self.estimator.predict(x_values))

    def __getattr__(self, attribute_name: str) -> object:
        return getattr(self.estimator, attribute_name)


def merge_model_params(
    default_params: dict[str, object],
    parameter_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    merged_params = dict(default_params)
    if parameter_overrides:
        merged_params.update(parameter_overrides)
    return merged_params


def normalize_single_target_regression_predictions(predictions: object) -> object:
    prediction_ndim = getattr(predictions, "ndim", None)
    if prediction_ndim != 2:
        return predictions

    prediction_shape = getattr(predictions, "shape", None)
    if prediction_shape is None or len(prediction_shape) != 2:
        return predictions
    if prediction_shape[1] != 1:
        raise ValueError(
            "Single-target regression predictions must be 1-dimensional or single-column. "
            f"Observed shape: {prediction_shape!r}"
        )

    if hasattr(predictions, "iloc"):
        return predictions.iloc[:, 0]
    if hasattr(predictions, "reshape"):
        return predictions.reshape(-1)
    return predictions
