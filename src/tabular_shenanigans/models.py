from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge

ModelBuilder = Callable[[int], tuple[object, dict[str, object]]]
FitKwargsBuilder = Callable[[object, list[str], list[str]], dict[str, object]]


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    model_name: str
    preprocessing_scheme_id: str
    builder: ModelBuilder
    fit_kwargs_builder: FitKwargsBuilder | None = None


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


def _build_ridge(random_state: int) -> tuple[Ridge, dict[str, object]]:
    del random_state
    return Ridge(), {}


def _build_elasticnet(random_state: int) -> tuple[ElasticNet, dict[str, object]]:
    del random_state
    return ElasticNet(), {}


def _build_random_forest_regressor(random_state: int) -> tuple[RandomForestRegressor, dict[str, object]]:
    params = {"n_estimators": 500, "n_jobs": -1, "random_state": random_state}
    return RandomForestRegressor(**params), params


def _build_extra_trees_regressor(random_state: int) -> tuple[ExtraTreesRegressor, dict[str, object]]:
    params = {"n_estimators": 500, "n_jobs": -1, "random_state": random_state}
    return ExtraTreesRegressor(**params), params


def _build_hist_gradient_boosting_regressor(
    random_state: int,
) -> tuple[HistGradientBoostingRegressor, dict[str, object]]:
    params = {
        "early_stopping": False,
        "learning_rate": 0.05,
        "max_iter": 300,
        "random_state": random_state,
    }
    return HistGradientBoostingRegressor(**params), params


def _build_logreg(random_state: int) -> tuple[LogisticRegression, dict[str, object]]:
    del random_state
    params = {"max_iter": 1000}
    return LogisticRegression(**params), params


def _build_random_forest_classifier(random_state: int) -> tuple[RandomForestClassifier, dict[str, object]]:
    params = {"n_estimators": 500, "n_jobs": -1, "random_state": random_state}
    return RandomForestClassifier(**params), params


def _build_extra_trees_classifier(random_state: int) -> tuple[ExtraTreesClassifier, dict[str, object]]:
    params = {"n_estimators": 500, "n_jobs": -1, "random_state": random_state}
    return ExtraTreesClassifier(**params), params


def _build_hist_gradient_boosting_classifier(
    random_state: int,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    params = {
        "early_stopping": False,
        "learning_rate": 0.05,
        "max_iter": 300,
        "random_state": random_state,
    }
    return HistGradientBoostingClassifier(**params), params


def _build_lightgbm_regressor(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "n_jobs": -1,
        "random_state": random_state,
        "subsample": 0.8,
        "verbosity": -1,
    }
    return LGBMRegressor(**params), params


def _build_lightgbm_classifier(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "n_jobs": -1,
        "random_state": random_state,
        "subsample": 0.8,
        "verbosity": -1,
    }
    return LGBMClassifier(**params), params


def _build_catboost_regressor(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError(
            "CatBoost support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "allow_writing_files": False,
        "depth": 6,
        "iterations": 500,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
    }
    return CatBoostRegressor(**params), params


def _build_catboost_classifier(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "allow_writing_files": False,
        "depth": 6,
        "iterations": 500,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
    }
    return CatBoostClassifier(**params), params


def _build_xgboost_regressor(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "colsample_bytree": 0.8,
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 500,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "subsample": 0.8,
        "tree_method": "hist",
    }
    return XGBRegressor(**params), params


def _build_xgboost_classifier(random_state: int) -> tuple[object, dict[str, object]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    params = {
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 500,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "random_state": random_state,
        "subsample": 0.8,
        "tree_method": "hist",
    }
    return BinaryLabelEncodingClassifier(XGBClassifier(**params)), params


def _build_catboost_fit_kwargs(
    x_train_processed: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, object]:
    del numeric_columns
    if not isinstance(x_train_processed, pd.DataFrame):
        raise ValueError("CatBoost native preprocessing must produce a pandas DataFrame.")
    cat_feature_indices = [x_train_processed.columns.get_loc(column) for column in categorical_columns]
    return {"cat_features": cat_feature_indices}


DEFAULT_MODEL_ID_BY_TASK = {
    "regression": "onehot_elasticnet",
    "binary": "onehot_logreg",
}

MODEL_REGISTRY: dict[str, dict[str, ModelDefinition]] = {
    "regression": {
        "onehot_ridge": ModelDefinition(
            model_id="onehot_ridge",
            model_name="Ridge",
            preprocessing_scheme_id="onehot",
            builder=_build_ridge,
        ),
        "onehot_elasticnet": ModelDefinition(
            model_id="onehot_elasticnet",
            model_name="ElasticNet",
            preprocessing_scheme_id="onehot",
            builder=_build_elasticnet,
        ),
        "ordinal_randomforest": ModelDefinition(
            model_id="ordinal_randomforest",
            model_name="RandomForestRegressor",
            preprocessing_scheme_id="ordinal",
            builder=_build_random_forest_regressor,
        ),
        "ordinal_extratrees": ModelDefinition(
            model_id="ordinal_extratrees",
            model_name="ExtraTreesRegressor",
            preprocessing_scheme_id="ordinal",
            builder=_build_extra_trees_regressor,
        ),
        "ordinal_hgb": ModelDefinition(
            model_id="ordinal_hgb",
            model_name="HistGradientBoostingRegressor",
            preprocessing_scheme_id="ordinal",
            builder=_build_hist_gradient_boosting_regressor,
        ),
        "ordinal_lightgbm": ModelDefinition(
            model_id="ordinal_lightgbm",
            model_name="LGBMRegressor",
            preprocessing_scheme_id="ordinal",
            builder=_build_lightgbm_regressor,
        ),
        "native_catboost": ModelDefinition(
            model_id="native_catboost",
            model_name="CatBoostRegressor",
            preprocessing_scheme_id="native",
            builder=_build_catboost_regressor,
            fit_kwargs_builder=_build_catboost_fit_kwargs,
        ),
        "ordinal_xgboost": ModelDefinition(
            model_id="ordinal_xgboost",
            model_name="XGBRegressor",
            preprocessing_scheme_id="ordinal",
            builder=_build_xgboost_regressor,
        ),
    },
    "binary": {
        "onehot_logreg": ModelDefinition(
            model_id="onehot_logreg",
            model_name="LogisticRegression",
            preprocessing_scheme_id="onehot",
            builder=_build_logreg,
        ),
        "ordinal_randomforest": ModelDefinition(
            model_id="ordinal_randomforest",
            model_name="RandomForestClassifier",
            preprocessing_scheme_id="ordinal",
            builder=_build_random_forest_classifier,
        ),
        "ordinal_extratrees": ModelDefinition(
            model_id="ordinal_extratrees",
            model_name="ExtraTreesClassifier",
            preprocessing_scheme_id="ordinal",
            builder=_build_extra_trees_classifier,
        ),
        "ordinal_hgb": ModelDefinition(
            model_id="ordinal_hgb",
            model_name="HistGradientBoostingClassifier",
            preprocessing_scheme_id="ordinal",
            builder=_build_hist_gradient_boosting_classifier,
        ),
        "ordinal_lightgbm": ModelDefinition(
            model_id="ordinal_lightgbm",
            model_name="LGBMClassifier",
            preprocessing_scheme_id="ordinal",
            builder=_build_lightgbm_classifier,
        ),
        "native_catboost": ModelDefinition(
            model_id="native_catboost",
            model_name="CatBoostClassifier",
            preprocessing_scheme_id="native",
            builder=_build_catboost_classifier,
            fit_kwargs_builder=_build_catboost_fit_kwargs,
        ),
        "ordinal_xgboost": ModelDefinition(
            model_id="ordinal_xgboost",
            model_name="XGBClassifier",
            preprocessing_scheme_id="ordinal",
            builder=_build_xgboost_classifier,
        ),
    },
}

MODEL_ID_ALIASES: dict[str, dict[str, str]] = {
    "regression": {
        "catboost": "native_catboost",
        "catboost_native": "native_catboost",
        "elasticnet": "onehot_elasticnet",
        "lightgbm": "ordinal_lightgbm",
        "random_forest": "ordinal_randomforest",
        "xgb": "ordinal_xgboost",
        "xgboost": "ordinal_xgboost",
    },
    "binary": {
        "catboost": "native_catboost",
        "catboost_native": "native_catboost",
        "lightgbm": "ordinal_lightgbm",
        "logistic_regression": "onehot_logreg",
        "random_forest": "ordinal_randomforest",
        "xgb": "ordinal_xgboost",
        "xgboost": "ordinal_xgboost",
    },
}


def _get_task_model_registry(task_type: str) -> dict[str, ModelDefinition]:
    try:
        return MODEL_REGISTRY[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def _get_task_model_aliases(task_type: str) -> dict[str, str]:
    try:
        return MODEL_ID_ALIASES[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_default_model_id(task_type: str) -> str:
    try:
        return DEFAULT_MODEL_ID_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_supported_model_ids(task_type: str, include_aliases: bool = False) -> list[str]:
    supported_model_ids = sorted(_get_task_model_registry(task_type))
    if not include_aliases:
        return supported_model_ids
    return sorted(supported_model_ids + list(_get_task_model_aliases(task_type)))


def resolve_model_id(task_type: str, model_id: str) -> str:
    task_registry = _get_task_model_registry(task_type)
    if model_id in task_registry:
        return model_id

    task_aliases = _get_task_model_aliases(task_type)
    if model_id in task_aliases:
        return task_aliases[model_id]

    supported_model_ids = get_supported_model_ids(task_type, include_aliases=True)
    raise ValueError(
        f"Configured model_id '{model_id}' is not valid for task_type '{task_type}'. "
        f"Supported model_ids: {supported_model_ids}"
    )


def get_model_definition(task_type: str, model_id: str) -> ModelDefinition:
    resolved_model_id = resolve_model_id(task_type, model_id)
    return _get_task_model_registry(task_type)[resolved_model_id]


def is_model_id_valid_for_task(task_type: str, model_id: str) -> bool:
    try:
        resolve_model_id(task_type, model_id)
        return True
    except ValueError:
        return False


def build_model(
    task_type: str,
    model_id: str,
    random_state: int,
) -> tuple[ModelDefinition, object, dict[str, object]]:
    model_definition = get_model_definition(task_type, model_id)
    estimator, explicit_params = model_definition.builder(random_state)
    return model_definition, estimator, explicit_params


def build_model_fit_kwargs(
    model_definition: ModelDefinition,
    x_train_processed: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, object]:
    if model_definition.fit_kwargs_builder is None:
        return {}
    return model_definition.fit_kwargs_builder(x_train_processed, numeric_columns, categorical_columns)
