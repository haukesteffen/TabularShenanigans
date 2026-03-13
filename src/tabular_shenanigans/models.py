from dataclasses import dataclass
from collections.abc import Mapping
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

from tabular_shenanigans.runtime_execution import get_runtime_execution_context

ModelBuilder = Callable[[int, dict[str, object] | None], tuple[object, dict[str, object]]]
FitKwargsBuilder = Callable[[object, list[str], list[str]], dict[str, object]]
TuningSpaceBuilder = Callable[[object], dict[str, object]]


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    model_name: str
    builder: ModelBuilder
    fit_kwargs_builder: FitKwargsBuilder | None = None
    tuning_space_builder: TuningSpaceBuilder | None = None
    supports_native_categorical_preprocessing: bool = False
    supports_sparse_preprocessed_input: bool = False


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


def _merge_model_params(
    default_params: dict[str, object],
    parameter_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    merged_params = dict(default_params)
    if parameter_overrides:
        merged_params.update(parameter_overrides)
    return merged_params


def _resolve_booster_runtime_defaults(model_id: str) -> dict[str, object]:
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_compute_target != "gpu":
        return {}

    if model_id == "xgboost":
        return {"device": "cuda"}
    if model_id == "lightgbm":
        return {"device_type": "cuda"}
    if model_id == "catboost":
        return {"task_type": "GPU"}
    return {}


def _build_ridge(random_state: int, parameter_overrides: dict[str, object] | None = None) -> tuple[Ridge, dict[str, object]]:
    del random_state
    params = _merge_model_params({}, parameter_overrides)
    return Ridge(**params), params


def _build_elasticnet(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ElasticNet, dict[str, object]]:
    del random_state
    params = _merge_model_params({}, parameter_overrides)
    return ElasticNet(**params), params


def _build_random_forest_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[RandomForestRegressor, dict[str, object]]:
    params = _merge_model_params({"n_estimators": 500, "n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return RandomForestRegressor(**params), params


def _build_extra_trees_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ExtraTreesRegressor, dict[str, object]]:
    params = _merge_model_params({"n_estimators": 500, "n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return ExtraTreesRegressor(**params), params


def _build_hist_gradient_boosting_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[HistGradientBoostingRegressor, dict[str, object]]:
    params = _merge_model_params(
        {
            "early_stopping": False,
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": random_state,
        },
        parameter_overrides,
    )
    return HistGradientBoostingRegressor(**params), params


def _build_logreg(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    _validate_logreg_parameter_overrides(parameter_overrides)
    params = _merge_model_params({"solver": "saga", "max_iter": 1000}, parameter_overrides)
    estimator = LogisticRegression(**params)
    if get_runtime_execution_context().resolved_compute_target == "gpu":
        return BinaryLabelEncodingClassifier(estimator), params
    return estimator, params


def _build_random_forest_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[RandomForestClassifier, dict[str, object]]:
    params = _merge_model_params({"n_estimators": 500, "n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return RandomForestClassifier(**params), params


def _build_extra_trees_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ExtraTreesClassifier, dict[str, object]]:
    params = _merge_model_params({"n_estimators": 500, "n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return ExtraTreesClassifier(**params), params


def _build_hist_gradient_boosting_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    params = _merge_model_params(
        {
            "early_stopping": False,
            "learning_rate": 0.05,
            "max_iter": 300,
            "random_state": random_state,
        },
        parameter_overrides,
    )
    return HistGradientBoostingClassifier(**params), params


def _build_lightgbm_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("lightgbm"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return LGBMRegressor(**params), params


def _build_lightgbm_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("lightgbm"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return LGBMClassifier(**params), params


def _build_catboost_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("catboost"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return CatBoostRegressor(**params), params


def _build_catboost_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("catboost"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return CatBoostClassifier(**params), params


def _build_xgboost_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("xgboost"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return XGBRegressor(**params), params


def _build_xgboost_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
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
        **_resolve_booster_runtime_defaults("xgboost"),
    }
    params = _merge_model_params(params, parameter_overrides)
    return BinaryLabelEncodingClassifier(XGBClassifier(**params)), params


def _build_random_forest_tuning_space(trial: object) -> dict[str, object]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
    }


def _build_extra_trees_tuning_space(trial: object) -> dict[str, object]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
    }


def _build_hist_gradient_boosting_tuning_space(trial: object) -> dict[str, object]:
    return {
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-10, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
    }


def _build_logreg_tuning_space(trial: object) -> dict[str, object]:
    return {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05),
        "max_iter": 1000,
        "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
    }


def _is_numeric_logreg_param(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate_logreg_parameter_overrides(parameter_overrides: Mapping[str, object] | None) -> None:
    if not parameter_overrides:
        return

    if "penalty" in parameter_overrides:
        raise ValueError(
            "Logistic regression model_params no longer accept 'penalty'. "
            "Use l1_ratio only: 0.0 for L2, 1.0 for L1, and values between 0 and 1 for elastic net."
        )

    if "solver" in parameter_overrides:
        raise ValueError(
            "Logistic regression uses solver='saga' only. Remove model_params.solver from config.yaml."
        )

    if "n_jobs" in parameter_overrides:
        raise ValueError(
            "Logistic regression model_params do not support n_jobs in this runtime. "
            "Remove model_params.n_jobs from config.yaml."
        )

    if "l1_ratio" in parameter_overrides:
        l1_ratio = parameter_overrides["l1_ratio"]
        if not _is_numeric_logreg_param(l1_ratio):
            raise ValueError("Logistic regression model_params.l1_ratio must be numeric.")
        l1_ratio_value = float(l1_ratio)
        if not np.isfinite(l1_ratio_value) or not 0.0 <= l1_ratio_value <= 1.0:
            raise ValueError(
                "Logistic regression model_params.l1_ratio must be finite and within [0, 1]."
            )

    if "C" in parameter_overrides:
        c_value = parameter_overrides["C"]
        if not _is_numeric_logreg_param(c_value):
            raise ValueError("Logistic regression model_params.C must be numeric.")
        c_value_float = float(c_value)
        if not np.isfinite(c_value_float) or c_value_float <= 0.0:
            raise ValueError("Logistic regression model_params.C must be finite and > 0.")

    if "tol" in parameter_overrides:
        tol_value = parameter_overrides["tol"]
        if not _is_numeric_logreg_param(tol_value):
            raise ValueError("Logistic regression model_params.tol must be numeric.")
        tol_value_float = float(tol_value)
        if not np.isfinite(tol_value_float) or tol_value_float <= 0.0:
            raise ValueError("Logistic regression model_params.tol must be finite and > 0.")

    if "max_iter" in parameter_overrides:
        max_iter = parameter_overrides["max_iter"]
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise ValueError("Logistic regression model_params.max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("Logistic regression model_params.max_iter must be > 0.")

    if "class_weight" not in parameter_overrides:
        return

    class_weight = parameter_overrides["class_weight"]
    if class_weight is None or isinstance(class_weight, Mapping):
        return
    if isinstance(class_weight, str) and class_weight == "balanced":
        return
    raise ValueError(
        "Logistic regression model_params.class_weight must be 'balanced', null, or a mapping."
    )


def validate_model_parameter_overrides(
    task_type: str,
    model_id: str,
    parameter_overrides: Mapping[str, object] | None,
) -> None:
    if not parameter_overrides:
        return

    resolved_model_id = resolve_model_id(task_type, model_id)
    if resolved_model_id == "logistic_regression":
        _validate_logreg_parameter_overrides(parameter_overrides)


def _build_lightgbm_tuning_space(trial: object) -> dict[str, object]:
    return {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _build_xgboost_tuning_space(trial: object) -> dict[str, object]:
    return {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-10, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _build_catboost_tuning_space(trial: object) -> dict[str, object]:
    return {
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "depth": trial.suggest_int("depth", 4, 10),
        "iterations": trial.suggest_int("iterations", 200, 1200, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-10, 10.0, log=True),
    }


def _build_catboost_fit_kwargs(
    x_train_processed: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, object]:
    del numeric_columns
    if not categorical_columns:
        return {}
    if not isinstance(x_train_processed, pd.DataFrame):
        raise ValueError("CatBoost native preprocessing must produce a pandas DataFrame.")
    cat_feature_indices = [x_train_processed.columns.get_loc(column) for column in categorical_columns]
    return {"cat_features": cat_feature_indices}


DEFAULT_MODEL_ID_BY_TASK = {
    "regression": "elasticnet",
    "binary": "logistic_regression",
}

MODEL_REGISTRY: dict[str, dict[str, ModelDefinition]] = {
    "regression": {
        "ridge": ModelDefinition(
            model_id="ridge",
            model_name="Ridge",
            builder=_build_ridge,
            supports_sparse_preprocessed_input=True,
        ),
        "elasticnet": ModelDefinition(
            model_id="elasticnet",
            model_name="ElasticNet",
            builder=_build_elasticnet,
            supports_sparse_preprocessed_input=True,
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestRegressor",
            builder=_build_random_forest_regressor,
            tuning_space_builder=_build_random_forest_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "extra_trees": ModelDefinition(
            model_id="extra_trees",
            model_name="ExtraTreesRegressor",
            builder=_build_extra_trees_regressor,
            tuning_space_builder=_build_extra_trees_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "hist_gradient_boosting": ModelDefinition(
            model_id="hist_gradient_boosting",
            model_name="HistGradientBoostingRegressor",
            builder=_build_hist_gradient_boosting_regressor,
            tuning_space_builder=_build_hist_gradient_boosting_tuning_space,
        ),
        "lightgbm": ModelDefinition(
            model_id="lightgbm",
            model_name="LGBMRegressor",
            builder=_build_lightgbm_regressor,
            tuning_space_builder=_build_lightgbm_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "catboost": ModelDefinition(
            model_id="catboost",
            model_name="CatBoostRegressor",
            builder=_build_catboost_regressor,
            fit_kwargs_builder=_build_catboost_fit_kwargs,
            tuning_space_builder=_build_catboost_tuning_space,
            supports_native_categorical_preprocessing=True,
            supports_sparse_preprocessed_input=True,
        ),
        "xgboost": ModelDefinition(
            model_id="xgboost",
            model_name="XGBRegressor",
            builder=_build_xgboost_regressor,
            tuning_space_builder=_build_xgboost_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
    },
    "binary": {
        "logistic_regression": ModelDefinition(
            model_id="logistic_regression",
            model_name="LogisticRegression",
            builder=_build_logreg,
            tuning_space_builder=_build_logreg_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestClassifier",
            builder=_build_random_forest_classifier,
            tuning_space_builder=_build_random_forest_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "extra_trees": ModelDefinition(
            model_id="extra_trees",
            model_name="ExtraTreesClassifier",
            builder=_build_extra_trees_classifier,
            tuning_space_builder=_build_extra_trees_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "hist_gradient_boosting": ModelDefinition(
            model_id="hist_gradient_boosting",
            model_name="HistGradientBoostingClassifier",
            builder=_build_hist_gradient_boosting_classifier,
            tuning_space_builder=_build_hist_gradient_boosting_tuning_space,
        ),
        "lightgbm": ModelDefinition(
            model_id="lightgbm",
            model_name="LGBMClassifier",
            builder=_build_lightgbm_classifier,
            tuning_space_builder=_build_lightgbm_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
        "catboost": ModelDefinition(
            model_id="catboost",
            model_name="CatBoostClassifier",
            builder=_build_catboost_classifier,
            fit_kwargs_builder=_build_catboost_fit_kwargs,
            tuning_space_builder=_build_catboost_tuning_space,
            supports_native_categorical_preprocessing=True,
            supports_sparse_preprocessed_input=True,
        ),
        "xgboost": ModelDefinition(
            model_id="xgboost",
            model_name="XGBClassifier",
            builder=_build_xgboost_classifier,
            tuning_space_builder=_build_xgboost_tuning_space,
            supports_sparse_preprocessed_input=True,
        ),
    },
}


def resolve_candidate_model_id(
    task_type: str,
    model_family: str,
) -> str:
    task_registry = _get_task_model_registry(task_type)
    if model_family in task_registry:
        return model_family

    supported_model_families = sorted(task_registry)
    raise ValueError(
        f"Candidate model_family '{model_family}' is not valid for task_type '{task_type}'. "
        f"Supported model families: {supported_model_families}"
    )


def _get_task_model_registry(task_type: str) -> dict[str, ModelDefinition]:
    try:
        return MODEL_REGISTRY[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_default_model_id(task_type: str) -> str:
    try:
        return DEFAULT_MODEL_ID_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_supported_model_ids(task_type: str) -> list[str]:
    return sorted(_get_task_model_registry(task_type))


def get_tunable_model_ids(task_type: str) -> list[str]:
    task_registry = _get_task_model_registry(task_type)
    return sorted(
        model_id
        for model_id, model_definition in task_registry.items()
        if model_definition.tuning_space_builder is not None
    )


def validate_model_preprocessing_compatibility(
    task_type: str,
    model_id: str,
    categorical_preprocessor_id: str,
) -> None:
    model_definition = get_model_definition(task_type, model_id)
    if categorical_preprocessor_id != "native":
        return
    if model_definition.supports_native_categorical_preprocessing:
        return
    raise ValueError(
        f"Model family '{model_definition.model_id}' does not support "
        "categorical_preprocessor='native'. Use model_family='catboost' for native categorical handling."
    )


def resolve_model_matrix_output_kind(
    task_type: str,
    model_id: str,
    categorical_preprocessor_id: str,
) -> str:
    model_definition = get_model_definition(task_type, model_id)
    if categorical_preprocessor_id == "native":
        return "native_frame"
    if categorical_preprocessor_id == "onehot" and model_definition.supports_sparse_preprocessed_input:
        return "sparse_csr"
    return "dense_array"


def resolve_model_id(task_type: str, model_id: str) -> str:
    task_registry = _get_task_model_registry(task_type)
    if model_id in task_registry:
        return model_id

    supported_model_ids = get_supported_model_ids(task_type)
    raise ValueError(
        f"Model id '{model_id}' is not valid for task_type '{task_type}'. "
        f"Use canonical model_ids only. Supported model_ids: {supported_model_ids}"
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


def is_model_tunable(task_type: str, model_id: str) -> bool:
    model_definition = get_model_definition(task_type, model_id)
    return model_definition.tuning_space_builder is not None


def build_tuning_space(task_type: str, model_id: str, trial: object) -> dict[str, object]:
    model_definition = get_model_definition(task_type, model_id)
    if model_definition.tuning_space_builder is None:
        supported_model_ids = get_tunable_model_ids(task_type)
        raise ValueError(
            f"Model id '{model_definition.model_id}' does not support tuning for task_type '{task_type}'. "
            f"Supported tunable model_ids: {supported_model_ids}"
        )
    return model_definition.tuning_space_builder(trial)


def build_model(
    task_type: str,
    model_id: str,
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ModelDefinition, object, dict[str, object]]:
    model_definition = get_model_definition(task_type, model_id)
    estimator, explicit_params = model_definition.builder(random_state, parameter_overrides)
    return model_definition, estimator, explicit_params


def build_model_fit_kwargs(
    model_definition: ModelDefinition,
    x_train_processed: object,
    numeric_columns: list[str],
    categorical_columns: list[str],
    uses_native_categorical_preprocessing: bool,
) -> dict[str, object]:
    if model_definition.fit_kwargs_builder is None or not uses_native_categorical_preprocessing:
        return {}
    return model_definition.fit_kwargs_builder(x_train_processed, numeric_columns, categorical_columns)
