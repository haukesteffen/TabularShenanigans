from dataclasses import dataclass
from typing import Callable

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


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    model_name: str
    preprocessing_scheme_id: str
    builder: ModelBuilder


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
    },
}

MODEL_ID_ALIASES: dict[str, dict[str, str]] = {
    "regression": {
        "elasticnet": "onehot_elasticnet",
        "random_forest": "ordinal_randomforest",
    },
    "binary": {
        "logistic_regression": "onehot_logreg",
        "random_forest": "ordinal_randomforest",
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
