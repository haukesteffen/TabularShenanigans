from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression

ModelBuilder = Callable[[int], tuple[object, dict[str, object]]]


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    model_name: str
    builder: ModelBuilder


def _build_elasticnet(random_state: int) -> tuple[ElasticNet, dict[str, object]]:
    del random_state
    return ElasticNet(), {}


def _build_random_forest_regressor(random_state: int) -> tuple[RandomForestRegressor, dict[str, object]]:
    params = {"random_state": random_state}
    return RandomForestRegressor(**params), params


def _build_logistic_regression(random_state: int) -> tuple[LogisticRegression, dict[str, object]]:
    del random_state
    return LogisticRegression(), {}


def _build_random_forest_classifier(random_state: int) -> tuple[RandomForestClassifier, dict[str, object]]:
    params = {"random_state": random_state}
    return RandomForestClassifier(**params), params


DEFAULT_MODEL_ID_BY_TASK = {
    "regression": "elasticnet",
    "binary": "logistic_regression",
}

MODEL_REGISTRY: dict[str, dict[str, ModelDefinition]] = {
    "regression": {
        "elasticnet": ModelDefinition(
            model_id="elasticnet",
            model_name="ElasticNet",
            builder=_build_elasticnet,
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestRegressor",
            builder=_build_random_forest_regressor,
        ),
    },
    "binary": {
        "logistic_regression": ModelDefinition(
            model_id="logistic_regression",
            model_name="LogisticRegression",
            builder=_build_logistic_regression,
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestClassifier",
            builder=_build_random_forest_classifier,
        ),
    },
}


def get_default_model_id(task_type: str) -> str:
    try:
        return DEFAULT_MODEL_ID_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_supported_model_ids(task_type: str) -> list[str]:
    try:
        return sorted(MODEL_REGISTRY[task_type])
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def is_model_id_valid_for_task(task_type: str, model_id: str) -> bool:
    try:
        return model_id in MODEL_REGISTRY[task_type]
    except KeyError:
        return False


def build_model(
    task_type: str,
    model_id: str,
    random_state: int,
) -> tuple[str, str, object, dict[str, object]]:
    if not is_model_id_valid_for_task(task_type, model_id):
        supported_model_ids = get_supported_model_ids(task_type)
        raise ValueError(
            f"Configured model_id '{model_id}' is not valid for task_type '{task_type}'. "
            f"Supported model_ids: {supported_model_ids}"
        )

    model_definition = MODEL_REGISTRY[task_type][model_id]
    estimator, explicit_params = model_definition.builder(random_state)
    return model_definition.model_id, model_definition.model_name, estimator, explicit_params
