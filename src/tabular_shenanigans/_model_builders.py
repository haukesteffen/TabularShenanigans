from collections.abc import Mapping

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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from tabular_shenanigans._model_types import (
    BinaryLabelEncodingClassifier,
    SingleTargetRegressionAdapter,
    merge_model_params,
)
from tabular_shenanigans.lightgbm_cuda_backend import RepositoryLightGbmEstimator
from tabular_shenanigans.runtime_execution import (
    NATIVE_GPU_BACKEND,
    get_runtime_execution_context,
)

GPU_NATIVE_RIDGE_SUPPORTED_PARAM_NAMES = frozenset({"alpha", "copy_X", "fit_intercept", "solver"})
GPU_NATIVE_RIDGE_SUPPORTED_SOLVERS = frozenset({"auto", "eig", "svd"})
GPU_NATIVE_ELASTICNET_SUPPORTED_PARAM_NAMES = frozenset(
    {"alpha", "fit_intercept", "l1_ratio", "max_iter", "selection", "solver", "tol"}
)
GPU_NATIVE_ELASTICNET_SUPPORTED_SOLVERS = frozenset({"cd", "qn"})
GPU_NATIVE_ELASTICNET_SUPPORTED_SELECTIONS = frozenset({"cyclic", "random"})
GPU_NATIVE_RANDOM_FOREST_SUPPORTED_PARAM_NAMES = frozenset(
    {
        "bootstrap",
        "criterion",
        "max_batch_size",
        "max_depth",
        "max_features",
        "max_leaf_nodes",
        "max_samples",
        "min_impurity_decrease",
        "min_samples_leaf",
        "min_samples_split",
        "n_bins",
        "n_estimators",
        "n_streams",
        "oob_score",
        "random_state",
    }
)
GPU_NATIVE_RANDOM_FOREST_MAX_FEATURE_KEYWORDS = frozenset({"auto", "log2", "sqrt"})
GPU_NATIVE_RANDOM_FOREST_CLASSIFIER_CRITERION_MAP = {
    "entropy": "entropy",
    "gini": "gini",
}
GPU_NATIVE_RANDOM_FOREST_REGRESSOR_CRITERION_MAP = {
    "mse": "mse",
    "poisson": "poisson",
    "squared_error": "mse",
}


def resolve_booster_runtime_defaults(model_id: str) -> dict[str, object]:
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_compute_target != "gpu":
        return {}

    if model_id == "xgboost":
        return {"device": "cuda"}
    if model_id == "lightgbm" and runtime_execution_context.resolved_gpu_backend == "gpu_native":
        return {"device_type": "cuda"}
    if model_id == "catboost":
        return {"task_type": "GPU"}
    return {}


def import_cuml_linear_model(
    model_class_name: str,
    *,
    model_label: str,
) -> type[object]:
    try:
        from cuml import linear_model as cuml_linear_model
    except ImportError as exc:
        raise ImportError(
            f"gpu_native {model_label} requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    model_class = getattr(cuml_linear_model, model_class_name, None)
    if model_class is None:
        raise ImportError(f"cuml.linear_model.{model_class_name} is unavailable in this environment.")
    return model_class


def import_cuml_ensemble_model(
    model_class_name: str,
    *,
    model_label: str,
) -> type[object]:
    try:
        from cuml import ensemble as cuml_ensemble
    except ImportError as exc:
        raise ImportError(
            f"gpu_native {model_label} requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    model_class = getattr(cuml_ensemble, model_class_name, None)
    if model_class is None:
        raise ImportError(f"cuml.ensemble.{model_class_name} is unavailable in this environment.")
    return model_class


def import_cuml_neighbors_model(
    model_class_name: str,
    *,
    model_label: str,
) -> type[object]:
    try:
        from cuml import neighbors as cuml_neighbors
    except ImportError as exc:
        raise ImportError(
            f"gpu_native {model_label} requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    model_class = getattr(cuml_neighbors, model_class_name, None)
    if model_class is None:
        raise ImportError(f"cuml.neighbors.{model_class_name} is unavailable in this environment.")
    return model_class


def import_cuml_svm_model(
    model_class_name: str,
    *,
    model_label: str,
) -> type[object]:
    try:
        from cuml import svm as cuml_svm
    except ImportError as exc:
        raise ImportError(
            f"gpu_native {model_label} requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    model_class = getattr(cuml_svm, model_class_name, None)
    if model_class is None:
        raise ImportError(f"cuml.svm.{model_class_name} is unavailable in this environment.")
    return model_class


def import_cuml_naive_bayes_model(
    model_class_name: str,
    *,
    model_label: str,
) -> type[object]:
    try:
        from cuml import naive_bayes as cuml_naive_bayes
    except ImportError as exc:
        raise ImportError(
            f"gpu_native {model_label} requires the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    model_class = getattr(cuml_naive_bayes, model_class_name, None)
    if model_class is None:
        raise ImportError(f"cuml.naive_bayes.{model_class_name} is unavailable in this environment.")
    return model_class


def is_numeric_scalar_param(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def resolve_supported_gpu_native_params(
    *,
    model_label: str,
    parameter_overrides: Mapping[str, object] | None,
    supported_param_names: frozenset[str],
) -> dict[str, object]:
    resolved_overrides = dict(parameter_overrides or {})
    unsupported_param_names = sorted(set(resolved_overrides) - supported_param_names)
    if unsupported_param_names:
        raise ValueError(
            f"gpu_native {model_label} only supports cuML-overlapping model_params {sorted(supported_param_names)}. "
            f"Unsupported params: {unsupported_param_names}"
        )
    return resolved_overrides


def build_gpu_native_ridge_params(
    parameter_overrides: Mapping[str, object] | None,
) -> dict[str, object]:
    resolved_overrides = resolve_supported_gpu_native_params(
        model_label="ridge",
        parameter_overrides=parameter_overrides,
        supported_param_names=GPU_NATIVE_RIDGE_SUPPORTED_PARAM_NAMES,
    )

    if "alpha" in resolved_overrides:
        alpha_value = resolved_overrides["alpha"]
        if not is_numeric_scalar_param(alpha_value):
            raise ValueError("gpu_native ridge model_params.alpha must be a numeric scalar.")
        alpha_float = float(alpha_value)
        if not np.isfinite(alpha_float) or alpha_float <= 0.0:
            raise ValueError("gpu_native ridge model_params.alpha must be finite and > 0.")

    if "fit_intercept" in resolved_overrides and not isinstance(resolved_overrides["fit_intercept"], bool):
        raise ValueError("gpu_native ridge model_params.fit_intercept must be a boolean.")

    if "copy_X" in resolved_overrides and not isinstance(resolved_overrides["copy_X"], bool):
        raise ValueError("gpu_native ridge model_params.copy_X must be a boolean.")

    if "solver" in resolved_overrides:
        solver = resolved_overrides["solver"]
        if not isinstance(solver, str) or solver not in GPU_NATIVE_RIDGE_SUPPORTED_SOLVERS:
            raise ValueError(
                "gpu_native ridge model_params.solver must be one of "
                f"{sorted(GPU_NATIVE_RIDGE_SUPPORTED_SOLVERS)}."
            )

    return merge_model_params({}, resolved_overrides)


def build_gpu_native_elasticnet_params(
    parameter_overrides: Mapping[str, object] | None,
) -> dict[str, object]:
    resolved_overrides = resolve_supported_gpu_native_params(
        model_label="elasticnet",
        parameter_overrides=parameter_overrides,
        supported_param_names=GPU_NATIVE_ELASTICNET_SUPPORTED_PARAM_NAMES,
    )

    if "alpha" in resolved_overrides:
        alpha_value = resolved_overrides["alpha"]
        if not is_numeric_scalar_param(alpha_value):
            raise ValueError("gpu_native elasticnet model_params.alpha must be a numeric scalar.")
        alpha_float = float(alpha_value)
        if not np.isfinite(alpha_float) or alpha_float < 0.0:
            raise ValueError("gpu_native elasticnet model_params.alpha must be finite and >= 0.")

    if "l1_ratio" in resolved_overrides:
        l1_ratio_value = resolved_overrides["l1_ratio"]
        if not is_numeric_scalar_param(l1_ratio_value):
            raise ValueError("gpu_native elasticnet model_params.l1_ratio must be numeric.")
        l1_ratio_float = float(l1_ratio_value)
        if not np.isfinite(l1_ratio_float) or not 0.0 <= l1_ratio_float <= 1.0:
            raise ValueError(
                "gpu_native elasticnet model_params.l1_ratio must be finite and within [0, 1]."
            )

    if "fit_intercept" in resolved_overrides and not isinstance(resolved_overrides["fit_intercept"], bool):
        raise ValueError("gpu_native elasticnet model_params.fit_intercept must be a boolean.")

    if "max_iter" in resolved_overrides:
        max_iter = resolved_overrides["max_iter"]
        if not isinstance(max_iter, int) or isinstance(max_iter, bool):
            raise ValueError("gpu_native elasticnet model_params.max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("gpu_native elasticnet model_params.max_iter must be > 0.")

    if "tol" in resolved_overrides:
        tol_value = resolved_overrides["tol"]
        if not is_numeric_scalar_param(tol_value):
            raise ValueError("gpu_native elasticnet model_params.tol must be numeric.")
        tol_float = float(tol_value)
        if not np.isfinite(tol_float) or tol_float <= 0.0:
            raise ValueError("gpu_native elasticnet model_params.tol must be finite and > 0.")

    if "solver" in resolved_overrides:
        solver = resolved_overrides["solver"]
        if not isinstance(solver, str) or solver not in GPU_NATIVE_ELASTICNET_SUPPORTED_SOLVERS:
            raise ValueError(
                "gpu_native elasticnet model_params.solver must be one of "
                f"{sorted(GPU_NATIVE_ELASTICNET_SUPPORTED_SOLVERS)}."
            )

    if "selection" in resolved_overrides:
        selection = resolved_overrides["selection"]
        if not isinstance(selection, str) or selection not in GPU_NATIVE_ELASTICNET_SUPPORTED_SELECTIONS:
            raise ValueError(
                "gpu_native elasticnet model_params.selection must be one of "
                f"{sorted(GPU_NATIVE_ELASTICNET_SUPPORTED_SELECTIONS)}."
            )

    return merge_model_params({}, resolved_overrides)


def normalize_gpu_native_random_forest_int_param(
    *,
    param_name: str,
    value: object,
    minimum_value: int,
) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"gpu_native random_forest model_params.{param_name} must be an integer.")
    int_value = int(value)
    if int_value < minimum_value:
        raise ValueError(
            f"gpu_native random_forest model_params.{param_name} must be >= {minimum_value}."
        )
    return int_value


def normalize_gpu_native_random_forest_bool_param(
    *,
    param_name: str,
    value: object,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"gpu_native random_forest model_params.{param_name} must be a boolean.")
    return value


def normalize_gpu_native_random_forest_non_negative_float_param(
    *,
    param_name: str,
    value: object,
) -> float:
    if not is_numeric_scalar_param(value):
        raise ValueError(f"gpu_native random_forest model_params.{param_name} must be numeric.")
    float_value = float(value)
    if not np.isfinite(float_value) or float_value < 0.0:
        raise ValueError(
            f"gpu_native random_forest model_params.{param_name} must be finite and >= 0."
        )
    return float_value


def normalize_gpu_native_random_forest_count_or_fraction_param(
    *,
    param_name: str,
    value: object,
    minimum_count: int,
) -> int | float:
    if isinstance(value, bool):
        raise ValueError(
            f"gpu_native random_forest model_params.{param_name} must be an integer count "
            "or a float fraction within (0, 1]."
        )

    if isinstance(value, (int, np.integer)):
        int_value = int(value)
        if int_value < minimum_count:
            raise ValueError(
                f"gpu_native random_forest model_params.{param_name} must be >= {minimum_count}."
            )
        return int_value

    if not is_numeric_scalar_param(value):
        raise ValueError(
            f"gpu_native random_forest model_params.{param_name} must be an integer count "
            "or a float fraction within (0, 1]."
        )

    float_value = float(value)
    if not np.isfinite(float_value):
        raise ValueError(
            f"gpu_native random_forest model_params.{param_name} must be finite."
        )
    if float_value.is_integer():
        int_value = int(float_value)
        if int_value < minimum_count:
            raise ValueError(
                f"gpu_native random_forest model_params.{param_name} must be >= {minimum_count}."
            )
        return int_value
    if 0.0 < float_value <= 1.0:
        return float_value
    raise ValueError(
        f"gpu_native random_forest model_params.{param_name} must be >= {minimum_count} "
        "when passed as a count or within (0, 1] when passed as a fraction."
    )


def normalize_gpu_native_random_forest_max_features(value: object) -> int | float | str:
    if isinstance(value, bool):
        raise ValueError(
            "gpu_native random_forest model_params.max_features must be an integer, a float "
            "within (0, 1], or one of ['auto', 'log2', 'sqrt']."
        )

    if isinstance(value, (int, np.integer)):
        int_value = int(value)
        if int_value <= 0:
            raise ValueError("gpu_native random_forest model_params.max_features must be > 0.")
        return int_value

    if is_numeric_scalar_param(value):
        float_value = float(value)
        if not np.isfinite(float_value):
            raise ValueError("gpu_native random_forest model_params.max_features must be finite.")
        if float_value.is_integer() and float_value > 0.0:
            return int(float_value)
        if 0.0 < float_value <= 1.0:
            return float_value
        raise ValueError(
            "gpu_native random_forest model_params.max_features must be > 0 when passed "
            "as a count or within (0, 1] when passed as a fraction."
        )

    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in GPU_NATIVE_RANDOM_FOREST_MAX_FEATURE_KEYWORDS:
            return normalized_value

    raise ValueError(
        "gpu_native random_forest model_params.max_features must be an integer, a float "
        "within (0, 1], or one of ['auto', 'log2', 'sqrt']."
    )


def normalize_gpu_native_random_forest_criterion(
    *,
    criterion_value: object,
    criterion_map: Mapping[str, str],
) -> str:
    if not isinstance(criterion_value, str):
        raise ValueError("gpu_native random_forest model_params.criterion must be a string.")

    normalized_value = criterion_value.strip().lower()
    normalized_criterion = criterion_map.get(normalized_value)
    if normalized_criterion is None:
        raise ValueError(
            "gpu_native random_forest model_params.criterion must be one of "
            f"{sorted(criterion_map)}."
        )
    return normalized_criterion


def build_gpu_native_random_forest_params(
    *,
    parameter_overrides: Mapping[str, object] | None,
    random_state: int,
    criterion_map: Mapping[str, str],
) -> dict[str, object]:
    raw_overrides = dict(parameter_overrides or {})
    if "n_jobs" in raw_overrides:
        raise ValueError(
            "gpu_native random_forest does not support model_params.n_jobs. "
            "Use model_params.n_streams or omit the override."
        )

    resolved_overrides = resolve_supported_gpu_native_params(
        model_label="random_forest",
        parameter_overrides=raw_overrides,
        supported_param_names=GPU_NATIVE_RANDOM_FOREST_SUPPORTED_PARAM_NAMES,
    )

    normalized_overrides: dict[str, object] = {}
    for param_name, param_value in resolved_overrides.items():
        if param_name == "criterion":
            normalized_overrides["split_criterion"] = normalize_gpu_native_random_forest_criterion(
                criterion_value=param_value,
                criterion_map=criterion_map,
            )
            continue
        if param_name == "max_leaf_nodes":
            normalized_overrides["max_leaves"] = normalize_gpu_native_random_forest_int_param(
                param_name="max_leaf_nodes",
                value=param_value,
                minimum_value=2,
            )
            continue
        if param_name == "max_features":
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_max_features(param_value)
            continue
        if param_name in {"bootstrap", "oob_score"}:
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_bool_param(
                param_name=param_name,
                value=param_value,
            )
            continue
        if param_name == "max_depth":
            if param_value is None:
                raise ValueError(
                    "gpu_native random_forest model_params.max_depth does not support null. "
                    "Set a positive integer or omit the override to use the cuML default depth."
                )
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_int_param(
                param_name=param_name,
                value=param_value,
                minimum_value=1,
            )
            continue
        if param_name in {"max_batch_size", "n_bins", "n_estimators", "n_streams", "random_state"}:
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_int_param(
                param_name=param_name,
                value=param_value,
                minimum_value=1,
            )
            continue
        if param_name == "min_samples_split":
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_count_or_fraction_param(
                param_name=param_name,
                value=param_value,
                minimum_count=2,
            )
            continue
        if param_name in {"max_samples", "min_samples_leaf"}:
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_count_or_fraction_param(
                param_name=param_name,
                value=param_value,
                minimum_count=1,
            )
            continue
        if param_name == "min_impurity_decrease":
            normalized_overrides[param_name] = normalize_gpu_native_random_forest_non_negative_float_param(
                param_name=param_name,
                value=param_value,
            )
            continue
        normalized_overrides[param_name] = param_value

    resolved_bootstrap = normalized_overrides.get("bootstrap", True)
    resolved_oob_score = normalized_overrides.get("oob_score", False)
    if resolved_oob_score and not resolved_bootstrap:
        raise ValueError(
            "gpu_native random_forest model_params.oob_score requires bootstrap=True."
        )
    if "max_samples" in normalized_overrides and not resolved_bootstrap:
        raise ValueError(
            "gpu_native random_forest model_params.max_samples requires bootstrap=True."
        )

    default_params = {
        "n_estimators": 500,
        "random_state": random_state,
    }
    return merge_model_params(default_params, normalized_overrides)


def build_ridge(random_state: int, parameter_overrides: dict[str, object] | None = None) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = build_gpu_native_ridge_params(parameter_overrides)
        estimator_class = import_cuml_linear_model("Ridge", model_label="ridge")
        return SingleTargetRegressionAdapter(estimator_class(**params)), params

    params = merge_model_params({}, parameter_overrides)
    return Ridge(**params), params


def build_elasticnet(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = build_gpu_native_elasticnet_params(parameter_overrides)
        estimator_class = import_cuml_linear_model("ElasticNet", model_label="elasticnet")
        return SingleTargetRegressionAdapter(estimator_class(**params)), params

    params = merge_model_params({}, parameter_overrides)
    return ElasticNet(**params), params


def build_random_forest_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == NATIVE_GPU_BACKEND:
        params = build_gpu_native_random_forest_params(
            parameter_overrides=parameter_overrides,
            random_state=random_state,
            criterion_map=GPU_NATIVE_RANDOM_FOREST_REGRESSOR_CRITERION_MAP,
        )
        estimator_class = import_cuml_ensemble_model(
            "RandomForestRegressor",
            model_label="random_forest",
        )
        return SingleTargetRegressionAdapter(estimator_class(**params)), params

    params = merge_model_params({"n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return RandomForestRegressor(**params), params


def build_extra_trees_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ExtraTreesRegressor, dict[str, object]]:
    params = merge_model_params({"n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return ExtraTreesRegressor(**params), params


def build_hist_gradient_boosting_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[HistGradientBoostingRegressor, dict[str, object]]:
    params = merge_model_params(
        {"random_state": random_state},
        parameter_overrides,
    )
    return HistGradientBoostingRegressor(**params), params


def build_logreg(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    validate_logreg_parameter_overrides(parameter_overrides)
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = build_gpu_native_logreg_params(parameter_overrides)
        try:
            from cuml.linear_model import LogisticRegression as CuMlLogisticRegression
        except ImportError as exc:
            raise ImportError(
                "gpu_native logistic regression requires the optional GPU dependencies. "
                "Install them with `uv sync --extra boosters --extra gpu`."
            ) from exc

        estimator = CuMlLogisticRegression(**params)
        return BinaryLabelEncodingClassifier(estimator), params

    params = merge_model_params({"solver": "saga", "max_iter": 1000}, parameter_overrides)
    estimator = LogisticRegression(**params)
    if runtime_execution_context.resolved_gpu_backend == "gpu_patch":
        return BinaryLabelEncodingClassifier(estimator), params
    return estimator, params


def build_gpu_native_logreg_params(
    parameter_overrides: dict[str, object] | None,
) -> dict[str, object]:
    resolved_overrides = dict(parameter_overrides or {})
    class_weight = resolved_overrides.get("class_weight")
    if class_weight not in (None,):
        raise ValueError(
            "gpu_native logistic regression currently does not support model_params.class_weight. "
            "Use null, force gpu_backend='patch', or force CPU execution."
        )
    resolved_overrides.pop("class_weight", None)

    l1_ratio = float(resolved_overrides.pop("l1_ratio", 0.0))
    if l1_ratio == 0.0:
        penalty = "l2"
    elif l1_ratio == 1.0:
        penalty = "l1"
    else:
        penalty = "elasticnet"

    default_params: dict[str, object] = {
        "penalty": penalty,
        "max_iter": 1000,
    }
    if penalty == "elasticnet":
        default_params["l1_ratio"] = l1_ratio
    params = merge_model_params(default_params, resolved_overrides)
    return params


def build_random_forest_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == NATIVE_GPU_BACKEND:
        params = build_gpu_native_random_forest_params(
            parameter_overrides=parameter_overrides,
            random_state=random_state,
            criterion_map=GPU_NATIVE_RANDOM_FOREST_CLASSIFIER_CRITERION_MAP,
        )
        estimator_class = import_cuml_ensemble_model(
            "RandomForestClassifier",
            model_label="random_forest",
        )
        return BinaryLabelEncodingClassifier(estimator_class(**params)), params

    params = merge_model_params({"n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return RandomForestClassifier(**params), params


def build_extra_trees_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[ExtraTreesClassifier, dict[str, object]]:
    params = merge_model_params({"n_jobs": -1, "random_state": random_state}, parameter_overrides)
    return ExtraTreesClassifier(**params), params


def build_hist_gradient_boosting_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[HistGradientBoostingClassifier, dict[str, object]]:
    params = merge_model_params(
        {"random_state": random_state},
        parameter_overrides,
    )
    return HistGradientBoostingClassifier(**params), params


def build_knn_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = merge_model_params({}, parameter_overrides)
        estimator_class = import_cuml_neighbors_model("KNeighborsRegressor", model_label="knn")
        return SingleTargetRegressionAdapter(estimator_class(**params)), params

    params = merge_model_params({}, parameter_overrides)
    return KNeighborsRegressor(**params), params


def build_knn_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = merge_model_params({}, parameter_overrides)
        estimator_class = import_cuml_neighbors_model("KNeighborsClassifier", model_label="knn")
        return BinaryLabelEncodingClassifier(estimator_class(**params)), params

    params = merge_model_params({}, parameter_overrides)
    estimator = KNeighborsClassifier(**params)
    if runtime_execution_context.resolved_gpu_backend == "gpu_patch":
        return BinaryLabelEncodingClassifier(estimator), params
    return estimator, params


def build_svm_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = merge_model_params({"kernel": "rbf"}, parameter_overrides)
        estimator_class = import_cuml_svm_model("SVR", model_label="svm")
        return SingleTargetRegressionAdapter(estimator_class(**params)), params

    params = merge_model_params({"kernel": "rbf"}, parameter_overrides)
    return SVR(**params), params


def build_svm_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = merge_model_params({"kernel": "rbf", "probability": True}, parameter_overrides)
        estimator_class = import_cuml_svm_model("SVC", model_label="svm")
        return BinaryLabelEncodingClassifier(estimator_class(**params)), params

    params = merge_model_params({"kernel": "rbf", "probability": True}, parameter_overrides)
    estimator = SVC(**params)
    if runtime_execution_context.resolved_gpu_backend == "gpu_patch":
        return BinaryLabelEncodingClassifier(estimator), params
    return estimator, params


def build_naive_bayes_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    del random_state
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        params = merge_model_params({}, parameter_overrides)
        estimator_class = import_cuml_naive_bayes_model("GaussianNB", model_label="naive_bayes")
        return BinaryLabelEncodingClassifier(estimator_class(**params)), params

    params = merge_model_params({}, parameter_overrides)
    estimator = GaussianNB(**params)
    if runtime_execution_context.resolved_gpu_backend == "gpu_patch":
        return BinaryLabelEncodingClassifier(estimator), params
    return estimator, params


def build_realmlp_regressor(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    try:
        from pytabkit import RealMLP_TD_Regressor
    except ImportError as exc:
        raise ImportError(
            "RealMLP support requires the optional neural dependencies. "
            "Install them with `uv sync --extra neural`."
        ) from exc

    runtime_execution_context = get_runtime_execution_context()
    params = merge_model_params(
        {
            "device": "cuda" if runtime_execution_context.resolved_compute_target == "gpu" else "cpu",
            "random_state": random_state,
        },
        parameter_overrides,
    )
    return RealMLP_TD_Regressor(**params), params


def build_realmlp_classifier(
    random_state: int,
    parameter_overrides: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    try:
        from pytabkit import RealMLP_TD_Classifier
    except ImportError as exc:
        raise ImportError(
            "RealMLP support requires the optional neural dependencies. "
            "Install them with `uv sync --extra neural`."
        ) from exc

    runtime_execution_context = get_runtime_execution_context()
    params = merge_model_params(
        {
            "device": "cuda" if runtime_execution_context.resolved_compute_target == "gpu" else "cpu",
            "random_state": random_state,
        },
        parameter_overrides,
    )
    return RealMLP_TD_Classifier(**params), params


def build_lightgbm_regressor(
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

    runtime_execution_context = get_runtime_execution_context()
    params = {
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": -1,
        **resolve_booster_runtime_defaults("lightgbm"),
    }
    params = merge_model_params(params, parameter_overrides)
    estimator = RepositoryLightGbmEstimator(
        LGBMRegressor(**params),
        requires_cuda_build=runtime_execution_context.resolved_gpu_backend == "gpu_native",
    )
    return estimator, params


def build_lightgbm_classifier(
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

    runtime_execution_context = get_runtime_execution_context()
    params = {
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": -1,
        **resolve_booster_runtime_defaults("lightgbm"),
    }
    params = merge_model_params(params, parameter_overrides)
    estimator = RepositoryLightGbmEstimator(
        LGBMClassifier(**params),
        requires_cuda_build=runtime_execution_context.resolved_gpu_backend == "gpu_native",
    )
    return estimator, params


def build_catboost_regressor(
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
        "loss_function": "RMSE",
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
        **resolve_booster_runtime_defaults("catboost"),
    }
    params = merge_model_params(params, parameter_overrides)
    return CatBoostRegressor(**params), params


def build_catboost_classifier(
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
        "loss_function": "Logloss",
        "random_seed": random_state,
        "thread_count": -1,
        "verbose": False,
        **resolve_booster_runtime_defaults("catboost"),
    }
    params = merge_model_params(params, parameter_overrides)
    return CatBoostClassifier(**params), params


def build_xgboost_regressor(
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
        "eval_metric": "rmse",
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "tree_method": "hist",
        **resolve_booster_runtime_defaults("xgboost"),
    }
    params = merge_model_params(params, parameter_overrides)
    return XGBRegressor(**params), params


def build_xgboost_classifier(
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
        "eval_metric": "logloss",
        "n_jobs": -1,
        "objective": "binary:logistic",
        "random_state": random_state,
        "tree_method": "hist",
        **resolve_booster_runtime_defaults("xgboost"),
    }
    params = merge_model_params(params, parameter_overrides)
    return BinaryLabelEncodingClassifier(XGBClassifier(**params)), params


def build_random_forest_tuning_space(trial: object) -> dict[str, object]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
    }


def build_extra_trees_tuning_space(trial: object) -> dict[str, object]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 24),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
    }


def build_hist_gradient_boosting_tuning_space(trial: object) -> dict[str, object]:
    return {
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-10, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
    }


def build_realmlp_tuning_space(trial: object) -> dict[str, object]:
    return {
        "n_epochs": trial.suggest_int("n_epochs", 50, 500, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_sizes": trial.suggest_categorical(
            "hidden_sizes",
            [[128], [256], [128, 128], [256, 128]],
        ),
        "p_drop": trial.suggest_float("p_drop", 0.0, 0.5),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
    }


def build_knn_tuning_space(trial: object) -> dict[str, object]:
    params: dict[str, object] = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    }
    if params["metric"] == "minkowski":
        params["p"] = trial.suggest_int("p", 1, 5)
    return params


def build_svm_regressor_tuning_space(trial: object) -> dict[str, object]:
    use_scale_gamma = trial.suggest_categorical("use_scale_gamma", [True, False])
    gamma = "scale" if use_scale_gamma else trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    return {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        "gamma": gamma,
        "kernel": "rbf",
    }


def build_svm_classifier_tuning_space(trial: object) -> dict[str, object]:
    use_scale_gamma = trial.suggest_categorical("use_scale_gamma", [True, False])
    gamma = "scale" if use_scale_gamma else trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    return {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "gamma": gamma,
        "kernel": "rbf",
        "probability": True,
    }


def build_naive_bayes_tuning_space(trial: object) -> dict[str, object]:
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-2, log=True),
    }


def build_logreg_tuning_space(trial: object) -> dict[str, object]:
    runtime_execution_context = get_runtime_execution_context()
    if runtime_execution_context.resolved_gpu_backend == "gpu_native":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05),
            "max_iter": 1000,
            "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
        }
    return {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.05),
        "max_iter": 1000,
        "tol": trial.suggest_float("tol", 1e-4, 1e-2, log=True),
    }


def is_numeric_logreg_param(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_logreg_parameter_overrides(parameter_overrides: Mapping[str, object] | None) -> None:
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
        if not is_numeric_logreg_param(l1_ratio):
            raise ValueError("Logistic regression model_params.l1_ratio must be numeric.")
        l1_ratio_value = float(l1_ratio)
        if not np.isfinite(l1_ratio_value) or not 0.0 <= l1_ratio_value <= 1.0:
            raise ValueError(
                "Logistic regression model_params.l1_ratio must be finite and within [0, 1]."
            )

    if "C" in parameter_overrides:
        c_value = parameter_overrides["C"]
        if not is_numeric_logreg_param(c_value):
            raise ValueError("Logistic regression model_params.C must be numeric.")
        c_value_float = float(c_value)
        if not np.isfinite(c_value_float) or c_value_float <= 0.0:
            raise ValueError("Logistic regression model_params.C must be finite and > 0.")

    if "tol" in parameter_overrides:
        tol_value = parameter_overrides["tol"]
        if not is_numeric_logreg_param(tol_value):
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

    if task_type == "binary" and model_id == "logistic_regression":
        validate_logreg_parameter_overrides(parameter_overrides)


def build_lightgbm_tuning_space(trial: object) -> dict[str, object]:
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


def build_xgboost_tuning_space(trial: object) -> dict[str, object]:
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


def build_catboost_tuning_space(trial: object) -> dict[str, object]:
    return {
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "depth": trial.suggest_int("depth", 4, 10),
        "iterations": trial.suggest_int("iterations", 200, 1200, step=100),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "random_strength": trial.suggest_float("random_strength", 1e-10, 10.0, log=True),
    }


def build_catboost_fit_kwargs(
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
