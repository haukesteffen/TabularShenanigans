from tabular_shenanigans._model_builders import (
    build_catboost_classifier,
    build_catboost_fit_kwargs,
    build_catboost_regressor,
    build_catboost_tuning_space,
    build_elasticnet,
    build_extra_trees_classifier,
    build_extra_trees_regressor,
    build_extra_trees_tuning_space,
    build_hist_gradient_boosting_classifier,
    build_hist_gradient_boosting_regressor,
    build_hist_gradient_boosting_tuning_space,
    build_knn_classifier,
    build_knn_regressor,
    build_knn_tuning_space,
    build_lightgbm_classifier,
    build_lightgbm_regressor,
    build_lightgbm_tuning_space,
    build_logreg,
    build_logreg_tuning_space,
    build_naive_bayes_classifier,
    build_naive_bayes_tuning_space,
    build_realmlp_classifier,
    build_realmlp_fit_kwargs,
    build_realmlp_regressor,
    build_realmlp_tuning_space,
    build_random_forest_classifier,
    build_random_forest_regressor,
    build_random_forest_tuning_space,
    build_ridge,
    build_svm_classifier,
    build_svm_classifier_tuning_space,
    build_svm_regressor,
    build_svm_regressor_tuning_space,
    build_xgboost_classifier,
    build_xgboost_regressor,
    build_xgboost_tuning_space,
)
from tabular_shenanigans._model_types import GpuRoutingRule, ModelDefinition
from tabular_shenanigans.representations.types import RepresentationContract
from tabular_shenanigans.runtime_execution import (
    NATIVE_GPU_BACKEND,
    PATCH_GPU_BACKEND,
    RuntimeExecutionContext,
)


DEFAULT_MODEL_ID_BY_TASK = {
    "regression": "elasticnet",
    "binary": "logistic_regression",
}

MODEL_REGISTRY: dict[str, dict[str, ModelDefinition]] = {
    "regression": {
        "ridge": ModelDefinition(
            model_id="ridge",
            model_name="Ridge",
            builder=build_ridge,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "elasticnet": ModelDefinition(
            model_id="elasticnet",
            model_name="ElasticNet",
            builder=build_elasticnet,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestRegressor",
            builder=build_random_forest_regressor,
            tuning_space_builder=build_random_forest_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), rejects_sparse=True),
            ),
        ),
        "extra_trees": ModelDefinition(
            model_id="extra_trees",
            model_name="ExtraTreesRegressor",
            builder=build_extra_trees_regressor,
            tuning_space_builder=build_extra_trees_tuning_space,
            supports_sparse_preprocessed_input=True,
            is_cpu_only=True,
        ),
        "hist_gradient_boosting": ModelDefinition(
            model_id="hist_gradient_boosting",
            model_name="HistGradientBoostingRegressor",
            builder=build_hist_gradient_boosting_regressor,
            tuning_space_builder=build_hist_gradient_boosting_tuning_space,
            is_cpu_only=True,
        ),
        "realmlp": ModelDefinition(
            model_id="realmlp",
            model_name="RealMLP_TD_Regressor",
            builder=build_realmlp_regressor,
            fit_kwargs_builder=build_realmlp_fit_kwargs,
            tuning_space_builder=build_realmlp_tuning_space,
            supports_native_categorical_preprocessing=True,
            is_cpu_only=True,
        ),
        "knn": ModelDefinition(
            model_id="knn",
            model_name="KNeighborsRegressor",
            builder=build_knn_regressor,
            tuning_space_builder=build_knn_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), rejects_sparse=True),
            ),
        ),
        "svm": ModelDefinition(
            model_id="svm",
            model_name="SVR",
            builder=build_svm_regressor,
            tuning_space_builder=build_svm_regressor_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "lightgbm": ModelDefinition(
            model_id="lightgbm",
            model_name="LGBMRegressor",
            builder=build_lightgbm_regressor,
            tuning_space_builder=build_lightgbm_tuning_space,
            supports_sparse_preprocessed_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "catboost": ModelDefinition(
            model_id="catboost",
            model_name="CatBoostRegressor",
            builder=build_catboost_regressor,
            fit_kwargs_builder=build_catboost_fit_kwargs,
            tuning_space_builder=build_catboost_tuning_space,
            supports_native_categorical_preprocessing=True,
            supports_sparse_preprocessed_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), requires_native_categorical=True),
            ),
        ),
        "xgboost": ModelDefinition(
            model_id="xgboost",
            model_name="XGBRegressor",
            builder=build_xgboost_regressor,
            tuning_space_builder=build_xgboost_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
    },
    "binary": {
        "logistic_regression": ModelDefinition(
            model_id="logistic_regression",
            model_name="LogisticRegression",
            builder=build_logreg,
            tuning_space_builder=build_logreg_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "random_forest": ModelDefinition(
            model_id="random_forest",
            model_name="RandomForestClassifier",
            builder=build_random_forest_classifier,
            tuning_space_builder=build_random_forest_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), rejects_sparse=True),
            ),
        ),
        "extra_trees": ModelDefinition(
            model_id="extra_trees",
            model_name="ExtraTreesClassifier",
            builder=build_extra_trees_classifier,
            tuning_space_builder=build_extra_trees_tuning_space,
            supports_sparse_preprocessed_input=True,
            is_cpu_only=True,
        ),
        "hist_gradient_boosting": ModelDefinition(
            model_id="hist_gradient_boosting",
            model_name="HistGradientBoostingClassifier",
            builder=build_hist_gradient_boosting_classifier,
            tuning_space_builder=build_hist_gradient_boosting_tuning_space,
            is_cpu_only=True,
        ),
        "realmlp": ModelDefinition(
            model_id="realmlp",
            model_name="RealMLP_TD_Classifier",
            builder=build_realmlp_classifier,
            fit_kwargs_builder=build_realmlp_fit_kwargs,
            tuning_space_builder=build_realmlp_tuning_space,
            supports_native_categorical_preprocessing=True,
            is_cpu_only=True,
        ),
        "knn": ModelDefinition(
            model_id="knn",
            model_name="KNeighborsClassifier",
            builder=build_knn_classifier,
            tuning_space_builder=build_knn_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), rejects_sparse=True),
            ),
        ),
        "svm": ModelDefinition(
            model_id="svm",
            model_name="SVC",
            builder=build_svm_classifier,
            tuning_space_builder=build_svm_classifier_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "naive_bayes": ModelDefinition(
            model_id="naive_bayes",
            model_name="GaussianNB",
            builder=build_naive_bayes_classifier,
            tuning_space_builder=build_naive_bayes_tuning_space,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "lightgbm": ModelDefinition(
            model_id="lightgbm",
            model_name="LGBMClassifier",
            builder=build_lightgbm_classifier,
            tuning_space_builder=build_lightgbm_tuning_space,
            supports_sparse_preprocessed_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
        "catboost": ModelDefinition(
            model_id="catboost",
            model_name="CatBoostClassifier",
            builder=build_catboost_classifier,
            fit_kwargs_builder=build_catboost_fit_kwargs,
            tuning_space_builder=build_catboost_tuning_space,
            supports_native_categorical_preprocessing=True,
            supports_sparse_preprocessed_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,), requires_native_categorical=True),
            ),
        ),
        "xgboost": ModelDefinition(
            model_id="xgboost",
            model_name="XGBClassifier",
            builder=build_xgboost_classifier,
            tuning_space_builder=build_xgboost_tuning_space,
            supports_sparse_preprocessed_input=True,
            supports_gpu_native_dense_onehot_input=True,
            gpu_routing_rules=(
                GpuRoutingRule(gpu_backends=(NATIVE_GPU_BACKEND,)),
            ),
        ),
    },
}


def get_task_model_registry(task_type: str) -> dict[str, ModelDefinition]:
    try:
        return MODEL_REGISTRY[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def resolve_candidate_model_id(
    task_type: str,
    model_family: str,
) -> str:
    task_registry = get_task_model_registry(task_type)
    if model_family in task_registry:
        return model_family

    supported_model_families = sorted(task_registry)
    raise ValueError(
        f"Candidate model_family '{model_family}' is not valid for task_type '{task_type}'. "
        f"Supported model families: {supported_model_families}"
    )


def get_default_model_id(task_type: str) -> str:
    try:
        return DEFAULT_MODEL_ID_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_type for model selection: {task_type}") from exc


def get_supported_model_ids(task_type: str) -> list[str]:
    return sorted(get_task_model_registry(task_type))


def get_tunable_model_ids(task_type: str) -> list[str]:
    task_registry = get_task_model_registry(task_type)
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
    native_model_families = sorted(
        candidate_model_id
        for candidate_model_id, candidate_definition in get_task_model_registry(task_type).items()
        if candidate_definition.supports_native_categorical_preprocessing
    )
    raise ValueError(
        f"Model family '{model_definition.model_id}' does not support "
        "categorical_preprocessor='native'. "
        f"Supported native categorical model families: {native_model_families}."
    )


def validate_model_representation_compatibility(
    task_type: str,
    model_id: str,
    representation_contract: RepresentationContract,
) -> None:
    validate_model_output_compatibility(
        task_type=task_type,
        model_id=model_id,
        has_native_categorical=representation_contract.has_native_categorical,
        has_sparse_numeric=representation_contract.has_sparse_numeric,
    )


def validate_model_output_compatibility(
    task_type: str,
    model_id: str,
    has_native_categorical: bool,
    has_sparse_numeric: bool,
) -> None:
    model_definition = get_model_definition(task_type, model_id)

    if has_native_categorical and not model_definition.supports_native_categorical_preprocessing:
        native_model_families = sorted(
            candidate_model_id
            for candidate_model_id, candidate_definition in get_task_model_registry(task_type).items()
            if candidate_definition.supports_native_categorical_preprocessing
        )
        raise ValueError(
            f"Model family '{model_definition.model_id}' does not support native categorical representations. "
            f"Supported native categorical model families: {native_model_families}."
        )

    if has_sparse_numeric and not model_definition.supports_sparse_preprocessed_input:
        raise ValueError(
            f"Model family '{model_definition.model_id}' does not support sparse numeric representations."
        )


def resolve_model_matrix_output_kind(
    task_type: str,
    model_id: str,
    categorical_preprocessor_id: str,
    runtime_execution_context: RuntimeExecutionContext | None = None,
) -> str:
    model_definition = get_model_definition(task_type, model_id)
    if categorical_preprocessor_id == "native":
        return "native_frame"
    if (
        categorical_preprocessor_id == "onehot"
        and runtime_execution_context is not None
        and runtime_execution_context.resolved_gpu_backend in (NATIVE_GPU_BACKEND, PATCH_GPU_BACKEND)
        and model_definition.supports_gpu_native_dense_onehot_input
    ):
        return "dense_array"
    if categorical_preprocessor_id == "onehot" and model_definition.supports_sparse_preprocessed_input:
        return "sparse_csr"
    return "dense_array"


def resolve_model_id(task_type: str, model_id: str) -> str:
    task_registry = get_task_model_registry(task_type)
    if model_id in task_registry:
        return model_id

    supported_model_ids = get_supported_model_ids(task_type)
    raise ValueError(
        f"Model id '{model_id}' is not valid for task_type '{task_type}'. "
        f"Use canonical model_ids only. Supported model_ids: {supported_model_ids}"
    )


def get_model_definition(task_type: str, model_id: str) -> ModelDefinition:
    resolved_model_id = resolve_model_id(task_type, model_id)
    return get_task_model_registry(task_type)[resolved_model_id]


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
    if model_definition.model_id == "realmlp":
        return model_definition.tuning_space_builder(trial, task_type=task_type)
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


