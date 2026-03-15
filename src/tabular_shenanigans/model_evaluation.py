import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from tabular_shenanigans.candidate_artifacts import build_target_summary
from tabular_shenanigans.lightgbm_cuda_backend import coerce_lightgbm_matrix_input
from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.feature_recipes import apply_feature_recipe
from tabular_shenanigans.models import (
    build_model,
    build_model_fit_kwargs,
    resolve_model_matrix_output_kind,
)
from tabular_shenanigans.preprocess import (
    ResolvedFeatureSchema,
    prepare_feature_frames,
    resolve_feature_schema,
)
from tabular_shenanigans.preprocess_execution import (
    PreprocessingExecutionPlan,
    build_preprocessor_for_execution_plan,
)


def _module_startswith(values: object, prefix: str) -> bool:
    return type(values).__module__.startswith(prefix)


@dataclass(frozen=True)
class CvSummary:
    metric_name: str
    metric_mean: float
    metric_std: float
    higher_is_better: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "metric_mean": self.metric_mean,
            "metric_std": self.metric_std,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class ModelRunResult:
    model_registry_key: str
    estimator_name: str
    preprocessing_scheme_id: str
    model_params: dict[str, object]
    cv_summary: CvSummary

    def to_fingerprint_entry(self) -> dict[str, object]:
        return {
            "model_registry_key": self.model_registry_key,
            "preprocessing_scheme_id": self.preprocessing_scheme_id,
            "model_params": self.model_params,
        }


@dataclass(frozen=True)
class TrainingModelSpec:
    model_registry_key: str
    parameter_overrides: dict[str, object] | None = None


@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    model_result: ModelRunResult
    fold_metrics_df: pd.DataFrame
    oof_predictions: np.ndarray
    final_test_predictions: np.ndarray
    test_prediction_probabilities: np.ndarray | None = None
    runtime_profile: dict[str, object] | None = None


@dataclass(frozen=True)
class ModelCvEvaluation:
    model_result: ModelRunResult
    fold_metrics_df: pd.DataFrame


@dataclass(frozen=True)
class PreparedTrainingContext:
    id_column: str
    label_column: str
    competition_manifest: dict[str, object]
    y_train: pd.Series
    x_train_features: pd.DataFrame
    x_test_features: pd.DataFrame
    split_indices: list[tuple[int, np.ndarray, np.ndarray]]
    fold_assignments: np.ndarray
    positive_label: object | None
    negative_label: object | None
    observed_label_pair: tuple[object, object] | None
    target_summary: dict[str, object]
    feature_schema: ResolvedFeatureSchema
    numeric_preprocessor: str
    categorical_preprocessor: str
    preprocessing_scheme_id: str
    matrix_output_kind: str
    uses_xgboost_gpu_native_inputs: bool
    preprocessing_backend: str
    model_compute_target: str


def _validate_gpu_native_matrix_output(
    uses_xgboost_gpu_native_inputs: bool,
    matrix_output_kind: str,
) -> None:
    if not uses_xgboost_gpu_native_inputs:
        return
    if matrix_output_kind != "sparse_csr":
        return
    raise ValueError(
        "XGBoost GPU execution currently requires dense fold-local preprocessing output in this runtime. "
        "The sparse CSR path produced by categorical_preprocessor='onehot' and related kbins compositions "
        "cannot be promoted to a supported GPU-native XGBoost input because cupyx CSR is not supported yet. "
        "Use categorical_preprocessor='ordinal' or 'frequency', or force CPU execution."
    )


def build_prepared_training_context(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> PreparedTrainingContext:
    if not config.is_model_candidate:
        raise ValueError("Prepared training context is only supported for model candidates.")

    competition = config.competition
    features = competition.features
    candidate = config.experiment.candidate
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=features.force_categorical,
        force_numeric=features.force_numeric,
        drop_columns=features.drop_columns,
    )

    positive_label = competition.positive_label
    observed_label_pair = None
    negative_label = None
    if competition.task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    prepared_context = ensure_prepared_competition_context(
        config=config,
        dataset_context=dataset_context,
        expected_feature_columns=x_train_raw.columns.tolist(),
    )
    x_train_features, x_test_features = apply_feature_recipe(
        recipe_id=candidate.feature_recipe_id,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
    )
    target_summary = build_target_summary(
        task_type=competition.task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )
    feature_schema = resolve_feature_schema(
        x_train_raw=x_train_features,
        force_categorical=features.force_categorical,
        force_numeric=features.force_numeric,
        low_cardinality_int_threshold=features.low_cardinality_int_threshold,
    )
    matrix_output_kind = resolve_model_matrix_output_kind(
        task_type=competition.task_type,
        model_id=config.resolved_model_registry_key,
        categorical_preprocessor_id=candidate.categorical_preprocessor,
        runtime_execution_context=config.runtime_execution_context,
    )
    preprocessing_execution_plan = config.preprocessing_execution_plan
    uses_xgboost_gpu_native_inputs = (
        config.resolved_model_registry_key == "xgboost"
        and config.runtime_execution_context.resolved_compute_target == "gpu"
    )
    _validate_gpu_native_matrix_output(
        uses_xgboost_gpu_native_inputs=uses_xgboost_gpu_native_inputs,
        matrix_output_kind=matrix_output_kind,
    )
    return PreparedTrainingContext(
        id_column=id_column,
        label_column=label_column,
        competition_manifest=prepared_context.manifest,
        y_train=y_train,
        x_train_features=x_train_features,
        x_test_features=x_test_features,
        split_indices=prepared_context.split_indices,
        fold_assignments=prepared_context.fold_assignments,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
        target_summary=target_summary,
        feature_schema=feature_schema,
        numeric_preprocessor=candidate.numeric_preprocessor,
        categorical_preprocessor=candidate.categorical_preprocessor,
        preprocessing_scheme_id=candidate.preprocessing_scheme_id,
        matrix_output_kind=preprocessing_execution_plan.matrix_output_kind,
        uses_xgboost_gpu_native_inputs=uses_xgboost_gpu_native_inputs,
        preprocessing_backend=preprocessing_execution_plan.preprocessing_backend,
        model_compute_target=config.runtime_execution_context.resolved_compute_target,
    )


def _coerce_processed_matrix(
    values: object,
    matrix_output_kind: str,
    model_compute_target: str,
) -> object:
    if model_compute_target != "gpu":
        if _module_startswith(values, "cupy"):
            import cupy as cp

            values = cp.asnumpy(values)
        elif _module_startswith(values, "cudf"):
            values = values.to_pandas()

    if _module_startswith(values, "cudf") or _module_startswith(values, "cupy"):
        return values
    if matrix_output_kind == "native_frame":
        if not isinstance(values, pd.DataFrame):
            raise ValueError("Native categorical preprocessing must produce a pandas DataFrame.")
        return values
    if matrix_output_kind == "sparse_csr":
        return sparse.csr_matrix(values)
    return np.asarray(values)


def _coerce_xgboost_gpu_input(values: object, matrix_output_kind: str) -> object:
    try:
        import cudf
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost GPU-native inputs require the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost GPU-native inputs require the optional GPU dependencies. "
            "Install them with `uv sync --extra boosters --extra gpu`."
        ) from exc

    if matrix_output_kind == "sparse_csr":
        raise ValueError(
            "XGBoost GPU execution currently does not support sparse CSR preprocessing output in this runtime."
        )

    if _module_startswith(values, "cudf") or _module_startswith(values, "cupy"):
        return values

    if isinstance(values, pd.DataFrame):
        return cudf.from_pandas(values)

    if hasattr(values, "to_pandas"):
        return cudf.from_pandas(values.to_pandas())

    return cp.asarray(values)


def _coerce_prediction_values(values: object) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values

    if _module_startswith(values, "cupy"):
        import cupy as cp

        return cp.asnumpy(values)

    if hasattr(values, "to_pandas"):
        return np.asarray(values.to_pandas())

    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())

    return np.asarray(values)


def _select_binary_positive_class_scores(
    probability_values: object,
    positive_class_index: int,
) -> object:
    if hasattr(probability_values, "iloc"):
        return probability_values.iloc[:, positive_class_index]

    if isinstance(probability_values, np.ndarray):
        if probability_values.ndim == 1:
            return probability_values
        return probability_values[:, positive_class_index]

    if _module_startswith(probability_values, "cupy"):
        return probability_values[:, positive_class_index]

    if hasattr(probability_values, "ndim") and getattr(probability_values, "ndim") == 1:
        return probability_values

    return probability_values[:, positive_class_index]


def _describe_matrix_residency(values: object) -> dict[str, object]:
    value_type = type(values)
    module_name = value_type.__module__
    type_name = value_type.__name__

    if module_name.startswith("cudf"):
        residency = "gpu_cudf"
    elif module_name.startswith("cupy"):
        residency = "gpu_cupy"
    elif sparse.issparse(values):
        residency = "cpu_scipy_sparse"
    elif isinstance(values, pd.DataFrame):
        residency = "cpu_pandas"
    elif isinstance(values, np.ndarray):
        residency = "cpu_numpy"
    else:
        residency = "unknown"

    return {
        "residency": residency,
        "type_name": f"{module_name}.{type_name}",
    }


def _run_cv_evaluation(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
    collect_prediction_artifacts: bool,
) -> tuple[
    ModelCvEvaluation,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, object],
]:
    model_definition, _, model_params = build_model(
        task_type,
        model_spec.model_registry_key,
        cv_random_state,
        parameter_overrides=model_spec.parameter_overrides,
    )
    resolved_model_registry_key = model_definition.model_id
    estimator_name = model_definition.model_name
    preprocessing_scheme_id = training_context.preprocessing_scheme_id
    matrix_output_kind = training_context.matrix_output_kind
    uses_xgboost_gpu_native_inputs = training_context.uses_xgboost_gpu_native_inputs
    preprocessing_backend = training_context.preprocessing_backend
    model_compute_target = training_context.model_compute_target

    oof_predictions = (
        np.zeros(training_context.x_train_features.shape[0], dtype=float)
        if collect_prediction_artifacts
        else None
    )
    test_predictions_per_fold = [] if collect_prediction_artifacts else None
    fold_metrics: list[dict[str, object]] = []
    binary_prediction_kind = None
    if task_type == "binary":
        binary_prediction_kind = get_binary_prediction_kind(primary_metric)
    preprocess_wall_seconds = 0.0
    fit_wall_seconds = 0.0
    predict_wall_seconds = 0.0
    first_fold_residency: dict[str, object] | None = None

    for fold_index, train_idx, valid_idx in training_context.split_indices:
        x_fold_train = training_context.x_train_features.iloc[train_idx]
        x_fold_valid = training_context.x_train_features.iloc[valid_idx]
        y_fold_train = training_context.y_train.iloc[train_idx]
        y_fold_valid = training_context.y_train.iloc[valid_idx]

        preprocess_started = time.perf_counter()
        preprocessor = build_preprocessor_for_execution_plan(
            feature_schema=training_context.feature_schema,
            numeric_preprocessor_id=training_context.numeric_preprocessor,
            categorical_preprocessor_id=training_context.categorical_preprocessor,
            execution_plan=PreprocessingExecutionPlan(
                preprocessing_backend=training_context.preprocessing_backend,
                matrix_output_kind=matrix_output_kind,
            ),
        )
        x_fold_train_processed = preprocessor.fit_transform(x_fold_train)
        x_fold_valid_processed = preprocessor.transform(x_fold_valid)
        x_test_processed = None
        if collect_prediction_artifacts:
            x_test_processed = preprocessor.transform(training_context.x_test_features)

        x_fold_train_processed = _coerce_processed_matrix(
            x_fold_train_processed,
            matrix_output_kind,
            model_compute_target,
        )
        x_fold_valid_processed = _coerce_processed_matrix(
            x_fold_valid_processed,
            matrix_output_kind,
            model_compute_target,
        )
        if x_test_processed is not None:
            x_test_processed = _coerce_processed_matrix(
                x_test_processed,
                matrix_output_kind,
                model_compute_target,
            )

        if uses_xgboost_gpu_native_inputs:
            # Convert fold-local preprocessing outputs to GPU-native inputs before XGBoost fit/predict.
            x_fold_train_processed = _coerce_xgboost_gpu_input(x_fold_train_processed, matrix_output_kind)
            x_fold_valid_processed = _coerce_xgboost_gpu_input(x_fold_valid_processed, matrix_output_kind)
            if x_test_processed is not None:
                x_test_processed = _coerce_xgboost_gpu_input(x_test_processed, matrix_output_kind)
        elif resolved_model_registry_key == "lightgbm":
            x_fold_train_processed = coerce_lightgbm_matrix_input(x_fold_train_processed)
            x_fold_valid_processed = coerce_lightgbm_matrix_input(x_fold_valid_processed)
            if x_test_processed is not None:
                x_test_processed = coerce_lightgbm_matrix_input(x_test_processed)
        preprocess_wall_seconds += time.perf_counter() - preprocess_started
        if first_fold_residency is None:
            first_fold_residency = {
                "train_processed": _describe_matrix_residency(x_fold_train_processed),
                "valid_processed": _describe_matrix_residency(x_fold_valid_processed),
                "test_processed": (
                    _describe_matrix_residency(x_test_processed) if x_test_processed is not None else None
                ),
            }

        _, model, _ = build_model(
            task_type,
            resolved_model_registry_key,
            cv_random_state,
            parameter_overrides=model_spec.parameter_overrides,
        )
        model_fit_kwargs = build_model_fit_kwargs(
            model_definition=model_definition,
            x_train_processed=x_fold_train_processed,
            numeric_columns=training_context.feature_schema.numeric_columns,
            categorical_columns=training_context.feature_schema.categorical_columns,
            uses_native_categorical_preprocessing=matrix_output_kind == "native_frame",
        )
        fit_started = time.perf_counter()
        model.fit(x_fold_train_processed, y_fold_train, **model_fit_kwargs)
        fit_wall_seconds += time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        if task_type == "binary":
            if training_context.positive_label is None or training_context.negative_label is None:
                raise ValueError("Binary training requires resolved class metadata.")
            positive_class_index = list(model.classes_).index(training_context.positive_label)
            fold_valid_predictions = _select_binary_positive_class_scores(
                model.predict_proba(x_fold_valid_processed),
                positive_class_index=positive_class_index,
            )
            fold_test_predictions = None
            if x_test_processed is not None:
                fold_test_predictions = _select_binary_positive_class_scores(
                    model.predict_proba(x_test_processed),
                    positive_class_index=positive_class_index,
                )
        else:
            fold_valid_predictions = model.predict(x_fold_valid_processed)
            fold_test_predictions = None
            if x_test_processed is not None:
                fold_test_predictions = model.predict(x_test_processed)

        fold_valid_predictions = _coerce_prediction_values(fold_valid_predictions)
        if fold_test_predictions is not None:
            fold_test_predictions = _coerce_prediction_values(fold_test_predictions)
        predict_wall_seconds += time.perf_counter() - predict_started

        fold_score = score_predictions(
            task_type=task_type,
            primary_metric=primary_metric,
            y_true=y_fold_valid,
            y_pred=fold_valid_predictions,
            positive_label=training_context.positive_label,
        )

        if oof_predictions is not None:
            oof_predictions[valid_idx] = fold_valid_predictions
        if test_predictions_per_fold is not None and fold_test_predictions is not None:
            test_predictions_per_fold.append(np.asarray(fold_test_predictions, dtype=float))
        fold_metrics.append(
            {
                "fold": fold_index,
                "metric_name": primary_metric,
                "metric_value": fold_score,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    cv_evaluation = ModelCvEvaluation(
        model_result=ModelRunResult(
            model_registry_key=resolved_model_registry_key,
            estimator_name=estimator_name,
            preprocessing_scheme_id=preprocessing_scheme_id,
            model_params=model_params,
            cv_summary=CvSummary(
                metric_name=primary_metric,
                metric_mean=float(fold_metrics_df["metric_value"].mean()),
                metric_std=float(fold_metrics_df["metric_value"].std(ddof=0)),
                higher_is_better=is_higher_better(primary_metric),
            ),
        ),
        fold_metrics_df=fold_metrics_df,
    )
    runtime_profile = {
        "fold_count": len(training_context.split_indices),
        "preprocessing_backend": preprocessing_backend,
        "cv_preprocess_wall_seconds": preprocess_wall_seconds,
        "cv_fit_wall_seconds": fit_wall_seconds,
        "cv_predict_wall_seconds": predict_wall_seconds,
        "cv_stage_wall_seconds": preprocess_wall_seconds + fit_wall_seconds + predict_wall_seconds,
        "first_fold_residency": first_fold_residency,
    }

    if not collect_prediction_artifacts:
        return cv_evaluation, None, None, None, runtime_profile

    if oof_predictions is None or test_predictions_per_fold is None:
        raise RuntimeError("Prediction artifacts were requested but not collected.")

    mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
    test_prediction_probabilities = None
    if task_type == "regression" and primary_metric == "rmsle":
        mean_test_predictions = np.clip(mean_test_predictions, a_min=0.0, a_max=None)
    if task_type == "binary" and binary_prediction_kind == "label":
        if training_context.positive_label is None or training_context.negative_label is None:
            raise ValueError("Binary label exports require resolved class metadata.")
        test_prediction_probabilities = np.asarray(mean_test_predictions, dtype=float)
        final_test_predictions = np.where(
            mean_test_predictions >= 0.5,
            training_context.positive_label,
            training_context.negative_label,
        )
    else:
        final_test_predictions = mean_test_predictions

    return (
        cv_evaluation,
        oof_predictions,
        np.asarray(final_test_predictions),
        test_prediction_probabilities,
        runtime_profile,
    )


def score_model_spec(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> ModelCvEvaluation:
    cv_evaluation, _, _, _, _ = _run_cv_evaluation(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
        collect_prediction_artifacts=False,
    )
    return cv_evaluation


def evaluate_model_spec(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> ModelEvaluationArtifacts:
    cv_evaluation, oof_predictions, final_test_predictions, test_prediction_probabilities, runtime_profile = _run_cv_evaluation(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
        collect_prediction_artifacts=True,
    )
    if oof_predictions is None or final_test_predictions is None:
        raise RuntimeError("Training evaluation must return OOF and test predictions.")

    return ModelEvaluationArtifacts(
        model_result=cv_evaluation.model_result,
        fold_metrics_df=cv_evaluation.fold_metrics_df,
        oof_predictions=oof_predictions,
        final_test_predictions=final_test_predictions,
        test_prediction_probabilities=test_prediction_probabilities,
        runtime_profile=runtime_profile,
    )
