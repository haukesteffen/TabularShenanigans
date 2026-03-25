import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from tabular_shenanigans.candidate_artifacts import build_target_summary
from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.models import (
    ModelDefinition,
    build_model,
    build_model_fit_kwargs,
)
from tabular_shenanigans.preprocess import prepare_feature_frames
from tabular_shenanigans.representations import (
    CompiledRepresentation,
    ResolvedFeatureSchema,
    build_representation_contract,
    compile_representation,
    resolve_feature_schema,
)
from tabular_shenanigans.runtime_execution import NATIVE_GPU_BACKEND


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
    representation_id: str
    model_params: dict[str, object]
    cv_summary: CvSummary

    def to_fingerprint_entry(self) -> dict[str, object]:
        return {
            "model_registry_key": self.model_registry_key,
            "representation_id": self.representation_id,
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
class FoldEvaluationResult:
    valid_predictions: np.ndarray
    test_predictions: np.ndarray | None
    metric_value: float
    train_rows: int
    valid_rows: int
    preprocess_wall_seconds: float
    fit_wall_seconds: float
    predict_wall_seconds: float
    residency_profile: dict[str, object]

    def to_metric_row(self, *, fold_index: int, metric_name: str) -> dict[str, object]:
        return {
            "fold": fold_index,
            "metric_name": metric_name,
            "metric_value": self.metric_value,
            "train_rows": self.train_rows,
            "valid_rows": self.valid_rows,
        }


@dataclass(frozen=True)
class CvEvaluationResult:
    cv_evaluation: ModelCvEvaluation
    runtime_profile: dict[str, object]


@dataclass(frozen=True)
class FullCvEvaluationResult:
    cv_evaluation: ModelCvEvaluation
    runtime_profile: dict[str, object]
    oof_predictions: np.ndarray
    final_test_predictions: np.ndarray
    test_prediction_probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class CvFoldLoopResult:
    model_definition: ModelDefinition
    model_params: dict[str, object]
    fold_results: list[FoldEvaluationResult]
    fold_metrics: list[dict[str, object]]
    runtime_profile: dict[str, object]


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
    compiled_representation: CompiledRepresentation
    representation_id: str
    matrix_output_kind: str
    preserves_gpu_preprocessed_inputs: bool
    preprocessing_backend: str


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
    target_summary = build_target_summary(
        task_type=competition.task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )
    representation_spec = candidate.representation.to_runtime_spec()
    representation_contract = build_representation_contract(representation_spec)
    runtime_execution_context = config.runtime_execution_context
    if runtime_execution_context.sparse_to_dense_coercion and representation_contract.matrix_output_kind == "sparse_csr":
        representation_contract = replace(representation_contract, matrix_output_kind="dense_array")
    representation_id = representation_spec.representation_id
    feature_schema = resolve_feature_schema(
        x_train_raw=x_train_raw,
        force_categorical=features.force_categorical,
        force_numeric=features.force_numeric,
        low_cardinality_int_threshold=features.low_cardinality_int_threshold,
    )
    compiled_representation = compile_representation(
        representation_spec=representation_spec,
        feature_schema=feature_schema,
        x_train_sample=x_train_raw,
        representation_contract=representation_contract,
    )
    preserves_gpu_preprocessed_inputs = (
        config.resolved_model_registry_key == "xgboost"
        and runtime_execution_context.resolved_gpu_backend == NATIVE_GPU_BACKEND
    )
    return PreparedTrainingContext(
        id_column=id_column,
        label_column=label_column,
        competition_manifest=prepared_context.manifest,
        y_train=y_train,
        x_train_features=x_train_raw,
        x_test_features=x_test_raw,
        split_indices=prepared_context.split_indices,
        fold_assignments=prepared_context.fold_assignments,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
        target_summary=target_summary,
        feature_schema=feature_schema,
        compiled_representation=compiled_representation,
        representation_id=representation_id,
        matrix_output_kind=compiled_representation.matrix_output_kind,
        preserves_gpu_preprocessed_inputs=preserves_gpu_preprocessed_inputs,
        preprocessing_backend=compiled_representation.preprocessing_backend,
    )


def _coerce_processed_matrix(
    values: object,
    matrix_output_kind: str,
    preserves_gpu_preprocessed_inputs: bool,
) -> object:
    if not preserves_gpu_preprocessed_inputs:
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


def _resolve_processed_feature_columns(
    training_context: PreparedTrainingContext,
    x_fold_train_processed: object,
) -> tuple[list[str], list[str]]:
    if training_context.matrix_output_kind != "native_frame" or not isinstance(x_fold_train_processed, pd.DataFrame):
        return (
            training_context.feature_schema.numeric_columns,
            training_context.feature_schema.categorical_columns,
        )

    processed_categorical_columns = [
        column_name
        for column_name in x_fold_train_processed.columns
        if x_fold_train_processed[column_name].dtype == object
        or pd.api.types.is_string_dtype(x_fold_train_processed[column_name])
    ]
    processed_numeric_columns = [
        column_name
        for column_name in x_fold_train_processed.columns
        if column_name not in processed_categorical_columns
    ]
    return processed_numeric_columns, processed_categorical_columns


def _coerce_model_inputs(
    model_definition: ModelDefinition,
    x_fold_train_processed: object,
    x_fold_valid_processed: object,
    x_test_processed: object | None,
) -> tuple[object, object, object | None]:
    x_fold_train_processed = model_definition.coerce_input(x_fold_train_processed)
    x_fold_valid_processed = model_definition.coerce_input(x_fold_valid_processed)
    if x_test_processed is not None:
        x_test_processed = model_definition.coerce_input(x_test_processed)
    return x_fold_train_processed, x_fold_valid_processed, x_test_processed


def _predict_fold(
    *,
    model: object,
    task_type: str,
    x_fold_valid_processed: object,
    x_test_processed: object | None,
    positive_label: object | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if task_type == "binary":
        if positive_label is None:
            raise ValueError("Binary training requires resolved class metadata.")
        positive_class_index = list(model.classes_).index(positive_label)
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

    resolved_valid_predictions = _coerce_prediction_values(fold_valid_predictions)
    resolved_test_predictions = None
    if fold_test_predictions is not None:
        resolved_test_predictions = _coerce_prediction_values(fold_test_predictions)
    return resolved_valid_predictions, resolved_test_predictions


def _evaluate_fold(
    *,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    task_type: str,
    primary_metric: str,
    model_definition: ModelDefinition,
    base_model: object,
    training_context: PreparedTrainingContext,
    collect_prediction_artifacts: bool,
) -> FoldEvaluationResult:
    x_fold_train = training_context.x_train_features.iloc[train_idx]
    x_fold_valid = training_context.x_train_features.iloc[valid_idx]
    y_fold_train = training_context.y_train.iloc[train_idx]
    y_fold_valid = training_context.y_train.iloc[valid_idx]

    preprocess_started = time.perf_counter()
    fitted_representation = training_context.compiled_representation.fit(x_fold_train, y_fold_train)
    x_fold_train_processed = fitted_representation.transform(x_fold_train)
    x_fold_valid_processed = fitted_representation.transform(x_fold_valid)
    x_test_processed = None
    if collect_prediction_artifacts:
        x_test_processed = fitted_representation.transform(training_context.x_test_features)

    x_fold_train_processed = _coerce_processed_matrix(
        x_fold_train_processed,
        training_context.matrix_output_kind,
        training_context.preserves_gpu_preprocessed_inputs,
    )
    x_fold_valid_processed = _coerce_processed_matrix(
        x_fold_valid_processed,
        training_context.matrix_output_kind,
        training_context.preserves_gpu_preprocessed_inputs,
    )
    if x_test_processed is not None:
        x_test_processed = _coerce_processed_matrix(
            x_test_processed,
            training_context.matrix_output_kind,
            training_context.preserves_gpu_preprocessed_inputs,
        )
    x_fold_train_processed, x_fold_valid_processed, x_test_processed = _coerce_model_inputs(
        model_definition,
        x_fold_train_processed,
        x_fold_valid_processed,
        x_test_processed,
    )
    preprocess_wall_seconds = time.perf_counter() - preprocess_started

    processed_numeric_columns, processed_categorical_columns = _resolve_processed_feature_columns(
        training_context,
        x_fold_train_processed,
    )
    model_fit_kwargs = build_model_fit_kwargs(
        model_definition=model_definition,
        x_train_processed=x_fold_train_processed,
        numeric_columns=processed_numeric_columns,
        categorical_columns=processed_categorical_columns,
        uses_native_categorical_preprocessing=training_context.matrix_output_kind == "native_frame",
    )

    model = clone(base_model)
    fit_started = time.perf_counter()
    model.fit(x_fold_train_processed, y_fold_train, **model_fit_kwargs)
    fit_wall_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    fold_valid_predictions, fold_test_predictions = _predict_fold(
        model=model,
        task_type=task_type,
        x_fold_valid_processed=x_fold_valid_processed,
        x_test_processed=x_test_processed,
        positive_label=training_context.positive_label,
    )
    predict_wall_seconds = time.perf_counter() - predict_started

    fold_score = score_predictions(
        task_type=task_type,
        primary_metric=primary_metric,
        y_true=y_fold_valid,
        y_pred=fold_valid_predictions,
        positive_label=training_context.positive_label,
    )
    residency_profile = {
        "train_processed": _describe_matrix_residency(x_fold_train_processed),
        "valid_processed": _describe_matrix_residency(x_fold_valid_processed),
        "test_processed": _describe_matrix_residency(x_test_processed) if x_test_processed is not None else None,
    }
    return FoldEvaluationResult(
        valid_predictions=fold_valid_predictions,
        test_predictions=fold_test_predictions,
        metric_value=fold_score,
        train_rows=int(len(train_idx)),
        valid_rows=int(len(valid_idx)),
        preprocess_wall_seconds=preprocess_wall_seconds,
        fit_wall_seconds=fit_wall_seconds,
        predict_wall_seconds=predict_wall_seconds,
        residency_profile=residency_profile,
    )


def _build_cv_evaluation(
    *,
    model_definition: ModelDefinition,
    model_params: dict[str, object],
    representation_id: str,
    primary_metric: str,
    fold_metrics: list[dict[str, object]],
) -> ModelCvEvaluation:
    fold_metrics_df = pd.DataFrame(fold_metrics)
    return ModelCvEvaluation(
        model_result=ModelRunResult(
            model_registry_key=model_definition.model_id,
            estimator_name=model_definition.model_name,
            representation_id=representation_id,
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


def _build_runtime_profile(
    *,
    fold_count: int,
    preprocessing_backend: str,
    preprocess_wall_seconds: float,
    fit_wall_seconds: float,
    predict_wall_seconds: float,
    first_fold_residency: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "fold_count": fold_count,
        "preprocessing_backend": preprocessing_backend,
        "cv_preprocess_wall_seconds": preprocess_wall_seconds,
        "cv_fit_wall_seconds": fit_wall_seconds,
        "cv_predict_wall_seconds": predict_wall_seconds,
        "cv_stage_wall_seconds": preprocess_wall_seconds + fit_wall_seconds + predict_wall_seconds,
        "first_fold_residency": first_fold_residency,
    }


def _finalize_test_predictions(
    *,
    task_type: str,
    primary_metric: str,
    training_context: PreparedTrainingContext,
    test_predictions_per_fold: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
    test_prediction_probabilities = None
    if task_type == "regression" and primary_metric == "rmsle":
        mean_test_predictions = np.clip(mean_test_predictions, a_min=0.0, a_max=None)

    if task_type == "binary" and get_binary_prediction_kind(primary_metric) == "label":
        if training_context.positive_label is None or training_context.negative_label is None:
            raise ValueError("Binary label exports require resolved class metadata.")
        test_prediction_probabilities = np.asarray(mean_test_predictions, dtype=float)
        final_test_predictions = np.where(
            mean_test_predictions >= 0.5,
            training_context.positive_label,
            training_context.negative_label,
        )
        return np.asarray(final_test_predictions), test_prediction_probabilities

    return np.asarray(mean_test_predictions), test_prediction_probabilities


def _run_cv_fold_loop(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
    collect_prediction_artifacts: bool,
) -> CvFoldLoopResult:
    model_definition, base_model, model_params = build_model(
        task_type,
        model_spec.model_registry_key,
        cv_random_state,
        parameter_overrides=model_spec.parameter_overrides,
    )
    fold_results: list[FoldEvaluationResult] = []
    fold_metrics: list[dict[str, object]] = []
    preprocess_wall_seconds = 0.0
    fit_wall_seconds = 0.0
    predict_wall_seconds = 0.0
    first_fold_residency: dict[str, object] | None = None

    for fold_index, train_idx, valid_idx in training_context.split_indices:
        fold_result = _evaluate_fold(
            train_idx=train_idx,
            valid_idx=valid_idx,
            task_type=task_type,
            primary_metric=primary_metric,
            model_definition=model_definition,
            base_model=base_model,
            training_context=training_context,
            collect_prediction_artifacts=collect_prediction_artifacts,
        )
        fold_results.append(fold_result)
        fold_metrics.append(fold_result.to_metric_row(fold_index=fold_index, metric_name=primary_metric))
        preprocess_wall_seconds += fold_result.preprocess_wall_seconds
        fit_wall_seconds += fold_result.fit_wall_seconds
        predict_wall_seconds += fold_result.predict_wall_seconds
        if first_fold_residency is None:
            first_fold_residency = fold_result.residency_profile

    runtime_profile = _build_runtime_profile(
        fold_count=len(training_context.split_indices),
        preprocessing_backend=training_context.preprocessing_backend,
        preprocess_wall_seconds=preprocess_wall_seconds,
        fit_wall_seconds=fit_wall_seconds,
        predict_wall_seconds=predict_wall_seconds,
        first_fold_residency=first_fold_residency,
    )
    return CvFoldLoopResult(
        model_params=model_params,
        model_definition=model_definition,
        fold_results=fold_results,
        fold_metrics=fold_metrics,
        runtime_profile=runtime_profile,
    )


def _run_cv_evaluation(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> CvEvaluationResult:
    fold_loop_result = _run_cv_fold_loop(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
        collect_prediction_artifacts=False,
    )
    return CvEvaluationResult(
        cv_evaluation=_build_cv_evaluation(
            model_definition=fold_loop_result.model_definition,
            model_params=fold_loop_result.model_params,
            representation_id=training_context.representation_id,
            primary_metric=primary_metric,
            fold_metrics=fold_loop_result.fold_metrics,
        ),
        runtime_profile=fold_loop_result.runtime_profile,
    )


def _run_full_cv_evaluation(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> FullCvEvaluationResult:
    fold_loop_result = _run_cv_fold_loop(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
        collect_prediction_artifacts=True,
    )
    oof_predictions = np.zeros(training_context.x_train_features.shape[0], dtype=float)
    test_predictions_per_fold: list[np.ndarray] = []
    for (_, _, valid_idx), fold_result in zip(training_context.split_indices, fold_loop_result.fold_results):
        oof_predictions[valid_idx] = fold_result.valid_predictions
        if fold_result.test_predictions is None:
            raise RuntimeError("Full evaluation must collect test predictions for every fold.")
        test_predictions_per_fold.append(np.asarray(fold_result.test_predictions, dtype=float))

    final_test_predictions, test_prediction_probabilities = _finalize_test_predictions(
        task_type=task_type,
        primary_metric=primary_metric,
        training_context=training_context,
        test_predictions_per_fold=test_predictions_per_fold,
    )
    return FullCvEvaluationResult(
        cv_evaluation=_build_cv_evaluation(
            model_definition=fold_loop_result.model_definition,
            model_params=fold_loop_result.model_params,
            representation_id=training_context.representation_id,
            primary_metric=primary_metric,
            fold_metrics=fold_loop_result.fold_metrics,
        ),
        runtime_profile=fold_loop_result.runtime_profile,
        oof_predictions=oof_predictions,
        final_test_predictions=final_test_predictions,
        test_prediction_probabilities=test_prediction_probabilities,
    )


def score_model_spec(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> ModelCvEvaluation:
    evaluation_result = _run_cv_evaluation(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
    )
    return evaluation_result.cv_evaluation


def evaluate_model_spec(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    training_context: PreparedTrainingContext,
    cv_random_state: int,
) -> ModelEvaluationArtifacts:
    evaluation_result = _run_full_cv_evaluation(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        training_context=training_context,
        cv_random_state=cv_random_state,
    )

    return ModelEvaluationArtifacts(
        model_result=evaluation_result.cv_evaluation.model_result,
        fold_metrics_df=evaluation_result.cv_evaluation.fold_metrics_df,
        oof_predictions=evaluation_result.oof_predictions,
        final_test_predictions=evaluation_result.final_test_predictions,
        test_prediction_probabilities=evaluation_result.test_prediction_probabilities,
        runtime_profile=evaluation_result.runtime_profile,
    )
