import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import build_splitter, is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.models import build_model, build_model_fit_kwargs
from tabular_shenanigans.preprocess import build_preprocessor, prepare_feature_frames

RUN_LEDGER_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "competition_slug",
    "task_type",
    "primary_metric",
    "best_model_id",
    "best_model_name",
    "cv_mean",
    "cv_std",
    "higher_is_better",
    "model_count",
    "cv_n_splits",
    "cv_shuffle",
    "cv_random_state",
    "config_fingerprint",
    "target_mean",
    "target_std",
    "target_min",
    "target_max",
    "positive_count",
    "negative_count",
    "target_prevalence",
    "positive_label",
    "observed_label_1",
    "observed_label_2",
    "negative_label",
]


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
    model_id: str
    model_name: str
    preprocessing_scheme_id: str
    model_params: dict[str, object]
    cv_summary: CvSummary
    rank: int | None = None
    is_best_model: bool = False

    def require_rank(self) -> int:
        if self.rank is None:
            raise ValueError("Model run result must be ranked before writing run-level artifacts.")
        return self.rank

    def to_model_summary_row(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "preprocessing_scheme_id": self.preprocessing_scheme_id,
            "metric_name": self.cv_summary.metric_name,
            "cv_mean": self.cv_summary.metric_mean,
            "cv_std": self.cv_summary.metric_std,
            "higher_is_better": self.cv_summary.higher_is_better,
            "rank": self.require_rank(),
            "is_best_model": self.is_best_model,
        }

    def to_manifest_model_entry(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "preprocessing_scheme_id": self.preprocessing_scheme_id,
            "model_params": self.model_params,
            "cv_summary": self.cv_summary.to_dict(),
            "rank": self.require_rank(),
            "is_best_model": self.is_best_model,
        }

    def to_fingerprint_model_entry(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_params": self.model_params,
        }


@dataclass(frozen=True)
class TrainingModelSpec:
    model_id: str
    parameter_overrides: dict[str, object] | None = None


@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    model_result: ModelRunResult
    fold_metrics_df: pd.DataFrame
    oof_predictions: np.ndarray
    final_test_predictions: np.ndarray


@dataclass(frozen=True)
class TrainingRunContext:
    run_id: str
    generated_at_utc: str
    competition_slug: str
    task_type: str
    primary_metric: str
    config_snapshot: dict[str, object]
    config_fingerprint: str
    model_results: list[ModelRunResult]
    best_model_result: ModelRunResult
    observed_label_pair: tuple[object, object] | None
    negative_label: object | None
    positive_label: object | None
    id_column: str
    label_column: str
    target_summary: dict[str, object]
    train_rows: int
    train_cols: int
    test_rows: int
    test_cols: int
    cv_n_splits: int
    cv_shuffle: bool
    cv_random_state: int
    tuning_provenance: dict[str, object] | None = None


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_training_model_specs(
    config: AppConfig,
    model_specs: list[TrainingModelSpec] | None = None,
) -> list[TrainingModelSpec]:
    if model_specs is not None:
        if not model_specs:
            raise ValueError("Training requires at least one model specification.")
        return model_specs
    return [TrainingModelSpec(model_id=model_id) for model_id in config.model_ids]


def _read_run_ledger(ledger_path: Path) -> pd.DataFrame:
    ledger_df = pd.read_csv(ledger_path)
    if "best_model_id" not in ledger_df.columns:
        legacy_model_id = ledger_df["model_id"] if "model_id" in ledger_df.columns else pd.Series("", index=ledger_df.index)
        ledger_df["best_model_id"] = legacy_model_id
    if "best_model_name" not in ledger_df.columns:
        legacy_model_name = (
            ledger_df["model_name"] if "model_name" in ledger_df.columns else pd.Series("", index=ledger_df.index)
        )
        ledger_df["best_model_name"] = legacy_model_name
    if "model_count" not in ledger_df.columns:
        ledger_df["model_count"] = 1
    return ledger_df


def _resolve_run_ledger_output_columns(
    existing_columns: list[str],
    row_columns: list[str],
) -> list[str]:
    extra_columns: list[str] = []
    seen_extra_columns: set[str] = set()
    for columns in (existing_columns, row_columns):
        for column in columns:
            if column in RUN_LEDGER_COLUMNS or column in seen_extra_columns:
                continue
            seen_extra_columns.add(column)
            extra_columns.append(column)
    return [*RUN_LEDGER_COLUMNS, *extra_columns]


def _append_run_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row])
    if ledger_path.exists():
        existing_df = _read_run_ledger(ledger_path)
        output_columns = _resolve_run_ledger_output_columns(
            existing_columns=existing_df.columns.tolist(),
            row_columns=ledger_df.columns.tolist(),
        )
        merged_df = pd.concat([existing_df, ledger_df], ignore_index=True, sort=False)
        merged_df = merged_df.reindex(columns=output_columns)
        merged_df.to_csv(ledger_path, index=False)
        return
    output_columns = _resolve_run_ledger_output_columns(existing_columns=[], row_columns=ledger_df.columns.tolist())
    ledger_df = ledger_df.reindex(columns=output_columns)
    ledger_df.to_csv(ledger_path, index=False)


def _build_target_summary(
    task_type: str,
    y_train: pd.Series,
    positive_label: object | None = None,
    negative_label: object | None = None,
    observed_label_pair: tuple[object, object] | None = None,
) -> dict[str, object]:
    if task_type == "regression":
        return {
            "target_mean": float(y_train.mean()),
            "target_std": float(y_train.std(ddof=0)),
            "target_min": float(y_train.min()),
            "target_max": float(y_train.max()),
        }

    if task_type == "binary":
        if positive_label is None or negative_label is None or observed_label_pair is None:
            raise ValueError("Binary target summary requires resolved label metadata.")
        positive_count = int((y_train == positive_label).sum())
        row_count = int(y_train.shape[0])
        negative_count = row_count - positive_count
        return {
            "observed_label_1": str(observed_label_pair[0]),
            "observed_label_2": str(observed_label_pair[1]),
            "negative_label": str(negative_label),
            "positive_label": str(positive_label),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "target_prevalence": float(positive_count / row_count),
        }

    raise ValueError(f"Unsupported task_type for target summary: {task_type}")


def _build_diagnostic_rows(
    task_type: str,
    fold_index: int,
    split_name: str,
    y_values: pd.Series,
    positive_label: object | None = None,
) -> list[dict[str, object]]:
    row = {
        "task_type": task_type,
        "fold": fold_index,
        "split": split_name,
        "row_count": int(y_values.shape[0]),
        "target_mean": np.nan,
        "target_std": np.nan,
        "target_min": np.nan,
        "target_max": np.nan,
        "positive_count": np.nan,
        "negative_count": np.nan,
        "positive_rate": np.nan,
    }

    if task_type == "regression":
        row.update(
            {
                "diagnostic_type": "target_distribution",
                "target_mean": float(y_values.mean()),
                "target_std": float(y_values.std(ddof=0)),
                "target_min": float(y_values.min()),
                "target_max": float(y_values.max()),
            }
        )
        return [row]

    if task_type == "binary":
        if positive_label is None:
            raise ValueError("Binary diagnostics require positive_label.")
        positive_count = int((y_values == positive_label).sum())
        negative_count = int(y_values.shape[0] - positive_count)
        row.update(
            {
                "diagnostic_type": "class_balance",
                "positive_label": str(positive_label),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": float(positive_count / y_values.shape[0]),
            }
        )
        return [row]

    raise ValueError(f"Unsupported task_type for diagnostics: {task_type}")


def _materialize_split_indices(
    task_type: str,
    x_train_raw: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    splitter = build_splitter(
        task_type=task_type,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    split_indices: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train_raw, y_train), start=1):
        split_indices.append((fold_index, train_idx, valid_idx))
    return split_indices


def _build_fold_assignments(
    row_count: int,
    split_indices: list[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    fold_assignments = np.full(row_count, fill_value=-1, dtype=int)
    for fold_index, _, valid_idx in split_indices:
        if (fold_assignments[valid_idx] >= 0).any():
            raise ValueError("Fold assignment failed: at least one training row received multiple validation folds.")
        fold_assignments[valid_idx] = fold_index
    if (fold_assignments < 0).any():
        raise ValueError("Fold assignment failed: at least one training row did not receive a validation fold.")
    return fold_assignments


def _build_run_diagnostics(
    task_type: str,
    y_train: pd.Series,
    split_indices: list[tuple[int, np.ndarray, np.ndarray]],
    positive_label: object | None = None,
) -> pd.DataFrame:
    run_diagnostics: list[dict[str, object]] = []
    run_diagnostics.extend(
        _build_diagnostic_rows(
            task_type=task_type,
            fold_index=0,
            split_name="all",
            y_values=y_train,
            positive_label=positive_label,
        )
    )
    for fold_index, train_idx, valid_idx in split_indices:
        run_diagnostics.extend(
            _build_diagnostic_rows(
                task_type=task_type,
                fold_index=fold_index,
                split_name="train",
                y_values=y_train.iloc[train_idx],
                positive_label=positive_label,
            )
        )
        run_diagnostics.extend(
            _build_diagnostic_rows(
                task_type=task_type,
                fold_index=fold_index,
                split_name="valid",
                y_values=y_train.iloc[valid_idx],
                positive_label=positive_label,
            )
        )
    return pd.DataFrame(run_diagnostics)


def _evaluate_model_spec(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    y_train: pd.Series,
    split_indices: list[tuple[int, np.ndarray, np.ndarray]],
    force_categorical: list[str] | None,
    force_numeric: list[str] | None,
    low_cardinality_int_threshold: int | None,
    cv_random_state: int,
    positive_label: object | None,
    negative_label: object | None,
) -> ModelEvaluationArtifacts:
    model_definition, _, model_params = build_model(
        task_type,
        model_spec.model_id,
        cv_random_state,
        parameter_overrides=model_spec.parameter_overrides,
    )
    resolved_model_id = model_definition.model_id
    model_name = model_definition.model_name
    preprocessing_scheme_id = model_definition.preprocessing_scheme_id

    oof_predictions = np.zeros(x_train_raw.shape[0], dtype=float)
    test_predictions_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict[str, object]] = []
    use_named_columns = model_name.startswith("LGBM")
    binary_prediction_kind = None
    if task_type == "binary":
        binary_prediction_kind = get_binary_prediction_kind(primary_metric)

    for fold_index, train_idx, valid_idx in split_indices:
        x_fold_train = x_train_raw.iloc[train_idx]
        x_fold_valid = x_train_raw.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        preprocessor, numeric_columns, categorical_columns = build_preprocessor(
            scheme_id=preprocessing_scheme_id,
            x_train_raw=x_fold_train,
            force_categorical=force_categorical,
            force_numeric=force_numeric,
            low_cardinality_int_threshold=low_cardinality_int_threshold,
        )
        if use_named_columns and hasattr(preprocessor, "set_output"):
            preprocessor.set_output(transform="pandas")
        x_fold_train_processed = preprocessor.fit_transform(x_fold_train)
        x_fold_valid_processed = preprocessor.transform(x_fold_valid)
        x_test_processed = preprocessor.transform(x_test_raw)

        if preprocessing_scheme_id != "native" and not use_named_columns:
            x_fold_train_processed = np.asarray(x_fold_train_processed)
            x_fold_valid_processed = np.asarray(x_fold_valid_processed)
            x_test_processed = np.asarray(x_test_processed)

        _, model, _ = build_model(
            task_type,
            resolved_model_id,
            cv_random_state,
            parameter_overrides=model_spec.parameter_overrides,
        )
        model_fit_kwargs = build_model_fit_kwargs(
            model_definition=model_definition,
            x_train_processed=x_fold_train_processed,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        model.fit(x_fold_train_processed, y_fold_train, **model_fit_kwargs)

        if task_type == "binary":
            if positive_label is None or negative_label is None:
                raise ValueError("Binary training requires resolved class metadata.")
            positive_class_index = list(model.classes_).index(positive_label)
            fold_valid_predictions = model.predict_proba(x_fold_valid_processed)[:, positive_class_index]
            fold_test_predictions = model.predict_proba(x_test_processed)[:, positive_class_index]
        else:
            fold_valid_predictions = model.predict(x_fold_valid_processed)
            fold_test_predictions = model.predict(x_test_processed)

        fold_score = score_predictions(
            task_type=task_type,
            primary_metric=primary_metric,
            y_true=y_fold_valid,
            y_pred=fold_valid_predictions,
            positive_label=positive_label,
        )

        oof_predictions[valid_idx] = fold_valid_predictions
        test_predictions_per_fold.append(np.asarray(fold_test_predictions, dtype=float))
        fold_metrics.append(
            {
                "model_id": resolved_model_id,
                "model_name": model_name,
                "fold": fold_index,
                "metric_name": primary_metric,
                "metric_value": fold_score,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
    if task_type == "regression" and primary_metric == "rmsle":
        mean_test_predictions = np.clip(mean_test_predictions, a_min=0.0, a_max=None)
    if task_type == "binary" and binary_prediction_kind == "label":
        if positive_label is None or negative_label is None:
            raise ValueError("Binary label exports require resolved class metadata.")
        final_test_predictions = np.where(mean_test_predictions >= 0.5, positive_label, negative_label)
    else:
        final_test_predictions = mean_test_predictions

    fold_metrics_df = pd.DataFrame(fold_metrics)
    return ModelEvaluationArtifacts(
        model_result=ModelRunResult(
            model_id=resolved_model_id,
            model_name=model_name,
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
        oof_predictions=oof_predictions,
        final_test_predictions=np.asarray(final_test_predictions),
    )


def _train_single_model(
    task_type: str,
    primary_metric: str,
    model_spec: TrainingModelSpec,
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    y_train: pd.Series,
    test_ids: pd.Series,
    id_column: str,
    label_column: str,
    split_indices: list[tuple[int, np.ndarray, np.ndarray]],
    fold_assignments: np.ndarray,
    run_dir: Path,
    force_categorical: list[str] | None,
    force_numeric: list[str] | None,
    low_cardinality_int_threshold: int | None,
    cv_random_state: int,
    positive_label: object | None,
    negative_label: object | None,
) -> ModelRunResult:
    evaluation_artifacts = _evaluate_model_spec(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=model_spec,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        y_train=y_train,
        split_indices=split_indices,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        low_cardinality_int_threshold=low_cardinality_int_threshold,
        cv_random_state=cv_random_state,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    model_result = evaluation_artifacts.model_result

    model_dir = run_dir / model_result.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_artifacts.fold_metrics_df.to_csv(model_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(x_train_raw.shape[0], dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": evaluation_artifacts.oof_predictions,
            "fold": fold_assignments,
            "model_id": model_result.model_id,
            "model_name": model_result.model_name,
        }
    )
    oof_df.to_csv(model_dir / "oof_predictions.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            label_column: evaluation_artifacts.final_test_predictions,
        }
    )
    test_predictions_df.to_csv(model_dir / "test_predictions.csv", index=False)

    return model_result


def _rank_model_results(
    model_results: list[ModelRunResult],
    configured_model_ids: list[str],
) -> tuple[list[ModelRunResult], ModelRunResult]:
    model_order = {model_id: index for index, model_id in enumerate(configured_model_ids)}

    sorted_results = sorted(
        model_results,
        key=lambda result: (
            -result.cv_summary.metric_mean if result.cv_summary.higher_is_better else result.cv_summary.metric_mean,
            model_order[result.model_id],
        ),
    )

    ranked_results_by_model_id: dict[str, ModelRunResult] = {}
    for rank, result in enumerate(sorted_results, start=1):
        ranked_results_by_model_id[result.model_id] = replace(
            result,
            rank=rank,
            is_best_model=rank == 1,
        )

    ranked_results = [ranked_results_by_model_id[result.model_id] for result in model_results]
    best_model_result = ranked_results_by_model_id[sorted_results[0].model_id]
    return ranked_results, best_model_result


def _build_model_summary_rows(
    model_results: list[ModelRunResult],
) -> list[dict[str, object]]:
    ranked_results = sorted(model_results, key=lambda result: result.require_rank())
    return [result.to_model_summary_row() for result in ranked_results]


def _build_run_manifest(
    run_context: TrainingRunContext,
) -> dict[str, object]:
    model_ids = [result.model_id for result in run_context.model_results]
    models = [result.to_manifest_model_entry() for result in run_context.model_results]
    run_manifest = {
        "run_id": run_context.run_id,
        "generated_at_utc": run_context.generated_at_utc,
        "competition_slug": run_context.competition_slug,
        "task_type": run_context.task_type,
        "primary_metric": run_context.primary_metric,
        "config_fingerprint": run_context.config_fingerprint,
        "config_snapshot": run_context.config_snapshot,
        "model_ids": model_ids,
        "best_model_id": run_context.best_model_result.model_id,
        "models": models,
        "observed_label_pair": list(run_context.observed_label_pair) if run_context.observed_label_pair is not None else None,
        "negative_label": run_context.negative_label,
        "positive_label": run_context.positive_label,
        "id_column": run_context.id_column,
        "label_column": run_context.label_column,
        "target_summary": run_context.target_summary,
        "train_rows": run_context.train_rows,
        "train_cols": run_context.train_cols,
        "test_rows": run_context.test_rows,
        "test_cols": run_context.test_cols,
        "tuning_provenance": run_context.tuning_provenance,
    }

    if len(models) == 1:
        single_model = models[0]
        run_manifest["model_id"] = single_model["model_id"]
        run_manifest["model_name"] = single_model["model_name"]
        run_manifest["preprocessing_scheme_id"] = single_model["preprocessing_scheme_id"]
        run_manifest["model_params"] = single_model["model_params"]
        run_manifest["cv_summary"] = single_model["cv_summary"]

    return run_manifest


def _build_run_ledger_row(
    run_context: TrainingRunContext,
) -> dict[str, object]:
    ledger_row = {
        "run_id": run_context.run_id,
        "timestamp_utc": run_context.generated_at_utc,
        "competition_slug": run_context.competition_slug,
        "task_type": run_context.task_type,
        "primary_metric": run_context.primary_metric,
        "best_model_id": run_context.best_model_result.model_id,
        "best_model_name": run_context.best_model_result.model_name,
        "cv_mean": run_context.best_model_result.cv_summary.metric_mean,
        "cv_std": run_context.best_model_result.cv_summary.metric_std,
        "higher_is_better": run_context.best_model_result.cv_summary.higher_is_better,
        "model_count": len(run_context.model_results),
        "cv_n_splits": run_context.cv_n_splits,
        "cv_shuffle": run_context.cv_shuffle,
        "cv_random_state": run_context.cv_random_state,
        "config_fingerprint": run_context.config_fingerprint,
    }
    ledger_row.update(run_context.target_summary)
    return ledger_row


def _build_config_snapshot(
    config: AppConfig,
    model_specs: list[TrainingModelSpec],
    positive_label: object | None,
    id_column: str,
    label_column: str,
    tuning_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    config_snapshot = {
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "model_ids": [model_spec.model_id for model_spec in model_specs],
        "positive_label": positive_label,
        "id_column": id_column,
        "label_column": label_column,
        "force_categorical": config.force_categorical,
        "force_numeric": config.force_numeric,
        "drop_columns": config.drop_columns,
        "low_cardinality_int_threshold": config.low_cardinality_int_threshold,
        "cv_n_splits": config.cv_n_splits,
        "cv_shuffle": config.cv_shuffle,
        "cv_random_state": config.cv_random_state,
    }
    parameter_overrides = {
        model_spec.model_id: model_spec.parameter_overrides
        for model_spec in model_specs
        if model_spec.parameter_overrides
    }
    if parameter_overrides:
        config_snapshot["model_parameter_overrides"] = parameter_overrides
    if tuning_provenance is not None:
        config_snapshot["tuning_provenance"] = tuning_provenance
    return config_snapshot


def _build_config_fingerprint(
    config_snapshot: dict[str, object],
    model_results: list[ModelRunResult],
) -> str:
    fingerprint_payload = {
        "config_snapshot": config_snapshot,
        "models": [result.to_fingerprint_model_entry() for result in model_results],
    }
    config_snapshot_json = json.dumps(_json_ready(fingerprint_payload), sort_keys=True)
    return hashlib.sha256(config_snapshot_json.encode("utf-8")).hexdigest()[:12]


def run_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    model_specs: list[TrainingModelSpec] | None = None,
    tuning_provenance: dict[str, object] | None = None,
) -> Path:
    competition_slug = config.competition_slug
    task_type = config.task_type
    primary_metric = config.primary_metric
    positive_label = config.positive_label
    resolved_model_specs = _resolve_training_model_specs(config=config, model_specs=model_specs)
    configured_model_ids = [model_spec.model_id for model_spec in resolved_model_specs]

    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )

    observed_label_pair = None
    negative_label = None
    if task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    split_indices = _materialize_split_indices(
        task_type=task_type,
        x_train_raw=x_train_raw,
        y_train=y_train,
        n_splits=config.cv_n_splits,
        shuffle=config.cv_shuffle,
        random_state=config.cv_random_state,
    )
    fold_assignments = _build_fold_assignments(x_train_raw.shape[0], split_indices)
    run_diagnostics_df = _build_run_diagnostics(
        task_type=task_type,
        y_train=y_train,
        split_indices=split_indices,
        positive_label=positive_label,
    )
    target_summary = _build_target_summary(
        task_type=task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )

    run_id = _make_run_id()
    run_dir = Path("artifacts") / competition_slug / "train" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_diagnostics_df.to_csv(run_dir / "run_diagnostics.csv", index=False)

    model_results: list[ModelRunResult] = []
    for model_spec in resolved_model_specs:
        model_result = _train_single_model(
            task_type=task_type,
            primary_metric=primary_metric,
            model_spec=model_spec,
            x_train_raw=x_train_raw,
            x_test_raw=x_test_raw,
            y_train=y_train,
            test_ids=test_df[id_column],
            id_column=id_column,
            label_column=label_column,
            split_indices=split_indices,
            fold_assignments=fold_assignments,
            run_dir=run_dir,
            force_categorical=config.force_categorical,
            force_numeric=config.force_numeric,
            low_cardinality_int_threshold=config.low_cardinality_int_threshold,
            cv_random_state=config.cv_random_state,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        model_results.append(model_result)
        print(
            f"Training model: {model_result.model_id} ({model_result.model_name}) | "
            f"preprocessing={model_result.preprocessing_scheme_id} | "
            f"CV {primary_metric}: mean={model_result.cv_summary.metric_mean:.6f}, "
            f"std={model_result.cv_summary.metric_std:.6f}"
        )

    model_results, best_model_result = _rank_model_results(model_results, configured_model_ids)
    model_summary_rows = _build_model_summary_rows(model_results)
    model_summary_df = pd.DataFrame(model_summary_rows)
    model_summary_df.to_csv(run_dir / "model_summary.csv", index=False)

    config_snapshot = _build_config_snapshot(
        config=config,
        model_specs=resolved_model_specs,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        tuning_provenance=tuning_provenance,
    )
    config_fingerprint = _build_config_fingerprint(
        config_snapshot=config_snapshot,
        model_results=model_results,
    )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    run_context = TrainingRunContext(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        competition_slug=competition_slug,
        task_type=task_type,
        primary_metric=primary_metric,
        config_snapshot=config_snapshot,
        config_fingerprint=config_fingerprint,
        model_results=model_results,
        best_model_result=best_model_result,
        observed_label_pair=observed_label_pair,
        negative_label=negative_label,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        target_summary=target_summary,
        train_rows=int(x_train_raw.shape[0]),
        train_cols=int(x_train_raw.shape[1]),
        test_rows=int(x_test_raw.shape[0]),
        test_cols=int(x_test_raw.shape[1]),
        cv_n_splits=config.cv_n_splits,
        cv_shuffle=config.cv_shuffle,
        cv_random_state=config.cv_random_state,
        tuning_provenance=tuning_provenance,
    )
    run_manifest = _build_run_manifest(run_context)
    run_manifest_json = json.dumps(_json_ready(run_manifest), indent=2)
    (run_dir / "run_manifest.json").write_text(run_manifest_json, encoding="utf-8")

    ledger_row = _build_run_ledger_row(run_context)
    ledger_path = Path("artifacts") / competition_slug / "train" / "runs.csv"
    _append_run_ledger(ledger_path, ledger_row)

    print(
        f"Best model: {best_model_result.model_id} ({best_model_result.model_name}) | "
        f"CV {primary_metric}: mean={best_model_result.cv_summary.metric_mean:.6f}, "
        f"std={best_model_result.cv_summary.metric_std:.6f}"
    )

    return run_dir
