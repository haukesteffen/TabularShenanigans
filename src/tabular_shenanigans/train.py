import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.models import build_model, build_model_fit_kwargs
from tabular_shenanigans.preprocess import build_preprocessor, prepare_feature_frames


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

    def to_manifest_entry(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "preprocessing_scheme_id": self.preprocessing_scheme_id,
            "model_params": self.model_params,
            "cv_summary": self.cv_summary.to_dict(),
        }

    def to_fingerprint_entry(self) -> dict[str, object]:
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
class OptimizationArtifacts:
    summary: dict[str, object]
    trials_df: pd.DataFrame


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


def _resolve_training_model_spec(
    config: AppConfig,
    model_spec: TrainingModelSpec | None = None,
) -> TrainingModelSpec:
    if model_spec is not None:
        return model_spec
    return TrainingModelSpec(
        model_id=config.resolved_model_id,
        parameter_overrides=config.model_parameter_overrides,
    )


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


def _candidate_dir(competition_slug: str, candidate_id: str) -> Path:
    return Path("artifacts") / competition_slug / "candidates" / candidate_id


def _build_config_snapshot(
    config: AppConfig,
    model_spec: TrainingModelSpec,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    tuning_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    config_snapshot = {
        "competition": {
            **config.competition.model_dump(mode="python"),
            "primary_metric": config.primary_metric,
            "positive_label": positive_label,
            "id_column": id_column,
            "label_column": label_column,
        },
        "experiment": config.experiment.model_dump(mode="python"),
        "resolved_model_id": model_spec.model_id,
    }
    if model_spec.parameter_overrides:
        config_snapshot["resolved_model_parameter_overrides"] = model_spec.parameter_overrides
    if tuning_provenance is not None:
        config_snapshot["tuning_provenance"] = tuning_provenance
    return config_snapshot


def _build_config_fingerprint(
    config_snapshot: dict[str, object],
    model_result: ModelRunResult,
) -> str:
    fingerprint_payload = {
        "config_snapshot": config_snapshot,
        "model": model_result.to_fingerprint_entry(),
    }
    fingerprint_payload_json = json.dumps(_json_ready(fingerprint_payload), sort_keys=True)
    return hashlib.sha256(fingerprint_payload_json.encode("utf-8")).hexdigest()[:12]


def _build_candidate_manifest(
    config: AppConfig,
    generated_at_utc: str,
    model_result: ModelRunResult,
    config_snapshot: dict[str, object],
    config_fingerprint: str,
    observed_label_pair: tuple[object, object] | None,
    negative_label: object | None,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    target_summary: dict[str, object],
    train_rows: int,
    train_cols: int,
    test_rows: int,
    test_cols: int,
    tuning_provenance: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "artifact_type": "candidate",
        "candidate_id": config.candidate_id,
        "candidate_type": config.candidate_type,
        "generated_at_utc": generated_at_utc,
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "model_family": config.model_family,
        "preprocessor": config.preprocessor,
        "model_id": model_result.model_id,
        "model_name": model_result.model_name,
        "preprocessing_scheme_id": model_result.preprocessing_scheme_id,
        "model_params": model_result.model_params,
        "cv_summary": model_result.cv_summary.to_dict(),
        "observed_label_pair": list(observed_label_pair) if observed_label_pair is not None else None,
        "negative_label": negative_label,
        "positive_label": positive_label,
        "id_column": id_column,
        "label_column": label_column,
        "target_summary": target_summary,
        "train_rows": train_rows,
        "train_cols": train_cols,
        "test_rows": test_rows,
        "test_cols": test_cols,
        "tuning_provenance": tuning_provenance,
    }


def _write_candidate_artifacts(
    candidate_dir: Path,
    manifest: dict[str, object],
    evaluation_artifacts: ModelEvaluationArtifacts,
    y_train: pd.Series,
    fold_assignments: np.ndarray,
    test_ids: pd.Series,
    id_column: str,
    label_column: str,
) -> None:
    (candidate_dir / "candidate.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evaluation_artifacts.fold_metrics_df.to_csv(candidate_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(y_train.shape[0], dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": evaluation_artifacts.oof_predictions,
            "fold": fold_assignments,
        }
    )
    oof_df.to_csv(candidate_dir / "oof_predictions.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            label_column: evaluation_artifacts.final_test_predictions,
        }
    )
    test_predictions_df.to_csv(candidate_dir / "test_predictions.csv", index=False)


def _write_optimization_artifacts(
    candidate_dir: Path,
    optimization_artifacts: OptimizationArtifacts,
) -> None:
    (candidate_dir / "optimization_summary.json").write_text(
        json.dumps(_json_ready(optimization_artifacts.summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    optimization_artifacts.trials_df.to_csv(candidate_dir / "optimization_trials.csv", index=False)

    best_params = optimization_artifacts.summary.get("best_params")
    if isinstance(best_params, dict):
        (candidate_dir / "optimization_best_params.json").write_text(
            json.dumps(_json_ready(best_params), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    model_spec: TrainingModelSpec | None = None,
    tuning_provenance: dict[str, object] | None = None,
) -> Path:
    resolved_model_spec = _resolve_training_model_spec(config=config, model_spec=model_spec)
    candidate_dir = _candidate_dir(config.competition_slug, config.candidate_id)
    if candidate_dir.exists():
        raise ValueError(
            "Candidate artifacts already exist for this candidate_id. "
            f"Choose a new experiment.candidate.candidate_id or remove {candidate_dir}"
        )

    task_type = config.task_type
    primary_metric = config.primary_metric
    positive_label = config.positive_label

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

    prepared_context = ensure_prepared_competition_context(
        config=config,
        dataset_context=dataset_context,
        expected_feature_columns=x_train_raw.columns.tolist(),
    )
    split_indices = prepared_context.split_indices
    fold_assignments = prepared_context.fold_assignments
    target_summary = _build_target_summary(
        task_type=task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )

    evaluation_artifacts = _evaluate_model_spec(
        task_type=task_type,
        primary_metric=primary_metric,
        model_spec=resolved_model_spec,
        x_train_raw=x_train_raw,
        x_test_raw=x_test_raw,
        y_train=y_train,
        split_indices=split_indices,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        low_cardinality_int_threshold=config.low_cardinality_int_threshold,
        cv_random_state=config.cv_random_state,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    model_result = evaluation_artifacts.model_result
    print(
        f"Training candidate: {config.candidate_id} | "
        f"model={model_result.model_id} ({model_result.model_name}) | "
        f"preprocessing={model_result.preprocessing_scheme_id} | "
        f"CV {primary_metric}: mean={model_result.cv_summary.metric_mean:.6f}, "
        f"std={model_result.cv_summary.metric_std:.6f}"
    )

    config_snapshot = _build_config_snapshot(
        config=config,
        model_spec=resolved_model_spec,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        tuning_provenance=tuning_provenance,
    )
    config_fingerprint = _build_config_fingerprint(
        config_snapshot=config_snapshot,
        model_result=model_result,
    )
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    candidate_manifest = _build_candidate_manifest(
        config=config,
        generated_at_utc=generated_at_utc,
        model_result=model_result,
        config_snapshot=config_snapshot,
        config_fingerprint=config_fingerprint,
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
        tuning_provenance=tuning_provenance,
    )

    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=False, exist_ok=False)
    _write_candidate_artifacts(
        candidate_dir=candidate_dir,
        manifest=candidate_manifest,
        evaluation_artifacts=evaluation_artifacts,
        y_train=y_train,
        fold_assignments=fold_assignments,
        test_ids=test_df[id_column],
        id_column=id_column,
        label_column=label_column,
    )
    return candidate_dir


def run_training_workflow(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> Path:
    candidate_dir = _candidate_dir(config.competition_slug, config.candidate_id)
    if candidate_dir.exists():
        raise ValueError(
            "Candidate artifacts already exist for this candidate_id. "
            f"Choose a new experiment.candidate.candidate_id or remove {candidate_dir}"
        )

    optimization = config.experiment.candidate.optimization
    if not optimization.enabled:
        return run_training(config=config, dataset_context=dataset_context)

    from tabular_shenanigans.tune import run_optimization

    optimization_result = run_optimization(config=config, dataset_context=dataset_context)
    candidate_dir = run_training(
        config=config,
        dataset_context=dataset_context,
        model_spec=optimization_result.best_model_spec,
        tuning_provenance=optimization_result.tuning_provenance,
    )
    _write_optimization_artifacts(
        candidate_dir=candidate_dir,
        optimization_artifacts=OptimizationArtifacts(
            summary=optimization_result.optimization_summary,
            trials_df=optimization_result.trials_df,
        ),
    )
    print(
        f"Optimization complete: best_trial={optimization_result.best_trial_number}, "
        f"best_{config.primary_metric}={optimization_result.best_value:.6f}, "
        f"candidate={candidate_dir.name}"
    )
    return candidate_dir
