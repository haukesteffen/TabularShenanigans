import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tabular_shenanigans.candidate_artifacts import (
    build_base_config_snapshot,
    build_binary_accuracy_artifact_metadata,
    build_config_fingerprint,
    candidate_dir as resolve_candidate_dir,
    json_ready,
    write_candidate_artifacts,
)
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.model_evaluation import (
    ModelRunResult,
    PreparedTrainingContext,
    TrainingModelSpec,
    build_prepared_training_context,
    evaluate_model_spec,
)


@dataclass(frozen=True)
class OptimizationArtifacts:
    summary: dict[str, object]
    trials_df: pd.DataFrame


def _resolve_training_model_spec(
    config: AppConfig,
    model_spec: TrainingModelSpec | None = None,
) -> TrainingModelSpec:
    if model_spec is not None:
        return model_spec
    candidate = config.experiment.candidate
    return TrainingModelSpec(
        model_id=config.resolved_model_id,
        parameter_overrides=candidate.model_params or None,
    )


def _build_config_snapshot(
    config: AppConfig,
    model_spec: TrainingModelSpec,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    tuning_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    candidate = config.experiment.candidate
    config_snapshot = build_base_config_snapshot(
        config=config,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
    )
    config_snapshot["resolved_model_id"] = model_spec.model_id
    config_snapshot["resolved_numeric_preprocessor"] = candidate.numeric_preprocessor
    config_snapshot["resolved_categorical_preprocessor"] = candidate.categorical_preprocessor
    config_snapshot["resolved_preprocessing_scheme_id"] = candidate.preprocessing_scheme_id
    if model_spec.parameter_overrides:
        config_snapshot["resolved_model_parameter_overrides"] = model_spec.parameter_overrides
    if tuning_provenance is not None:
        config_snapshot["tuning_provenance"] = tuning_provenance
    return config_snapshot


def _build_config_fingerprint(
    config_snapshot: dict[str, object],
    model_result: ModelRunResult,
) -> str:
    return build_config_fingerprint(
        {
            "config_snapshot": config_snapshot,
            "model": model_result.to_fingerprint_entry(),
        }
    )


def _build_candidate_manifest(
    config: AppConfig,
    generated_at_utc: str,
    model_result: ModelRunResult,
    config_snapshot: dict[str, object],
    config_fingerprint: str,
    training_context: PreparedTrainingContext,
    tuning_provenance: dict[str, object] | None,
) -> dict[str, object]:
    competition = config.competition
    candidate = config.experiment.candidate
    manifest = {
        "artifact_type": "candidate",
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "generated_at_utc": generated_at_utc,
        "competition_slug": competition.slug,
        "task_type": competition.task_type,
        "primary_metric": competition.primary_metric,
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "model_family": candidate.model_family,
        "feature_recipe_id": candidate.feature_recipe_id,
        "feature_columns": training_context.x_train_features.columns.tolist(),
        "numeric_preprocessor": candidate.numeric_preprocessor,
        "categorical_preprocessor": candidate.categorical_preprocessor,
        "model_id": model_result.model_id,
        "model_name": model_result.model_name,
        "preprocessing_scheme_id": model_result.preprocessing_scheme_id,
        "model_params": model_result.model_params,
        "cv_summary": model_result.cv_summary.to_dict(),
        "observed_label_pair": (
            list(training_context.observed_label_pair)
            if training_context.observed_label_pair is not None
            else None
        ),
        "negative_label": training_context.negative_label,
        "positive_label": training_context.positive_label,
        "id_column": training_context.id_column,
        "label_column": training_context.label_column,
        "target_summary": training_context.target_summary,
        "train_rows": int(training_context.x_train_features.shape[0]),
        "train_cols": int(training_context.x_train_features.shape[1]),
        "test_rows": int(training_context.x_test_features.shape[0]),
        "test_cols": int(training_context.x_test_features.shape[1]),
        "tuning_provenance": tuning_provenance,
    }
    manifest.update(
        build_binary_accuracy_artifact_metadata(
            task_type=competition.task_type,
            primary_metric=competition.primary_metric,
        )
    )
    return manifest


def _write_optimization_artifacts(
    candidate_dir: Path,
    optimization_artifacts: OptimizationArtifacts,
) -> None:
    (candidate_dir / "optimization_summary.json").write_text(
        json.dumps(json_ready(optimization_artifacts.summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    optimization_artifacts.trials_df.to_csv(candidate_dir / "optimization_trials.csv", index=False)

    best_params = optimization_artifacts.summary.get("best_params")
    if isinstance(best_params, dict):
        (candidate_dir / "optimization_best_params.json").write_text(
            json.dumps(json_ready(best_params), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    model_spec: TrainingModelSpec | None = None,
    tuning_provenance: dict[str, object] | None = None,
    prepared_training_context: PreparedTrainingContext | None = None,
) -> Path:
    if not config.is_model_candidate:
        raise ValueError("run_training only supports experiment.candidate.candidate_type=model.")

    competition = config.competition
    candidate = config.experiment.candidate
    resolved_model_spec = _resolve_training_model_spec(config=config, model_spec=model_spec)
    candidate_dir = resolve_candidate_dir(competition.slug, candidate.candidate_id)
    if candidate_dir.exists():
        raise ValueError(
            "Candidate artifacts already exist for this candidate_id. "
            f"Choose a new experiment.candidate.candidate_id or remove {candidate_dir}"
        )

    training_context = prepared_training_context
    if training_context is None:
        training_context = build_prepared_training_context(
            config=config,
            dataset_context=dataset_context,
        )

    evaluation_artifacts = evaluate_model_spec(
        task_type=competition.task_type,
        primary_metric=competition.primary_metric,
        model_spec=resolved_model_spec,
        training_context=training_context,
        cv_random_state=competition.cv.random_state,
    )
    model_result = evaluation_artifacts.model_result
    print(
        f"Training candidate: {candidate.candidate_id} | "
        f"feature_recipe={candidate.feature_recipe_id} | "
        f"model={model_result.model_id} ({model_result.model_name}) | "
        f"preprocessing={model_result.preprocessing_scheme_id} | "
        f"features={training_context.x_train_features.shape[1]} | "
        f"CV {competition.primary_metric}: mean={model_result.cv_summary.metric_mean:.6f}, "
        f"std={model_result.cv_summary.metric_std:.6f}"
    )

    config_snapshot = _build_config_snapshot(
        config=config,
        model_spec=resolved_model_spec,
        positive_label=training_context.positive_label,
        id_column=training_context.id_column,
        label_column=training_context.label_column,
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
        training_context=training_context,
        tuning_provenance=tuning_provenance,
    )

    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=False, exist_ok=False)
    write_candidate_artifacts(
        candidate_dir_path=candidate_dir,
        manifest=candidate_manifest,
        fold_metrics_df=evaluation_artifacts.fold_metrics_df,
        y_train=training_context.y_train,
        oof_predictions=evaluation_artifacts.oof_predictions,
        fold_assignments=training_context.fold_assignments,
        test_ids=dataset_context.test_df[training_context.id_column],
        test_predictions=evaluation_artifacts.final_test_predictions,
        id_column=training_context.id_column,
        label_column=training_context.label_column,
        test_prediction_probabilities=evaluation_artifacts.test_prediction_probabilities,
    )
    return candidate_dir


def run_training_workflow(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> Path:
    competition = config.competition
    candidate = config.experiment.candidate
    candidate_dir = resolve_candidate_dir(competition.slug, candidate.candidate_id)
    if candidate_dir.exists():
        raise ValueError(
            "Candidate artifacts already exist for this candidate_id. "
            f"Choose a new experiment.candidate.candidate_id or remove {candidate_dir}"
        )

    if config.is_blend_candidate:
        from tabular_shenanigans.blend import run_blend_training

        return run_blend_training(config=config, dataset_context=dataset_context)

    optimization = config.experiment.candidate.optimization
    if not optimization.enabled:
        return run_training(config=config, dataset_context=dataset_context)

    from tabular_shenanigans.tune import run_optimization

    prepared_training_context = build_prepared_training_context(
        config=config,
        dataset_context=dataset_context,
    )
    optimization_result = run_optimization(
        config=config,
        dataset_context=dataset_context,
        prepared_training_context=prepared_training_context,
    )
    candidate_dir = run_training(
        config=config,
        dataset_context=dataset_context,
        model_spec=optimization_result.best_model_spec,
        tuning_provenance=optimization_result.tuning_provenance,
        prepared_training_context=prepared_training_context,
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
        f"best_{competition.primary_metric}={optimization_result.best_value:.6f}, "
        f"candidate={candidate_dir.name}"
    )
    return candidate_dir
