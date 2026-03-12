import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tabular_shenanigans.candidate_artifacts import (
    CANDIDATE_ARTIFACT_DIRNAME,
    build_base_config_snapshot,
    build_binary_accuracy_artifact_metadata,
    build_config_fingerprint,
    json_ready,
    write_candidate_artifacts,
    write_context_artifacts,
)
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.mlflow_store import (
    CandidateRunRef,
    create_candidate_run,
    log_candidate_run,
    terminate_run,
)
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
        model_registry_key=config.resolved_model_registry_key,
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
    config_snapshot["resolved_model_registry_key"] = model_spec.model_registry_key
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
    mlflow_run_id: str,
) -> dict[str, object]:
    competition = config.competition
    candidate = config.experiment.candidate
    manifest = {
        "artifact_type": "candidate",
        "candidate_id": config.resolved_candidate_id,
        "candidate_type": config.experiment.candidate.candidate_type,
        "generated_at_utc": generated_at_utc,
        "competition_slug": competition.slug,
        "task_type": competition.task_type,
        "primary_metric": competition.primary_metric,
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "mlflow_run_id": mlflow_run_id,
        "model_family": candidate.model_family,
        "feature_recipe_id": candidate.feature_recipe_id,
        "feature_columns": training_context.x_train_features.columns.tolist(),
        "numeric_preprocessor": candidate.numeric_preprocessor,
        "categorical_preprocessor": candidate.categorical_preprocessor,
        "model_registry_key": model_result.model_registry_key,
        "estimator_name": model_result.estimator_name,
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
    candidate_artifact_dir: Path,
    optimization_artifacts: OptimizationArtifacts,
) -> None:
    (candidate_artifact_dir / "optimization_summary.json").write_text(
        json.dumps(json_ready(optimization_artifacts.summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    optimization_artifacts.trials_df.to_csv(candidate_artifact_dir / "optimization_trials.csv", index=False)

    best_params = optimization_artifacts.summary.get("best_params")
    if isinstance(best_params, dict):
        (candidate_artifact_dir / "optimization_best_params.json").write_text(
            json.dumps(json_ready(best_params), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_runtime_config(bundle_root: Path, config: AppConfig) -> None:
    config_dir = bundle_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "runtime_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _stage_candidate_bundle(
    bundle_root: Path,
    config: AppConfig,
    candidate_manifest: dict[str, object],
    training_context: PreparedTrainingContext,
    dataset_context: CompetitionDatasetContext,
    evaluation_artifacts,
    optimization_artifacts: OptimizationArtifacts | None = None,
) -> None:
    _write_runtime_config(bundle_root=bundle_root, config=config)
    write_context_artifacts(
        bundle_root=bundle_root,
        competition_manifest=training_context.competition_manifest,
        fold_assignments=training_context.fold_assignments,
    )
    candidate_artifact_dir = bundle_root / CANDIDATE_ARTIFACT_DIRNAME
    write_candidate_artifacts(
        candidate_artifact_dir=candidate_artifact_dir,
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
    if optimization_artifacts is not None:
        _write_optimization_artifacts(
            candidate_artifact_dir=candidate_artifact_dir,
            optimization_artifacts=optimization_artifacts,
        )


def run_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    model_spec: TrainingModelSpec | None = None,
    tuning_provenance: dict[str, object] | None = None,
    optimization_artifacts: OptimizationArtifacts | None = None,
    prepared_training_context: PreparedTrainingContext | None = None,
) -> CandidateRunRef:
    if not config.is_model_candidate:
        raise ValueError("run_training only supports experiment.candidate.candidate_type=model.")

    competition = config.competition
    candidate = config.experiment.candidate
    candidate_id = config.resolved_candidate_id
    resolved_model_spec = _resolve_training_model_spec(config=config, model_spec=model_spec)

    training_context = prepared_training_context
    if training_context is None:
        training_context = build_prepared_training_context(
            config=config,
            dataset_context=dataset_context,
        )

    fit_started = time.perf_counter()
    evaluation_artifacts = evaluate_model_spec(
        task_type=competition.task_type,
        primary_metric=competition.primary_metric,
        model_spec=resolved_model_spec,
        training_context=training_context,
        cv_random_state=competition.cv.random_state,
    )
    fit_wall_seconds = time.perf_counter() - fit_started
    model_result = evaluation_artifacts.model_result
    print(
        f"Training candidate: {candidate_id} | "
        f"feature_recipe={candidate.feature_recipe_id} | "
        f"estimator={model_result.estimator_name} | "
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
    candidate_run = create_candidate_run(
        config=config,
        candidate_id=candidate_id,
        candidate_type=candidate.candidate_type,
    )
    candidate_manifest = _build_candidate_manifest(
        config=config,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        model_result=model_result,
        config_snapshot=config_snapshot,
        config_fingerprint=config_fingerprint,
        training_context=training_context,
        tuning_provenance=tuning_provenance,
        mlflow_run_id=candidate_run.run_id,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-candidate-") as temp_dir:
            bundle_root = Path(temp_dir)
            _stage_candidate_bundle(
                bundle_root=bundle_root,
                config=config,
                candidate_manifest=candidate_manifest,
                training_context=training_context,
                dataset_context=dataset_context,
                evaluation_artifacts=evaluation_artifacts,
                optimization_artifacts=optimization_artifacts,
            )
            optimization_summary = optimization_artifacts.summary if optimization_artifacts is not None else None
            log_candidate_run(
                config=config,
                candidate_run=candidate_run,
                bundle_root=bundle_root,
                manifest=candidate_manifest,
                fit_wall_seconds=fit_wall_seconds,
                optimization_summary=optimization_summary,
            )
        terminate_run(config=config, run_id=candidate_run.run_id, status="FINISHED")
        return candidate_run
    except Exception:
        terminate_run(config=config, run_id=candidate_run.run_id, status="FAILED")
        raise


def run_training_workflow(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> CandidateRunRef:
    competition = config.competition
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
    candidate_run = run_training(
        config=config,
        dataset_context=dataset_context,
        model_spec=optimization_result.best_model_spec,
        tuning_provenance=optimization_result.tuning_provenance,
        optimization_artifacts=OptimizationArtifacts(
            summary=optimization_result.optimization_summary,
            trials_df=optimization_result.trials_df,
        ),
        prepared_training_context=prepared_training_context,
    )
    print(
        f"Optimization complete: best_trial={optimization_result.best_trial_number}, "
        f"best_{competition.primary_metric}={optimization_result.best_value:.6f}, "
        f"candidate={candidate_run.candidate_id}"
    )
    return candidate_run
