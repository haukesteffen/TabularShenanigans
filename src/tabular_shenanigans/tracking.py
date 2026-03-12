import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from tabular_shenanigans.candidate_artifacts import json_ready, load_candidate_manifest
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.submission_history import SubmissionRefreshResult

StageResult = TypeVar("StageResult")


def make_pipeline_invocation_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking requires the optional tracking dependencies. "
            "Install them with `uv sync --extra tracking`."
        ) from exc

    return mlflow


def _coerce_tags(tags: Mapping[str, object]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in tags.items()
        if value is not None
    }


def _build_stage_run_name(
    config: AppConfig,
    stage: str,
    extra_tags: Mapping[str, object] | None = None,
) -> str:
    run_name_parts = [stage, config.competition.slug]
    if extra_tags is not None and extra_tags.get("candidate_id") is not None:
        run_name_parts.append(str(extra_tags["candidate_id"]))
    return " | ".join(run_name_parts)


@contextmanager
def start_stage_run(
    config: AppConfig,
    stage: str,
    pipeline_invocation_id: str,
    extra_tags: Mapping[str, object] | None = None,
) -> Iterator[None]:
    mlflow = _load_mlflow()
    tracking = config.experiment.tracking
    if tracking.tracking_uri is None or tracking.experiment_name is None:
        raise ValueError("Tracking run setup requires experiment.tracking.enabled=true with both tracking fields set.")
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(tracking.experiment_name)
    mlflow.start_run(run_name=_build_stage_run_name(config=config, stage=stage, extra_tags=extra_tags))

    competition = config.competition
    base_tags = {
        "app": "tabular_shenanigans",
        "tracking_schema_version": "1",
        "pipeline_invocation_id": pipeline_invocation_id,
        "stage": stage,
        "competition_slug": competition.slug,
        "task_type": competition.task_type,
        "primary_metric": competition.primary_metric,
        "local_experiment_name": config.experiment.name,
    }
    mlflow.set_tags(_coerce_tags(base_tags))
    if config.experiment.notes:
        mlflow.set_tag("mlflow.note.content", config.experiment.notes)
    if extra_tags:
        mlflow.set_tags(_coerce_tags(extra_tags))

    try:
        yield
    except Exception:
        mlflow.set_tag("run_outcome", "failed")
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.set_tag("run_outcome", "succeeded")
        mlflow.end_run(status="FINISHED")


def build_train_tracking_tags(config: AppConfig) -> dict[str, object]:
    candidate = config.experiment.candidate
    tags: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
    }
    if config.is_model_candidate:
        tags["model_registry_key"] = config.resolved_model_registry_key
        tags["model_family"] = candidate.model_family
        tags["feature_recipe_id"] = candidate.feature_recipe_id
        tags["numeric_preprocessor"] = candidate.numeric_preprocessor
        tags["categorical_preprocessor"] = candidate.categorical_preprocessor
        tags["preprocessing_scheme_id"] = candidate.preprocessing_scheme_id
        return tags

    tags["base_candidate_count"] = len(candidate.base_candidate_ids)
    return tags


def run_stage_with_tracking(
    config: AppConfig,
    stage: str,
    pipeline_invocation_id: str,
    stage_fn: Callable[[], StageResult],
    extra_tags: Mapping[str, object] | None = None,
    result_logger: Callable[[StageResult], None] | None = None,
) -> StageResult:
    tracking_enabled = config.experiment.tracking.enabled
    tracking_context = nullcontext()
    if tracking_enabled:
        tracking_context = start_stage_run(
            config=config,
            stage=stage,
            pipeline_invocation_id=pipeline_invocation_id,
            extra_tags=extra_tags,
        )

    with tracking_context:
        if tracking_enabled:
            log_runtime_config(config)
        stage_result = stage_fn()
        if tracking_enabled and result_logger is not None:
            result_logger(stage_result)
        return stage_result


def log_runtime_config(config: AppConfig) -> None:
    mlflow = _load_mlflow()
    mlflow.log_dict(config.model_dump(mode="json"), "config/runtime_config.json")


def log_prepare_outputs(prepared_context) -> None:
    mlflow = _load_mlflow()
    manifest = prepared_context.manifest
    mlflow.log_metric("train_rows", float(manifest["train_rows"]))
    mlflow.log_metric("test_rows", float(manifest["test_rows"]))
    mlflow.log_metric("model_feature_count", float(len(manifest["feature_columns"])))
    mlflow.log_artifact(str(prepared_context.manifest_path), "prepared_context")
    mlflow.log_artifact(str(prepared_context.folds_path), "prepared_context")
    mlflow.log_artifacts(str(prepared_context.report_dir), "reports")


def log_prepare_stage_outputs(stage_result: tuple[object, object]) -> None:
    _, prepared_context = stage_result
    log_prepare_outputs(prepared_context)


def log_train_outputs(candidate_dir: Path) -> None:
    mlflow = _load_mlflow()
    manifest = load_candidate_manifest(candidate_dir_path=candidate_dir)
    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError(f"Candidate manifest cv_summary must be a mapping: {candidate_dir / 'candidate.json'}")

    candidate_tags = {
        "candidate_id": manifest.get("candidate_id"),
        "candidate_type": manifest.get("candidate_type"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "model_registry_key": manifest.get("model_registry_key"),
        "estimator_name": manifest.get("estimator_name"),
        "model_family": manifest.get("model_family"),
        "feature_recipe_id": manifest.get("feature_recipe_id"),
        "preprocessing_scheme_id": manifest.get("preprocessing_scheme_id"),
        "cv_metric_name": cv_summary.get("metric_name"),
    }
    mlflow.set_tags(_coerce_tags(candidate_tags))

    metric_values = {
        "cv_metric_mean": cv_summary.get("metric_mean"),
        "cv_metric_std": cv_summary.get("metric_std"),
        "train_rows": manifest.get("train_rows"),
        "train_cols": manifest.get("train_cols"),
        "test_rows": manifest.get("test_rows"),
        "test_cols": manifest.get("test_cols"),
    }
    for metric_name, metric_value in metric_values.items():
        if metric_value is None:
            continue
        mlflow.log_metric(metric_name, float(metric_value))

    optimization_summary_path = candidate_dir / "optimization_summary.json"
    if optimization_summary_path.exists():
        optimization_summary = json.loads(optimization_summary_path.read_text(encoding="utf-8"))
        if isinstance(optimization_summary, dict):
            optimization_metrics = {
                "optimization_best_value": optimization_summary.get("best_value"),
                "optimization_trial_count": optimization_summary.get("trial_count"),
                "optimization_completed_trial_count": optimization_summary.get("completed_trial_count"),
                "optimization_best_trial_number": optimization_summary.get("best_trial_number"),
            }
            for metric_name, metric_value in optimization_metrics.items():
                if metric_value is None:
                    continue
                mlflow.log_metric(metric_name, float(metric_value))

    mlflow.log_artifacts(str(candidate_dir), "candidate")


def log_submit_outputs(submission_result) -> None:
    mlflow = _load_mlflow()
    submission_path = submission_result.submission_path
    candidate_dir = submission_path.parent
    manifest = load_candidate_manifest(candidate_dir_path=candidate_dir)
    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError(f"Candidate manifest cv_summary must be a mapping: {candidate_dir / 'candidate.json'}")

    submit_tags = {
        "competition_slug": manifest.get("competition_slug"),
        "candidate_id": manifest.get("candidate_id"),
        "candidate_type": manifest.get("candidate_type"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "model_registry_key": manifest.get("model_registry_key"),
        "estimator_name": manifest.get("estimator_name"),
        "submission_status": submission_result.submission_status,
        "cv_metric_name": cv_summary.get("metric_name"),
    }
    if submission_result.submission_event is not None:
        submit_tags["submission_event_id"] = submission_result.submission_event.submission_event_id
    mlflow.set_tags(_coerce_tags(submit_tags))
    if cv_summary.get("metric_mean") is not None:
        mlflow.log_metric("candidate_cv_metric_mean", float(cv_summary["metric_mean"]))
    if cv_summary.get("metric_std") is not None:
        mlflow.log_metric("candidate_cv_metric_std", float(cv_summary["metric_std"]))
    if submission_result.submission_refresh_result is not None:
        mlflow.log_metric(
            "submit_refresh_appended_observation_count",
            float(submission_result.submission_refresh_result.appended_observation_count),
        )
        mlflow.log_metric(
            "submit_refresh_matched_submission_event_count",
            float(submission_result.submission_refresh_result.matched_submission_event_count),
        )

    submit_summary = {
        "competition_slug": manifest.get("competition_slug"),
        "candidate_id": manifest.get("candidate_id"),
        "submission_filename": submission_path.name,
        "submission_status": submission_result.submission_status,
        "submission_message": submission_result.submission_message,
        "immediate_refresh_error": submission_result.immediate_refresh_error,
    }
    mlflow.log_artifact(str(submission_path), "submission")
    if (
        submission_result.submission_event_ledger_path is not None
        and submission_result.submission_event_ledger_path.exists()
    ):
        mlflow.log_artifact(str(submission_result.submission_event_ledger_path), "submission")
    if (
        submission_result.submission_refresh_result is not None
        and submission_result.submission_refresh_result.submission_score_ledger_path.exists()
    ):
        mlflow.log_artifact(str(submission_result.submission_refresh_result.submission_score_ledger_path), "submission")
    if submission_result.submission_event is not None:
        mlflow.log_dict(json_ready(asdict(submission_result.submission_event)), "submission/submission_event.json")
    if submission_result.submission_refresh_result is not None:
        mlflow.log_dict(json_ready(asdict(submission_result.submission_refresh_result)), "submission/submission_refresh.json")
    mlflow.log_dict(submit_summary, "submission/submission_summary.json")


def log_submission_refresh_outputs(refresh_result: SubmissionRefreshResult) -> None:
    mlflow = _load_mlflow()
    mlflow.set_tags(
        _coerce_tags(
            {
                "competition_slug": refresh_result.competition_slug,
                "submission_refresh_observation_source": refresh_result.observation_source,
            }
        )
    )
    mlflow.log_metric("tracked_submission_event_count", float(refresh_result.tracked_submission_event_count))
    mlflow.log_metric("matched_submission_event_count", float(refresh_result.matched_submission_event_count))
    mlflow.log_metric("appended_submission_score_count", float(refresh_result.appended_observation_count))
    mlflow.log_metric("scanned_remote_submission_count", float(refresh_result.scanned_remote_submission_count))
    if refresh_result.submission_score_ledger_path.exists():
        mlflow.log_artifact(str(refresh_result.submission_score_ledger_path), "submission_refresh")
    mlflow.log_dict(
        json_ready(asdict(refresh_result)),
        "submission_refresh/submission_refresh.json",
    )
