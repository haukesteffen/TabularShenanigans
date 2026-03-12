import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from tabular_shenanigans.candidate_artifacts import load_candidate_manifest
from tabular_shenanigans.config import AppConfig


def is_tracking_enabled(config: AppConfig) -> bool:
    return config.tracking_enabled


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
    run_name_parts = [stage, config.competition_slug]
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
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.tracking_experiment_name)
    mlflow.start_run(run_name=_build_stage_run_name(config=config, stage=stage, extra_tags=extra_tags))

    base_tags = {
        "app": "tabular_shenanigans",
        "tracking_schema_version": "1",
        "pipeline_invocation_id": pipeline_invocation_id,
        "stage": stage,
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
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
        "model_id": manifest.get("model_id"),
        "model_name": manifest.get("model_name"),
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


def log_submit_outputs(
    competition_slug: str,
    candidate_id: str,
    submission_path: Path,
    submission_status: str,
    message: str,
    submit_enabled: bool,
) -> None:
    mlflow = _load_mlflow()
    candidate_dir = submission_path.parent
    manifest = load_candidate_manifest(candidate_dir_path=candidate_dir)
    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError(f"Candidate manifest cv_summary must be a mapping: {candidate_dir / 'candidate.json'}")

    submit_tags = {
        "competition_slug": competition_slug,
        "candidate_id": candidate_id,
        "candidate_type": manifest.get("candidate_type"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "model_id": manifest.get("model_id"),
        "model_name": manifest.get("model_name"),
        "submission_status": submission_status,
        "cv_metric_name": cv_summary.get("metric_name"),
    }
    mlflow.set_tags(_coerce_tags(submit_tags))
    if cv_summary.get("metric_mean") is not None:
        mlflow.log_metric("candidate_cv_metric_mean", float(cv_summary["metric_mean"]))
    if cv_summary.get("metric_std") is not None:
        mlflow.log_metric("candidate_cv_metric_std", float(cv_summary["metric_std"]))

    submission_event = {
        "competition_slug": competition_slug,
        "candidate_id": candidate_id,
        "model_id": manifest.get("model_id"),
        "model_name": manifest.get("model_name"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "submission_filename": submission_path.name,
        "submit_enabled": submit_enabled,
        "status": submission_status,
        "message": message,
    }
    mlflow.log_artifact(str(submission_path), "submission")
    mlflow.log_dict(submission_event, "submission/submission_event.json")
