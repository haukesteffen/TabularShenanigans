import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabular_shenanigans.candidate_artifacts import (
    CANDIDATE_ARTIFACT_DIRNAME,
    CONTEXT_ARTIFACT_DIRNAME,
    json_ready,
    load_candidate_manifest,
)
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.submission_history import CandidateSubmissionHistory

TRACKING_SCHEMA_VERSION = "2"
RUN_KIND_CANDIDATE = "candidate"
SUBMISSION_HISTORY_ARTIFACT_PATH = "submissions/history.json"


@dataclass(frozen=True)
class CandidateRunRef:
    run_id: str
    experiment_id: str
    candidate_id: str


@dataclass(frozen=True)
class DownloadedCandidateBundle:
    run_id: str
    experiment_id: str
    candidate_id: str
    candidate_artifact_dir: Path
    manifest: dict[str, object]


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow support requires the project dependencies to be installed. "
            "Install them with `uv sync`."
        ) from exc
    return mlflow


def _client(config: AppConfig):
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(config.experiment.tracking.tracking_uri)
    return mlflow.tracking.MlflowClient()


def _experiment_id(config: AppConfig) -> str:
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(config.experiment.tracking.tracking_uri)
    experiment = mlflow.set_experiment(config.competition.slug)
    return experiment.experiment_id


def _git_output(args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_metadata() -> dict[str, str]:
    metadata: dict[str, str] = {}
    git_commit = _git_output(["rev-parse", "HEAD"])
    if git_commit is not None:
        metadata["git_commit"] = git_commit
    git_branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    if git_branch is not None:
        metadata["git_branch"] = git_branch
    return metadata


def _coerce_param_value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_ready(value), sort_keys=True)
    return str(value)


def _set_tags(client, run_id: str, tags: dict[str, object]) -> None:
    for key, value in tags.items():
        if value is None:
            continue
        client.set_tag(run_id, key, str(value))


def _log_params(client, run_id: str, params: dict[str, object]) -> None:
    for key, value in params.items():
        if value is None:
            continue
        client.log_param(run_id, key, _coerce_param_value(value))


def _log_metrics(client, run_id: str, metrics: dict[str, float | int | None]) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        client.log_metric(run_id, key, float(value))


def _candidate_search_filter(candidate_id: str) -> str:
    return (
        f"tags.run_kind = '{RUN_KIND_CANDIDATE}' "
        f"and tags.candidate_id = '{candidate_id}'"
    )


def find_candidate_run(config: AppConfig, candidate_id: str) -> CandidateRunRef:
    client = _client(config)
    experiment_id = _experiment_id(config)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=_candidate_search_filter(candidate_id),
        max_results=10,
    )
    if not runs:
        raise ValueError(
            f"Candidate '{candidate_id}' was not found in MLflow experiment '{config.competition.slug}'."
        )
    if len(runs) > 1:
        run_ids = [run.info.run_id for run in runs]
        raise ValueError(
            "Candidate id must map to exactly one MLflow run. "
            f"Candidate '{candidate_id}' matched runs: {run_ids}"
        )
    run = runs[0]
    return CandidateRunRef(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        candidate_id=candidate_id,
    )


def ensure_candidate_run_absent(config: AppConfig, candidate_id: str) -> None:
    client = _client(config)
    experiment_id = _experiment_id(config)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=_candidate_search_filter(candidate_id),
        max_results=1,
    )
    if runs:
        raise ValueError(
            "Candidate already exists in MLflow for this competition. "
            f"Change the candidate config so it derives a new candidate_id or delete candidate '{candidate_id}'."
        )


def _base_candidate_tags(config: AppConfig, candidate_id: str, candidate_type: str) -> dict[str, object]:
    runtime_execution_context = config.runtime_execution_context
    tags: dict[str, object] = {
        "run_kind": RUN_KIND_CANDIDATE,
        "tracking_schema_version": TRACKING_SCHEMA_VERSION,
        "competition_slug": config.competition.slug,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "task_type": config.competition.task_type,
        "primary_metric": config.competition.primary_metric,
        "runtime_requested_compute_target": runtime_execution_context.requested_compute_target,
        "runtime_resolved_compute_target": runtime_execution_context.resolved_compute_target,
        **_git_metadata(),
    }
    return tags


def create_candidate_run(
    config: AppConfig,
    candidate_id: str,
    candidate_type: str,
) -> CandidateRunRef:
    ensure_candidate_run_absent(config=config, candidate_id=candidate_id)
    client = _client(config)
    experiment_id = _experiment_id(config)
    run = client.create_run(
        experiment_id=experiment_id,
        tags=_base_candidate_tags(config=config, candidate_id=candidate_id, candidate_type=candidate_type),
        run_name=candidate_id,
    )
    return CandidateRunRef(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        candidate_id=candidate_id,
    )


def terminate_run(
    config: AppConfig,
    run_id: str,
    status: str,
) -> None:
    _client(config).set_terminated(run_id=run_id, status=status)


def _candidate_run_params(config: AppConfig, manifest: dict[str, object]) -> dict[str, object]:
    competition = config.competition
    candidate = config.experiment.candidate
    runtime_execution_context = config.runtime_execution_context
    params: dict[str, object] = {
        "cv__n_splits": competition.cv.n_splits,
        "cv__shuffle": competition.cv.shuffle,
        "cv__random_state": competition.cv.random_state,
        "runtime__requested_compute_target": runtime_execution_context.requested_compute_target,
        "runtime__resolved_compute_target": runtime_execution_context.resolved_compute_target,
        "runtime__gpu_available": runtime_execution_context.gpu_available,
    }
    if runtime_execution_context.fallback_reason is not None:
        params["runtime__fallback_reason"] = runtime_execution_context.fallback_reason
    if config.is_model_candidate:
        params.update(
            {
                "feature_recipe_id": manifest.get("feature_recipe_id"),
                "numeric_preprocessor": manifest.get("numeric_preprocessor"),
                "categorical_preprocessor": manifest.get("categorical_preprocessor"),
                "preprocessing_scheme_id": manifest.get("preprocessing_scheme_id"),
                "model_family": manifest.get("model_family"),
                "model_registry_key": manifest.get("model_registry_key"),
            }
        )
        model_params = manifest.get("model_params")
        if isinstance(model_params, dict):
            for key, value in model_params.items():
                params[f"model__{key}"] = value
        optimization = candidate.optimization
        params.update(
            {
                "opt__enabled": optimization.enabled,
                "opt__method": optimization.method,
                "opt__n_trials": optimization.n_trials,
                "opt__timeout_seconds": optimization.timeout_seconds,
                "opt__random_state": optimization.random_state,
            }
        )
        return params

    params["blend__base_candidate_ids_json"] = candidate.base_candidate_ids
    params["blend__configured_weights_json"] = candidate.weights
    return params


def _candidate_run_tags(config: AppConfig, manifest: dict[str, object]) -> dict[str, object]:
    tags = {
        "competition_slug": manifest.get("competition_slug"),
        "candidate_id": manifest.get("candidate_id"),
        "candidate_type": manifest.get("candidate_type"),
        "task_type": manifest.get("task_type"),
        "primary_metric": manifest.get("primary_metric"),
        "config_fingerprint": manifest.get("config_fingerprint"),
    }
    return tags


def _candidate_run_metrics(
    manifest: dict[str, object],
    fit_wall_seconds: float,
    optimization_summary: dict[str, object] | None = None,
) -> dict[str, float | int | None]:
    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError("Candidate manifest cv_summary must be a mapping.")
    metrics: dict[str, float | int | None] = {
        "cv_score_mean": cv_summary.get("metric_mean"),
        "cv_score_std": cv_summary.get("metric_std"),
        "train_rows": manifest.get("train_rows"),
        "test_rows": manifest.get("test_rows"),
        "feature_count": manifest.get("train_cols"),
        "fit_wall_seconds": fit_wall_seconds,
    }
    if optimization_summary is not None:
        metrics["optimization_best_value"] = optimization_summary.get("best_value")  # type: ignore[assignment]
        metrics["optimization_trial_count"] = optimization_summary.get("trial_count")  # type: ignore[assignment]
    return metrics


def log_candidate_run(
    config: AppConfig,
    candidate_run: CandidateRunRef,
    bundle_root: Path,
    manifest: dict[str, object],
    fit_wall_seconds: float,
    optimization_summary: dict[str, object] | None = None,
) -> None:
    client = _client(config)
    _set_tags(client, candidate_run.run_id, _candidate_run_tags(config=config, manifest=manifest))
    _log_params(client, candidate_run.run_id, _candidate_run_params(config=config, manifest=manifest))
    _log_metrics(
        client,
        candidate_run.run_id,
        _candidate_run_metrics(
            manifest=manifest,
            fit_wall_seconds=fit_wall_seconds,
            optimization_summary=optimization_summary,
        ),
    )
    client.log_artifacts(candidate_run.run_id, str(bundle_root / CONTEXT_ARTIFACT_DIRNAME), CONTEXT_ARTIFACT_DIRNAME)
    client.log_artifact(candidate_run.run_id, str(bundle_root / "config" / "runtime_config.json"), "config")
    client.log_artifacts(
        candidate_run.run_id,
        str(bundle_root / CANDIDATE_ARTIFACT_DIRNAME),
        CANDIDATE_ARTIFACT_DIRNAME,
    )


def upload_run_log(
    config: AppConfig,
    run_id: str,
    log_path: Path,
    artifact_dir: str = "logs",
) -> None:
    if not log_path.exists():
        raise ValueError(f"Runtime log artifact does not exist: {log_path}")
    _client(config).log_artifact(run_id, str(log_path), artifact_dir)


def download_candidate_bundle(
    config: AppConfig,
    candidate_id: str,
    destination_dir: Path,
) -> DownloadedCandidateBundle:
    candidate_run = find_candidate_run(config=config, candidate_id=candidate_id)
    client = _client(config)
    candidate_dir_path = Path(
        client.download_artifacts(
            run_id=candidate_run.run_id,
            path=CANDIDATE_ARTIFACT_DIRNAME,
            dst_path=str(destination_dir),
        )
    )
    manifest = load_candidate_manifest(candidate_artifact_dir=candidate_dir_path)
    return DownloadedCandidateBundle(
        run_id=candidate_run.run_id,
        experiment_id=candidate_run.experiment_id,
        candidate_id=candidate_id,
        candidate_artifact_dir=candidate_dir_path,
        manifest=manifest,
    )


def download_submission_history(
    config: AppConfig,
    run_id: str,
    destination_dir: Path,
) -> CandidateSubmissionHistory:
    client = _client(config)
    artifact_entries = client.list_artifacts(run_id, "submissions")
    if not any(entry.path == SUBMISSION_HISTORY_ARTIFACT_PATH for entry in artifact_entries):
        return CandidateSubmissionHistory.empty()

    history_path = Path(
        client.download_artifacts(
            run_id=run_id,
            path=SUBMISSION_HISTORY_ARTIFACT_PATH,
            dst_path=str(destination_dir),
        )
    )
    return CandidateSubmissionHistory.from_path(history_path)


def upload_submission_history(
    config: AppConfig,
    run_id: str,
    history: CandidateSubmissionHistory,
    updated_submission_event_ids: list[str] | None = None,
    submission_csv_paths: Mapping[str, Path] | None = None,
) -> None:
    client = _client(config)
    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-submission-history-") as temp_dir:
        temp_root = Path(temp_dir)
        submissions_dir = temp_root / "submissions"
        submissions_dir.mkdir(parents=True, exist_ok=True)
        history_path = submissions_dir / "history.json"
        history.write(history_path)
        client.log_artifact(run_id, str(history_path), "submissions")

        if not updated_submission_event_ids:
            return

        for submission_event_id in updated_submission_event_ids:
            event_dir = submissions_dir / submission_event_id
            event_dir.mkdir(parents=True, exist_ok=True)
            if submission_csv_paths is not None and submission_event_id in submission_csv_paths:
                client.log_artifact(
                    run_id,
                    str(submission_csv_paths[submission_event_id]),
                    f"submissions/{submission_event_id}",
                )
            event = history.get_event(submission_event_id)
            if event is None:
                raise ValueError(f"Submission history is missing event '{submission_event_id}'.")
            event_path = event_dir / "event.json"
            event_path.write_text(json.dumps(json_ready(event.to_dict()), indent=2, sort_keys=True), encoding="utf-8")
            client.log_artifact(run_id, str(event_path), f"submissions/{submission_event_id}")
            observations_path = event_dir / "observations.json"
            event_observations = [observation.to_dict() for observation in history.get_observations(submission_event_id)]
            observations_path.write_text(
                json.dumps(json_ready(event_observations), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            client.log_artifact(run_id, str(observations_path), f"submissions/{submission_event_id}")


def update_submission_metrics(
    config: AppConfig,
    run_id: str,
    score_metrics: dict[str, float | int | None],
) -> None:
    _log_metrics(_client(config), run_id, score_metrics)


def search_candidate_runs(config: AppConfig) -> list[CandidateRunRef]:
    client = _client(config)
    experiment_id = _experiment_id(config)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.run_kind = '{RUN_KIND_CANDIDATE}'",
        max_results=1000,
    )
    return [
        CandidateRunRef(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            candidate_id=str(run.data.tags.get("candidate_id")),
        )
        for run in runs
    ]
