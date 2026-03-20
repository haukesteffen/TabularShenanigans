import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tabular_shenanigans.candidate_artifacts import (
    CANDIDATE_ARTIFACT_DIRNAME,
    CANDIDATE_MANIFEST_FILENAME,
    CONTEXT_ARTIFACT_DIRNAME,
    json_ready,
    load_candidate_manifest,
)
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.submission_history import CandidateSubmissionHistory, SubmissionEvent

TRACKING_SCHEMA_VERSION = "4"
RUN_KIND_SCREENING = "screening"
RUN_KIND_CANONICAL = "canonical"
RUN_KIND_SUBMISSION = "submission"
RUN_KIND_OPTIMIZATION_TRIAL = "optimization_trial"
SUBMISSION_HISTORY_ARTIFACT_PATH = "submissions/history.json"
SUBMISSION_EVENT_ARTIFACT_PATH = "submissions/event.json"
CANDIDATE_BUNDLE_REQUIRED_ARTIFACT_PATHS = (
    "logs/runtime.log",
    "config/runtime_config.json",
    "context/competition.json",
    "context/folds.csv",
    "candidate/candidate.json",
    "candidate/fold_metrics.csv",
    "candidate/oof_predictions.csv",
    "candidate/test_predictions.csv",
)
CANDIDATE_BUNDLE_ROOT_PATHS = ("logs", "config", "context", "candidate")
ACTIVE_MLFLOW_RUN_STATUSES = {"RUNNING", "SCHEDULED"}
ExperimentRole = Literal["screening", "candidates", "submissions"]


@dataclass(frozen=True)
class CandidateRunRef:
    run_id: str
    experiment_id: str
    candidate_id: str
    run_kind: str


@dataclass(frozen=True)
class DownloadedCandidateBundle:
    run_id: str
    experiment_id: str
    candidate_id: str
    candidate_artifact_dir: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class TrialRunRef:
    run_id: str
    trial_number: int


@dataclass(frozen=True)
class SubmissionRunRef:
    run_id: str
    experiment_id: str
    candidate_id: str
    submission_event_id: str


@dataclass(frozen=True)
class CandidateRunAssessment:
    run_ref: CandidateRunRef
    run_status: str
    missing_artifact_paths: tuple[str, ...]
    bundle_error: str | None

    @property
    def is_canonical(self) -> bool:
        return (
            self.run_status == "FINISHED"
            and not self.missing_artifact_paths
            and self.bundle_error is None
        )

    @property
    def is_active(self) -> bool:
        return self.run_status in ACTIVE_MLFLOW_RUN_STATUSES


@dataclass(frozen=True)
class CandidateRunLookup:
    candidate_id: str
    matching_runs: tuple[CandidateRunAssessment, ...]
    canonical_run: CandidateRunRef | None

    @property
    def has_active_runs(self) -> bool:
        return any(run.is_active for run in self.matching_runs)


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


def _experiment_name(config: AppConfig, role: ExperimentRole) -> str:
    return f"{config.competition.slug}__{role}"


def _experiment_id(config: AppConfig, role: ExperimentRole) -> str:
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(config.experiment.tracking.tracking_uri)
    experiment = mlflow.set_experiment(_experiment_name(config, role))
    return experiment.experiment_id


def _training_experiment_role(config: AppConfig) -> ExperimentRole:
    if config.active_run_stage == "screening":
        return "screening"
    return "candidates"


def _training_run_kind(config: AppConfig) -> str:
    if config.active_run_stage == "screening":
        return RUN_KIND_SCREENING
    return RUN_KIND_CANONICAL


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


def _candidate_search_filter(candidate_id: str, run_kind: str) -> str:
    return (
        f"tags.run_kind = '{run_kind}' "
        f"and tags.candidate_id = '{candidate_id}'"
    )


def _submission_run_search_filter() -> str:
    return f"tags.run_kind = '{RUN_KIND_SUBMISSION}'"


def _candidate_run_ref_from_run(run) -> CandidateRunRef:
    candidate_id = run.data.tags.get("candidate_id")
    if candidate_id is None:
        raise ValueError(f"Candidate run {run.info.run_id} is missing tag 'candidate_id'.")
    run_kind = run.data.tags.get("run_kind")
    if run_kind is None:
        raise ValueError(f"Candidate run {run.info.run_id} is missing tag 'run_kind'.")
    return CandidateRunRef(
        run_id=run.info.run_id,
        experiment_id=run.info.experiment_id,
        candidate_id=str(candidate_id),
        run_kind=str(run_kind),
    )


def _collect_candidate_bundle_artifact_paths(client, run_id: str) -> set[str]:
    artifact_paths: set[str] = set()
    pending_paths = list(CANDIDATE_BUNDLE_ROOT_PATHS)
    visited_paths: set[str] = set()

    while pending_paths:
        current_path = pending_paths.pop()
        if current_path in visited_paths:
            continue
        visited_paths.add(current_path)
        for artifact in client.list_artifacts(run_id, current_path):
            artifact_paths.add(artifact.path)
            if artifact.is_dir:
                pending_paths.append(artifact.path)
    return artifact_paths


def _download_json_artifact(
    client,
    run_id: str,
    artifact_path: str,
    destination_dir: Path,
) -> dict[str, object]:
    local_path = Path(
        client.download_artifacts(
            run_id=run_id,
            path=artifact_path,
            dst_path=str(destination_dir),
        )
    )
    payload = json.loads(local_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact '{artifact_path}' for run '{run_id}' must contain a JSON object.")
    return payload


def _required_candidate_artifact_paths(manifest: Mapping[str, object] | None = None) -> list[str]:
    required_paths = list(CANDIDATE_BUNDLE_REQUIRED_ARTIFACT_PATHS)
    if manifest is None:
        return required_paths

    probability_artifact_path = manifest.get("binary_accuracy_test_probability_path")
    if probability_artifact_path is not None:
        required_paths.append(f"{CANDIDATE_ARTIFACT_DIRNAME}/{probability_artifact_path}")
    return required_paths


def _assess_candidate_run(client, run, destination_dir: Path) -> CandidateRunAssessment:
    run_ref = _candidate_run_ref_from_run(run)
    artifact_paths = _collect_candidate_bundle_artifact_paths(client, run_ref.run_id)
    required_paths = _required_candidate_artifact_paths()
    bundle_error = None

    manifest_artifact_path = f"{CANDIDATE_ARTIFACT_DIRNAME}/{CANDIDATE_MANIFEST_FILENAME}"
    if manifest_artifact_path in artifact_paths:
        try:
            manifest = _download_json_artifact(
                client=client,
                run_id=run_ref.run_id,
                artifact_path=manifest_artifact_path,
                destination_dir=destination_dir,
            )
        except Exception as exc:
            bundle_error = str(exc)
        else:
            required_paths = _required_candidate_artifact_paths(manifest)

    missing_artifact_paths = tuple(sorted(path for path in required_paths if path not in artifact_paths))
    return CandidateRunAssessment(
        run_ref=run_ref,
        run_status=str(run.info.status),
        missing_artifact_paths=missing_artifact_paths,
        bundle_error=bundle_error,
    )


def _format_candidate_run_assessment(assessment: CandidateRunAssessment) -> str:
    parts = [
        f"run_id={assessment.run_ref.run_id}",
        f"status={assessment.run_status}",
    ]
    if assessment.missing_artifact_paths:
        parts.append(f"missing_artifacts={list(assessment.missing_artifact_paths)}")
    if assessment.bundle_error is not None:
        parts.append(f"bundle_error={assessment.bundle_error}")
    return ", ".join(parts)


def _candidate_run_guidance(lookup: CandidateRunLookup) -> str:
    if lookup.has_active_runs:
        return "Wait for the active run to finish or terminate it before retrying."
    return "Retry training to create a fresh canonical run or repair/delete the broken runs."


def _build_candidate_lookup_from_runs(candidate_id: str, client, runs: list[Any]) -> CandidateRunLookup:
    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-candidate-lookup-") as temp_dir:
        temp_root = Path(temp_dir)
        assessments = tuple(
            _assess_candidate_run(
                client=client,
                run=run,
                destination_dir=temp_root / run.info.run_id,
            )
            for run in runs
        )

    canonical_runs = [assessment.run_ref for assessment in assessments if assessment.is_canonical]
    if len(canonical_runs) > 1:
        matching_runs = "; ".join(_format_candidate_run_assessment(assessment) for assessment in assessments)
        raise ValueError(
            "Candidate id resolved to multiple completed MLflow runs with the required artifact bundle. "
            f"Candidate '{candidate_id}' matched runs: {matching_runs}. Manual cleanup is required."
        )

    return CandidateRunLookup(
        candidate_id=candidate_id,
        matching_runs=assessments,
        canonical_run=canonical_runs[0] if canonical_runs else None,
    )


def _candidate_run_lookup(config: AppConfig, candidate_id: str) -> CandidateRunLookup:
    client = _client(config)
    experiment_id = _experiment_id(config, "candidates")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=_candidate_search_filter(candidate_id, RUN_KIND_CANONICAL),
        max_results=100,
    )
    return _build_candidate_lookup_from_runs(
        candidate_id=candidate_id,
        client=client,
        runs=runs,
    )


def find_candidate_run(config: AppConfig, candidate_id: str) -> CandidateRunRef:
    lookup = _candidate_run_lookup(config=config, candidate_id=candidate_id)
    if not lookup.matching_runs:
        raise ValueError(
            f"Candidate '{candidate_id}' was not found in MLflow experiment '{_experiment_name(config, 'candidates')}'."
        )
    if lookup.canonical_run is None:
        matching_runs = "; ".join(_format_candidate_run_assessment(assessment) for assessment in lookup.matching_runs)
        raise ValueError(
            "Candidate did not resolve to a completed MLflow run with the required artifact bundle. "
            f"Candidate '{candidate_id}' matched runs: {matching_runs}. "
            f"{_candidate_run_guidance(lookup)}"
        )
    return lookup.canonical_run


def ensure_candidate_run_absent(config: AppConfig, candidate_id: str) -> None:
    lookup = _candidate_run_lookup(config=config, candidate_id=candidate_id)
    if lookup.canonical_run is not None:
        raise ValueError(
            "Candidate already exists in MLflow for this competition. "
            f"Candidate '{candidate_id}' resolved to canonical run '{lookup.canonical_run.run_id}'. "
            "Change the candidate config so it derives a new candidate_id or delete the canonical candidate run."
        )
    if lookup.has_active_runs:
        matching_runs = "; ".join(_format_candidate_run_assessment(assessment) for assessment in lookup.matching_runs)
        raise ValueError(
            "Candidate has an active MLflow run and cannot be retried safely yet. "
            f"Candidate '{candidate_id}' matched runs: {matching_runs}. "
            "Wait for the active run to finish or terminate it before retrying."
        )


def candidate_run_exists(config: AppConfig, candidate_id: str) -> bool:
    lookup = _candidate_run_lookup(config=config, candidate_id=candidate_id)
    if lookup.has_active_runs:
        matching_runs = "; ".join(_format_candidate_run_assessment(assessment) for assessment in lookup.matching_runs)
        raise ValueError(
            "Candidate has an active MLflow run and cannot be resolved for --skip-existing yet. "
            f"Candidate '{candidate_id}' matched runs: {matching_runs}. "
            "Wait for the active run to finish or terminate it before retrying."
        )
    return lookup.canonical_run is not None


def _base_candidate_tags(config: AppConfig, candidate_id: str, candidate_type: str) -> dict[str, object]:
    runtime_execution_context = config.runtime_execution_context
    tags: dict[str, object] = {
        "run_kind": _training_run_kind(config),
        "tracking_schema_version": TRACKING_SCHEMA_VERSION,
        "competition_slug": config.competition.slug,
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "task_type": config.competition.task_type,
        "primary_metric": config.competition.primary_metric,
        "runtime_requested_compute_target": runtime_execution_context.requested_compute_target,
        "runtime_resolved_compute_target": runtime_execution_context.resolved_compute_target,
        "runtime_requested_gpu_backend": runtime_execution_context.requested_gpu_backend,
        "runtime_resolved_gpu_backend": runtime_execution_context.resolved_gpu_backend,
        "runtime_acceleration_backend": runtime_execution_context.acceleration_backend,
        **_git_metadata(),
    }
    if config.is_model_candidate:
        tags["runtime_preprocessing_backend"] = config.preprocessing_execution_plan.preprocessing_backend
    return tags


def create_candidate_run(config: AppConfig, candidate_id: str, candidate_type: str) -> CandidateRunRef:
    if config.active_run_stage != "screening":
        ensure_candidate_run_absent(config=config, candidate_id=candidate_id)
    client = _client(config)
    role = _training_experiment_role(config)
    experiment_id = _experiment_id(config, role)
    run = client.create_run(
        experiment_id=experiment_id,
        tags=_base_candidate_tags(config=config, candidate_id=candidate_id, candidate_type=candidate_type),
        run_name=candidate_id,
    )
    return CandidateRunRef(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        run_kind=_training_run_kind(config),
    )


def terminate_run(config: AppConfig, run_id: str, status: str) -> None:
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
        "runtime__requested_gpu_backend": runtime_execution_context.requested_gpu_backend,
        "runtime__resolved_gpu_backend": runtime_execution_context.resolved_gpu_backend,
        "runtime__gpu_available": runtime_execution_context.gpu_available,
        "runtime__acceleration_backend": runtime_execution_context.acceleration_backend,
        "runtime__rapids_hooks_installed": runtime_execution_context.rapids_hooks_installed,
    }
    if runtime_execution_context.fallback_reason is not None:
        params["runtime__fallback_reason"] = runtime_execution_context.fallback_reason
    if config.is_model_candidate:
        params["runtime__preprocessing_backend"] = config.preprocessing_execution_plan.preprocessing_backend
        params.update(
            {
                "representation_id": manifest.get("representation_id"),
                "model_family": manifest.get("model_family"),
                "model_registry_key": manifest.get("model_registry_key"),
            }
        )
        model_params = manifest.get("model_params")
        if isinstance(model_params, dict):
            for key, value in model_params.items():
                params[f"model__{key}"] = value
        optimization = candidate.optimization
        if optimization is not None:
            params.update(
                {
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
    return {
        "run_kind": _training_run_kind(config),
        "competition_slug": manifest.get("competition_slug"),
        "candidate_id": manifest.get("candidate_id"),
        "candidate_type": manifest.get("candidate_type"),
        "task_type": manifest.get("task_type"),
        "primary_metric": manifest.get("primary_metric"),
        "config_fingerprint": manifest.get("config_fingerprint"),
    }


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
    runtime_profile = manifest.get("runtime_profile")
    if isinstance(runtime_profile, dict):
        for metric_name in (
            "prepare_training_context_wall_seconds",
            "cv_preprocess_wall_seconds",
            "cv_fit_wall_seconds",
            "cv_predict_wall_seconds",
            "cv_stage_wall_seconds",
            "artifact_staging_wall_seconds",
        ):
            metric_value = runtime_profile.get(metric_name)
            if isinstance(metric_value, int | float):
                metrics[metric_name] = metric_value
    if optimization_summary is not None:
        metrics["optimization_best_value"] = optimization_summary.get("best_value")  # type: ignore[assignment]
        metrics["optimization_trial_count"] = optimization_summary.get("trial_count")  # type: ignore[assignment]
    return metrics


def create_trial_run(
    config: AppConfig,
    candidate_run: CandidateRunRef,
    trial_number: int,
    hyperparams: dict[str, object],
    representation_id: str,
    model_family: str,
    model_registry_key: str,
    preprocessing_backend: str,
) -> TrialRunRef:
    client = _client(config)
    run = client.create_run(
        experiment_id=candidate_run.experiment_id,
        run_name=f"trial_{trial_number}",
        tags={
            "run_kind": RUN_KIND_OPTIMIZATION_TRIAL,
            "tracking_schema_version": TRACKING_SCHEMA_VERSION,
            "mlflow.parentRunId": candidate_run.run_id,
            "candidate_id": candidate_run.candidate_id,
            "model_family": model_family,
            "trial_state": "RUNNING",
        },
    )
    trial_run_id = run.info.run_id
    _log_params(
        client,
        trial_run_id,
        {
            "trial_number": trial_number,
            "representation_id": representation_id,
            "model_family": model_family,
            "model_registry_key": model_registry_key,
            "runtime__resolved_compute_target": config.runtime_execution_context.resolved_compute_target,
            "runtime__resolved_gpu_backend": config.runtime_execution_context.resolved_gpu_backend,
            "runtime__preprocessing_backend": preprocessing_backend,
            **{f"hp__{key}": value for key, value in hyperparams.items()},
        },
    )
    return TrialRunRef(run_id=trial_run_id, trial_number=trial_number)


def finalize_trial_run(
    config: AppConfig,
    trial_run: TrialRunRef,
    model_params: dict[str, object] | None,
    cv_score_mean: float | None,
    cv_score_std: float | None,
    duration_seconds: float | None,
    trial_state: str,
) -> None:
    client = _client(config)
    if isinstance(model_params, dict):
        _log_params(
            client,
            trial_run.run_id,
            {f"mp__{key}": value for key, value in model_params.items()},
        )
    _log_metrics(
        client,
        trial_run.run_id,
        {
            "cv_score_mean": cv_score_mean,
            "cv_score_std": cv_score_std,
            "duration_seconds": duration_seconds,
        },
    )
    _set_tags(client, trial_run.run_id, {"trial_state": trial_state})
    status = "FINISHED" if trial_state == "COMPLETE" else "FAILED"
    client.set_terminated(trial_run.run_id, status=status)


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


def upload_run_log(config: AppConfig, run_id: str, log_path: Path, artifact_dir: str = "logs") -> None:
    if not log_path.exists():
        raise ValueError(f"Runtime log artifact does not exist: {log_path}")
    _client(config).log_artifact(run_id, str(log_path), artifact_dir)


def download_candidate_manifest(config: AppConfig, run_id: str, destination_dir: Path) -> dict[str, object]:
    return _download_json_artifact(
        client=_client(config),
        run_id=run_id,
        artifact_path=f"{CANDIDATE_ARTIFACT_DIRNAME}/{CANDIDATE_MANIFEST_FILENAME}",
        destination_dir=destination_dir,
    )


def _validate_downloaded_candidate_artifact_dir(
    candidate_id: str,
    run_id: str,
    candidate_artifact_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    required_filenames = [
        CANDIDATE_MANIFEST_FILENAME,
        "fold_metrics.csv",
        "oof_predictions.csv",
        "test_predictions.csv",
    ]
    probability_artifact_path = manifest.get("binary_accuracy_test_probability_path")
    if probability_artifact_path is not None:
        required_filenames.append(str(probability_artifact_path))

    missing_files = sorted(
        filename
        for filename in required_filenames
        if not (candidate_artifact_dir / filename).exists()
    )
    if missing_files:
        raise ValueError(
            "Candidate bundle is incomplete after download. "
            f"Candidate '{candidate_id}' resolved to run '{run_id}' but candidate artifacts were missing: "
            f"{missing_files}"
        )


def download_candidate_bundle(config: AppConfig, candidate_id: str, destination_dir: Path) -> DownloadedCandidateBundle:
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
    _validate_downloaded_candidate_artifact_dir(
        candidate_id=candidate_id,
        run_id=candidate_run.run_id,
        candidate_artifact_dir=candidate_dir_path,
        manifest=manifest,
    )
    return DownloadedCandidateBundle(
        run_id=candidate_run.run_id,
        experiment_id=candidate_run.experiment_id,
        candidate_id=candidate_id,
        candidate_artifact_dir=candidate_dir_path,
        manifest=manifest,
    )


def create_submission_run(config: AppConfig, submission_event: SubmissionEvent) -> SubmissionRunRef:
    client = _client(config)
    experiment_id = _experiment_id(config, "submissions")
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=submission_event.submission_event_id,
        tags={
            "run_kind": RUN_KIND_SUBMISSION,
            "tracking_schema_version": TRACKING_SCHEMA_VERSION,
            "competition_slug": submission_event.competition_slug,
            "candidate_id": submission_event.candidate_id,
            "submission_event_id": submission_event.submission_event_id,
            **_git_metadata(),
        },
    )
    _log_params(
        client,
        run.info.run_id,
        {
            "candidate_id": submission_event.candidate_id,
            "submission_file_name": submission_event.submission_file_name,
        },
    )
    return SubmissionRunRef(
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        candidate_id=submission_event.candidate_id,
        submission_event_id=submission_event.submission_event_id,
    )


def list_submission_runs(config: AppConfig) -> list[SubmissionRunRef]:
    client = _client(config)
    experiment_id = _experiment_id(config, "submissions")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=_submission_run_search_filter(),
        max_results=1000,
    )
    results: list[SubmissionRunRef] = []
    for run in runs:
        candidate_id = run.data.tags.get("candidate_id")
        submission_event_id = run.data.tags.get("submission_event_id")
        if candidate_id is None or submission_event_id is None:
            continue
        results.append(
            SubmissionRunRef(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                candidate_id=str(candidate_id),
                submission_event_id=str(submission_event_id),
            )
        )
    return results


def download_submission_history(config: AppConfig, run_id: str, destination_dir: Path) -> CandidateSubmissionHistory:
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
    if updated_submission_event_ids is not None and len(updated_submission_event_ids) > 1:
        raise ValueError(
            "Submission runs support exactly one submission event per MLflow run. "
            f"Got updated_submission_event_ids={updated_submission_event_ids}."
        )
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
            event = history.get_event(submission_event_id)
            if event is None:
                raise ValueError(f"Submission history is missing event '{submission_event_id}'.")
            if submission_csv_paths is not None and submission_event_id in submission_csv_paths:
                client.log_artifact(run_id, str(submission_csv_paths[submission_event_id]), "submissions")
            event_path = submissions_dir / "event.json"
            event_path.write_text(json.dumps(json_ready(event.to_dict()), indent=2, sort_keys=True), encoding="utf-8")
            client.log_artifact(run_id, str(event_path), "submissions")
            observations_path = submissions_dir / "observations.json"
            event_observations = [observation.to_dict() for observation in history.get_observations(submission_event_id)]
            observations_path.write_text(
                json.dumps(json_ready(event_observations), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            client.log_artifact(run_id, str(observations_path), "submissions")


def update_submission_metrics(config: AppConfig, run_id: str, score_metrics: dict[str, float | int | None]) -> None:
    _log_metrics(_client(config), run_id, score_metrics)


def search_candidate_runs(config: AppConfig) -> list[CandidateRunRef]:
    client = _client(config)
    experiment_id = _experiment_id(config, "candidates")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.run_kind = '{RUN_KIND_CANONICAL}'",
        max_results=1000,
    )
    runs_by_candidate_id: dict[str, list[Any]] = {}
    for run in runs:
        candidate_id = run.data.tags.get("candidate_id")
        if candidate_id is None:
            continue
        runs_by_candidate_id.setdefault(str(candidate_id), []).append(run)

    resolved_runs: list[CandidateRunRef] = []
    for candidate_id, candidate_id_runs in sorted(runs_by_candidate_id.items()):
        lookup = _build_candidate_lookup_from_runs(
            candidate_id=candidate_id,
            client=client,
            runs=candidate_id_runs,
        )
        if lookup.canonical_run is None:
            continue
        resolved_runs.append(lookup.canonical_run)
    return resolved_runs
