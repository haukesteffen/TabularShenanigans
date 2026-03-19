import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import (
    get_binary_prediction_kind,
    load_sample_submission_template,
    validate_sample_submission_schema,
)
from tabular_shenanigans.mlflow_store import (
    CandidateRunRef,
    download_candidate_bundle,
    download_submission_history,
    search_candidate_runs,
    update_submission_metrics,
    upload_submission_history,
)
from tabular_shenanigans.submission_history import (
    CandidateSubmissionHistory,
    SubmissionEvent,
    SubmissionRefreshResult,
    SubmissionScoreObservation,
    build_submission_score_metrics,
    extract_candidate_id,
    extract_submission_event_id,
    iter_kaggle_submissions,
    make_submission_event_id,
    utc_now_iso,
)


@dataclass(frozen=True)
class SubmissionContext:
    candidate_run: CandidateRunRef
    candidate_id: str
    candidate_type: str
    competition_slug: str
    task_type: str
    primary_metric: str
    feature_recipe_id: str | None
    preprocessing_scheme_id: str | None
    model_registry_key: str
    estimator_name: str
    cv_metric_name: str
    cv_metric_mean: float
    cv_metric_std: float
    id_column: str
    label_column: str
    observed_label_pair: tuple[object, object] | None
    config_fingerprint: str | None
    prediction_path: Path


@dataclass(frozen=True)
class SubmissionRunResult:
    candidate_id: str
    candidate_run_id: str
    submission_status: str
    submission_message: str
    submission_event_id: str | None
    submission_artifact_path: str | None
    submission_refresh_result: SubmissionRefreshResult | None
    immediate_refresh_error: str | None = None


def _require_manifest_value(manifest: dict[str, object], field_name: str) -> object:
    field_value = manifest.get(field_name)
    if field_value is None:
        raise ValueError(f"Candidate manifest is missing required field '{field_name}'.")
    return field_value


def _parse_observed_label_pair(manifest: dict[str, object]) -> tuple[object, object] | None:
    observed_label_pair_raw = manifest.get("observed_label_pair")
    if observed_label_pair_raw is None:
        return None
    if not isinstance(observed_label_pair_raw, list) or len(observed_label_pair_raw) != 2:
        raise ValueError("Candidate manifest observed_label_pair must contain exactly two labels when present.")
    return (observed_label_pair_raw[0], observed_label_pair_raw[1])


def _resolve_prediction_path(candidate_artifact_dir: Path) -> Path:
    prediction_path = candidate_artifact_dir / "test_predictions.csv"
    if prediction_path.exists():
        return prediction_path
    raise ValueError(
        "Missing test predictions file for submission. "
        f"Expected current candidate artifact at {prediction_path}"
    )


def _load_submission_context(
    config: AppConfig,
    candidate_id: str,
    destination_dir: Path,
) -> SubmissionContext:
    downloaded_bundle = download_candidate_bundle(
        config=config,
        candidate_id=candidate_id,
        destination_dir=destination_dir,
    )
    candidate_manifest = downloaded_bundle.manifest
    cv_summary = candidate_manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError("Candidate manifest cv_summary must be a mapping.")

    return SubmissionContext(
        candidate_run=CandidateRunRef(
            run_id=downloaded_bundle.run_id,
            experiment_id=downloaded_bundle.experiment_id,
            candidate_id=candidate_id,
        ),
        candidate_id=str(_require_manifest_value(candidate_manifest, "candidate_id")),
        candidate_type=str(_require_manifest_value(candidate_manifest, "candidate_type")),
        competition_slug=str(_require_manifest_value(candidate_manifest, "competition_slug")),
        task_type=str(_require_manifest_value(candidate_manifest, "task_type")),
        primary_metric=str(_require_manifest_value(candidate_manifest, "primary_metric")),
        feature_recipe_id=(
            str(candidate_manifest["feature_recipe_id"])
            if candidate_manifest.get("feature_recipe_id") is not None
            else None
        ),
        preprocessing_scheme_id=(
            str(candidate_manifest["preprocessing_scheme_id"])
            if candidate_manifest.get("preprocessing_scheme_id") is not None
            else None
        ),
        model_registry_key=str(_require_manifest_value(candidate_manifest, "model_registry_key")),
        estimator_name=str(_require_manifest_value(candidate_manifest, "estimator_name")),
        cv_metric_name=str(cv_summary["metric_name"]),
        cv_metric_mean=float(cv_summary["metric_mean"]),
        cv_metric_std=float(cv_summary["metric_std"]),
        id_column=str(_require_manifest_value(candidate_manifest, "id_column")),
        label_column=str(_require_manifest_value(candidate_manifest, "label_column")),
        observed_label_pair=_parse_observed_label_pair(candidate_manifest),
        config_fingerprint=candidate_manifest.get("config_fingerprint"),
        prediction_path=_resolve_prediction_path(downloaded_bundle.candidate_artifact_dir),
    )


def _validate_submission_ids(
    prediction_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    id_column: str,
) -> None:
    prediction_ids = prediction_df[id_column]
    sample_ids = sample_submission_df[id_column]

    if prediction_ids.duplicated().any():
        duplicate_ids = prediction_ids[prediction_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "Submission ID column contains duplicate values in test_predictions.csv. "
            f"Duplicate IDs: {duplicate_ids[:10]}"
        )

    if sample_ids.duplicated().any():
        duplicate_ids = sample_ids[sample_ids.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "sample_submission.csv ID column contains duplicate values. "
            f"Duplicate IDs: {duplicate_ids[:10]}"
        )

    prediction_id_list = prediction_ids.tolist()
    sample_id_list = sample_ids.tolist()
    if prediction_id_list == sample_id_list:
        return

    prediction_id_set = set(prediction_id_list)
    sample_id_set = set(sample_id_list)
    if prediction_id_set == sample_id_set:
        raise ValueError(
            "Submission ID order does not match sample_submission.csv. "
            "IDs contain the same values but appear in a different order."
        )

    missing_ids = sorted(sample_id_set - prediction_id_set)
    extra_ids = sorted(prediction_id_set - sample_id_set)
    raise ValueError(
        "Submission ID values do not match sample_submission.csv. "
        f"Missing IDs: {missing_ids[:10]}; extra IDs: {extra_ids[:10]}"
    )


def _normalize_binary_label(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _validate_binary_probability_predictions(prediction_values: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(prediction_values):
        raise ValueError("Binary probability submissions must be numeric.")
    if not prediction_values.map(pd.notna).all():
        raise ValueError("Binary probability submissions contain missing values.")
    if not np.isfinite(prediction_values.to_numpy(dtype=float)).all():
        raise ValueError("Binary probability submissions must be finite.")
    if ((prediction_values < 0.0) | (prediction_values > 1.0)).any():
        raise ValueError("Binary probability submissions must be within [0, 1].")


def _validate_binary_label_predictions(
    prediction_values: pd.Series,
    observed_label_pair: tuple[object, object] | None,
) -> None:
    if observed_label_pair is None:
        raise ValueError(
            "Binary label submissions require observed_label_pair metadata in the candidate manifest."
        )
    if not prediction_values.map(pd.notna).all():
        raise ValueError("Binary label submissions contain missing values.")

    allowed_labels = {_normalize_binary_label(label) for label in observed_label_pair}
    normalized_predictions = prediction_values.map(_normalize_binary_label)
    invalid_labels = sorted(set(normalized_predictions) - allowed_labels)
    if invalid_labels:
        raise ValueError(
            "Binary label submissions must contain only observed class labels. "
            f"Allowed labels: {sorted(allowed_labels)}; invalid labels: {invalid_labels[:10]}"
        )


def _validate_regression_predictions(prediction_values: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(prediction_values):
        raise ValueError("Regression submissions must be numeric.")
    if not prediction_values.map(pd.notna).all():
        raise ValueError("Regression submissions contain missing values.")
    if not np.isfinite(prediction_values.to_numpy(dtype=float)).all():
        raise ValueError("Regression submissions must be finite.")


def _prepare_submission_file_from_context(
    submission_context: SubmissionContext,
    output_dir: Path,
) -> Path:
    prediction_df = pd.read_csv(submission_context.prediction_path)
    sample_submission_df = load_sample_submission_template(submission_context.competition_slug)
    validate_sample_submission_schema(
        sample_submission_df=sample_submission_df,
        id_column=submission_context.id_column,
        label_column=submission_context.label_column,
    )

    expected_columns = [submission_context.id_column, submission_context.label_column]
    actual_columns = prediction_df.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            "Submission columns do not match sample_submission.csv. "
            f"Expected {expected_columns}, got {actual_columns}"
        )
    if prediction_df.shape[0] != sample_submission_df.shape[0]:
        raise ValueError(
            "Submission row count does not match sample_submission.csv. "
            f"Expected {sample_submission_df.shape[0]}, got {prediction_df.shape[0]}"
        )

    prediction_values = prediction_df[submission_context.label_column]
    if submission_context.task_type == "regression":
        _validate_regression_predictions(prediction_values)
    elif submission_context.task_type == "binary":
        binary_prediction_kind = get_binary_prediction_kind(submission_context.primary_metric)
        if binary_prediction_kind == "probability":
            _validate_binary_probability_predictions(prediction_values)
        else:
            _validate_binary_label_predictions(
                prediction_values=prediction_values,
                observed_label_pair=submission_context.observed_label_pair,
            )

    _validate_submission_ids(
        prediction_df=prediction_df,
        sample_submission_df=sample_submission_df,
        id_column=submission_context.id_column,
    )

    submission_path = output_dir / "submission.csv"
    prediction_df.to_csv(submission_path, index=False)
    return submission_path


def _build_submission_message_from_context(
    submission_context: SubmissionContext,
    submission_event_id: str,
    submit_message_prefix: str | None = None,
) -> str:
    message_parts = []
    if submit_message_prefix:
        message_parts.append(submit_message_prefix.strip())
    message_parts.append(f"candidate={submission_context.candidate_id}")
    message_parts.append(f"submit={submission_event_id}")
    message_parts.append(f"{submission_context.cv_metric_name}={submission_context.cv_metric_mean:.6f}")
    return " | ".join(message_parts)


def _collect_submit_response_message(completed: subprocess.CompletedProcess[str]) -> str:
    response_parts = []
    if completed.stdout.strip():
        response_parts.append(completed.stdout.strip())
    if completed.stderr.strip():
        response_parts.append(completed.stderr.strip())
    return "\n".join(response_parts)


def _build_submission_event(
    submission_context: SubmissionContext,
    submission_event_id: str,
    submission_file_name: str,
    submit_message: str,
    submit_response_message: str,
) -> SubmissionEvent:
    return SubmissionEvent(
        submission_event_id=submission_event_id,
        submitted_at_utc=utc_now_iso(),
        competition_slug=submission_context.competition_slug,
        candidate_id=submission_context.candidate_id,
        candidate_type=submission_context.candidate_type,
        config_fingerprint=submission_context.config_fingerprint,
        feature_recipe_id=submission_context.feature_recipe_id,
        preprocessing_scheme_id=submission_context.preprocessing_scheme_id,
        model_registry_key=submission_context.model_registry_key,
        estimator_name=submission_context.estimator_name,
        cv_metric_name=submission_context.cv_metric_name,
        cv_metric_mean=submission_context.cv_metric_mean,
        cv_metric_std=submission_context.cv_metric_std,
        submission_file_name=submission_file_name,
        submit_message=submit_message,
        submit_response_message=submit_response_message,
    )


def _build_recovered_submission_event(
    submission_context: SubmissionContext,
    submission_event_id: str,
    kaggle_submitted_at: str,
    kaggle_file_name: str,
    kaggle_description: str,
) -> SubmissionEvent:
    return SubmissionEvent(
        submission_event_id=submission_event_id,
        submitted_at_utc=kaggle_submitted_at or utc_now_iso(),
        competition_slug=submission_context.competition_slug,
        candidate_id=submission_context.candidate_id,
        candidate_type=submission_context.candidate_type,
        config_fingerprint=submission_context.config_fingerprint,
        feature_recipe_id=submission_context.feature_recipe_id,
        preprocessing_scheme_id=submission_context.preprocessing_scheme_id,
        model_registry_key=submission_context.model_registry_key,
        estimator_name=submission_context.estimator_name,
        cv_metric_name=submission_context.cv_metric_name,
        cv_metric_mean=submission_context.cv_metric_mean,
        cv_metric_std=submission_context.cv_metric_std,
        submission_file_name=kaggle_file_name or "submission.csv",
        submit_message=kaggle_description,
        submit_response_message=(
            "Recovered by refresh-submissions from Kaggle metadata after the original "
            "post-submit MLflow write did not complete."
        ),
    )


def run_submission(
    config: AppConfig,
    candidate_id: str,
    execute: bool = False,
    message_prefix: str | None = None,
) -> SubmissionRunResult:
    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-submit-") as temp_dir:
        temp_root = Path(temp_dir)
        submission_context = _load_submission_context(
            config=config,
            candidate_id=candidate_id,
            destination_dir=temp_root / "candidate",
        )
        submission_path = _prepare_submission_file_from_context(
            submission_context=submission_context,
            output_dir=temp_root,
        )

        if execute:
            submission_event_id = make_submission_event_id()
            submission_message = _build_submission_message_from_context(
                submission_context,
                submission_event_id=submission_event_id,
                submit_message_prefix=message_prefix,
            )
            completed = subprocess.run(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    submission_context.competition_slug,
                    "-f",
                    str(submission_path),
                    "-m",
                    submission_message,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                print(completed.stdout.strip())
            if completed.stderr.strip():
                print(completed.stderr.strip())
            submit_response_message = _collect_submit_response_message(completed)

            history = download_submission_history(
                config=config,
                run_id=submission_context.candidate_run.run_id,
                destination_dir=temp_root / "history",
            )
            submission_event = _build_submission_event(
                submission_context=submission_context,
                submission_event_id=submission_event_id,
                submission_file_name=submission_path.name,
                submit_message=submission_message,
                submit_response_message=submit_response_message,
            )
            history = history.with_submission_event(submission_event)
            upload_submission_history(
                config=config,
                run_id=submission_context.candidate_run.run_id,
                history=history,
                updated_submission_event_ids=[submission_event_id],
                submission_csv_paths={submission_event_id: submission_path},
            )
            update_submission_metrics(
                config=config,
                run_id=submission_context.candidate_run.run_id,
                score_metrics=build_submission_score_metrics(
                    primary_metric=submission_context.primary_metric,
                    history=history,
                ),
            )

            immediate_refresh_error = None
            try:
                submission_refresh_result = run_submission_refresh(
                    config=config,
                    target_candidate_ids={submission_context.candidate_id},
                    target_submission_event_ids={submission_event_id},
                    observation_source="submit_immediate_refresh",
                )
            except Exception as exc:
                immediate_refresh_error = str(exc)
                submission_refresh_result = None
                print(
                    "Immediate submission score refresh failed after a successful Kaggle submission. "
                    "Run `uv run python main.py refresh-submissions` later. "
                    f"Original error: {exc}"
                )

            return SubmissionRunResult(
                candidate_id=submission_context.candidate_id,
                candidate_run_id=submission_context.candidate_run.run_id,
                submission_status="submitted",
                submission_message=submission_message,
                submission_event_id=submission_event_id,
                submission_artifact_path=f"submissions/{submission_event_id}/submission.csv",
                submission_refresh_result=submission_refresh_result,
                immediate_refresh_error=immediate_refresh_error,
            )

        print("Submission dry-run mode: validation complete, Kaggle submit skipped.")
        return SubmissionRunResult(
            candidate_id=submission_context.candidate_id,
            candidate_run_id=submission_context.candidate_run.run_id,
            submission_status="prepared",
            submission_message="",
            submission_event_id=None,
            submission_artifact_path=None,
            submission_refresh_result=None,
        )


def run_submission_refresh(
    config: AppConfig,
    target_candidate_ids: set[str] | None = None,
    target_submission_event_ids: set[str] | None = None,
    observation_source: str = "manual_refresh",
) -> SubmissionRefreshResult:
    candidate_runs = search_candidate_runs(config)
    tracked_candidate_count = 0
    tracked_submission_event_count = 0
    matched_submission_event_ids: set[str] = set()
    updated_candidate_count = 0
    appended_observation_count = 0
    scanned_remote_submission_count = 0

    histories_by_run_id: dict[str, CandidateSubmissionHistory] = {}
    event_to_candidate_run: dict[str, CandidateRunRef] = {}
    candidate_run_by_candidate_id: dict[str, CandidateRunRef] = {}
    submission_contexts_by_run_id: dict[str, SubmissionContext] = {}

    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-refresh-submissions-") as temp_dir:
        temp_root = Path(temp_dir)
        for candidate_run in candidate_runs:
            if target_candidate_ids is not None and candidate_run.candidate_id not in target_candidate_ids:
                continue
            candidate_run_by_candidate_id[candidate_run.candidate_id] = candidate_run
            history = download_submission_history(
                config=config,
                run_id=candidate_run.run_id,
                destination_dir=temp_root / candidate_run.candidate_id,
            )
            histories_by_run_id[candidate_run.run_id] = history
            relevant_event_ids = {
                event.submission_event_id
                for event in history.events
                if target_submission_event_ids is None or event.submission_event_id in target_submission_event_ids
            }
            if relevant_event_ids:
                tracked_candidate_count += 1
                tracked_submission_event_count += len(relevant_event_ids)
            for submission_event_id in relevant_event_ids:
                if submission_event_id in event_to_candidate_run:
                    existing_candidate_id = event_to_candidate_run[submission_event_id].candidate_id
                    raise ValueError(
                        "Submission event ids must be unique across candidate runs. "
                        f"Event '{submission_event_id}' was found under candidates "
                        f"'{existing_candidate_id}' and '{candidate_run.candidate_id}'."
                    )
                event_to_candidate_run[submission_event_id] = candidate_run

        if not histories_by_run_id:
            return SubmissionRefreshResult(
                competition_slug=config.competition.slug,
                tracked_candidate_count=tracked_candidate_count,
                tracked_submission_event_count=tracked_submission_event_count,
                matched_submission_event_count=0,
                updated_candidate_count=0,
                appended_observation_count=0,
                scanned_remote_submission_count=0,
                observation_source=observation_source,
            )

        observations_by_run_id: dict[str, list[SubmissionScoreObservation]] = {}
        updated_submission_event_ids_by_run_id: dict[str, set[str]] = {}
        for remote_submission in iter_kaggle_submissions(config.competition.slug):
            scanned_remote_submission_count += 1
            submission_event_id = extract_submission_event_id(remote_submission.kaggle_description)
            if submission_event_id is None:
                continue
            if target_submission_event_ids is not None and submission_event_id not in target_submission_event_ids:
                continue

            candidate_id = extract_candidate_id(remote_submission.kaggle_description)
            candidate_run = event_to_candidate_run.get(submission_event_id)
            if candidate_run is not None:
                if target_candidate_ids is not None and candidate_run.candidate_id not in target_candidate_ids:
                    continue
                if candidate_id is not None and candidate_id != candidate_run.candidate_id:
                    raise ValueError(
                        "Kaggle submission metadata did not match the tracked MLflow candidate. "
                        f"submission_event_id={submission_event_id}, "
                        f"kaggle_candidate_id={candidate_id}, "
                        f"tracked_candidate_id={candidate_run.candidate_id}"
                    )
            else:
                if target_candidate_ids is not None and candidate_id not in target_candidate_ids:
                    continue
                if candidate_id is None:
                    print(
                        "Submission refresh could not recover orphaned Kaggle submission: "
                        f"submission_event_id={submission_event_id}, "
                        "reason=missing_candidate_id_in_kaggle_description"
                    )
                    continue
                candidate_run = candidate_run_by_candidate_id.get(candidate_id)
                if candidate_run is None:
                    print(
                        "Submission refresh could not recover orphaned Kaggle submission: "
                        f"candidate_id={candidate_id}, "
                        f"submission_event_id={submission_event_id}, "
                        "reason=no_canonical_candidate_run"
                    )
                    continue

                history = histories_by_run_id[candidate_run.run_id]
                if history.get_event(submission_event_id) is None:
                    submission_context = submission_contexts_by_run_id.get(candidate_run.run_id)
                    if submission_context is None:
                        submission_context = _load_submission_context(
                            config=config,
                            candidate_id=candidate_run.candidate_id,
                            destination_dir=temp_root / "recovered-events" / candidate_run.candidate_id,
                        )
                        submission_contexts_by_run_id[candidate_run.run_id] = submission_context

                    history = history.with_submission_event(
                        _build_recovered_submission_event(
                            submission_context=submission_context,
                            submission_event_id=submission_event_id,
                            kaggle_submitted_at=remote_submission.kaggle_submitted_at,
                            kaggle_file_name=remote_submission.kaggle_file_name,
                            kaggle_description=remote_submission.kaggle_description,
                        )
                    )
                    histories_by_run_id[candidate_run.run_id] = history
                    event_to_candidate_run[submission_event_id] = candidate_run
                    updated_submission_event_ids_by_run_id.setdefault(candidate_run.run_id, set()).add(
                        submission_event_id
                    )
                    print(
                        "Recovered orphaned Kaggle submission onto candidate run: "
                        f"candidate_id={candidate_run.candidate_id}, "
                        f"submission_event_id={submission_event_id}, "
                        f"mlflow_run_id={candidate_run.run_id}"
                    )

            matched_submission_event_ids.add(submission_event_id)
            observations_by_run_id.setdefault(candidate_run.run_id, []).append(
                SubmissionScoreObservation(
                    observed_at_utc=utc_now_iso(),
                    submission_event_id=submission_event_id,
                    kaggle_submitted_at=remote_submission.kaggle_submitted_at,
                    kaggle_file_name=remote_submission.kaggle_file_name,
                    kaggle_description=remote_submission.kaggle_description,
                    kaggle_status=remote_submission.kaggle_status,
                    public_score=remote_submission.public_score,
                    private_score=remote_submission.private_score,
                    observation_source=observation_source,
                )
            )
            updated_submission_event_ids_by_run_id.setdefault(candidate_run.run_id, set()).add(submission_event_id)

        for run_id, history in histories_by_run_id.items():
            observations = observations_by_run_id.get(run_id, [])
            updated_history, appended_count = history.with_submission_observations(observations)
            updated_submission_event_ids = sorted(updated_submission_event_ids_by_run_id.get(run_id, set()))
            if appended_count == 0 and not updated_submission_event_ids:
                continue

            upload_submission_history(
                config=config,
                run_id=run_id,
                history=updated_history,
                updated_submission_event_ids=updated_submission_event_ids,
            )
            update_submission_metrics(
                config=config,
                run_id=run_id,
                score_metrics=build_submission_score_metrics(
                    primary_metric=config.competition.primary_metric,
                    history=updated_history,
                ),
            )
            histories_by_run_id[run_id] = updated_history
            updated_candidate_count += 1
            appended_observation_count += appended_count

    return SubmissionRefreshResult(
        competition_slug=config.competition.slug,
        tracked_candidate_count=tracked_candidate_count,
        tracked_submission_event_count=tracked_submission_event_count,
        matched_submission_event_count=len(matched_submission_event_ids),
        updated_candidate_count=updated_candidate_count,
        appended_observation_count=appended_observation_count,
        scanned_remote_submission_count=scanned_remote_submission_count,
        observation_source=observation_source,
    )
