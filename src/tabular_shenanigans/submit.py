import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.data import get_binary_prediction_kind, load_sample_submission_template, validate_sample_submission_schema

SUBMISSION_LEDGER_COLUMNS = [
    "timestamp_utc",
    "competition_slug",
    "run_id",
    "model_id",
    "model_name",
    "config_fingerprint",
    "submission_path",
    "submit_enabled",
    "status",
    "message",
]


@dataclass(frozen=True)
class SubmissionContext:
    artifact_kind: str
    artifact_id: str
    competition_slug: str
    task_type: str
    primary_metric: str
    model_id: str
    model_name: str
    metric_name: str
    metric_mean: float
    id_column: str
    label_column: str
    observed_label_pair: tuple[object, object] | None
    config_fingerprint: str | None
    prediction_path: Path


def _read_submission_ledger(ledger_path: Path) -> pd.DataFrame:
    ledger_df = pd.read_csv(ledger_path)
    ledger_columns = ledger_df.columns.tolist()
    if ledger_columns != SUBMISSION_LEDGER_COLUMNS:
        raise ValueError(
            "Submission ledger does not match the supported schema. "
            f"Expected columns {SUBMISSION_LEDGER_COLUMNS}, got {ledger_columns}"
        )
    return ledger_df


def _append_submission_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row]).reindex(columns=SUBMISSION_LEDGER_COLUMNS)
    if ledger_path.exists():
        existing_df = _read_submission_ledger(ledger_path)
        merged_df = pd.concat([existing_df, ledger_df], ignore_index=True, sort=False)
        merged_df = merged_df.reindex(columns=SUBMISSION_LEDGER_COLUMNS)
        merged_df.to_csv(ledger_path, index=False)
        return
    ledger_df.to_csv(ledger_path, index=False)


def _load_json_manifest(manifest_path: Path, description: str) -> dict[str, object]:
    if not manifest_path.exists():
        raise ValueError(f"Missing {description}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {manifest_path}")
    return manifest


def _require_manifest_value(manifest: dict[str, object], field_name: str, description: str) -> object:
    field_value = manifest.get(field_name)
    if field_value is None:
        raise ValueError(f"{description.capitalize()} is missing required field '{field_name}'.")
    return field_value


def _parse_observed_label_pair(manifest: dict[str, object], description: str) -> tuple[object, object] | None:
    observed_label_pair_raw = manifest.get("observed_label_pair")
    if observed_label_pair_raw is None:
        return None
    if not isinstance(observed_label_pair_raw, list) or len(observed_label_pair_raw) != 2:
        raise ValueError(f"{description.capitalize()} observed_label_pair must contain exactly two labels when present.")
    return (observed_label_pair_raw[0], observed_label_pair_raw[1])


def _load_manifest_models(run_manifest: dict[str, object]) -> list[dict[str, object]]:
    manifest_models = run_manifest.get("models")
    if not isinstance(manifest_models, list) or not manifest_models:
        raise ValueError(
            "Run manifest must contain a non-empty 'models' list. "
            "Submission requires current multi-model-aware run artifacts."
        )
    if not all(isinstance(model, dict) for model in manifest_models):
        raise ValueError("Run manifest models must be a list of mappings.")
    return manifest_models


def _resolve_selected_legacy_model(
    run_manifest: dict[str, object],
    requested_model_id: str | None = None,
) -> tuple[str, dict[str, object]]:
    manifest_models = _load_manifest_models(run_manifest)
    available_model_ids = [str(model["model_id"]) for model in manifest_models]

    if requested_model_id is not None:
        selected_model_id = requested_model_id
        if selected_model_id not in available_model_ids:
            raise ValueError(
                f"Requested model_id '{selected_model_id}' is not present in the run manifest. "
                f"Available model_ids: {available_model_ids}"
            )
    else:
        best_model_id = run_manifest.get("best_model_id")
        if best_model_id is None:
            raise ValueError(
                "Run manifest is missing required submission field 'best_model_id'. "
                f"Available model_ids: {available_model_ids}"
            )
        selected_model_id = str(best_model_id)
        if selected_model_id not in available_model_ids:
            raise ValueError(
                "Run manifest best_model_id must match one of the available model entries. "
                f"Available model_ids: {available_model_ids}"
            )

    selected_model = next(
        (model for model in manifest_models if str(model["model_id"]) == selected_model_id),
        None,
    )
    if selected_model is None:
        raise ValueError(
            f"Requested model_id '{selected_model_id}' is missing from the run manifest model entries."
        )
    return selected_model_id, selected_model


def _resolve_legacy_prediction_path(run_dir: Path, model_id: str) -> Path:
    prediction_path = run_dir / model_id / "test_predictions.csv"
    if prediction_path.exists():
        return prediction_path
    raise ValueError(
        "Missing test predictions file for submission. "
        f"Expected current per-model artifact at {prediction_path}"
    )


def _load_legacy_submission_context(
    run_dir: Path,
    model_id: str | None = None,
) -> SubmissionContext:
    run_manifest = _load_json_manifest(
        manifest_path=run_dir / "run_manifest.json",
        description="run manifest",
    )
    selected_model_id, selected_model = _resolve_selected_legacy_model(run_manifest, requested_model_id=model_id)
    cv_summary = selected_model.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError("Run manifest model cv_summary must be a mapping.")
    prediction_path = _resolve_legacy_prediction_path(run_dir=run_dir, model_id=selected_model_id)

    return SubmissionContext(
        artifact_kind="run",
        artifact_id=str(_require_manifest_value(run_manifest, "run_id", "run manifest")),
        competition_slug=str(_require_manifest_value(run_manifest, "competition_slug", "run manifest")),
        task_type=str(_require_manifest_value(run_manifest, "task_type", "run manifest")),
        primary_metric=str(_require_manifest_value(run_manifest, "primary_metric", "run manifest")),
        model_id=selected_model_id,
        model_name=str(selected_model["model_name"]),
        metric_name=str(cv_summary["metric_name"]),
        metric_mean=float(cv_summary["metric_mean"]),
        id_column=str(_require_manifest_value(run_manifest, "id_column", "run manifest")),
        label_column=str(_require_manifest_value(run_manifest, "label_column", "run manifest")),
        observed_label_pair=_parse_observed_label_pair(run_manifest, "run manifest"),
        config_fingerprint=run_manifest.get("config_fingerprint"),
        prediction_path=prediction_path,
    )


def _resolve_candidate_prediction_path(candidate_dir: Path) -> Path:
    prediction_path = candidate_dir / "test_predictions.csv"
    if prediction_path.exists():
        return prediction_path
    raise ValueError(
        "Missing test predictions file for submission. "
        f"Expected current candidate artifact at {prediction_path}"
    )


def _load_candidate_submission_context(
    candidate_dir: Path,
    model_id: str | None = None,
) -> SubmissionContext:
    candidate_manifest = _load_json_manifest(
        manifest_path=candidate_dir / "candidate.json",
        description="candidate manifest",
    )
    selected_model_id = str(_require_manifest_value(candidate_manifest, "model_id", "candidate manifest"))
    if model_id is not None and model_id != selected_model_id:
        raise ValueError(
            "Candidate artifacts contain exactly one model. "
            f"Requested model_id '{model_id}', available model_id '{selected_model_id}'."
        )

    cv_summary = candidate_manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError("Candidate manifest cv_summary must be a mapping.")
    prediction_path = _resolve_candidate_prediction_path(candidate_dir)
    return SubmissionContext(
        artifact_kind="candidate",
        artifact_id=str(_require_manifest_value(candidate_manifest, "candidate_id", "candidate manifest")),
        competition_slug=str(_require_manifest_value(candidate_manifest, "competition_slug", "candidate manifest")),
        task_type=str(_require_manifest_value(candidate_manifest, "task_type", "candidate manifest")),
        primary_metric=str(_require_manifest_value(candidate_manifest, "primary_metric", "candidate manifest")),
        model_id=selected_model_id,
        model_name=str(_require_manifest_value(candidate_manifest, "model_name", "candidate manifest")),
        metric_name=str(cv_summary["metric_name"]),
        metric_mean=float(cv_summary["metric_mean"]),
        id_column=str(_require_manifest_value(candidate_manifest, "id_column", "candidate manifest")),
        label_column=str(_require_manifest_value(candidate_manifest, "label_column", "candidate manifest")),
        observed_label_pair=_parse_observed_label_pair(candidate_manifest, "candidate manifest"),
        config_fingerprint=candidate_manifest.get("config_fingerprint"),
        prediction_path=prediction_path,
    )


def _load_submission_context(
    artifact_dir: Path,
    model_id: str | None = None,
) -> SubmissionContext:
    if (artifact_dir / "candidate.json").exists():
        return _load_candidate_submission_context(candidate_dir=artifact_dir, model_id=model_id)
    return _load_legacy_submission_context(run_dir=artifact_dir, model_id=model_id)


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
            "Binary label submissions require observed_label_pair metadata in the artifact manifest."
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


def _prepare_submission_file_from_context(submission_context: SubmissionContext) -> Path:
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

    if submission_context.task_type == "binary":
        prediction_values = prediction_df[submission_context.label_column]
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

    submission_path = submission_context.prediction_path.parent / "submission.csv"
    prediction_df.to_csv(submission_path, index=False)
    return submission_path


def prepare_submission_file(
    artifact_dir: Path,
    model_id: str | None = None,
) -> Path:
    submission_context = _load_submission_context(artifact_dir=artifact_dir, model_id=model_id)
    return _prepare_submission_file_from_context(submission_context)


def _build_submission_message_from_context(
    submission_context: SubmissionContext,
    submit_message_prefix: str | None = None,
) -> str:
    message_parts = []
    if submit_message_prefix:
        message_parts.append(submit_message_prefix.strip())
    message_parts.append(f"{submission_context.artifact_kind}={submission_context.artifact_id}")
    message_parts.append(f"model={submission_context.model_id}")
    message_parts.append(f"{submission_context.metric_name}={submission_context.metric_mean:.6f}")
    return " | ".join(message_parts)


def build_submission_message(
    artifact_dir: Path,
    submit_message_prefix: str | None = None,
    model_id: str | None = None,
) -> str:
    submission_context = _load_submission_context(artifact_dir=artifact_dir, model_id=model_id)
    return _build_submission_message_from_context(submission_context, submit_message_prefix=submit_message_prefix)


def run_submission(
    config: AppConfig,
    artifact_dir: Path,
    model_id: str | None = None,
) -> tuple[Path, str]:
    submission_context = _load_submission_context(artifact_dir=artifact_dir, model_id=model_id)
    submission_path = _prepare_submission_file_from_context(submission_context)
    message = _build_submission_message_from_context(
        submission_context,
        submit_message_prefix=config.submit_message_prefix,
    )

    if config.submit_enabled:
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
                message,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.stderr.strip():
            print(completed.stderr.strip())
        status = "submitted"
    else:
        print("Submission dry-run mode: validation complete, Kaggle submit skipped.")
        status = "prepared"

    ledger_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": submission_context.competition_slug,
        "run_id": submission_context.artifact_id,
        "model_id": submission_context.model_id,
        "model_name": submission_context.model_name,
        "config_fingerprint": submission_context.config_fingerprint,
        "submission_path": str(submission_path),
        "submit_enabled": config.submit_enabled,
        "status": status,
        "message": message,
    }
    ledger_path = Path("artifacts") / submission_context.competition_slug / "train" / "submissions.csv"
    _append_submission_ledger(ledger_path=ledger_path, row=ledger_row)
    return submission_path, status
