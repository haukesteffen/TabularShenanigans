import csv
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
class SubmissionRunContext:
    run_id: str
    competition_slug: str
    task_type: str
    primary_metric: str
    model_id: str
    id_column: str
    label_column: str
    observed_label_pair: tuple[object, object] | None
    config_fingerprint: str | None


def _read_submission_ledger(ledger_path: Path) -> pd.DataFrame:
    with ledger_path.open(encoding="utf-8", newline="") as ledger_file:
        rows = list(csv.reader(ledger_file))

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    if not header:
        return pd.DataFrame()

    max_column_count = max(len(row) for row in rows[1:]) if len(rows) > 1 else len(header)
    normalized_header = header.copy()
    legacy_extra_columns = ["positive_label", "negative_label"]
    while len(normalized_header) < max_column_count:
        extra_index = len(normalized_header) - len(header)
        if extra_index < len(legacy_extra_columns):
            normalized_header.append(legacy_extra_columns[extra_index])
        else:
            normalized_header.append(f"extra_col_{extra_index - len(legacy_extra_columns) + 1}")

    normalized_rows = []
    for row in rows[1:]:
        padded_row = row + [""] * (len(normalized_header) - len(row))
        normalized_rows.append(dict(zip(normalized_header, padded_row)))
    return pd.DataFrame(normalized_rows, columns=normalized_header)


def _append_submission_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row])
    ledger_df = ledger_df.reindex(columns=SUBMISSION_LEDGER_COLUMNS)
    if ledger_path.exists():
        existing_df = _read_submission_ledger(ledger_path)
        merged_df = pd.concat([existing_df, ledger_df], ignore_index=True, sort=False)
        merged_df = merged_df.reindex(columns=SUBMISSION_LEDGER_COLUMNS)
        merged_df.to_csv(ledger_path, index=False)
        return
    ledger_df.to_csv(ledger_path, index=False)


def _load_legacy_cv_summary(run_dir: Path) -> dict[str, object]:
    summary_path = run_dir / "cv_summary.csv"
    if not summary_path.exists():
        raise ValueError(f"Missing CV summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    if summary_df.shape[0] != 1:
        raise ValueError(f"Expected exactly one row in CV summary, got {summary_df.shape[0]}")
    return {
        "model_name": str(summary_df.loc[0, "model_name"]),
        "metric_name": str(summary_df.loc[0, "metric_name"]),
        "metric_mean": float(summary_df.loc[0, "metric_mean"]),
        "metric_std": float(summary_df.loc[0, "metric_std"]),
        "higher_is_better": bool(summary_df.loc[0, "higher_is_better"]),
    }


def _load_run_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _require_manifest_value(manifest: dict[str, object], field_name: str) -> object:
    field_value = manifest.get(field_name)
    if field_value is not None:
        return field_value

    config_snapshot = manifest.get("config_snapshot", {})
    if isinstance(config_snapshot, dict):
        field_value = config_snapshot.get(field_name)
    if field_value is None:
        raise ValueError(
            f"Run manifest is missing required submission field '{field_name}'. "
            "Submission requires a manifest-backed artifact contract."
        )
    return field_value


def _infer_legacy_model_id(task_type: str, model_name: object | None) -> str | None:
    resolved_model_name = str(model_name).strip()
    legacy_model_map = {
        ("regression", "Ridge"): "onehot_ridge",
        ("regression", "ElasticNet"): "onehot_elasticnet",
        ("regression", "RandomForestRegressor"): "ordinal_randomforest",
        ("regression", "ExtraTreesRegressor"): "ordinal_extratrees",
        ("regression", "HistGradientBoostingRegressor"): "ordinal_hgb",
        ("regression", "LGBMRegressor"): "ordinal_lightgbm",
        ("regression", "CatBoostRegressor"): "native_catboost",
        ("regression", "XGBRegressor"): "ordinal_xgboost",
        ("binary", "LogisticRegression"): "onehot_logreg",
        ("binary", "RandomForestClassifier"): "ordinal_randomforest",
        ("binary", "ExtraTreesClassifier"): "ordinal_extratrees",
        ("binary", "HistGradientBoostingClassifier"): "ordinal_hgb",
        ("binary", "LGBMClassifier"): "ordinal_lightgbm",
        ("binary", "CatBoostClassifier"): "native_catboost",
        ("binary", "XGBClassifier"): "ordinal_xgboost",
        ("regression", "onehot_ridge"): "onehot_ridge",
        ("regression", "onehot_elasticnet"): "onehot_elasticnet",
        ("regression", "ordinal_randomforest"): "ordinal_randomforest",
        ("regression", "ordinal_extratrees"): "ordinal_extratrees",
        ("regression", "ordinal_hgb"): "ordinal_hgb",
        ("regression", "ordinal_lightgbm"): "ordinal_lightgbm",
        ("regression", "native_catboost"): "native_catboost",
        ("regression", "ordinal_xgboost"): "ordinal_xgboost",
        ("binary", "onehot_logreg"): "onehot_logreg",
        ("binary", "ordinal_randomforest"): "ordinal_randomforest",
        ("binary", "ordinal_extratrees"): "ordinal_extratrees",
        ("binary", "ordinal_hgb"): "ordinal_hgb",
        ("binary", "ordinal_lightgbm"): "ordinal_lightgbm",
        ("binary", "native_catboost"): "native_catboost",
        ("binary", "ordinal_xgboost"): "ordinal_xgboost",
        ("regression", "elasticnet"): "onehot_elasticnet",
        ("regression", "catboost"): "native_catboost",
        ("regression", "lightgbm"): "ordinal_lightgbm",
        ("regression", "random_forest"): "ordinal_randomforest",
        ("regression", "xgb"): "ordinal_xgboost",
        ("binary", "logistic_regression"): "onehot_logreg",
        ("binary", "catboost"): "native_catboost",
        ("binary", "lightgbm"): "ordinal_lightgbm",
        ("binary", "random_forest"): "ordinal_randomforest",
        ("binary", "xgb"): "ordinal_xgboost",
    }
    return legacy_model_map.get((task_type, resolved_model_name))


def _resolve_legacy_model_name(manifest: dict[str, object]) -> str | None:
    model_name = manifest.get("model_name")
    if model_name is not None:
        return str(model_name)

    config_snapshot = manifest.get("config_snapshot", {})
    if isinstance(config_snapshot, dict):
        config_model_name = config_snapshot.get("model_name")
        if config_model_name is not None:
            return str(config_model_name)

    return None


def _load_manifest_models(manifest: dict[str, object]) -> list[dict[str, object]]:
    manifest_models = manifest.get("models")
    if manifest_models is None:
        return []
    if not isinstance(manifest_models, list) or not all(isinstance(model, dict) for model in manifest_models):
        raise ValueError("Run manifest models must be a list of mappings.")
    return manifest_models


def _resolve_legacy_manifest_model_id(manifest: dict[str, object]) -> str:
    model_id = manifest.get("model_id")
    if model_id is None:
        config_snapshot = manifest.get("config_snapshot", {})
        if isinstance(config_snapshot, dict):
            model_id = config_snapshot.get("model_id")
    if model_id is not None:
        return str(model_id)

    task_type = str(_require_manifest_value(manifest, "task_type"))
    legacy_model_name = _resolve_legacy_model_name(manifest)
    legacy_model_id = _infer_legacy_model_id(task_type, legacy_model_name)
    if legacy_model_id is not None:
        return legacy_model_id

    raise ValueError(
        "Run manifest is missing required submission field 'model_id'. "
        f"Could not infer it from legacy model metadata for task_type '{task_type}' and model_name "
        f"{legacy_model_name!r}."
    )


def _resolve_selected_model_id(
    manifest: dict[str, object],
    requested_model_id: str | None = None,
) -> str:
    manifest_models = _load_manifest_models(manifest)
    if manifest_models:
        available_model_ids = [str(model["model_id"]) for model in manifest_models]
        if requested_model_id is not None:
            if requested_model_id not in available_model_ids:
                raise ValueError(
                    f"Requested model_id '{requested_model_id}' is not present in the run manifest. "
                    f"Available model_ids: {available_model_ids}"
                )
            return requested_model_id

        best_model_id = manifest.get("best_model_id")
        if best_model_id is None and len(available_model_ids) == 1:
            return available_model_ids[0]
        if best_model_id not in available_model_ids:
            raise ValueError(
                "Run manifest best_model_id must match one of the available model entries. "
                f"Available model_ids: {available_model_ids}"
            )
        return str(best_model_id)

    resolved_model_id = _resolve_legacy_manifest_model_id(manifest)
    if requested_model_id is not None and requested_model_id != resolved_model_id:
        raise ValueError(
            f"Requested model_id '{requested_model_id}' does not match the legacy run model_id '{resolved_model_id}'."
        )
    return resolved_model_id


def _load_submission_run_context(
    run_dir: Path,
    model_id: str | None = None,
) -> SubmissionRunContext:
    manifest = _load_run_manifest(run_dir)
    run_id = str(manifest["run_id"])
    selected_model_id = _resolve_selected_model_id(manifest, requested_model_id=model_id)
    observed_label_pair_raw = manifest.get("observed_label_pair")
    observed_label_pair = None
    if isinstance(observed_label_pair_raw, list):
        if len(observed_label_pair_raw) != 2:
            raise ValueError("Run manifest observed_label_pair must contain exactly two labels when present.")
        observed_label_pair = (observed_label_pair_raw[0], observed_label_pair_raw[1])
    elif manifest.get("negative_label") is not None and manifest.get("positive_label") is not None:
        observed_label_pair = (manifest["negative_label"], manifest["positive_label"])
    return SubmissionRunContext(
        run_id=run_id,
        competition_slug=str(_require_manifest_value(manifest, "competition_slug")),
        task_type=str(_require_manifest_value(manifest, "task_type")),
        primary_metric=str(_require_manifest_value(manifest, "primary_metric")),
        model_id=selected_model_id,
        id_column=str(_require_manifest_value(manifest, "id_column")),
        label_column=str(_require_manifest_value(manifest, "label_column")),
        observed_label_pair=observed_label_pair,
        config_fingerprint=manifest.get("config_fingerprint"),
    )


def _load_run_metadata(
    run_dir: Path,
    model_id: str | None = None,
) -> dict[str, object]:
    manifest = _load_run_manifest(run_dir)
    selected_model_id = _resolve_selected_model_id(manifest, requested_model_id=model_id)
    manifest_models = _load_manifest_models(manifest)

    if manifest_models:
        selected_model = next(
            (model for model in manifest_models if str(model["model_id"]) == selected_model_id),
            None,
        )
        if selected_model is None:
            raise ValueError(
                f"Requested model_id '{selected_model_id}' is missing from the run manifest model entries."
            )
        cv_summary = selected_model.get("cv_summary")
        if not isinstance(cv_summary, dict):
            raise ValueError("Run manifest model cv_summary must be a mapping.")
        return {
            "run_id": str(manifest["run_id"]),
            "model_id": selected_model_id,
            "config_fingerprint": manifest.get("config_fingerprint"),
            "model_name": str(selected_model["model_name"]),
            "metric_name": str(cv_summary["metric_name"]),
            "metric_mean": float(cv_summary["metric_mean"]),
            "metric_std": float(cv_summary["metric_std"]) if cv_summary.get("metric_std") is not None else None,
            "higher_is_better": cv_summary.get("higher_is_better"),
        }

    model_name = _resolve_legacy_model_name(manifest)
    cv_summary = manifest.get("cv_summary")
    if isinstance(cv_summary, dict):
        metric_name = cv_summary.get("metric_name")
        metric_mean = cv_summary.get("metric_mean")
        metric_std = cv_summary.get("metric_std")
        higher_is_better = cv_summary.get("higher_is_better")
        if model_name is not None and metric_name is not None and metric_mean is not None:
            return {
                "run_id": str(manifest["run_id"]),
                "model_id": selected_model_id,
                "config_fingerprint": manifest.get("config_fingerprint"),
                "model_name": str(model_name),
                "metric_name": str(metric_name),
                "metric_mean": float(metric_mean),
                "metric_std": float(metric_std) if metric_std is not None else None,
                "higher_is_better": higher_is_better,
            }

    legacy_summary = _load_legacy_cv_summary(run_dir)
    resolved_model_name = model_name if model_name is not None else legacy_summary["model_name"]
    return {
        "run_id": str(manifest["run_id"]),
        "model_id": selected_model_id,
        "config_fingerprint": manifest.get("config_fingerprint"),
        "model_name": str(resolved_model_name),
        "metric_name": str(legacy_summary["metric_name"]),
        "metric_mean": float(legacy_summary["metric_mean"]),
        "metric_std": float(legacy_summary["metric_std"]),
        "higher_is_better": legacy_summary["higher_is_better"],
    }


def _resolve_prediction_path(
    run_dir: Path,
    model_id: str,
) -> Path:
    manifest = _load_run_manifest(run_dir)
    manifest_models = _load_manifest_models(manifest)
    model_prediction_path = run_dir / model_id / "test_predictions.csv"
    if model_prediction_path.exists():
        return model_prediction_path

    legacy_prediction_path = run_dir / "test_predictions.csv"
    if not manifest_models and legacy_prediction_path.exists():
        return legacy_prediction_path

    raise ValueError(
        "Missing test predictions file for submission. "
        f"Checked {model_prediction_path} and {legacy_prediction_path}"
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
            "Binary label submissions require observed_label_pair metadata in the run manifest."
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


def prepare_submission_file(
    run_dir: Path,
    model_id: str | None = None,
) -> Path:
    run_context = _load_submission_run_context(run_dir=run_dir, model_id=model_id)
    prediction_path = _resolve_prediction_path(run_dir=run_dir, model_id=run_context.model_id)
    prediction_df = pd.read_csv(prediction_path)
    sample_submission_df = load_sample_submission_template(run_context.competition_slug)
    validate_sample_submission_schema(
        sample_submission_df=sample_submission_df,
        id_column=run_context.id_column,
        label_column=run_context.label_column,
    )

    expected_columns = [run_context.id_column, run_context.label_column]
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
    if run_context.task_type == "binary":
        prediction_values = prediction_df[run_context.label_column]
        binary_prediction_kind = get_binary_prediction_kind(run_context.primary_metric)
        if binary_prediction_kind == "probability":
            _validate_binary_probability_predictions(prediction_values)
        else:
            _validate_binary_label_predictions(
                prediction_values=prediction_values,
                observed_label_pair=run_context.observed_label_pair,
            )
    _validate_submission_ids(
        prediction_df=prediction_df,
        sample_submission_df=sample_submission_df,
        id_column=run_context.id_column,
    )

    submission_path = prediction_path.parent / "submission.csv"
    prediction_df.to_csv(submission_path, index=False)
    return submission_path


def build_submission_message(
    run_dir: Path,
    submit_message_prefix: str | None = None,
    model_id: str | None = None,
) -> str:
    run_metadata = _load_run_metadata(run_dir=run_dir, model_id=model_id)
    message_parts = []
    if submit_message_prefix:
        message_parts.append(submit_message_prefix.strip())
    message_parts.append(f"run={run_metadata['run_id']}")
    message_parts.append(f"model={run_metadata['model_id']}")
    message_parts.append(f"{run_metadata['metric_name']}={run_metadata['metric_mean']:.6f}")
    return " | ".join(message_parts)


def run_submission(
    config: AppConfig,
    run_dir: Path,
    model_id: str | None = None,
) -> tuple[Path, str]:
    run_context = _load_submission_run_context(run_dir=run_dir, model_id=model_id)
    run_metadata = _load_run_metadata(run_dir=run_dir, model_id=model_id)
    submission_path = prepare_submission_file(run_dir=run_dir, model_id=model_id)
    message = build_submission_message(
        run_dir=run_dir,
        submit_message_prefix=config.submit_message_prefix,
        model_id=model_id,
    )

    if config.submit_enabled:
        completed = subprocess.run(
            [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                run_context.competition_slug,
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
        "competition_slug": run_context.competition_slug,
        "run_id": run_context.run_id,
        "model_id": run_context.model_id,
        "model_name": run_metadata["model_name"],
        "config_fingerprint": run_context.config_fingerprint,
        "submission_path": str(submission_path),
        "submit_enabled": config.submit_enabled,
        "status": status,
        "message": message,
    }
    ledger_path = Path("artifacts") / run_context.competition_slug / "train" / "submissions.csv"
    _append_submission_ledger(ledger_path=ledger_path, row=ledger_row)

    return submission_path, status
