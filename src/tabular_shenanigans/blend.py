import json
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.candidate_artifacts import (
    BINARY_ACCURACY_BLEND_RULE,
    CANDIDATE_ARTIFACT_DIRNAME,
    build_base_config_snapshot,
    build_binary_accuracy_artifact_metadata,
    build_config_fingerprint as make_candidate_config_fingerprint,
    build_target_summary,
    write_candidate_artifacts as write_common_candidate_artifacts,
    write_context_artifacts,
)
from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.mlflow_store import (
    CandidateRunRef,
    create_candidate_run,
    download_candidate_bundle,
    log_candidate_run,
    terminate_run,
)
from tabular_shenanigans.preprocess import prepare_feature_frames

BLEND_REGISTRY_KEY = "blend_weighted_average"
BLEND_ESTIMATOR_NAME = "WeightedAverageBlend"
BLEND_PREPROCESSING_SCHEME_ID = "blend"


@dataclass(frozen=True)
class BlendComponent:
    candidate_id: str
    candidate_type: str
    mlflow_run_id: str
    config_fingerprint: str | None
    model_registry_key: str
    estimator_name: str
    feature_recipe_id: str | None
    cv_metric_mean: float
    cv_metric_std: float
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


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


def _series_matches_expected(observed: pd.Series, expected: pd.Series) -> bool:
    observed_reset = observed.reset_index(drop=True)
    expected_reset = expected.reset_index(drop=True)

    if pd.api.types.is_numeric_dtype(observed_reset) and pd.api.types.is_numeric_dtype(expected_reset):
        observed_values = observed_reset.to_numpy(dtype=float)
        expected_values = expected_reset.to_numpy(dtype=float)
        return np.allclose(observed_values, expected_values, equal_nan=True)

    observed_values = observed_reset.map(_normalize_binary_label)
    expected_values = expected_reset.map(_normalize_binary_label)
    return observed_values.equals(expected_values)


def _load_oof_predictions(candidate_artifact_dir: Path) -> pd.DataFrame:
    oof_path = candidate_artifact_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise ValueError(f"Missing OOF predictions required for blending: {oof_path}")
    oof_df = pd.read_csv(oof_path)
    expected_columns = ["row_idx", "y_true", "y_pred", "fold"]
    if oof_df.columns.tolist() != expected_columns:
        raise ValueError(
            f"OOF predictions must have columns {expected_columns} for blending: {oof_path}"
        )
    return oof_df


def _load_test_predictions(candidate_artifact_dir: Path, id_column: str, label_column: str) -> pd.DataFrame:
    prediction_path = candidate_artifact_dir / "test_predictions.csv"
    if not prediction_path.exists():
        raise ValueError(f"Missing test predictions required for blending: {prediction_path}")
    prediction_df = pd.read_csv(prediction_path)
    expected_columns = [id_column, label_column]
    if prediction_df.columns.tolist() != expected_columns:
        raise ValueError(
            "Blend base candidate test predictions do not match the resolved submission schema. "
            f"Expected columns {expected_columns}, got {prediction_df.columns.tolist()} in {prediction_path}"
        )
    return prediction_df


def _load_binary_accuracy_test_probabilities(
    manifest: dict[str, object],
    candidate_artifact_dir: Path,
    candidate_id: str,
    id_column: str,
    label_column: str,
) -> pd.DataFrame:
    blend_rule = manifest.get("binary_accuracy_blend_rule")
    if blend_rule != BINARY_ACCURACY_BLEND_RULE:
        raise ValueError(
            "Binary accuracy blends require base candidates written with the current probability-based "
            f"aggregation contract. Candidate {candidate_id} had binary_accuracy_blend_rule={blend_rule!r}."
        )

    probability_path_name = manifest.get("binary_accuracy_test_probability_path")
    if not isinstance(probability_path_name, str) or not probability_path_name:
        raise ValueError(
            "Binary accuracy blends require base candidates with test probability artifacts from the current "
            f"runtime. Candidate {candidate_id} is missing binary_accuracy_test_probability_path; retrain it."
        )

    probability_path = candidate_artifact_dir / probability_path_name
    if not probability_path.exists():
        raise ValueError(
            "Binary accuracy blends require the configured test probability artifact to exist. "
            f"Candidate {candidate_id} is missing {probability_path}."
        )

    probability_df = pd.read_csv(probability_path)
    expected_columns = [id_column, label_column]
    if probability_df.columns.tolist() != expected_columns:
        raise ValueError(
            "Binary accuracy blend probability artifacts must match the resolved schema. "
            f"Expected columns {expected_columns}, got {probability_df.columns.tolist()} in {probability_path}"
        )
    return probability_df


def _resolve_blend_weights(
    base_candidate_ids: list[str],
    configured_weights: list[float] | None,
) -> list[float]:
    if configured_weights is None:
        equal_weight = 1.0 / len(base_candidate_ids)
        return [equal_weight] * len(base_candidate_ids)

    if not np.isfinite(np.asarray(configured_weights, dtype=float)).all():
        raise ValueError("Blend candidate weights must be finite.")

    weight_sum = float(sum(configured_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Blend candidate weights must sum to a positive value.")
    return [float(weight / weight_sum) for weight in configured_weights]


def _validate_binary_test_labels(
    prediction_values: pd.Series,
    positive_label: object,
    negative_label: object,
    candidate_id: str,
) -> None:
    normalized_positive_label = _normalize_binary_label(positive_label)
    normalized_negative_label = _normalize_binary_label(negative_label)
    normalized_predictions = prediction_values.map(_normalize_binary_label)
    invalid_labels = sorted(
        set(normalized_predictions) - {normalized_positive_label, normalized_negative_label}
    )
    if invalid_labels:
        raise ValueError(
            "Binary accuracy blends require base candidate test predictions to contain only the observed "
            f"class labels. Candidate {candidate_id} had invalid labels: {invalid_labels[:10]}"
        )


def _validate_binary_probability_values(
    prediction_values: pd.Series,
    candidate_id: str,
    artifact_name: str,
) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(prediction_values):
        raise ValueError(
            f"Binary probability artifacts must be numeric. Candidate {candidate_id}: {artifact_name}"
        )
    if not prediction_values.map(pd.notna).all():
        raise ValueError(
            f"Binary probability artifacts cannot contain missing values. Candidate {candidate_id}: {artifact_name}"
        )

    prediction_array = prediction_values.to_numpy(dtype=float)
    if not np.isfinite(prediction_array).all():
        raise ValueError(
            f"Binary probability artifacts must be finite. Candidate {candidate_id}: {artifact_name}"
        )
    if ((prediction_array < 0.0) | (prediction_array > 1.0)).any():
        raise ValueError(
            "Binary probability artifacts must stay within [0, 1]. "
            f"Candidate {candidate_id}: {artifact_name}"
        )
    return prediction_array


def _validate_binary_probability_label_contract(
    manifest: dict[str, object],
    candidate_id: str,
    positive_label: object | None,
    negative_label: object | None,
    expected_observed_label_pair: tuple[object, object] | None,
) -> None:
    if positive_label is None or negative_label is None or expected_observed_label_pair is None:
        raise ValueError("Binary probability blending requires resolved class metadata.")

    manifest_positive_label = manifest.get("positive_label")
    manifest_negative_label = manifest.get("negative_label")
    observed_label_pair = manifest.get("observed_label_pair")
    if (
        manifest_positive_label is None
        or manifest_negative_label is None
        or not isinstance(observed_label_pair, list)
        or len(observed_label_pair) != 2
    ):
        raise ValueError(
            "Binary probability blend candidates must include positive_label, negative_label, "
            f"and observed_label_pair metadata. Candidate {candidate_id} is missing that contract."
        )

    expected_contract = {
        "positive_label": _normalize_binary_label(positive_label),
        "negative_label": _normalize_binary_label(negative_label),
        "observed_label_pair": [
            _normalize_binary_label(label) for label in expected_observed_label_pair
        ],
    }
    observed_contract = {
        "positive_label": _normalize_binary_label(manifest_positive_label),
        "negative_label": _normalize_binary_label(manifest_negative_label),
        "observed_label_pair": [_normalize_binary_label(label) for label in observed_label_pair],
    }
    if observed_contract != expected_contract:
        raise ValueError(
            "Binary probability blend candidate label contract does not match the configured blend context. "
            f"Candidate {candidate_id}: expected {expected_contract}, observed {observed_contract}"
        )


def _load_blend_component(
    config: AppConfig,
    candidate_id: str,
    destination_dir: Path,
    task_type: str,
    primary_metric: str,
    id_column: str,
    label_column: str,
    expected_y_train: pd.Series,
    expected_fold_assignments: np.ndarray,
    expected_test_ids: pd.Series,
    positive_label: object | None,
    negative_label: object | None,
    observed_label_pair: tuple[object, object] | None,
) -> BlendComponent:
    downloaded_bundle = download_candidate_bundle(
        config=config,
        candidate_id=candidate_id,
        destination_dir=destination_dir,
    )
    manifest = downloaded_bundle.manifest
    candidate_artifact_dir = downloaded_bundle.candidate_artifact_dir

    manifest_competition_slug = manifest.get("competition_slug")
    manifest_task_type = manifest.get("task_type")
    manifest_primary_metric = manifest.get("primary_metric")
    manifest_id_column = manifest.get("id_column")
    manifest_label_column = manifest.get("label_column")
    if manifest_competition_slug != config.competition.slug:
        raise ValueError(
            "Blend base candidate competition_slug does not match the configured competition. "
            f"Candidate {candidate_id}: {manifest_competition_slug!r}"
        )
    if manifest_task_type != task_type:
        raise ValueError(
            "Blend base candidate task_type does not match the configured task. "
            f"Candidate {candidate_id}: {manifest_task_type!r}"
        )
    if manifest_primary_metric != primary_metric:
        raise ValueError(
            "Blend base candidate primary_metric does not match the configured metric. "
            f"Candidate {candidate_id}: {manifest_primary_metric!r}"
        )
    if manifest_id_column != id_column or manifest_label_column != label_column:
        raise ValueError(
            "Blend base candidate schema does not match the configured schema. "
            f"Candidate {candidate_id}: id_column={manifest_id_column!r}, label_column={manifest_label_column!r}"
        )

    oof_df = _load_oof_predictions(candidate_artifact_dir=candidate_artifact_dir)
    row_idx = oof_df["row_idx"].to_numpy(dtype=int)
    expected_row_idx = np.arange(expected_fold_assignments.shape[0], dtype=int)
    if not np.array_equal(row_idx, expected_row_idx):
        raise ValueError(
            "Blend base candidate OOF row_idx values must be sequential and aligned. "
            f"Candidate {candidate_id} had unexpected row_idx values."
        )
    if not _series_matches_expected(oof_df["y_true"], expected_y_train):
        raise ValueError(
            "Blend base candidate OOF targets do not match the configured training target. "
            f"Candidate {candidate_id} is not compatible."
        )
    observed_fold_assignments = oof_df["fold"].to_numpy(dtype=int)
    if not np.array_equal(observed_fold_assignments, expected_fold_assignments):
        raise ValueError(
            "Blend base candidate fold assignments do not match the resolved competition folds. "
            f"Candidate {candidate_id} is not compatible."
        )

    prediction_df = _load_test_predictions(
        candidate_artifact_dir=candidate_artifact_dir,
        id_column=id_column,
        label_column=label_column,
    )
    if not _series_matches_expected(prediction_df[id_column], expected_test_ids):
        raise ValueError(
            "Blend base candidate test IDs do not match the configured competition test set. "
            f"Candidate {candidate_id} is not compatible."
        )

    binary_prediction_kind = None
    if task_type == "binary":
        binary_prediction_kind = get_binary_prediction_kind(primary_metric)

    if binary_prediction_kind == "probability":
        _validate_binary_probability_label_contract(
            manifest=manifest,
            candidate_id=candidate_id,
            positive_label=positive_label,
            negative_label=negative_label,
            expected_observed_label_pair=observed_label_pair,
        )

    if binary_prediction_kind == "label":
        if positive_label is None or negative_label is None:
            raise ValueError("Binary accuracy blending requires resolved class metadata.")
        _validate_binary_test_labels(
            prediction_values=prediction_df[label_column],
            positive_label=positive_label,
            negative_label=negative_label,
            candidate_id=candidate_id,
        )
        _validate_binary_probability_label_contract(
            manifest=manifest,
            candidate_id=candidate_id,
            positive_label=positive_label,
            negative_label=negative_label,
            expected_observed_label_pair=observed_label_pair,
        )
        probability_df = _load_binary_accuracy_test_probabilities(
            manifest=manifest,
            candidate_artifact_dir=candidate_artifact_dir,
            candidate_id=candidate_id,
            id_column=id_column,
            label_column=label_column,
        )
        if not _series_matches_expected(probability_df[id_column], expected_test_ids):
            raise ValueError(
                "Binary accuracy blend probability artifact IDs do not match the configured competition test set. "
                f"Candidate {candidate_id} is not compatible."
            )
        test_predictions = _validate_binary_probability_values(
            prediction_values=probability_df[label_column],
            candidate_id=candidate_id,
            artifact_name="test_prediction_probabilities.csv",
        )
    else:
        if not pd.api.types.is_numeric_dtype(prediction_df[label_column]):
            raise ValueError(
                "Blend base candidate test predictions must be numeric for regression and binary "
                f"probability metrics. Candidate {candidate_id} was not numeric."
            )
        test_predictions = prediction_df[label_column].to_numpy(dtype=float)

    if task_type == "regression" or binary_prediction_kind == "probability":
        if not np.isfinite(test_predictions).all():
            raise ValueError(
                "Blend base candidate test predictions must be finite. "
                f"Candidate {candidate_id} contained non-finite values."
            )

    cv_summary = manifest.get("cv_summary")
    if not isinstance(cv_summary, dict):
        raise ValueError(f"Blend base candidate manifest must contain cv_summary. Candidate: {candidate_id}")

    return BlendComponent(
        candidate_id=candidate_id,
        candidate_type=str(manifest.get("candidate_type") or "model"),
        mlflow_run_id=downloaded_bundle.run_id,
        config_fingerprint=manifest.get("config_fingerprint"),
        model_registry_key=str(manifest.get("model_registry_key") or "candidate"),
        estimator_name=str(manifest.get("estimator_name") or "Candidate"),
        feature_recipe_id=manifest.get("feature_recipe_id"),
        cv_metric_mean=float(cv_summary["metric_mean"]),
        cv_metric_std=float(cv_summary["metric_std"]),
        oof_predictions=oof_df["y_pred"].to_numpy(dtype=float),
        test_predictions=test_predictions,
    )


def _build_fold_metrics(
    task_type: str,
    primary_metric: str,
    y_train: pd.Series,
    oof_predictions: np.ndarray,
    fold_assignments: np.ndarray,
    positive_label: object | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_indices = np.arange(fold_assignments.shape[0], dtype=int)
    for fold_index in sorted(int(fold) for fold in np.unique(fold_assignments).tolist()):
        valid_idx = row_indices[fold_assignments == fold_index]
        train_idx = row_indices[fold_assignments != fold_index]
        fold_score = score_predictions(
            task_type=task_type,
            primary_metric=primary_metric,
            y_true=y_train.iloc[valid_idx],
            y_pred=oof_predictions[valid_idx],
            positive_label=positive_label,
        )
        rows.append(
            {
                "fold": fold_index,
                "metric_name": primary_metric,
                "metric_value": fold_score,
                "train_rows": int(train_idx.shape[0]),
                "valid_rows": int(valid_idx.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _build_blend_summary(
    candidate_id: str,
    metric_name: str,
    metric_mean: float,
    metric_std: float,
    components: list[BlendComponent],
    normalized_weights: list[float],
) -> pd.DataFrame:
    if len(components) == 1:
        correlation_matrix = np.ones((1, 1), dtype=float)
    else:
        component_predictions = np.vstack([component.oof_predictions for component in components])
        correlation_matrix = np.corrcoef(component_predictions)

    rows: list[dict[str, object]] = []
    for component_index, component in enumerate(components):
        other_correlations = [
            float(correlation_matrix[component_index, other_index])
            for other_index in range(len(components))
            if other_index != component_index
        ]
        rows.append(
            {
                "blend_candidate_id": candidate_id,
                "blend_registry_key": BLEND_REGISTRY_KEY,
                "blend_metric_name": metric_name,
                "blend_metric_mean": metric_mean,
                "blend_metric_std": metric_std,
                "component_rank": component_index + 1,
                "component_candidate_id": component.candidate_id,
                "component_candidate_type": component.candidate_type,
                "component_mlflow_run_id": component.mlflow_run_id,
                "component_model_registry_key": component.model_registry_key,
                "component_estimator_name": component.estimator_name,
                "component_feature_recipe_id": component.feature_recipe_id,
                "component_config_fingerprint": component.config_fingerprint,
                "component_weight": normalized_weights[component_index],
                "component_cv_metric_mean": component.cv_metric_mean,
                "component_cv_metric_std": component.cv_metric_std,
                "avg_oof_correlation_to_others": (
                    float(np.mean(other_correlations)) if other_correlations else None
                ),
                "min_oof_correlation_to_others": (
                    float(np.min(other_correlations)) if other_correlations else None
                ),
                "max_oof_correlation_to_others": (
                    float(np.max(other_correlations)) if other_correlations else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_config_snapshot(
    config: AppConfig,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    normalized_weights: list[float],
) -> dict[str, object]:
    candidate = config.experiment.candidate
    config_snapshot = build_base_config_snapshot(
        config=config,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
    )
    config_snapshot["resolved_blend_components"] = [
        {
            "candidate_id": candidate_id,
            "weight": weight,
        }
        for candidate_id, weight in zip(candidate.base_candidate_ids, normalized_weights, strict=True)
    ]
    return config_snapshot


def _build_config_fingerprint(
    config_snapshot: dict[str, object],
    components: list[BlendComponent],
    normalized_weights: list[float],
) -> str:
    return make_candidate_config_fingerprint(
        {
            "config_snapshot": config_snapshot,
            "blend_components": [
                {
                    "candidate_id": component.candidate_id,
                    "config_fingerprint": component.config_fingerprint,
                    "weight": weight,
                }
                for component, weight in zip(components, normalized_weights, strict=True)
            ],
            "model_registry_key": BLEND_REGISTRY_KEY,
        }
    )


def _build_candidate_manifest(
    config: AppConfig,
    generated_at_utc: str,
    config_snapshot: dict[str, object],
    config_fingerprint: str,
    metric_mean: float,
    metric_std: float,
    observed_label_pair: tuple[object, object] | None,
    negative_label: object | None,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    target_summary: dict[str, object],
    feature_columns: list[str],
    train_rows: int,
    train_cols: int,
    test_rows: int,
    test_cols: int,
    components: list[BlendComponent],
    normalized_weights: list[float],
    mlflow_run_id: str,
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
        "mlflow_run_id": mlflow_run_id,
        "feature_columns": feature_columns,
        "model_registry_key": BLEND_REGISTRY_KEY,
        "estimator_name": BLEND_ESTIMATOR_NAME,
        "preprocessing_scheme_id": BLEND_PREPROCESSING_SCHEME_ID,
        "cv_summary": {
            "metric_name": competition.primary_metric,
            "metric_mean": metric_mean,
            "metric_std": metric_std,
            "higher_is_better": is_higher_better(competition.primary_metric),
        },
        "component_candidates": [
            {
                "candidate_id": component.candidate_id,
                "candidate_type": component.candidate_type,
                "mlflow_run_id": component.mlflow_run_id,
                "config_fingerprint": component.config_fingerprint,
                "model_registry_key": component.model_registry_key,
                "estimator_name": component.estimator_name,
                "feature_recipe_id": component.feature_recipe_id,
                "weight": weight,
                "cv_summary": {
                    "metric_name": competition.primary_metric,
                    "metric_mean": component.cv_metric_mean,
                    "metric_std": component.cv_metric_std,
                    "higher_is_better": is_higher_better(competition.primary_metric),
                },
            }
            for component, weight in zip(components, normalized_weights, strict=True)
        ],
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
    }
    manifest.update(
        build_binary_accuracy_artifact_metadata(
            task_type=competition.task_type,
            primary_metric=competition.primary_metric,
        )
    )
    return manifest


def _write_runtime_config(bundle_root: Path, config: AppConfig) -> None:
    config_dir = bundle_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "runtime_config.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _stage_blend_bundle(
    bundle_root: Path,
    config: AppConfig,
    candidate_manifest: dict[str, object],
    competition_manifest: dict[str, object],
    fold_assignments: np.ndarray,
    fold_metrics_df: pd.DataFrame,
    y_train: pd.Series,
    oof_predictions: np.ndarray,
    test_ids: pd.Series,
    test_predictions: np.ndarray,
    id_column: str,
    label_column: str,
    blend_summary_df: pd.DataFrame,
    test_prediction_probabilities: np.ndarray | None,
) -> None:
    _write_runtime_config(bundle_root=bundle_root, config=config)
    write_context_artifacts(
        bundle_root=bundle_root,
        competition_manifest=competition_manifest,
        fold_assignments=fold_assignments,
    )
    candidate_artifact_dir = bundle_root / CANDIDATE_ARTIFACT_DIRNAME
    write_common_candidate_artifacts(
        candidate_artifact_dir=candidate_artifact_dir,
        manifest=candidate_manifest,
        fold_metrics_df=fold_metrics_df,
        y_train=y_train,
        oof_predictions=oof_predictions,
        fold_assignments=fold_assignments,
        test_ids=test_ids,
        test_predictions=test_predictions,
        id_column=id_column,
        label_column=label_column,
        test_prediction_probabilities=test_prediction_probabilities,
    )
    blend_summary_df.to_csv(candidate_artifact_dir / "blend_summary.csv", index=False)


def run_blend_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> CandidateRunRef:
    if not config.is_blend_candidate:
        raise ValueError("Blend training requires experiment.candidate.candidate_type=blend.")

    competition = config.competition
    features = competition.features
    candidate = config.experiment.candidate
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column
    y_train = train_df[label_column].reset_index(drop=True)

    x_train_raw, x_test_raw, _ = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=features.force_categorical,
        force_numeric=features.force_numeric,
        drop_columns=features.drop_columns,
    )
    prepared_context = ensure_prepared_competition_context(
        config=config,
        dataset_context=dataset_context,
        expected_feature_columns=x_train_raw.columns.tolist(),
    )
    fold_assignments = prepared_context.fold_assignments

    positive_label = competition.positive_label
    negative_label = None
    observed_label_pair = None
    if competition.task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    normalized_weights = _resolve_blend_weights(
        base_candidate_ids=candidate.base_candidate_ids,
        configured_weights=candidate.weights,
    )
    fit_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-blend-components-") as component_dir:
        components = [
            _load_blend_component(
                config=config,
                candidate_id=base_candidate_id,
                destination_dir=Path(component_dir) / base_candidate_id,
                task_type=competition.task_type,
                primary_metric=competition.primary_metric,
                id_column=id_column,
                label_column=label_column,
                expected_y_train=y_train,
                expected_fold_assignments=fold_assignments,
                expected_test_ids=test_df[id_column],
                positive_label=positive_label,
                negative_label=negative_label,
                observed_label_pair=observed_label_pair,
            )
            for base_candidate_id in candidate.base_candidate_ids
        ]

        component_oof_predictions = np.vstack([component.oof_predictions for component in components])
        component_test_predictions = np.vstack([component.test_predictions for component in components])
        weight_array = np.asarray(normalized_weights, dtype=float)
        blended_oof_predictions = np.average(component_oof_predictions, axis=0, weights=weight_array)
        blended_test_predictions = np.average(component_test_predictions, axis=0, weights=weight_array)
        fit_wall_seconds = time.perf_counter() - fit_started

    test_prediction_probabilities = None
    if competition.task_type == "regression":
        final_test_predictions: np.ndarray | list[object] = blended_test_predictions
        if competition.primary_metric == "rmsle":
            final_test_predictions = np.clip(blended_test_predictions, a_min=0.0, a_max=None)
    elif get_binary_prediction_kind(competition.primary_metric) == "label":
        if positive_label is None or negative_label is None:
            raise ValueError("Binary label blends require resolved class metadata.")
        test_prediction_probabilities = np.asarray(blended_test_predictions, dtype=float)
        final_test_predictions = np.where(
            blended_test_predictions >= 0.5,
            positive_label,
            negative_label,
        )
    else:
        final_test_predictions = blended_test_predictions

    fold_metrics_df = _build_fold_metrics(
        task_type=competition.task_type,
        primary_metric=competition.primary_metric,
        y_train=y_train,
        oof_predictions=blended_oof_predictions,
        fold_assignments=fold_assignments,
        positive_label=positive_label,
    )
    metric_mean = float(fold_metrics_df["metric_value"].mean())
    metric_std = float(fold_metrics_df["metric_value"].std(ddof=0))
    blend_summary_df = _build_blend_summary(
        candidate_id=candidate.candidate_id,
        metric_name=competition.primary_metric,
        metric_mean=metric_mean,
        metric_std=metric_std,
        components=components,
        normalized_weights=normalized_weights,
    )
    print(
        f"Blend candidate: {candidate.candidate_id} | "
        f"components={candidate.base_candidate_ids} | "
        f"weights={normalized_weights} | "
        f"CV {competition.primary_metric}: mean={metric_mean:.6f}, std={metric_std:.6f}"
    )

    target_summary = build_target_summary(
        task_type=competition.task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )
    config_snapshot = _build_config_snapshot(
        config=config,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        normalized_weights=normalized_weights,
    )
    config_fingerprint = _build_config_fingerprint(
        config_snapshot=config_snapshot,
        components=components,
        normalized_weights=normalized_weights,
    )
    candidate_run = create_candidate_run(
        config=config,
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
    )
    candidate_manifest = _build_candidate_manifest(
        config=config,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        config_snapshot=config_snapshot,
        config_fingerprint=config_fingerprint,
        metric_mean=metric_mean,
        metric_std=metric_std,
        observed_label_pair=observed_label_pair,
        negative_label=negative_label,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        target_summary=target_summary,
        feature_columns=x_train_raw.columns.tolist(),
        train_rows=int(train_df.shape[0]),
        train_cols=int(x_train_raw.shape[1]),
        test_rows=int(test_df.shape[0]),
        test_cols=int(x_test_raw.shape[1]),
        components=components,
        normalized_weights=normalized_weights,
        mlflow_run_id=candidate_run.run_id,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="tabular-shenanigans-blend-candidate-") as temp_dir:
            bundle_root = Path(temp_dir)
            _stage_blend_bundle(
                bundle_root=bundle_root,
                config=config,
                candidate_manifest=candidate_manifest,
                competition_manifest=prepared_context.manifest,
                fold_assignments=fold_assignments,
                fold_metrics_df=fold_metrics_df,
                y_train=y_train,
                oof_predictions=blended_oof_predictions,
                test_ids=test_df[id_column],
                test_predictions=np.asarray(final_test_predictions),
                id_column=id_column,
                label_column=label_column,
                blend_summary_df=blend_summary_df,
                test_prediction_probabilities=test_prediction_probabilities,
            )
            log_candidate_run(
                config=config,
                candidate_run=candidate_run,
                bundle_root=bundle_root,
                manifest=candidate_manifest,
                fit_wall_seconds=fit_wall_seconds,
            )
        terminate_run(config=config, run_id=candidate_run.run_id, status="FINISHED")
        return candidate_run
    except Exception:
        terminate_run(config=config, run_id=candidate_run.run_id, status="FAILED")
        raise
