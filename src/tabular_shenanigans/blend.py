import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.competition import ensure_prepared_competition_context
from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import CompetitionDatasetContext, get_binary_prediction_kind
from tabular_shenanigans.preprocess import prepare_feature_frames

BLEND_MODEL_ID = "blend_weighted_average"
BLEND_MODEL_NAME = "WeightedAverageBlend"
BLEND_PREPROCESSING_SCHEME_ID = "blend"


@dataclass(frozen=True)
class BlendComponent:
    candidate_id: str
    candidate_type: str
    config_fingerprint: str | None
    model_id: str
    model_name: str
    feature_recipe_id: str | None
    cv_metric_mean: float
    cv_metric_std: float
    oof_predictions: np.ndarray
    test_predictions: np.ndarray


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_dir(competition_slug: str, candidate_id: str) -> Path:
    return Path("artifacts") / competition_slug / "candidates" / candidate_id


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


def _load_candidate_manifest(candidate_dir: Path) -> dict[str, object]:
    manifest_path = candidate_dir / "candidate.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing candidate manifest required for blending: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Candidate manifest must be a JSON object: {manifest_path}")
    return manifest


def _load_oof_predictions(candidate_dir: Path) -> pd.DataFrame:
    oof_path = candidate_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise ValueError(f"Missing OOF predictions required for blending: {oof_path}")
    oof_df = pd.read_csv(oof_path)
    expected_columns = ["row_idx", "y_true", "y_pred", "fold"]
    if oof_df.columns.tolist() != expected_columns:
        raise ValueError(
            f"OOF predictions must have columns {expected_columns} for blending: {oof_path}"
        )
    return oof_df


def _load_test_predictions(candidate_dir: Path, id_column: str, label_column: str) -> pd.DataFrame:
    prediction_path = candidate_dir / "test_predictions.csv"
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


def _build_target_summary(
    task_type: str,
    y_train: pd.Series,
    positive_label: object | None = None,
    negative_label: object | None = None,
    observed_label_pair: tuple[object, object] | None = None,
) -> dict[str, object]:
    if task_type == "regression":
        return {
            "target_mean": float(y_train.mean()),
            "target_std": float(y_train.std(ddof=0)),
            "target_min": float(y_train.min()),
            "target_max": float(y_train.max()),
        }

    if task_type == "binary":
        if positive_label is None or negative_label is None or observed_label_pair is None:
            raise ValueError("Binary target summary requires resolved label metadata.")
        positive_count = int((y_train == positive_label).sum())
        row_count = int(y_train.shape[0])
        negative_count = row_count - positive_count
        return {
            "observed_label_1": str(observed_label_pair[0]),
            "observed_label_2": str(observed_label_pair[1]),
            "negative_label": str(negative_label),
            "positive_label": str(positive_label),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "target_prevalence": float(positive_count / row_count),
        }

    raise ValueError(f"Unsupported task_type for target summary: {task_type}")


def _encode_binary_test_labels(
    prediction_values: pd.Series,
    positive_label: object,
    negative_label: object,
    candidate_id: str,
) -> np.ndarray:
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
    return normalized_predictions.map(
        {
            normalized_negative_label: 0.0,
            normalized_positive_label: 1.0,
        }
    ).to_numpy(dtype=float)


def _load_blend_component(
    competition_slug: str,
    candidate_id: str,
    task_type: str,
    primary_metric: str,
    id_column: str,
    label_column: str,
    expected_y_train: pd.Series,
    expected_fold_assignments: np.ndarray,
    expected_test_ids: pd.Series,
    positive_label: object | None,
    negative_label: object | None,
) -> BlendComponent:
    candidate_dir = _candidate_dir(competition_slug=competition_slug, candidate_id=candidate_id)
    manifest = _load_candidate_manifest(candidate_dir=candidate_dir)

    manifest_competition_slug = manifest.get("competition_slug")
    manifest_task_type = manifest.get("task_type")
    manifest_primary_metric = manifest.get("primary_metric")
    manifest_id_column = manifest.get("id_column")
    manifest_label_column = manifest.get("label_column")
    if manifest_competition_slug != competition_slug:
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

    oof_df = _load_oof_predictions(candidate_dir=candidate_dir)
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
            "Blend base candidate fold assignments do not match the prepared competition folds. "
            f"Candidate {candidate_id} is not compatible."
        )

    prediction_df = _load_test_predictions(
        candidate_dir=candidate_dir,
        id_column=id_column,
        label_column=label_column,
    )
    if not _series_matches_expected(prediction_df[id_column], expected_test_ids):
        raise ValueError(
            "Blend base candidate test IDs do not match the configured competition test set. "
            f"Candidate {candidate_id} is not compatible."
        )

    if task_type == "binary" and get_binary_prediction_kind(primary_metric) == "label":
        if positive_label is None or negative_label is None:
            raise ValueError("Binary accuracy blending requires resolved class metadata.")
        test_predictions = _encode_binary_test_labels(
            prediction_values=prediction_df[label_column],
            positive_label=positive_label,
            negative_label=negative_label,
            candidate_id=candidate_id,
        )
    else:
        if not pd.api.types.is_numeric_dtype(prediction_df[label_column]):
            raise ValueError(
                "Blend base candidate test predictions must be numeric for regression and binary "
                f"probability metrics. Candidate {candidate_id} was not numeric."
            )
        test_predictions = prediction_df[label_column].to_numpy(dtype=float)

    if task_type == "regression" or (
        task_type == "binary" and get_binary_prediction_kind(primary_metric) == "probability"
    ):
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
        config_fingerprint=manifest.get("config_fingerprint"),
        model_id=str(manifest.get("model_id") or "candidate"),
        model_name=str(manifest.get("model_name") or "Candidate"),
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
                "blend_model_id": BLEND_MODEL_ID,
                "blend_metric_name": metric_name,
                "blend_metric_mean": metric_mean,
                "blend_metric_std": metric_std,
                "component_rank": component_index + 1,
                "component_candidate_id": component.candidate_id,
                "component_candidate_type": component.candidate_type,
                "component_model_id": component.model_id,
                "component_model_name": component.model_name,
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
    return {
        "competition": {
            **config.competition.model_dump(mode="python"),
            "primary_metric": config.primary_metric,
            "positive_label": positive_label,
            "id_column": id_column,
            "label_column": label_column,
        },
        "experiment": config.experiment.model_dump(mode="python"),
        "resolved_blend_components": [
            {
                "candidate_id": candidate_id,
                "weight": weight,
            }
            for candidate_id, weight in zip(config.base_candidate_ids, normalized_weights, strict=True)
        ],
    }


def _build_config_fingerprint(
    config_snapshot: dict[str, object],
    components: list[BlendComponent],
    normalized_weights: list[float],
) -> str:
    fingerprint_payload = {
        "config_snapshot": config_snapshot,
        "blend_components": [
            {
                "candidate_id": component.candidate_id,
                "config_fingerprint": component.config_fingerprint,
                "weight": weight,
            }
            for component, weight in zip(components, normalized_weights, strict=True)
        ],
        "model_id": BLEND_MODEL_ID,
    }
    fingerprint_payload_json = json.dumps(_json_ready(fingerprint_payload), sort_keys=True)
    return hashlib.sha256(fingerprint_payload_json.encode("utf-8")).hexdigest()[:12]


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
    train_rows: int,
    test_rows: int,
    components: list[BlendComponent],
    normalized_weights: list[float],
) -> dict[str, object]:
    return {
        "artifact_type": "candidate",
        "candidate_id": config.candidate_id,
        "candidate_type": config.candidate_type,
        "generated_at_utc": generated_at_utc,
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "model_id": BLEND_MODEL_ID,
        "model_name": BLEND_MODEL_NAME,
        "preprocessing_scheme_id": BLEND_PREPROCESSING_SCHEME_ID,
        "cv_summary": {
            "metric_name": config.primary_metric,
            "metric_mean": metric_mean,
            "metric_std": metric_std,
            "higher_is_better": is_higher_better(config.primary_metric),
        },
        "component_candidates": [
            {
                "candidate_id": component.candidate_id,
                "candidate_type": component.candidate_type,
                "config_fingerprint": component.config_fingerprint,
                "model_id": component.model_id,
                "model_name": component.model_name,
                "feature_recipe_id": component.feature_recipe_id,
                "weight": weight,
                "cv_summary": {
                    "metric_name": config.primary_metric,
                    "metric_mean": component.cv_metric_mean,
                    "metric_std": component.cv_metric_std,
                    "higher_is_better": is_higher_better(config.primary_metric),
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
        "train_cols": None,
        "test_rows": test_rows,
        "test_cols": None,
    }


def _write_candidate_artifacts(
    candidate_dir: Path,
    manifest: dict[str, object],
    fold_metrics_df: pd.DataFrame,
    y_train: pd.Series,
    oof_predictions: np.ndarray,
    fold_assignments: np.ndarray,
    test_ids: pd.Series,
    test_predictions: np.ndarray,
    id_column: str,
    label_column: str,
    blend_summary_df: pd.DataFrame,
) -> None:
    (candidate_dir / "candidate.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fold_metrics_df.to_csv(candidate_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(y_train.shape[0], dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": oof_predictions,
            "fold": fold_assignments,
        }
    )
    oof_df.to_csv(candidate_dir / "oof_predictions.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            label_column: test_predictions,
        }
    )
    test_predictions_df.to_csv(candidate_dir / "test_predictions.csv", index=False)
    blend_summary_df.to_csv(candidate_dir / "blend_summary.csv", index=False)


def run_blend_training(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> Path:
    if not config.is_blend_candidate:
        raise ValueError("Blend training requires experiment.candidate.candidate_type=blend.")

    candidate_dir = _candidate_dir(config.competition_slug, config.candidate_id)
    if candidate_dir.exists():
        raise ValueError(
            "Candidate artifacts already exist for this candidate_id. "
            f"Choose a new experiment.candidate.candidate_id or remove {candidate_dir}"
        )

    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column
    y_train = train_df[label_column].reset_index(drop=True)

    x_train_raw, _, _ = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )
    prepared_context = ensure_prepared_competition_context(
        config=config,
        dataset_context=dataset_context,
        expected_feature_columns=x_train_raw.columns.tolist(),
    )
    fold_assignments = prepared_context.fold_assignments

    positive_label = config.positive_label
    negative_label = None
    observed_label_pair = None
    if config.task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    normalized_weights = _resolve_blend_weights(
        base_candidate_ids=config.base_candidate_ids,
        configured_weights=config.blend_weights,
    )
    components = [
        _load_blend_component(
            competition_slug=config.competition_slug,
            candidate_id=base_candidate_id,
            task_type=config.task_type,
            primary_metric=config.primary_metric,
            id_column=id_column,
            label_column=label_column,
            expected_y_train=y_train,
            expected_fold_assignments=fold_assignments,
            expected_test_ids=test_df[id_column],
            positive_label=positive_label,
            negative_label=negative_label,
        )
        for base_candidate_id in config.base_candidate_ids
    ]

    component_oof_predictions = np.vstack([component.oof_predictions for component in components])
    component_test_predictions = np.vstack([component.test_predictions for component in components])
    weight_array = np.asarray(normalized_weights, dtype=float)
    blended_oof_predictions = np.average(component_oof_predictions, axis=0, weights=weight_array)
    blended_test_predictions = np.average(component_test_predictions, axis=0, weights=weight_array)

    if config.task_type == "regression":
        final_test_predictions: np.ndarray | list[object] = blended_test_predictions
        if config.primary_metric == "rmsle":
            final_test_predictions = np.clip(blended_test_predictions, a_min=0.0, a_max=None)
    elif get_binary_prediction_kind(config.primary_metric) == "label":
        if positive_label is None or negative_label is None:
            raise ValueError("Binary label blends require resolved class metadata.")
        final_test_predictions = np.where(
            blended_test_predictions >= 0.5,
            positive_label,
            negative_label,
        )
    else:
        final_test_predictions = blended_test_predictions

    fold_metrics_df = _build_fold_metrics(
        task_type=config.task_type,
        primary_metric=config.primary_metric,
        y_train=y_train,
        oof_predictions=blended_oof_predictions,
        fold_assignments=fold_assignments,
        positive_label=positive_label,
    )
    metric_mean = float(fold_metrics_df["metric_value"].mean())
    metric_std = float(fold_metrics_df["metric_value"].std(ddof=0))
    blend_summary_df = _build_blend_summary(
        candidate_id=config.candidate_id,
        metric_name=config.primary_metric,
        metric_mean=metric_mean,
        metric_std=metric_std,
        components=components,
        normalized_weights=normalized_weights,
    )

    target_summary = _build_target_summary(
        task_type=config.task_type,
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
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    candidate_manifest = _build_candidate_manifest(
        config=config,
        generated_at_utc=generated_at_utc,
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
        train_rows=int(train_df.shape[0]),
        test_rows=int(test_df.shape[0]),
        components=components,
        normalized_weights=normalized_weights,
    )

    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=False, exist_ok=False)
    _write_candidate_artifacts(
        candidate_dir=candidate_dir,
        manifest=candidate_manifest,
        fold_metrics_df=fold_metrics_df,
        y_train=y_train,
        oof_predictions=blended_oof_predictions,
        fold_assignments=fold_assignments,
        test_ids=test_df[id_column],
        test_predictions=np.asarray(final_test_predictions),
        id_column=id_column,
        label_column=label_column,
        blend_summary_df=blend_summary_df,
    )
    print(
        f"Blend candidate: {config.candidate_id} | "
        f"components={config.base_candidate_ids} | "
        f"weights={normalized_weights} | "
        f"CV {config.primary_metric}: mean={metric_mean:.6f}, std={metric_std:.6f}"
    )
    return candidate_dir
