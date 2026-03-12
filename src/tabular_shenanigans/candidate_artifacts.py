import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.config import AppConfig

CANDIDATE_ARTIFACT_DIRNAME = "candidate"
CONTEXT_ARTIFACT_DIRNAME = "context"
CANDIDATE_MANIFEST_FILENAME = "candidate.json"
COMPETITION_MANIFEST_FILENAME = "competition.json"
FOLDS_FILENAME = "folds.csv"
BINARY_ACCURACY_TEST_PROBABILITIES_FILENAME = "test_prediction_probabilities.csv"
BINARY_ACCURACY_BLEND_RULE = "average_positive_class_probability_then_threshold_0.5"


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_candidate_manifest(
    candidate_artifact_dir: Path,
    missing_message: str | None = None,
) -> dict[str, object]:
    manifest_path = candidate_artifact_dir / CANDIDATE_MANIFEST_FILENAME
    if not manifest_path.exists():
        if missing_message is not None:
            raise ValueError(missing_message)
        raise ValueError(f"Missing candidate manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Candidate manifest must be a JSON object: {manifest_path}")
    return manifest


def build_target_summary(
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


def build_base_config_snapshot(
    config: AppConfig,
    positive_label: object | None,
    id_column: str,
    label_column: str,
) -> dict[str, object]:
    competition = config.competition
    return {
        "competition": {
            **competition.model_dump(mode="python"),
            "primary_metric": competition.primary_metric,
            "positive_label": positive_label,
            "id_column": id_column,
            "label_column": label_column,
        },
        "experiment": config.experiment.model_dump(mode="python"),
    }


def build_config_fingerprint(fingerprint_payload: dict[str, object]) -> str:
    fingerprint_payload_json = json.dumps(json_ready(fingerprint_payload), sort_keys=True)
    return hashlib.sha256(fingerprint_payload_json.encode("utf-8")).hexdigest()[:12]


def build_binary_accuracy_artifact_metadata(
    task_type: str,
    primary_metric: str,
) -> dict[str, object]:
    if task_type != "binary" or primary_metric != "accuracy":
        return {}
    return {
        "binary_accuracy_blend_rule": BINARY_ACCURACY_BLEND_RULE,
        "binary_accuracy_test_probability_path": BINARY_ACCURACY_TEST_PROBABILITIES_FILENAME,
    }


def write_context_artifacts(
    bundle_root: Path,
    competition_manifest: dict[str, object],
    fold_assignments: np.ndarray,
) -> None:
    context_dir = bundle_root / CONTEXT_ARTIFACT_DIRNAME
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / COMPETITION_MANIFEST_FILENAME).write_text(
        json.dumps(json_ready(competition_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    folds_df = pd.DataFrame(
        {
            "row_idx": np.arange(fold_assignments.shape[0], dtype=int),
            "fold": fold_assignments.astype(int),
        }
    )
    folds_df.to_csv(context_dir / FOLDS_FILENAME, index=False)


def write_candidate_artifacts(
    candidate_artifact_dir: Path,
    manifest: dict[str, object],
    fold_metrics_df: pd.DataFrame,
    y_train: pd.Series,
    oof_predictions: np.ndarray,
    fold_assignments: np.ndarray,
    test_ids: pd.Series,
    test_predictions: np.ndarray,
    id_column: str,
    label_column: str,
    test_prediction_probabilities: np.ndarray | None = None,
) -> None:
    candidate_artifact_dir.mkdir(parents=True, exist_ok=True)
    (candidate_artifact_dir / CANDIDATE_MANIFEST_FILENAME).write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fold_metrics_df.to_csv(candidate_artifact_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(y_train.shape[0], dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": oof_predictions,
            "fold": fold_assignments,
        }
    )
    oof_df.to_csv(candidate_artifact_dir / "oof_predictions.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            id_column: test_ids.to_numpy(),
            label_column: test_predictions,
        }
    )
    test_predictions_df.to_csv(candidate_artifact_dir / "test_predictions.csv", index=False)

    if test_prediction_probabilities is not None:
        test_probability_df = pd.DataFrame(
            {
                id_column: test_ids.to_numpy(),
                label_column: test_prediction_probabilities,
            }
        )
        test_probability_df.to_csv(
            candidate_artifact_dir / BINARY_ACCURACY_TEST_PROBABILITIES_FILENAME,
            index=False,
        )
