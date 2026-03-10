import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.config import AppConfig
from tabular_shenanigans.cv import build_splitter, resolve_positive_label
from tabular_shenanigans.data import CompetitionDatasetContext
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.preprocess import prepare_feature_frames


@dataclass(frozen=True)
class PreparedCompetitionContext:
    competition_dir: Path
    report_dir: Path
    manifest_path: Path
    folds_path: Path
    manifest: dict[str, object]
    fold_assignments: np.ndarray
    split_indices: list[tuple[int, np.ndarray, np.ndarray]]


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


def _competition_dir(competition_slug: str) -> Path:
    return Path("artifacts") / competition_slug


def _competition_manifest_path(competition_slug: str) -> Path:
    return _competition_dir(competition_slug) / "competition.json"


def _competition_folds_path(competition_slug: str) -> Path:
    return _competition_dir(competition_slug) / "folds.csv"


def has_prepared_competition_context(competition_slug: str) -> bool:
    manifest_path = _competition_manifest_path(competition_slug)
    folds_path = _competition_folds_path(competition_slug)
    return manifest_path.exists() and folds_path.exists()


def materialize_split_indices(
    task_type: str,
    x_train_raw: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    splitter = build_splitter(
        task_type=task_type,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    split_indices: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train_raw, y_train), start=1):
        split_indices.append((fold_index, train_idx, valid_idx))
    return split_indices


def build_fold_assignments(
    row_count: int,
    split_indices: list[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    fold_assignments = np.full(row_count, fill_value=-1, dtype=int)
    for fold_index, _, valid_idx in split_indices:
        if (fold_assignments[valid_idx] >= 0).any():
            raise ValueError("Fold assignment failed: at least one training row received multiple validation folds.")
        fold_assignments[valid_idx] = fold_index
    if (fold_assignments < 0).any():
        raise ValueError("Fold assignment failed: at least one training row did not receive a validation fold.")
    return fold_assignments


def build_split_indices_from_fold_assignments(
    fold_assignments: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if fold_assignments.ndim != 1:
        raise ValueError("Fold assignments must be one-dimensional.")
    if fold_assignments.size == 0:
        raise ValueError("Fold assignments cannot be empty.")
    if (fold_assignments < 1).any():
        raise ValueError("Fold assignments must be positive 1-based fold numbers.")

    fold_values = sorted(int(fold) for fold in np.unique(fold_assignments).tolist())
    expected_folds = list(range(1, len(fold_values) + 1))
    if fold_values != expected_folds:
        raise ValueError(
            "Fold assignments must contain contiguous 1-based fold numbers. "
            f"Expected {expected_folds}, got {fold_values}"
        )

    row_indices = np.arange(fold_assignments.shape[0], dtype=int)
    split_indices: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_index in fold_values:
        valid_idx = row_indices[fold_assignments == fold_index]
        train_idx = row_indices[fold_assignments != fold_index]
        if valid_idx.size == 0:
            raise ValueError(f"Fold assignment failed: fold {fold_index} has no validation rows.")
        if train_idx.size == 0:
            raise ValueError(f"Fold assignment failed: fold {fold_index} has no training rows.")
        split_indices.append((fold_index, train_idx, valid_idx))
    return split_indices


def _build_competition_manifest(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    x_train_raw: pd.DataFrame,
    positive_label: object | None,
    observed_label_pair: tuple[object, object] | None,
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": config.competition_slug,
        "task_type": config.task_type,
        "primary_metric": config.primary_metric,
        "id_column": dataset_context.id_column,
        "label_column": dataset_context.label_column,
        "positive_label": positive_label,
        "observed_label_pair": list(observed_label_pair) if observed_label_pair is not None else None,
        "train_rows": int(dataset_context.train_df.shape[0]),
        "test_rows": int(dataset_context.test_df.shape[0]),
        "feature_columns": x_train_raw.columns.tolist(),
        "cv": config.competition.cv.model_dump(mode="python"),
        "features": config.competition.features.model_dump(mode="python"),
    }


def _write_competition_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_fold_assignments(folds_path: Path, fold_assignments: np.ndarray) -> None:
    folds_df = pd.DataFrame(
        {
            "row_idx": np.arange(fold_assignments.shape[0], dtype=int),
            "fold": fold_assignments.astype(int),
        }
    )
    folds_df.to_csv(folds_path, index=False)


def prepare_competition(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
) -> PreparedCompetitionContext:
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, _, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=config.force_categorical,
        force_numeric=config.force_numeric,
        drop_columns=config.drop_columns,
    )

    positive_label = config.positive_label
    observed_label_pair = None
    if config.task_type == "binary":
        _, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    split_indices = materialize_split_indices(
        task_type=config.task_type,
        x_train_raw=x_train_raw,
        y_train=y_train,
        n_splits=config.cv_n_splits,
        shuffle=config.cv_shuffle,
        random_state=config.cv_random_state,
    )
    fold_assignments = build_fold_assignments(row_count=x_train_raw.shape[0], split_indices=split_indices)
    report_dir = run_eda(config=config, dataset_context=dataset_context)

    competition_dir = _competition_dir(config.competition_slug)
    competition_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _competition_manifest_path(config.competition_slug)
    folds_path = _competition_folds_path(config.competition_slug)
    manifest = _build_competition_manifest(
        config=config,
        dataset_context=dataset_context,
        x_train_raw=x_train_raw,
        positive_label=positive_label,
        observed_label_pair=observed_label_pair,
    )
    _write_competition_manifest(manifest_path=manifest_path, manifest=manifest)
    _write_fold_assignments(folds_path=folds_path, fold_assignments=fold_assignments)
    return PreparedCompetitionContext(
        competition_dir=competition_dir,
        report_dir=report_dir,
        manifest_path=manifest_path,
        folds_path=folds_path,
        manifest=manifest,
        fold_assignments=fold_assignments,
        split_indices=split_indices,
    )


def _load_competition_manifest(manifest_path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing prepared competition context: {manifest_path}. Run `uv run python main.py prepare` first."
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Prepared competition manifest must be a JSON object: {manifest_path}")
    return manifest


def _load_fold_assignments(folds_path: Path) -> np.ndarray:
    try:
        folds_df = pd.read_csv(folds_path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Missing prepared fold assignments: {folds_path}. Run `uv run python main.py prepare` first."
        ) from exc

    expected_columns = ["row_idx", "fold"]
    if folds_df.columns.tolist() != expected_columns:
        raise ValueError(f"Prepared folds file must have columns {expected_columns}: {folds_path}")
    expected_row_idx = np.arange(folds_df.shape[0], dtype=int)
    if not np.array_equal(folds_df["row_idx"].to_numpy(dtype=int), expected_row_idx):
        raise ValueError(f"Prepared folds file must contain sequential row_idx values starting at 0: {folds_path}")
    return folds_df["fold"].to_numpy(dtype=int)


def _validate_prepared_context(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    manifest: dict[str, object],
    fold_assignments: np.ndarray,
    expected_feature_columns: list[str] | None,
) -> None:
    expected_cv = config.competition.cv.model_dump(mode="python")
    expected_features = config.competition.features.model_dump(mode="python")
    train_rows = int(dataset_context.train_df.shape[0])
    test_rows = int(dataset_context.test_df.shape[0])

    if manifest.get("competition_slug") != config.competition_slug:
        raise ValueError("Prepared competition context does not match the configured competition slug.")
    if manifest.get("task_type") != config.task_type:
        raise ValueError("Prepared competition context does not match the configured task_type. Re-run prepare.")
    if manifest.get("primary_metric") != config.primary_metric:
        raise ValueError("Prepared competition context does not match the configured primary_metric. Re-run prepare.")
    if manifest.get("id_column") != dataset_context.id_column:
        raise ValueError("Prepared competition context does not match the resolved id_column. Re-run prepare.")
    if manifest.get("label_column") != dataset_context.label_column:
        raise ValueError("Prepared competition context does not match the resolved label_column. Re-run prepare.")
    if manifest.get("cv") != expected_cv:
        raise ValueError("Prepared competition context does not match competition.cv. Re-run prepare.")
    if manifest.get("features") != expected_features:
        raise ValueError("Prepared competition context does not match competition.features. Re-run prepare.")
    if manifest.get("train_rows") != train_rows:
        raise ValueError("Prepared competition context does not match train row count. Re-run prepare.")
    if manifest.get("test_rows") != test_rows:
        raise ValueError("Prepared competition context does not match test row count. Re-run prepare.")
    if fold_assignments.shape[0] != train_rows:
        raise ValueError("Prepared fold assignments do not match the training row count. Re-run prepare.")
    if expected_feature_columns is not None and manifest.get("feature_columns") != expected_feature_columns:
        raise ValueError("Prepared competition context does not match the resolved feature columns. Re-run prepare.")

    split_indices = build_split_indices_from_fold_assignments(fold_assignments)
    if len(split_indices) != config.cv_n_splits:
        raise ValueError("Prepared fold assignments do not match competition.cv.n_splits. Re-run prepare.")


def load_prepared_competition_context(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    expected_feature_columns: list[str] | None = None,
) -> PreparedCompetitionContext:
    manifest_path = _competition_manifest_path(config.competition_slug)
    folds_path = _competition_folds_path(config.competition_slug)
    manifest = _load_competition_manifest(manifest_path=manifest_path)
    fold_assignments = _load_fold_assignments(folds_path=folds_path)
    _validate_prepared_context(
        config=config,
        dataset_context=dataset_context,
        manifest=manifest,
        fold_assignments=fold_assignments,
        expected_feature_columns=expected_feature_columns,
    )
    return PreparedCompetitionContext(
        competition_dir=_competition_dir(config.competition_slug),
        report_dir=Path("reports") / config.competition_slug,
        manifest_path=manifest_path,
        folds_path=folds_path,
        manifest=manifest,
        fold_assignments=fold_assignments,
        split_indices=build_split_indices_from_fold_assignments(fold_assignments),
    )


def ensure_prepared_competition_context(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    expected_feature_columns: list[str] | None = None,
) -> PreparedCompetitionContext:
    if has_prepared_competition_context(config.competition_slug):
        return load_prepared_competition_context(
            config=config,
            dataset_context=dataset_context,
            expected_feature_columns=expected_feature_columns,
        )

    print("Prepared competition context missing; running prepare.")
    return prepare_competition(config=config, dataset_context=dataset_context)
