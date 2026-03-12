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
    report_dir: Path | None
    manifest: dict[str, object]
    fold_assignments: np.ndarray
    split_indices: list[tuple[int, np.ndarray, np.ndarray]]


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
    competition = config.competition
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_slug": competition.slug,
        "task_type": competition.task_type,
        "primary_metric": competition.primary_metric,
        "id_column": dataset_context.id_column,
        "label_column": dataset_context.label_column,
        "positive_label": positive_label,
        "observed_label_pair": list(observed_label_pair) if observed_label_pair is not None else None,
        "train_rows": int(dataset_context.train_df.shape[0]),
        "test_rows": int(dataset_context.test_df.shape[0]),
        "feature_columns": x_train_raw.columns.tolist(),
        "cv": competition.cv.model_dump(mode="python"),
        "features": competition.features.model_dump(mode="python"),
    }


def prepare_competition(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    run_reports: bool = True,
) -> PreparedCompetitionContext:
    competition = config.competition
    features = competition.features
    cv = competition.cv
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, _, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=features.force_categorical,
        force_numeric=features.force_numeric,
        drop_columns=features.drop_columns,
    )

    positive_label = competition.positive_label
    observed_label_pair = None
    if competition.task_type == "binary":
        _, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    split_indices = materialize_split_indices(
        task_type=competition.task_type,
        x_train_raw=x_train_raw,
        y_train=y_train,
        n_splits=cv.n_splits,
        shuffle=cv.shuffle,
        random_state=cv.random_state,
    )
    fold_assignments = build_fold_assignments(row_count=x_train_raw.shape[0], split_indices=split_indices)
    report_dir = run_eda(config=config, dataset_context=dataset_context) if run_reports else None
    manifest = _build_competition_manifest(
        config=config,
        dataset_context=dataset_context,
        x_train_raw=x_train_raw,
        positive_label=positive_label,
        observed_label_pair=observed_label_pair,
    )
    return PreparedCompetitionContext(
        report_dir=report_dir,
        manifest=manifest,
        fold_assignments=fold_assignments,
        split_indices=split_indices,
    )


def ensure_prepared_competition_context(
    config: AppConfig,
    dataset_context: CompetitionDatasetContext,
    expected_feature_columns: list[str] | None = None,
) -> PreparedCompetitionContext:
    prepared_context = prepare_competition(
        config=config,
        dataset_context=dataset_context,
        run_reports=False,
    )
    if expected_feature_columns is not None and prepared_context.manifest.get("feature_columns") != expected_feature_columns:
        raise ValueError("Prepared competition context does not match the resolved feature columns.")
    return prepared_context
