from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def _split_dir(data_dir: Path, competition: str, split_version: str) -> Path:
    return data_dir / competition / "splits" / split_version


def _split_path(data_dir: Path, competition: str, split_version: str) -> Path:
    return _split_dir(data_dir, competition, split_version) / "folds.csv"


def build_or_load_splits(
    *,
    data_dir: Path,
    competition: str,
    y: pd.Series,
    task_type: str,
    n_splits: int,
    seed: int,
    split_version: str = "v1",
    force_rebuild: bool = False,
) -> np.ndarray:
    path = _split_path(data_dir, competition, split_version)

    if path.exists() and not force_rebuild:
        df = pd.read_csv(path)
        if "row_idx" not in df.columns or "fold" not in df.columns:
            raise ValueError(f"Invalid split file format: {path}")
        df = df.sort_values("row_idx")
        if len(df) != len(y):
            raise ValueError(
                f"Split length mismatch for {competition}:{split_version}. "
                f"Expected {len(y)}, got {len(df)}."
            )
        return df["fold"].to_numpy(dtype=int)

    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(y)))

    folds = np.full(len(y), -1, dtype=int)
    for fold_id, (_, valid_idx) in enumerate(split_iter):
        folds[valid_idx] = fold_id

    out_df = pd.DataFrame({"row_idx": np.arange(len(y), dtype=int), "fold": folds})
    out_dir = _split_dir(data_dir, competition, split_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)
    return folds
