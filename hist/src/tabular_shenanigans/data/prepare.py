from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _schema_columns(runtime_cfg: dict[str, Any]) -> tuple[str, str]:
    schema_cfg = runtime_cfg.get("schema", {})
    submission_cfg = runtime_cfg.get("submission", {})

    target_col = schema_cfg.get("target") or submission_cfg.get("target_column")
    id_col = schema_cfg.get("id_column") or submission_cfg.get("id_column")
    if not target_col:
        raise ValueError("Missing target column. Set `schema.target` in competition config.")
    if not id_col:
        raise ValueError("Missing id column. Set `schema.id_column` or `submission.id_column`.")
    return str(target_col), str(id_col)


def _task_type(train_target: pd.Series, runtime_cfg: dict[str, Any]) -> str:
    configured = runtime_cfg.get("schema", {}).get("task_type")
    if configured in {"classification", "regression"}:
        return configured

    if train_target.dtype == "object" or train_target.nunique(dropna=False) <= 20:
        return "classification"
    return "regression"


def prepare_competition_data(
    runtime_cfg: dict[str, Any],
    competition: str,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, str, str, str]:
    target_col, id_col = _schema_columns(runtime_cfg)
    train_file = runtime_cfg.get("schema", {}).get("train_file", "train.csv")
    test_file = runtime_cfg.get("schema", {}).get("test_file", "test.csv")

    raw_dir = data_dir / competition / "raw"
    processed_dir = data_dir / competition / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / train_file
    test_path = raw_dir / test_file
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected training/test files at {train_path} and {test_path}. Run fetch-data first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {train_path}")
    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in {test_path}")

    y = train_df[target_col]
    X_train = train_df.drop(columns=[target_col], errors="ignore")
    X_test = test_df.copy()

    if id_col in X_train.columns:
        X_train = X_train.drop(columns=[id_col])
    X_test_ids = X_test[[id_col]].copy()
    X_test = X_test.drop(columns=[id_col])

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y.to_frame(name=target_col).to_csv(processed_dir / "y_train.csv", index=False)
    X_test_ids.to_csv(processed_dir / "test_ids.csv", index=False)

    task_type = _task_type(y, runtime_cfg)
    common_cols = sorted(set(X_train.columns).intersection(set(X_test.columns)))
    train_only_cols = sorted(set(X_train.columns) - set(X_test.columns))
    test_only_cols = sorted(set(X_test.columns) - set(X_train.columns))
    dtype_mismatches = [
        col for col in common_cols if str(X_train[col].dtype) != str(X_test[col].dtype)
    ]
    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
    train_missing = {col: int(val) for col, val in X_train.isna().sum().items() if int(val) > 0}
    test_missing = {col: int(val) for col, val in X_test.isna().sum().items() if int(val) > 0}

    report = {
        "task_type": task_type,
        "target_column": target_col,
        "id_column": id_col,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns_train": int(len(X_train.columns)),
        "feature_columns_test": int(len(X_test.columns)),
        "train_only_columns": train_only_cols,
        "test_only_columns": test_only_cols,
        "dtype_mismatches": dtype_mismatches,
        "constant_columns_train": constant_cols,
        "target_present_in_test": bool(target_col in test_df.columns),
        "missing_values_train": train_missing,
        "missing_values_test": test_missing,
    }
    (processed_dir / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return X_train, y, X_test, X_test_ids, target_col, id_col, task_type
