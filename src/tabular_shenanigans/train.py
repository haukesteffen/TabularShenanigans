import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression

from tabular_shenanigans.cv import build_splitter, is_higher_better, score_predictions
from tabular_shenanigans.data import find_competition_zip, read_csv_from_zip, resolve_id_and_label_columns
from tabular_shenanigans.preprocess import build_preprocessor, prepare_feature_frames


def _build_model(task_type: str, random_state: int) -> tuple[str, object, dict[str, object]]:
    if task_type == "regression":
        params = {
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 10000,
            "random_state": random_state,
        }
        return "ElasticNet", ElasticNet(**params), params

    if task_type == "binary":
        params = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": random_state,
        }
        return "LogisticRegression", LogisticRegression(**params), params

    raise ValueError(f"Unsupported task_type for baseline model: {task_type}")


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _append_run_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row])
    if ledger_path.exists():
        existing_df = pd.read_csv(ledger_path)
        merged_df = pd.concat([existing_df, ledger_df], ignore_index=True, sort=False)
        merged_df.to_csv(ledger_path, index=False)
        return
    ledger_df.to_csv(ledger_path, index=False)


def _build_target_summary(task_type: str, y_train: pd.Series) -> dict[str, object]:
    if task_type == "regression":
        return {
            "target_mean": float(y_train.mean()),
            "target_std": float(y_train.std(ddof=0)),
            "target_min": float(y_train.min()),
            "target_max": float(y_train.max()),
        }

    if task_type == "binary":
        positive_count = int(y_train.sum())
        row_count = int(y_train.shape[0])
        negative_count = row_count - positive_count
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "target_prevalence": float(positive_count / row_count),
        }

    raise ValueError(f"Unsupported task_type for target summary: {task_type}")


def _build_diagnostic_rows(
    task_type: str,
    fold_index: int,
    split_name: str,
    y_values: pd.Series,
) -> list[dict[str, object]]:
    row = {
        "task_type": task_type,
        "fold": fold_index,
        "split": split_name,
        "row_count": int(y_values.shape[0]),
        "target_mean": np.nan,
        "target_std": np.nan,
        "target_min": np.nan,
        "target_max": np.nan,
        "positive_count": np.nan,
        "negative_count": np.nan,
        "positive_rate": np.nan,
    }

    if task_type == "regression":
        row.update(
            {
                "diagnostic_type": "target_distribution",
                "target_mean": float(y_values.mean()),
                "target_std": float(y_values.std(ddof=0)),
                "target_min": float(y_values.min()),
                "target_max": float(y_values.max()),
            }
        )
        return [row]

    if task_type == "binary":
        positive_count = int(y_values.sum())
        negative_count = int(y_values.shape[0] - positive_count)
        row.update(
            {
                "diagnostic_type": "class_balance",
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": float(positive_count / y_values.shape[0]),
            }
        )
        return [row]

    raise ValueError(f"Unsupported task_type for diagnostics: {task_type}")


def run_training(
    competition_slug: str,
    task_type: str,
    primary_metric: str,
    id_column: str | None = None,
    label_column: str | None = None,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    drop_columns: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
    cv_n_splits: int = 7,
    cv_shuffle: bool = True,
    cv_random_state: int = 42,
) -> Path:
    zip_path = find_competition_zip(competition_slug)
    train_df = read_csv_from_zip(zip_path, "train.csv")
    test_df = read_csv_from_zip(zip_path, "test.csv")
    sample_submission_df = read_csv_from_zip(zip_path, "sample_submission.csv")
    id_column, label_column = resolve_id_and_label_columns(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
        configured_id_column=id_column,
        configured_label_column=label_column,
    )

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        label_column=label_column,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        drop_columns=drop_columns,
    )

    model_name, _, model_params = _build_model(task_type, cv_random_state)
    splitter = build_splitter(
        task_type=task_type,
        n_splits=cv_n_splits,
        shuffle=cv_shuffle,
        random_state=cv_random_state,
    )

    n_rows = x_train_raw.shape[0]
    oof_predictions = np.zeros(n_rows, dtype=float)
    fold_assignments = np.full(n_rows, fill_value=-1, dtype=int)
    test_predictions_per_fold: list[np.ndarray] = []
    fold_metrics: list[dict[str, object]] = []
    run_diagnostics: list[dict[str, object]] = []
    run_diagnostics.extend(_build_diagnostic_rows(task_type=task_type, fold_index=0, split_name="all", y_values=y_train))

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train_raw, y_train), start=1):
        x_fold_train = x_train_raw.iloc[train_idx]
        x_fold_valid = x_train_raw.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        run_diagnostics.extend(
            _build_diagnostic_rows(task_type=task_type, fold_index=fold_index, split_name="train", y_values=y_fold_train)
        )
        run_diagnostics.extend(
            _build_diagnostic_rows(task_type=task_type, fold_index=fold_index, split_name="valid", y_values=y_fold_valid)
        )

        preprocessor, _, _ = build_preprocessor(
            x_train_raw=x_fold_train,
            force_categorical=force_categorical,
            force_numeric=force_numeric,
            low_cardinality_int_threshold=low_cardinality_int_threshold,
        )

        x_fold_train_processed = preprocessor.fit_transform(x_fold_train)
        x_fold_valid_processed = preprocessor.transform(x_fold_valid)
        x_test_processed = preprocessor.transform(x_test_raw)

        _, model, _ = _build_model(task_type, cv_random_state)
        model.fit(x_fold_train_processed, y_fold_train)

        if task_type == "binary":
            fold_valid_predictions = model.predict_proba(x_fold_valid_processed)[:, 1]
            fold_test_predictions = model.predict_proba(x_test_processed)[:, 1]
        else:
            fold_valid_predictions = model.predict(x_fold_valid_processed)
            fold_test_predictions = model.predict(x_test_processed)

        fold_score = score_predictions(
            task_type=task_type,
            primary_metric=primary_metric,
            y_true=y_fold_valid,
            y_pred=fold_valid_predictions,
        )

        oof_predictions[valid_idx] = fold_valid_predictions
        fold_assignments[valid_idx] = fold_index
        test_predictions_per_fold.append(np.asarray(fold_test_predictions, dtype=float))
        fold_metrics.append(
            {
                "model_name": model_name,
                "fold": fold_index,
                "metric_name": primary_metric,
                "metric_value": fold_score,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )

    if (fold_assignments < 0).any():
        raise ValueError("Fold assignment failed: at least one training row did not receive a validation fold.")

    mean_test_predictions = np.mean(np.vstack(test_predictions_per_fold), axis=0)
    if task_type == "regression" and primary_metric == "rmsle":
        mean_test_predictions = np.clip(mean_test_predictions, a_min=0.0, a_max=None)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    run_diagnostics_df = pd.DataFrame(run_diagnostics)
    cv_mean = float(fold_metrics_df["metric_value"].mean())
    cv_std = float(fold_metrics_df["metric_value"].std(ddof=0))
    target_summary = _build_target_summary(task_type=task_type, y_train=y_train)

    run_id = _make_run_id()
    run_dir = Path("artifacts") / competition_slug / "train" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_df.to_csv(run_dir / "fold_metrics.csv", index=False)
    run_diagnostics_df.to_csv(run_dir / "run_diagnostics.csv", index=False)

    cv_summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "metric_name": primary_metric,
                "metric_mean": cv_mean,
                "metric_std": cv_std,
                "higher_is_better": is_higher_better(primary_metric),
            }
        ]
    )
    cv_summary_df.to_csv(run_dir / "cv_summary.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(n_rows, dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": oof_predictions,
            "fold": fold_assignments,
            "model_name": model_name,
        }
    )
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)

    test_predictions_df = pd.DataFrame(
        {
            id_column: test_df[id_column].to_numpy(),
            label_column: mean_test_predictions,
        }
    )
    test_predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    config_snapshot = {
        "competition_slug": competition_slug,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "id_column": id_column,
        "label_column": label_column,
        "force_categorical": force_categorical or [],
        "force_numeric": force_numeric or [],
        "drop_columns": drop_columns or [],
        "low_cardinality_int_threshold": low_cardinality_int_threshold,
        "cv_n_splits": cv_n_splits,
        "cv_shuffle": cv_shuffle,
        "cv_random_state": cv_random_state,
        "model_name": model_name,
        "model_params": model_params,
    }
    config_snapshot_json = json.dumps(config_snapshot, sort_keys=True)
    config_fingerprint = hashlib.sha256(config_snapshot_json.encode("utf-8")).hexdigest()[:12]
    artifact_paths = {
        "run_manifest": "run_manifest.json",
        "fold_metrics": "fold_metrics.csv",
        "cv_summary": "cv_summary.csv",
        "run_diagnostics": "run_diagnostics.csv",
        "oof_predictions": "oof_predictions.csv",
        "test_predictions": "test_predictions.csv",
    }

    run_manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "id_column": id_column,
        "label_column": label_column,
        "target_summary": target_summary,
        "train_rows": int(x_train_raw.shape[0]),
        "train_cols": int(x_train_raw.shape[1]),
        "test_rows": int(x_test_raw.shape[0]),
        "test_cols": int(x_test_raw.shape[1]),
        "artifacts": artifact_paths,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    ledger_row = {
        "run_id": run_id,
        "timestamp_utc": run_manifest["generated_at_utc"],
        "competition_slug": competition_slug,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "model_name": model_name,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_n_splits": cv_n_splits,
        "cv_random_state": cv_random_state,
        "config_fingerprint": config_fingerprint,
        "artifact_dir": str(run_dir),
        "run_manifest_path": str(run_dir / artifact_paths["run_manifest"]),
        "fold_metrics_path": str(run_dir / artifact_paths["fold_metrics"]),
        "cv_summary_path": str(run_dir / artifact_paths["cv_summary"]),
        "run_diagnostics_path": str(run_dir / artifact_paths["run_diagnostics"]),
        "oof_predictions_path": str(run_dir / artifact_paths["oof_predictions"]),
        "test_predictions_path": str(run_dir / artifact_paths["test_predictions"]),
    }
    ledger_row.update(target_summary)
    ledger_path = Path("artifacts") / competition_slug / "train" / "runs.csv"
    _append_run_ledger(ledger_path, ledger_row)

    print(f"Training model: {model_name}")
    print(f"CV {primary_metric}: mean={cv_mean:.6f}, std={cv_std:.6f}")

    return run_dir
