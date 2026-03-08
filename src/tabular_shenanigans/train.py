import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_shenanigans.cv import build_splitter, is_higher_better, resolve_positive_label, score_predictions
from tabular_shenanigans.data import load_competition_dataset_context
from tabular_shenanigans.models import build_model
from tabular_shenanigans.preprocess import build_preprocessor, prepare_feature_frames

RUN_LEDGER_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "competition_slug",
    "task_type",
    "primary_metric",
    "model_id",
    "model_name",
    "cv_mean",
    "cv_std",
    "higher_is_better",
    "cv_n_splits",
    "cv_shuffle",
    "cv_random_state",
    "config_fingerprint",
    "target_mean",
    "target_std",
    "target_min",
    "target_max",
    "positive_count",
    "negative_count",
    "target_prevalence",
    "positive_label",
    "observed_label_1",
    "observed_label_2",
    "negative_label",
]


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


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _append_run_ledger(ledger_path: Path, row: dict[str, object]) -> None:
    ledger_df = pd.DataFrame([row])
    ledger_df = ledger_df.reindex(columns=RUN_LEDGER_COLUMNS)
    if ledger_path.exists():
        existing_df = pd.read_csv(ledger_path)
        merged_df = pd.concat([existing_df, ledger_df], ignore_index=True, sort=False)
        merged_df = merged_df.reindex(columns=RUN_LEDGER_COLUMNS)
        merged_df.to_csv(ledger_path, index=False)
        return
    ledger_df.to_csv(ledger_path, index=False)


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


def _build_diagnostic_rows(
    task_type: str,
    fold_index: int,
    split_name: str,
    y_values: pd.Series,
    positive_label: object | None = None,
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
        if positive_label is None:
            raise ValueError("Binary diagnostics require positive_label.")
        positive_count = int((y_values == positive_label).sum())
        negative_count = int(y_values.shape[0] - positive_count)
        row.update(
            {
                "diagnostic_type": "class_balance",
                "positive_label": str(positive_label),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_rate": float(positive_count / y_values.shape[0]),
            }
        )
        return [row]

    raise ValueError(f"Unsupported task_type for diagnostics: {task_type}")


def _build_run_manifest(
    run_id: str,
    generated_at_utc: str,
    competition_slug: str,
    task_type: str,
    primary_metric: str,
    config_fingerprint: str,
    config_snapshot: dict[str, object],
    model_id: str,
    model_name: str,
    model_params: dict[str, object],
    cv_mean: float,
    cv_std: float,
    observed_label_pair: tuple[object, object] | None,
    negative_label: object | None,
    positive_label: object | None,
    id_column: str,
    label_column: str,
    target_summary: dict[str, object],
    train_rows: int,
    train_cols: int,
    test_rows: int,
    test_cols: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "competition_slug": competition_slug,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "config_fingerprint": config_fingerprint,
        "config_snapshot": config_snapshot,
        "model_id": model_id,
        "model_name": model_name,
        "model_params": model_params,
        "cv_summary": {
            "metric_name": primary_metric,
            "metric_mean": cv_mean,
            "metric_std": cv_std,
            "higher_is_better": is_higher_better(primary_metric),
        },
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


def _build_run_ledger_row(
    run_manifest: dict[str, object],
    cv_n_splits: int,
    cv_shuffle: bool,
    cv_random_state: int,
) -> dict[str, object]:
    cv_summary = run_manifest["cv_summary"]
    if not isinstance(cv_summary, dict):
        raise ValueError("Run manifest cv_summary must be a mapping.")

    ledger_row = {
        "run_id": run_manifest["run_id"],
        "timestamp_utc": run_manifest["generated_at_utc"],
        "competition_slug": run_manifest["competition_slug"],
        "task_type": run_manifest["task_type"],
        "primary_metric": run_manifest["primary_metric"],
        "model_id": run_manifest["model_id"],
        "model_name": run_manifest["model_name"],
        "cv_mean": cv_summary["metric_mean"],
        "cv_std": cv_summary["metric_std"],
        "higher_is_better": cv_summary["higher_is_better"],
        "cv_n_splits": cv_n_splits,
        "cv_shuffle": cv_shuffle,
        "cv_random_state": cv_random_state,
        "config_fingerprint": run_manifest["config_fingerprint"],
    }
    target_summary = run_manifest.get("target_summary", {})
    if isinstance(target_summary, dict):
        ledger_row.update(target_summary)
    return ledger_row


def run_training(
    competition_slug: str,
    task_type: str,
    primary_metric: str,
    model_id: str,
    id_column: str | None = None,
    label_column: str | None = None,
    force_categorical: list[str] | None = None,
    force_numeric: list[str] | None = None,
    drop_columns: list[str] | None = None,
    low_cardinality_int_threshold: int | None = None,
    cv_n_splits: int = 7,
    cv_shuffle: bool = True,
    cv_random_state: int = 42,
    positive_label: str | int | bool | None = None,
) -> Path:
    dataset_context = load_competition_dataset_context(
        competition_slug=competition_slug,
        configured_id_column=id_column,
        configured_label_column=label_column,
    )
    train_df = dataset_context.train_df
    test_df = dataset_context.test_df
    id_column = dataset_context.id_column
    label_column = dataset_context.label_column

    x_train_raw, x_test_raw, y_train = prepare_feature_frames(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        label_column=label_column,
        force_categorical=force_categorical,
        force_numeric=force_numeric,
        drop_columns=drop_columns,
    )
    observed_label_pair = None
    negative_label = None
    if task_type == "binary":
        negative_label, positive_label, observed_label_pair = resolve_positive_label(
            y_values=y_train,
            configured_positive_label=positive_label,
        )

    model_id, model_name, _, model_params = build_model(task_type, model_id, cv_random_state)
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
    run_diagnostics.extend(
        _build_diagnostic_rows(
            task_type=task_type,
            fold_index=0,
            split_name="all",
            y_values=y_train,
            positive_label=positive_label,
        )
    )

    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(x_train_raw, y_train), start=1):
        x_fold_train = x_train_raw.iloc[train_idx]
        x_fold_valid = x_train_raw.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        run_diagnostics.extend(
            _build_diagnostic_rows(
                task_type=task_type,
                fold_index=fold_index,
                split_name="train",
                y_values=y_fold_train,
                positive_label=positive_label,
            )
        )
        run_diagnostics.extend(
            _build_diagnostic_rows(
                task_type=task_type,
                fold_index=fold_index,
                split_name="valid",
                y_values=y_fold_valid,
                positive_label=positive_label,
            )
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

        _, _, model, _ = build_model(task_type, model_id, cv_random_state)
        model.fit(x_fold_train_processed, y_fold_train)

        if task_type == "binary":
            if positive_label is None:
                raise ValueError("Binary training requires positive_label.")
            if negative_label is None:
                raise ValueError("Binary training requires negative_label.")
            positive_class_index = list(model.classes_).index(positive_label)
            fold_valid_predictions = model.predict_proba(x_fold_valid_processed)[:, positive_class_index]
            fold_test_predictions = model.predict_proba(x_test_processed)[:, positive_class_index]
        else:
            fold_valid_predictions = model.predict(x_fold_valid_processed)
            fold_test_predictions = model.predict(x_test_processed)

        fold_score = score_predictions(
            task_type=task_type,
            primary_metric=primary_metric,
            y_true=y_fold_valid,
            y_pred=fold_valid_predictions,
            positive_label=positive_label,
        )

        oof_predictions[valid_idx] = fold_valid_predictions
        fold_assignments[valid_idx] = fold_index
        test_predictions_per_fold.append(np.asarray(fold_test_predictions, dtype=float))
        fold_metrics.append(
            {
                "model_id": model_id,
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
    target_summary = _build_target_summary(
        task_type=task_type,
        y_train=y_train,
        positive_label=positive_label,
        negative_label=negative_label,
        observed_label_pair=observed_label_pair,
    )

    run_id = _make_run_id()
    run_dir = Path("artifacts") / competition_slug / "train" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_df.to_csv(run_dir / "fold_metrics.csv", index=False)
    run_diagnostics_df.to_csv(run_dir / "run_diagnostics.csv", index=False)

    oof_df = pd.DataFrame(
        {
            "row_idx": np.arange(n_rows, dtype=int),
            "y_true": y_train.to_numpy(),
            "y_pred": oof_predictions,
            "fold": fold_assignments,
            "model_id": model_id,
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
        "model_id": model_id,
        "positive_label": positive_label,
        "id_column": id_column,
        "label_column": label_column,
        "force_categorical": force_categorical or [],
        "force_numeric": force_numeric or [],
        "drop_columns": drop_columns or [],
        "low_cardinality_int_threshold": low_cardinality_int_threshold,
        "cv_n_splits": cv_n_splits,
        "cv_shuffle": cv_shuffle,
        "cv_random_state": cv_random_state,
    }
    fingerprint_payload = {
        "config_snapshot": config_snapshot,
        "model_name": model_name,
        "model_params": model_params,
    }
    config_snapshot_json = json.dumps(_json_ready(fingerprint_payload), sort_keys=True)
    config_fingerprint = hashlib.sha256(config_snapshot_json.encode("utf-8")).hexdigest()[:12]

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    run_manifest = _build_run_manifest(
        run_id=run_id,
        generated_at_utc=generated_at_utc,
        competition_slug=competition_slug,
        task_type=task_type,
        primary_metric=primary_metric,
        config_fingerprint=config_fingerprint,
        config_snapshot=config_snapshot,
        model_id=model_id,
        model_name=model_name,
        model_params=model_params,
        cv_mean=cv_mean,
        cv_std=cv_std,
        observed_label_pair=observed_label_pair,
        negative_label=negative_label,
        positive_label=positive_label,
        id_column=id_column,
        label_column=label_column,
        target_summary=target_summary,
        train_rows=int(x_train_raw.shape[0]),
        train_cols=int(x_train_raw.shape[1]),
        test_rows=int(x_test_raw.shape[0]),
        test_cols=int(x_test_raw.shape[1]),
    )
    run_manifest_json = json.dumps(_json_ready(run_manifest), indent=2)
    (run_dir / "run_manifest.json").write_text(run_manifest_json, encoding="utf-8")

    ledger_row = _build_run_ledger_row(
        run_manifest=run_manifest,
        cv_n_splits=cv_n_splits,
        cv_shuffle=cv_shuffle,
        cv_random_state=cv_random_state,
    )
    ledger_path = Path("artifacts") / competition_slug / "train" / "runs.csv"
    _append_run_ledger(ledger_path, ledger_row)

    print(f"Training model: {model_id} ({model_name})")
    print(f"CV {primary_metric}: mean={cv_mean:.6f}, std={cv_std:.6f}")

    return run_dir
