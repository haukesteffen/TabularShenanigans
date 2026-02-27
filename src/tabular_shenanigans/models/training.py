from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tabular_shenanigans.core.metrics import (
    compute_metrics_bundle,
    compute_metric,
    resolve_metric,
    validate_metric_compatibility,
)
from tabular_shenanigans.core.splits import build_or_load_splits
from tabular_shenanigans.data.prepare import prepare_competition_data


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    ).set_output(transform="default")


def _build_model(task_type: str, runtime_cfg: dict[str, Any], seed: int) -> Any:
    training_cfg = runtime_cfg.get("training", {})
    model_family = str(training_cfg.get("model_family", "sklearn")).lower()
    model_params = training_cfg.get("model_params", {})

    if model_family == "sklearn":
        if task_type == "classification":
            return LogisticRegression(max_iter=2000, random_state=seed, **model_params)
        return Ridge(**model_params)

    if model_family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "lightgbm is not installed. Install with `uv sync --extra train`."
            ) from exc
        params = {"random_state": seed, "verbose": -1}
        params.update(model_params)
        if task_type == "classification":
            return LGBMClassifier(**params)
        return LGBMRegressor(**params)

    if model_family == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed. Install with `uv sync --extra train`."
            ) from exc
        base_params = {"random_state": seed, "n_estimators": 300}
        if task_type == "classification":
            base_params["eval_metric"] = "logloss"
        base_params.update(model_params)
        if task_type == "classification":
            return XGBClassifier(**base_params)
        return XGBRegressor(**base_params)

    if model_family == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError(
                "catboost is not installed. Install with `uv sync --extra train`."
            ) from exc
        params = {"random_seed": seed, "verbose": False}
        params.update(model_params)
        if task_type == "classification":
            return CatBoostClassifier(**params)
        return CatBoostRegressor(**params)

    raise ValueError(
        f"Unsupported training.model_family '{model_family}'. "
        "Use one of: sklearn, lightgbm, xgboost, catboost."
    )


def _is_lightgbm_model(model: Any) -> bool:
    return model.__class__.__module__.startswith("lightgbm")


def _align_feature_names(model: Any, Xt: Any) -> Any:
    if isinstance(Xt, pd.DataFrame):
        return Xt
    if hasattr(model, "feature_names_in_"):
        feature_names = list(getattr(model, "feature_names_in_"))
        if len(feature_names) == np.shape(Xt)[1]:
            return pd.DataFrame(Xt, columns=feature_names)
    return Xt


def _predict_with_pipeline(pipeline: Pipeline, X: pd.DataFrame, proba: bool) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    Xt = _align_feature_names(model, preprocessor.transform(X))

    if proba:
        if _is_lightgbm_model(model):
            preds = model.predict_proba(Xt, validate_features=False)
        else:
            preds = model.predict_proba(Xt)
        return preds

    if _is_lightgbm_model(model):
        return model.predict(Xt, validate_features=False)
    return model.predict(Xt)


def train_baseline(
    runtime_cfg: dict[str, Any],
    output_dir: Path,
    data_dir: Path,
    competition: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y, _, _, target_col, id_col, task_type = prepare_competition_data(
        runtime_cfg=runtime_cfg,
        competition=competition,
        data_dir=data_dir,
    )

    seed = int(runtime_cfg.get("seed", 42))
    n_splits = int(runtime_cfg.get("training", {}).get("cv_folds", 5))
    model_family = str(runtime_cfg.get("training", {}).get("model_family", "sklearn")).lower()
    metric_name, metric_direction = resolve_metric(runtime_cfg, task_type)
    n_classes = int(y.nunique()) if task_type == "classification" else None
    validate_metric_compatibility(metric_name, task_type, n_classes)
    classification_use_proba = task_type == "classification" and metric_name in {
        "roc_auc",
        "logloss",
    }

    split_cfg = runtime_cfg.get("cv", {})
    split_version = str(split_cfg.get("split_version", "v1"))
    force_rebuild = bool(split_cfg.get("rebuild_splits", False))
    fold_ids = build_or_load_splits(
        data_dir=data_dir,
        competition=competition,
        y=y,
        task_type=task_type,
        n_splits=n_splits,
        seed=seed,
        split_version=split_version,
        force_rebuild=force_rebuild,
    )

    if task_type == "classification":
        oof_scores = np.full(len(X_train), np.nan, dtype=float)
        oof_labels = np.full(len(X_train), np.nan, dtype=float)
    else:
        oof_values = np.zeros(len(X_train), dtype=float)

    fold_scores: list[float] = []
    fold_metrics: dict[str, list[float]] = {}

    unique_folds = sorted(set(int(v) for v in fold_ids.tolist()))
    for fold, fold_id in enumerate(unique_folds, start=1):
        valid_idx = np.where(fold_ids == fold_id)[0]
        train_idx = np.where(fold_ids != fold_id)[0]
        X_tr = X_train.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X_train.iloc[valid_idx]
        y_va = y.iloc[valid_idx]

        preprocessor = _build_preprocessor(X_tr)
        model = _build_model(task_type=task_type, runtime_cfg=runtime_cfg, seed=seed)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_tr, y_tr)

        if task_type == "classification":
            fold_pred_labels: np.ndarray
            fold_pred_scores: np.ndarray | None = None
            if classification_use_proba:
                if not hasattr(model, "predict_proba"):
                    raise ValueError(
                        f"evaluation.metric='{metric_name}' requires probability predictions, "
                        f"but model '{model.__class__.__name__}' does not implement predict_proba."
                    )
            if hasattr(model, "predict_proba"):
                fold_pred_scores = _predict_with_pipeline(pipeline, X_va, proba=True)[:, 1]
                fold_pred_labels = (fold_pred_scores >= 0.5).astype(int)
            else:
                fold_pred_labels = np.asarray(_predict_with_pipeline(pipeline, X_va, proba=False))

            primary_pred = (
                np.asarray(fold_pred_scores)
                if classification_use_proba and fold_pred_scores is not None
                else np.asarray(fold_pred_labels)
            )
            score = compute_metric(
                metric_name=metric_name,
                task_type=task_type,
                y_true=y_va.to_numpy(),
                y_pred=primary_pred,
            )
            bundle = compute_metrics_bundle(
                task_type=task_type,
                n_classes=n_classes,
                y_true=y_va.to_numpy(),
                y_pred_labels=np.asarray(fold_pred_labels),
                y_pred_scores=np.asarray(fold_pred_scores) if fold_pred_scores is not None else None,
            )
            for key, value in bundle.items():
                fold_metrics.setdefault(key, []).append(float(value))
            if fold_pred_scores is not None:
                oof_scores[valid_idx] = fold_pred_scores
            oof_labels[valid_idx] = fold_pred_labels
        else:
            pred = np.asarray(_predict_with_pipeline(pipeline, X_va, proba=False))
            score = compute_metric(
                metric_name=metric_name,
                task_type=task_type,
                y_true=y_va.to_numpy(),
                y_pred=pred,
            )
            bundle = compute_metrics_bundle(
                task_type=task_type,
                n_classes=None,
                y_true=y_va.to_numpy(),
                y_pred_labels=None,
                y_pred_scores=pred,
            )
            for key, value in bundle.items():
                fold_metrics.setdefault(key, []).append(float(value))
            oof_values[valid_idx] = pred

        fold_scores.append(float(score))
        joblib.dump(pipeline, output_dir / f"model_fold_{fold}.joblib")

    if task_type == "classification":
        overall_metrics = compute_metrics_bundle(
            task_type=task_type,
            n_classes=n_classes,
            y_true=y.to_numpy(),
            y_pred_labels=oof_labels,
            y_pred_scores=None if np.isnan(oof_scores).all() else oof_scores,
        )
        oof_for_primary = oof_scores if classification_use_proba else oof_labels
    else:
        overall_metrics = compute_metrics_bundle(
            task_type=task_type,
            n_classes=None,
            y_true=y.to_numpy(),
            y_pred_labels=None,
            y_pred_scores=oof_values,
        )
        oof_for_primary = oof_values

    cv_score = compute_metric(
        metric_name=metric_name,
        task_type=task_type,
        y_true=y.to_numpy(),
        y_pred=oof_for_primary,
    )

    pd.DataFrame({"oof_prediction": oof_for_primary}).to_csv(output_dir / "oof_predictions.csv", index=False)

    results = {
        "status": "ok",
        "task_type": task_type,
        "metric": metric_name,
        "metric_direction": metric_direction,
        "cv_score": cv_score,
        "fold_scores": fold_scores,
        "metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "model_family": model_family,
        "n_folds": n_splits,
        "split_version": split_version,
        "target_column": target_col,
        "id_column": id_col,
    }
    (output_dir / "train_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
