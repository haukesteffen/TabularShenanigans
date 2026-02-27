from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _is_lightgbm_model(model: object) -> bool:
    return model.__class__.__module__.startswith("lightgbm")


def _align_feature_names(model: object, Xt: object) -> object:
    if isinstance(Xt, pd.DataFrame):
        return Xt
    if hasattr(model, "feature_names_in_"):
        feature_names = list(getattr(model, "feature_names_in_"))
        if len(feature_names) == np.shape(Xt)[1]:
            return pd.DataFrame(Xt, columns=feature_names)
    return Xt


def _predict_with_pipeline(pipeline: object, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    Xt = _align_feature_names(model, preprocessor.transform(X))

    if hasattr(model, "predict_proba"):
        if _is_lightgbm_model(model):
            preds = model.predict_proba(Xt, validate_features=False)
        else:
            preds = model.predict_proba(Xt)
        if preds.ndim == 2 and preds.shape[1] >= 2:
            return preds[:, 1]
        return preds.reshape(-1)

    if _is_lightgbm_model(model):
        return model.predict(Xt, validate_features=False)
    return model.predict(Xt)


def generate_predictions(
    run_dir: Path,
    processed_test_path: Path,
    pred_path: Path,
) -> Path:
    if not processed_test_path.exists():
        raise FileNotFoundError(
            f"Processed test data not found at {processed_test_path}. Run train first to prepare data."
        )

    X_test = pd.read_csv(processed_test_path)
    model_paths = sorted(run_dir.glob("model_fold_*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No fold models found in {run_dir}. Run train first.")

    fold_preds = []
    for path in model_paths:
        pipeline = joblib.load(path)
        fold_preds.append(_predict_with_pipeline(pipeline, X_test))

    pred_values = np.mean(np.vstack(fold_preds), axis=0)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": pred_values}).to_csv(pred_path, index=False)

    info_path = run_dir / "predict_info.json"
    info_path.write_text(
        json.dumps({"num_models": len(model_paths), "num_rows": int(len(X_test))}, indent=2),
        encoding="utf-8",
    )
    return pred_path
