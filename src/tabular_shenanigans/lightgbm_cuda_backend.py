from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse

LIGHTGBM_CUDA_BUILD_DISABLED_FRAGMENT = "CUDA Tree Learner was not enabled in this build."


def _module_startswith(values: object, prefix: str) -> bool:
    return type(values).__module__.startswith(prefix)


def coerce_lightgbm_matrix_input(values: object) -> object:
    if sparse.issparse(values):
        return sparse.csr_matrix(values)

    if isinstance(values, pd.DataFrame | pd.Series):
        return values.to_numpy()

    if _module_startswith(values, "cudf"):
        return values.to_pandas().to_numpy()

    if _module_startswith(values, "cupy"):
        import cupy as cp

        return cp.asnumpy(values)

    if hasattr(values, "to_numpy"):
        return np.asarray(values.to_numpy())

    return np.asarray(values)


def _coerce_lightgbm_eval_set(eval_set: object) -> object:
    if eval_set is None:
        return None

    coerced_eval_set = []
    for x_values, y_values in eval_set:
        coerced_eval_set.append((coerce_lightgbm_matrix_input(x_values), y_values))
    return coerced_eval_set


@dataclass(frozen=True)
class LightGbmCudaValidationResult:
    lightgbm_version: str
    validated: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lightgbm_version": self.lightgbm_version,
            "validated": self.validated,
            "detail": self.detail,
        }


@lru_cache(maxsize=1)
def probe_lightgbm_cuda_build() -> LightGbmCudaValidationResult:
    try:
        import lightgbm
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM support requires the optional boosters dependencies. "
            "Install them with `uv sync --extra boosters`."
        ) from exc

    x_probe = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y_probe = np.asarray([0, 1, 1, 0], dtype=np.int64)
    estimator = LGBMClassifier(
        device_type="cuda",
        min_child_samples=1,
        min_child_weight=0.0,
        min_data_in_bin=1,
        n_estimators=1,
        num_leaves=4,
        verbosity=-1,
    )
    try:
        estimator.fit(x_probe, y_probe)
    except Exception as exc:
        message = str(exc)
        if LIGHTGBM_CUDA_BUILD_DISABLED_FRAGMENT in message:
            return LightGbmCudaValidationResult(
                lightgbm_version=lightgbm.__version__,
                validated=False,
                detail=message,
            )
        raise RuntimeError(
            "The LightGBM CUDA validation probe failed before a training run could start. "
            f"Reason: {message}"
        ) from exc

    return LightGbmCudaValidationResult(
        lightgbm_version=lightgbm.__version__,
        validated=True,
        detail="LightGBM accepted a CUDA training probe on this host.",
    )


def ensure_lightgbm_cuda_build() -> None:
    validation_result = probe_lightgbm_cuda_build()
    if validation_result.validated:
        return

    raise RuntimeError(
        "gpu_native LightGBM requires a CUDA-enabled LightGBM build. "
        "The stock wheel does not satisfy `device_type=\"cuda\"`. "
        "Reinstall LightGBM from source with "
        "`uv pip install --python .venv/bin/python --reinstall-package lightgbm "
        "--no-binary lightgbm -C cmake.define.USE_CUDA=ON \"lightgbm>=4.6.0\"` "
        "and validate it on the GPU host with "
        "`PYTHONPATH=src uv run python scripts/validate_lightgbm_cuda_build.py`. "
        f"Probe detail: {validation_result.detail}"
    )


class RepositoryLightGbmEstimator:
    def __init__(self, estimator: object, *, requires_cuda_build: bool) -> None:
        self.estimator = estimator
        self.requires_cuda_build = requires_cuda_build

    def fit(self, x_train: object, y_train: object, **fit_kwargs: object) -> "RepositoryLightGbmEstimator":
        if self.requires_cuda_build:
            ensure_lightgbm_cuda_build()

        resolved_fit_kwargs = dict(fit_kwargs)
        if "eval_set" in resolved_fit_kwargs:
            resolved_fit_kwargs["eval_set"] = _coerce_lightgbm_eval_set(resolved_fit_kwargs["eval_set"])

        self.estimator.fit(coerce_lightgbm_matrix_input(x_train), y_train, **resolved_fit_kwargs)
        return self

    def _predict_with_booster(
        self,
        x_values: object,
        *,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **kwargs: object,
    ) -> object:
        if not hasattr(self.estimator, "booster_"):
            raise ValueError("LightGBM estimator must be fit before predict.")

        return self.estimator.booster_.predict(
            coerce_lightgbm_matrix_input(x_values),
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=False,
            **kwargs,
        )

    def predict(
        self,
        x_values: object,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **kwargs: object,
    ) -> object:
        raw_predictions = self._predict_with_booster(
            x_values,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        if not hasattr(self.estimator, "classes_") or raw_score or pred_leaf or pred_contrib:
            return raw_predictions

        probability_values = np.asarray(
            self.predict_proba(
                x_values,
                start_iteration=start_iteration,
                num_iteration=num_iteration,
                **kwargs,
            )
        )
        if probability_values.ndim == 1:
            predicted_indices = (probability_values >= 0.5).astype(int)
        else:
            predicted_indices = np.argmax(probability_values, axis=1)
        return self.estimator.classes_[predicted_indices]

    def predict_proba(
        self,
        x_values: object,
        raw_score: bool = False,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **kwargs: object,
    ) -> object:
        raw_predictions = self._predict_with_booster(
            x_values,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs,
        )
        if raw_score or pred_leaf or pred_contrib:
            return raw_predictions

        prediction_array = np.asarray(raw_predictions)
        if prediction_array.ndim != 1:
            return prediction_array
        return np.vstack((1.0 - prediction_array, prediction_array)).transpose()

    def __getattr__(self, name: str) -> object:
        return getattr(self.estimator, name)
