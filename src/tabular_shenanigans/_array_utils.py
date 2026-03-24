from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_categorical_series(series: pd.Series, missing_value: str) -> pd.Series:
    return series.astype(object).where(series.notna(), missing_value).astype(str)


def _ensure_dense_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)
