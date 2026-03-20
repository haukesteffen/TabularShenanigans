from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from tabular_shenanigans.representations.types import FittedOutputAdapter, OutputContract


@dataclass
class _FittedDenseArrayAdapter:
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X.values, dtype=float)


@dataclass
class _FittedSparseCSRAdapter:
    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(X.values.astype(float))


@dataclass
class _FittedNativeFrameAdapter:
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


@dataclass(frozen=True)
class DenseArrayAdapter:
    output_contract: OutputContract = "dense_array"

    def fit(self, X: pd.DataFrame) -> FittedOutputAdapter:
        return _FittedDenseArrayAdapter()


@dataclass(frozen=True)
class SparseCSRAdapter:
    output_contract: OutputContract = "sparse_csr"

    def fit(self, X: pd.DataFrame) -> FittedOutputAdapter:
        return _FittedSparseCSRAdapter()


@dataclass(frozen=True)
class NativeFrameAdapter:
    output_contract: OutputContract = "native_frame"

    def fit(self, X: pd.DataFrame) -> FittedOutputAdapter:
        return _FittedNativeFrameAdapter()
