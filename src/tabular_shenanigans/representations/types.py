from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

FitMode = Literal["stateless", "unsupervised", "supervised"]
OutputContract = Literal["native_frame", "dense_array", "sparse_csr"]


class FittedStep(Protocol):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...


class RepresentationStep(Protocol):
    step_id: str
    fit_mode: FitMode

    def fit(self, X: pd.DataFrame, y: pd.Series | None) -> FittedStep: ...


class FittedOutputAdapter(Protocol):
    def transform(self, X: pd.DataFrame) -> object: ...


class OutputAdapter(Protocol):
    output_contract: OutputContract

    def fit(self, X: pd.DataFrame) -> FittedOutputAdapter: ...


@dataclass(frozen=True)
class RepresentationDefinition:
    representation_id: str
    representation_name: str
    steps: tuple[RepresentationStep, ...]
    output_adapter: OutputAdapter
    numeric_preprocessor_id: str
    categorical_preprocessor_id: str
