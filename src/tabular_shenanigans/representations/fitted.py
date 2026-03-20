from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tabular_shenanigans.representations.types import FittedOutputAdapter, FittedStep


@dataclass(frozen=True)
class FittedRepresentation:
    representation_id: str
    fitted_steps: tuple[FittedStep, ...]
    fitted_output_adapter: FittedOutputAdapter

    def transform(self, X: pd.DataFrame) -> object:
        current = X.copy()
        for fitted_step in self.fitted_steps:
            current = fitted_step.transform(current)
        return self.fitted_output_adapter.transform(current)
