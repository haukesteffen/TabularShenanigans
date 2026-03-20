from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tabular_shenanigans.representations.feature_schema import ResolvedFeatureSchema
from tabular_shenanigans.representations.fitted import FittedRepresentation
from tabular_shenanigans.representations.types import FittedStep, RepresentationDefinition


@dataclass(frozen=True)
class CompiledRepresentation:
    definition: RepresentationDefinition
    feature_schema: ResolvedFeatureSchema
    matrix_output_kind: str
    preprocessing_backend: str

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FittedRepresentation:
        current = X_train.copy()
        fitted_steps: list[FittedStep] = []
        for step in self.definition.steps:
            y_arg = y_train if step.fit_mode == "supervised" else None
            fitted_step = step.fit(current, y_arg)
            current = fitted_step.transform(current)
            fitted_steps.append(fitted_step)
        fitted_output_adapter = self.definition.output_adapter.fit(current)
        return FittedRepresentation(
            representation_id=self.definition.representation_id,
            fitted_steps=tuple(fitted_steps),
            fitted_output_adapter=fitted_output_adapter,
        )


def compile_representation(
    definition: RepresentationDefinition,
    feature_schema: ResolvedFeatureSchema,
    runtime_execution_context: object,
    task_type: str,
    model_id: str,
) -> CompiledRepresentation:
    from tabular_shenanigans.models import (
        resolve_model_matrix_output_kind,
        validate_model_preprocessing_compatibility,
    )
    from tabular_shenanigans.preprocess_execution import resolve_preprocessing_execution_plan

    validate_model_preprocessing_compatibility(
        task_type=task_type,
        model_id=model_id,
        categorical_preprocessor_id=definition.categorical_preprocessor_id,
    )
    matrix_output_kind = resolve_model_matrix_output_kind(
        task_type=task_type,
        model_id=model_id,
        categorical_preprocessor_id=definition.categorical_preprocessor_id,
        runtime_execution_context=runtime_execution_context,
    )
    execution_plan = resolve_preprocessing_execution_plan(
        runtime_execution_context=runtime_execution_context,
        numeric_preprocessor_id=definition.numeric_preprocessor_id,
        categorical_preprocessor_id=definition.categorical_preprocessor_id,
        matrix_output_kind=matrix_output_kind,
    )
    return CompiledRepresentation(
        definition=definition,
        feature_schema=feature_schema,
        matrix_output_kind=execution_plan.matrix_output_kind,
        preprocessing_backend=execution_plan.preprocessing_backend,
    )
