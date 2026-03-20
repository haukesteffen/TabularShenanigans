from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tabular_shenanigans.representations.feature_schema import ResolvedFeatureSchema
from tabular_shenanigans.representations.fitted import FittedRepresentation
from tabular_shenanigans.representations.steps import (
    bind_step_columns,
    is_categorical_step,
    is_numeric_step,
)
from tabular_shenanigans.representations.types import FittedStep, RepresentationDefinition


@dataclass(frozen=True)
class CompiledRepresentation:
    definition: RepresentationDefinition
    feature_schema: ResolvedFeatureSchema
    bound_steps: tuple[object, ...]
    matrix_output_kind: str
    preprocessing_backend: str

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FittedRepresentation:
        current = X_train.copy()
        fitted_steps: list[FittedStep] = []
        for step in self.bound_steps:
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

    bound_steps: list[object] = []
    for step in definition.steps:
        if is_numeric_step(step) and hasattr(step, "columns") and step.columns is None:
            bound_steps.append(bind_step_columns(step, list(feature_schema.numeric_columns)))
        elif is_categorical_step(step) and hasattr(step, "columns") and step.columns is None:
            bound_steps.append(bind_step_columns(step, list(feature_schema.categorical_columns)))
        else:
            bound_steps.append(step)

    return CompiledRepresentation(
        definition=definition,
        feature_schema=feature_schema,
        bound_steps=tuple(bound_steps),
        matrix_output_kind=execution_plan.matrix_output_kind,
        preprocessing_backend=execution_plan.preprocessing_backend,
    )
