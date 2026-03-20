import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric
from tabular_shenanigans.models import (
    get_tunable_model_ids,
    is_model_tunable,
    resolve_candidate_model_id,
    validate_model_parameter_overrides,
    validate_model_preprocessing_compatibility,
)
from tabular_shenanigans.naming import (
    build_blend_candidate_id,
    build_model_candidate_id,
    normalize_blend_weights,
)
from tabular_shenanigans.representations import (
    get_representation_definition,
    resolve_representation_id,
)
from tabular_shenanigans.preprocess_execution import resolve_preprocessing_execution_plan


class ConfigError(ValueError):
    pass


def _competition_identity_fingerprint_payload(competition: "CompetitionConfig") -> dict[str, object]:
    return {
        "slug": competition.slug,
        "task_type": competition.task_type,
        "primary_metric": competition.primary_metric,
        "id_column": competition.id_column,
        "label_column": competition.label_column,
        "features": competition.features.model_dump(mode="python"),
        "positive_label": competition.positive_label,
    }


class CompetitionCvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_splits: int = Field(default=7, ge=2)
    shuffle: bool = True
    random_state: int = 42


class CompetitionFeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force_categorical: list[str] = Field(default_factory=list)
    force_numeric: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    low_cardinality_int_threshold: int | None = Field(default=None, ge=1)


class CompetitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1)
    task_type: Literal["regression", "binary"]
    primary_metric: str
    positive_label: str | int | bool | None = None
    id_column: str | None = None
    label_column: str | None = None
    cv: CompetitionCvConfig = Field(default_factory=CompetitionCvConfig)
    features: CompetitionFeaturesConfig = Field(default_factory=CompetitionFeaturesConfig)


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representation_ids: list[str] = Field(min_length=1)
    model_families: list[str] = Field(min_length=1)
    optimization: "CandidateOptimizationConfig | None" = None
    cv: CompetitionCvConfig = Field(
        default_factory=lambda: CompetitionCvConfig(n_splits=2, shuffle=True, random_state=42)
    )
    promote_top_k: int = Field(default=3, ge=1)
    candidates: list["ModelCandidateConfig"] = Field(default_factory=list, exclude=True)
    active_candidate_index: int = Field(default=0, exclude=True, repr=False, ge=0)

    @model_validator(mode="before")
    @classmethod
    def reject_user_provided_candidates(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        if "candidates" in values:
            raise ValueError(
                "screening.candidates is not a user-facing field. "
                "Use screening.representation_ids and screening.model_families instead. "
                "The candidates list is built automatically from the cross-product."
            )
        return values

    @property
    def candidate(self) -> "ModelCandidateConfig":
        return self.candidates[self.active_candidate_index]


class CandidateOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["optuna"] = "optuna"
    n_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    random_state: int = 42


class BaseCandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCandidateConfig(BaseCandidateConfig):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["model"] = "model"
    representation_id: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    model_params: dict[str, object] = Field(default_factory=dict)
    optimization: CandidateOptimizationConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        legacy_fields = {"preprocessor", "numeric_preprocessor", "categorical_preprocessor", "feature_recipe_id"}
        found_legacy = sorted(legacy_fields.intersection(values))
        if found_legacy:
            raise ValueError(
                f"Legacy candidate fields {found_legacy} are no longer supported. "
                "Use candidate.representation_id instead."
            )
        return values

    @model_validator(mode="after")
    def validate_model_candidate(self) -> "ModelCandidateConfig":
        if self.model_params and self.optimization is not None:
            raise ValueError(
                "candidate.model_params and candidate.optimization are mutually exclusive. "
                "Use model_params for fixed training or optimization for tuning, not both."
            )
        return self


class BlendCandidateConfig(BaseCandidateConfig):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["blend"] = "blend"
    base_candidate_ids: list[str] = Field(min_length=2)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def validate_blend_candidate(self) -> "BlendCandidateConfig":
        duplicate_candidate_ids = sorted(
            candidate_id
            for candidate_id in set(self.base_candidate_ids)
            if self.base_candidate_ids.count(candidate_id) > 1
        )
        if duplicate_candidate_ids:
            raise ValueError(
                "Blend candidates require distinct base_candidate_ids. "
                f"Duplicates: {duplicate_candidate_ids}"
            )

        if self.weights is None:
            return self

        if len(self.weights) != len(self.base_candidate_ids):
            raise ValueError(
                "Blend candidate weights must match the number of base_candidate_ids. "
                f"Got {len(self.weights)} weights for {len(self.base_candidate_ids)} base candidates."
            )

        invalid_weights = [
            weight for weight in self.weights if not math.isfinite(weight) or weight <= 0
        ]
        if invalid_weights:
            raise ValueError(
                "Blend candidate weights must all be positive. "
                f"Invalid weights: {invalid_weights}"
            )

        return self


ExperimentCandidateConfig = Annotated[
    ModelCandidateConfig | BlendCandidateConfig,
    Field(discriminator="candidate_type"),
]


class ExperimentTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str = Field(min_length=1)


class ExperimentRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compute_target: Literal["auto", "cpu", "gpu"] = "auto"
    gpu_backend: Literal["auto", "patch", "native"] = "auto"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ExperimentCandidateConfig] = Field(min_length=1)
    runtime: ExperimentRuntimeConfig = Field(default_factory=ExperimentRuntimeConfig)
    tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)
    active_candidate_index: int = Field(default=0, exclude=True, repr=False, ge=0)
    legacy_candidate_contract_used: bool = Field(default=False, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_candidate_contract(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values

        if values.get("candidate") is not None and values.get("candidates") is not None:
            raise ValueError("Use either experiment.candidate or experiment.candidates, not both.")

        if values.get("candidate") is None:
            return values

        migrated_values = dict(values)
        migrated_values["candidates"] = [migrated_values.pop("candidate")]
        migrated_values["legacy_candidate_contract_used"] = True
        return migrated_values

    @model_validator(mode="after")
    def validate_active_candidate_index(self) -> "ExperimentConfig":
        if self.active_candidate_index >= len(self.candidates):
            raise ValueError(
                "experiment.active_candidate_index must reference a configured candidate. "
                f"Got {self.active_candidate_index} for {len(self.candidates)} candidates."
            )
        return self

    @property
    def candidate(self) -> ExperimentCandidateConfig:
        return self.candidates[self.active_candidate_index]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition: CompetitionConfig
    experiment: ExperimentConfig
    screening: ScreeningConfig | None = None
    active_run_stage: Literal["canonical", "screening"] = Field(
        default="canonical",
        exclude=True,
        repr=False,
    )

    @model_validator(mode="after")
    def validate_config(self) -> "AppConfig":
        competition = self.competition
        normalized_primary_metric = normalize_primary_metric(competition.primary_metric)
        if normalized_primary_metric is None:
            raise ValueError(
                "Configured competition.primary_metric is not supported. "
                f"Supported values: {sorted(set(SUPPORTED_PRIMARY_METRICS.values()))}"
            )
        if not is_metric_valid_for_task(competition.task_type, normalized_primary_metric):
            raise ValueError(
                "Configured competition.primary_metric "
                f"'{normalized_primary_metric}' is not valid for task_type '{competition.task_type}'."
            )
        competition.primary_metric = normalized_primary_metric

        if competition.task_type != "binary" and competition.positive_label is not None:
            raise ValueError("competition.positive_label is only supported for binary task_type.")

        resolved_candidate_ids: dict[str, list[int]] = {}
        for candidate_index, candidate in enumerate(self.experiment.candidates, start=1):
            try:
                if isinstance(candidate, ModelCandidateConfig):
                    candidate.representation_id = resolve_representation_id(candidate.representation_id)
                    representation_definition = get_representation_definition(candidate.representation_id)
                    resolved_model_registry_key = self.resolve_model_registry_key_for_index(candidate_index - 1)
                    validate_model_preprocessing_compatibility(
                        task_type=competition.task_type,
                        model_id=resolved_model_registry_key,
                        categorical_preprocessor_id=representation_definition.categorical_preprocessor_id,
                    )
                    validate_model_parameter_overrides(
                        task_type=competition.task_type,
                        model_id=resolved_model_registry_key,
                        parameter_overrides=candidate.model_params,
                    )
                    if candidate.optimization is not None:
                        optimization = candidate.optimization
                        if optimization.n_trials is None and optimization.timeout_seconds is None:
                            raise ValueError(
                                "At least one candidate.optimization stopping condition is required. "
                                "Set candidate.optimization.n_trials or "
                                "candidate.optimization.timeout_seconds."
                            )
                        if not is_model_tunable(competition.task_type, resolved_model_registry_key):
                            supported_tunable_model_families = get_tunable_model_ids(competition.task_type)
                            raise ValueError(
                                f"Configured model_family '{candidate.model_family}' does not support optimization "
                                f"for task_type '{competition.task_type}'. Supported tunable model families: "
                                f"{supported_tunable_model_families}"
                            )
            except ValueError as exc:
                raise ValueError(f"experiment.candidates[{candidate_index}] is invalid: {exc}") from exc

            resolved_candidate_id = self.resolve_candidate_id_for_index(candidate_index - 1)
            resolved_candidate_ids.setdefault(resolved_candidate_id, []).append(candidate_index)

        duplicate_candidate_ids = {
            candidate_id: candidate_indices
            for candidate_id, candidate_indices in resolved_candidate_ids.items()
            if len(candidate_indices) > 1
        }
        if duplicate_candidate_ids:
            duplicate_summary = "; ".join(
                f"{candidate_id} at candidates {candidate_indices}"
                for candidate_id, candidate_indices in sorted(duplicate_candidate_ids.items())
            )
            raise ValueError(
                "Configured candidates must derive distinct candidate_id values. "
                f"Duplicates: {duplicate_summary}"
            )

        if self.screening is not None:
            screening = self.screening

            resolved_representation_ids = []
            for raw_repr_id in screening.representation_ids:
                resolved_repr_id = resolve_representation_id(raw_repr_id)
                get_representation_definition(resolved_repr_id)
                resolved_representation_ids.append(resolved_repr_id)

            resolved_model_families: list[tuple[str, str]] = []
            for raw_model_family in screening.model_families:
                resolved_model_id = resolve_candidate_model_id(
                    task_type=competition.task_type,
                    model_family=raw_model_family,
                )
                resolved_model_families.append((raw_model_family, resolved_model_id))

            expanded_candidates: list[ModelCandidateConfig] = []
            for repr_id in resolved_representation_ids:
                representation_definition = get_representation_definition(repr_id)
                for model_family, model_registry_key in resolved_model_families:
                    try:
                        validate_model_preprocessing_compatibility(
                            task_type=competition.task_type,
                            model_id=model_registry_key,
                            categorical_preprocessor_id=representation_definition.categorical_preprocessor_id,
                        )
                    except ValueError as exc:
                        print(
                            f"Screening: skipping incompatible pair "
                            f"(model_family={model_family!r}, representation_id={repr_id!r}): {exc}"
                        )
                        continue
                    expanded_candidates.append(
                        ModelCandidateConfig(
                            representation_id=repr_id,
                            model_family=model_family,
                            optimization=screening.optimization.model_copy(deep=True)
                            if screening.optimization is not None
                            else None,
                        )
                    )

            if not expanded_candidates:
                raise ValueError(
                    "screening: no valid (model_family, representation_id) pairs survived "
                    "compatibility checks. All cross-product pairs were incompatible."
                )

            screening.candidates = expanded_candidates

            if screening.active_candidate_index >= len(screening.candidates):
                raise ValueError(
                    "screening.active_candidate_index must reference a valid screening candidate. "
                    f"Got {screening.active_candidate_index} for {len(screening.candidates)} candidates."
                )

            if screening.promote_top_k > len(screening.candidates):
                raise ValueError(
                    "screening.promote_top_k cannot exceed the number of valid screening candidates. "
                    f"Got promote_top_k={screening.promote_top_k} for "
                    f"{len(screening.candidates)} valid candidates."
                )

            resolved_screening_candidate_ids: dict[str, list[int]] = {}
            for candidate_index, candidate in enumerate(screening.candidates, start=1):
                resolved_candidate_id = self.resolve_screening_candidate_id_for_index(candidate_index - 1)
                resolved_screening_candidate_ids.setdefault(resolved_candidate_id, []).append(candidate_index)

            duplicate_screening_candidate_ids = {
                candidate_id: candidate_indices
                for candidate_id, candidate_indices in resolved_screening_candidate_ids.items()
                if len(candidate_indices) > 1
            }
            if duplicate_screening_candidate_ids:
                duplicate_summary = "; ".join(
                    f"{candidate_id} at screening candidates {candidate_indices}"
                    for candidate_id, candidate_indices in sorted(duplicate_screening_candidate_ids.items())
                )
                raise ValueError(
                    "Configured screening candidates must derive distinct candidate_id values. "
                    f"Duplicates: {duplicate_summary}"
                )

            if screening.optimization is not None:
                optimization = screening.optimization
                if optimization.n_trials is None and optimization.timeout_seconds is None:
                    raise ValueError(
                        "At least one screening.optimization stopping condition is required. "
                        "Set screening.optimization.n_trials or screening.optimization.timeout_seconds."
                    )
                for model_family, model_registry_key in resolved_model_families:
                    has_expanded_candidate = any(
                        c.model_family == model_family for c in expanded_candidates
                    )
                    if not has_expanded_candidate:
                        continue
                    if not is_model_tunable(competition.task_type, model_registry_key):
                        supported_tunable_model_families = get_tunable_model_ids(competition.task_type)
                        raise ValueError(
                            f"Configured screening model_family '{model_family}' does not support "
                            f"optimization for task_type '{competition.task_type}'. Supported tunable "
                            f"model families: {supported_tunable_model_families}"
                        )
        return self

    @property
    def is_model_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, ModelCandidateConfig)

    @property
    def is_blend_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, BlendCandidateConfig)

    @property
    def candidate_count(self) -> int:
        return len(self.experiment.candidates)

    @property
    def screening_candidate_count(self) -> int:
        if self.screening is None:
            return 0
        return len(self.screening.candidates)

    @property
    def active_candidate_index(self) -> int:
        return self.experiment.active_candidate_index

    @property
    def configured_candidate_ids(self) -> list[str]:
        return [self.resolve_candidate_id_for_index(candidate_index) for candidate_index in range(self.candidate_count)]

    def get_candidate(self, candidate_index: int | None = None) -> ExperimentCandidateConfig:
        if candidate_index is None:
            return self.experiment.candidate
        return self.experiment.candidates[candidate_index]

    def get_screening_candidate(self, candidate_index: int | None = None) -> ModelCandidateConfig:
        if self.screening is None:
            raise ValueError("screening is not configured in config.yaml.")
        if candidate_index is None:
            return self.screening.candidate
        return self.screening.candidates[candidate_index]

    def with_candidate_index(self, candidate_index: int) -> "AppConfig":
        if candidate_index < 0 or candidate_index >= self.candidate_count:
            raise ValueError(
                f"Candidate index must be between 0 and {self.candidate_count - 1}. Got {candidate_index}."
            )
        copied_config = self.model_copy(deep=True)
        copied_config.experiment.active_candidate_index = candidate_index
        return copied_config

    def with_screening_candidate_index(self, candidate_index: int) -> "AppConfig":
        if self.screening is None:
            raise ValueError("screening is not configured in config.yaml.")
        if candidate_index < 0 or candidate_index >= self.screening_candidate_count:
            raise ValueError(
                f"Screening candidate index must be between 0 and {self.screening_candidate_count - 1}. "
                f"Got {candidate_index}."
            )
        copied_config = self.model_copy(deep=True)
        copied_config.screening.active_candidate_index = candidate_index
        copied_config.experiment.candidates = [copied_config.screening.candidate]
        copied_config.experiment.active_candidate_index = 0
        copied_config.competition.cv = copied_config.screening.cv.model_copy(deep=True)
        copied_config.active_run_stage = "screening"
        return copied_config

    def resolve_candidate_indices(
        self,
        candidate_id: str | None = None,
        index: int | None = None,
        require_explicit: bool = False,
    ) -> list[int]:
        if candidate_id is not None and index is not None:
            raise ValueError("Use either --candidate-id or --index, not both.")

        if index is not None:
            resolved_index = index - 1
            if resolved_index < 0 or resolved_index >= self.candidate_count:
                raise ValueError(
                    f"--index must be between 1 and {self.candidate_count}. Got {index}."
                )
            return [resolved_index]

        if candidate_id is not None:
            configured_candidate_ids = self.configured_candidate_ids
            try:
                return [configured_candidate_ids.index(candidate_id)]
            except ValueError as exc:
                raise ValueError(
                    f"Configured candidate_id '{candidate_id}' was not found in config.yaml."
                ) from exc

        if require_explicit:
            raise ValueError("Select a configured candidate with --candidate-id or --index.")

        return list(range(self.candidate_count))

    def resolve_screening_candidate_indices(
        self,
        candidate_id: str | None = None,
        index: int | None = None,
        require_explicit: bool = False,
    ) -> list[int]:
        if self.screening is None:
            raise ValueError("screening is not configured in config.yaml.")
        if candidate_id is not None and index is not None:
            raise ValueError("Use either --candidate-id or --index, not both.")

        if index is not None:
            resolved_index = index - 1
            if resolved_index < 0 or resolved_index >= self.screening_candidate_count:
                raise ValueError(
                    f"--index must be between 1 and {self.screening_candidate_count}. Got {index}."
                )
            return [resolved_index]

        if candidate_id is not None:
            configured_candidate_ids = self.configured_screening_candidate_ids
            try:
                return [configured_candidate_ids.index(candidate_id)]
            except ValueError as exc:
                raise ValueError(
                    f"Configured screening candidate_id '{candidate_id}' was not found in config.yaml."
                ) from exc

        if require_explicit:
            raise ValueError("Select a configured screening candidate with --candidate-id or --index.")

        return list(range(self.screening_candidate_count))

    def resolve_model_registry_key_for_index(self, candidate_index: int | None = None) -> str:
        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            raise ValueError("resolved_model_registry_key is only available for model candidates.")
        return resolve_candidate_model_id(
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
        )

    def resolve_screening_model_registry_key_for_index(self, candidate_index: int | None = None) -> str:
        candidate = self.get_screening_candidate(candidate_index)
        return resolve_candidate_model_id(
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
        )

    def resolve_blend_weights_for_index(self, candidate_index: int | None = None) -> list[float]:
        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, BlendCandidateConfig):
            raise ValueError("resolved_blend_weights is only available for blend candidates.")
        return normalize_blend_weights(
            base_candidate_ids=candidate.base_candidate_ids,
            configured_weights=candidate.weights,
        )

    def runtime_execution_context_for_index(self, candidate_index: int | None = None):
        from tabular_shenanigans.execution_routing import resolve_model_candidate_runtime_execution
        from tabular_shenanigans.runtime_execution import (
            get_exported_runtime_execution_context,
            get_runtime_execution_context_override,
            resolve_runtime_execution,
        )

        override_runtime_execution_context = get_runtime_execution_context_override()
        if override_runtime_execution_context is not None:
            return override_runtime_execution_context

        exported_runtime_execution_context = get_exported_runtime_execution_context()
        exported_capabilities = None
        rapids_hooks_installed = False
        if exported_runtime_execution_context is not None:
            exported_capabilities = exported_runtime_execution_context.capabilities
            rapids_hooks_installed = exported_runtime_execution_context.rapids_hooks_installed

        runtime_execution_context = resolve_runtime_execution(
            self.experiment.runtime.compute_target,
            self.experiment.runtime.gpu_backend,
            capabilities=exported_capabilities,
        )
        runtime_execution_context = runtime_execution_context.__class__(
            requested_compute_target=runtime_execution_context.requested_compute_target,
            resolved_compute_target=runtime_execution_context.resolved_compute_target,
            requested_gpu_backend=runtime_execution_context.requested_gpu_backend,
            resolved_gpu_backend=runtime_execution_context.resolved_gpu_backend,
            capabilities=runtime_execution_context.capabilities,
            fallback_reason=runtime_execution_context.fallback_reason,
            rapids_hooks_installed=rapids_hooks_installed,
        )

        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            return runtime_execution_context

        representation_definition = get_representation_definition(candidate.representation_id)
        resolved_runtime_execution_context = resolve_model_candidate_runtime_execution(
            requested_compute_target=self.experiment.runtime.compute_target,
            requested_gpu_backend=self.experiment.runtime.gpu_backend,
            capabilities=runtime_execution_context.capabilities,
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
            numeric_preprocessor=representation_definition.numeric_preprocessor_id,
            categorical_preprocessor=representation_definition.categorical_preprocessor_id,
        )
        return resolved_runtime_execution_context.__class__(
            requested_compute_target=resolved_runtime_execution_context.requested_compute_target,
            resolved_compute_target=resolved_runtime_execution_context.resolved_compute_target,
            requested_gpu_backend=resolved_runtime_execution_context.requested_gpu_backend,
            resolved_gpu_backend=resolved_runtime_execution_context.resolved_gpu_backend,
            capabilities=resolved_runtime_execution_context.capabilities,
            fallback_reason=resolved_runtime_execution_context.fallback_reason,
            rapids_hooks_installed=rapids_hooks_installed,
        )

    def preprocessing_execution_plan_for_index(self, candidate_index: int | None = None):
        from tabular_shenanigans.models import resolve_model_matrix_output_kind

        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            raise ValueError("preprocessing_execution_plan is only available for model candidates.")

        representation_definition = get_representation_definition(candidate.representation_id)
        runtime_execution_context = self.runtime_execution_context_for_index(candidate_index)
        matrix_output_kind = resolve_model_matrix_output_kind(
            task_type=self.competition.task_type,
            model_id=self.resolve_model_registry_key_for_index(candidate_index),
            categorical_preprocessor_id=representation_definition.categorical_preprocessor_id,
            runtime_execution_context=runtime_execution_context,
        )
        return resolve_preprocessing_execution_plan(
            runtime_execution_context=runtime_execution_context,
            numeric_preprocessor_id=representation_definition.numeric_preprocessor_id,
            categorical_preprocessor_id=representation_definition.categorical_preprocessor_id,
            matrix_output_kind=matrix_output_kind,
        )

    def resolve_candidate_id_for_index(self, candidate_index: int | None = None) -> str:
        competition = self.competition
        candidate = self.get_candidate(candidate_index)
        if isinstance(candidate, ModelCandidateConfig):
            optimization_payload: dict[str, object] | None = None
            if candidate.optimization is not None:
                optimization_payload = candidate.optimization.model_dump(mode="python")
            fingerprint_payload = {
                "competition": _competition_identity_fingerprint_payload(competition),
                "candidate": {
                    "representation_id": candidate.representation_id,
                    "model_family": candidate.model_family,
                    "model_registry_key": self.resolve_model_registry_key_for_index(candidate_index),
                    "model_params": candidate.model_params,
                    "optimization": optimization_payload,
                },
            }
            return build_model_candidate_id(
                model_registry_key=self.resolve_model_registry_key_for_index(candidate_index),
                representation_id=candidate.representation_id,
                fingerprint_payload=fingerprint_payload,
            )

        normalized_weights = self.resolve_blend_weights_for_index(candidate_index)
        fingerprint_payload = {
            "competition": _competition_identity_fingerprint_payload(competition),
            "components": [
                {
                    "candidate_id": component_candidate_id,
                    "weight": weight,
                }
                for component_candidate_id, weight in sorted(
                    zip(candidate.base_candidate_ids, normalized_weights, strict=True),
                    key=lambda item: item[0],
                )
            ],
        }
        return build_blend_candidate_id(
            base_candidate_ids=candidate.base_candidate_ids,
            normalized_weights=normalized_weights,
            fingerprint_payload=fingerprint_payload,
        )

    def resolve_screening_candidate_id_for_index(self, candidate_index: int | None = None) -> str:
        competition = self.competition
        candidate = self.get_screening_candidate(candidate_index)
        optimization_payload: dict[str, object] | None = None
        if candidate.optimization is not None:
            optimization_payload = candidate.optimization.model_dump(mode="python")
        fingerprint_payload = {
            "competition": _competition_identity_fingerprint_payload(competition),
            "candidate": {
                "representation_id": candidate.representation_id,
                "model_family": candidate.model_family,
                "model_registry_key": self.resolve_screening_model_registry_key_for_index(candidate_index),
                "model_params": candidate.model_params,
                "optimization": optimization_payload,
            },
        }
        return build_model_candidate_id(
            model_registry_key=self.resolve_screening_model_registry_key_for_index(candidate_index),
            representation_id=candidate.representation_id,
            fingerprint_payload=fingerprint_payload,
        )

    @property
    def resolved_model_registry_key(self) -> str:
        return self.resolve_model_registry_key_for_index()

    @property
    def resolved_blend_weights(self) -> list[float]:
        return self.resolve_blend_weights_for_index()

    @property
    def runtime_execution_context(self):
        return self.runtime_execution_context_for_index()

    @property
    def preprocessing_execution_plan(self):
        return self.preprocessing_execution_plan_for_index()

    @property
    def resolved_candidate_id(self) -> str:
        return self.resolve_candidate_id_for_index()

    @property
    def configured_screening_candidate_ids(self) -> list[str]:
        return [
            self.resolve_screening_candidate_id_for_index(candidate_index)
            for candidate_index in range(self.screening_candidate_count)
        ]

    @property
    def resolved_screening_candidate_id(self) -> str:
        if self.screening is None:
            raise ValueError("screening is not configured in config.yaml.")
        return self.resolve_screening_candidate_id_for_index(self.screening.active_candidate_index)


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(
            f"Config file not found: {config_path}. "
            "Create repository-root config.yaml from config.binary.example.yaml "
            "or config.regression.example.yaml."
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError("Config must be a top-level mapping.")

    return AppConfig.model_validate(raw_data)
