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
    validate_model_representation_compatibility,
)
from tabular_shenanigans.naming import (
    build_blend_candidate_id,
    build_model_candidate_id,
    normalize_blend_weights,
)
from tabular_shenanigans.representations import (
    build_representation_contract,
    build_representation_spec_from_payload,
    validate_representation_spec,
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


class RepresentationComponentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)

    def params(self) -> dict[str, object]:
        return dict(self.model_extra or {})

    def to_payload(self) -> dict[str, object]:
        return {"id": self.id, **self.params()}


class RepresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operators: list[RepresentationComponentConfig] = Field(min_length=1)
    pruners: list[RepresentationComponentConfig] = Field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "operators": [operator.to_payload() for operator in self.operators],
            "pruners": [pruner.to_payload() for pruner in self.pruners],
        }

    def to_runtime_spec(self):
        representation_spec = build_representation_spec_from_payload(self.to_payload())
        validate_representation_spec(representation_spec)
        return representation_spec


class ScreeningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cv: CompetitionCvConfig = Field(
        default_factory=lambda: CompetitionCvConfig(n_splits=2, shuffle=True, random_state=42)
    )
    optimization: "CandidateOptimizationConfig | None" = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        if "representations" in values or "model_families" in values:
            raise ValueError(
                "screening.representations and screening.model_families are no longer supported. "
                "Screening candidates now live in experiment.candidates. "
                "Use screening only for mode overrides (cv, optimization)."
            )
        if "representation_ids" in values:
            raise ValueError(
                "screening.representation_ids is no longer supported. "
                "Screening candidates now live in experiment.candidates."
            )
        if "candidates" in values:
            raise ValueError(
                "screening.candidates is no longer supported. "
                "All candidates now live in experiment.candidates. "
                "Use screening only for mode overrides (cv, optimization). "
                "Run with: uv run python main.py train --screening"
            )
        if "promote_top_k" in values:
            raise ValueError(
                "screening.promote_top_k is no longer supported. "
                "The promotion snippet has been removed."
            )
        return values


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
    representation: RepresentationConfig
    representation_id: str = Field(default="", exclude=True)
    model_family: str = Field(min_length=1)
    model_params: dict[str, object] = Field(default_factory=dict)
    optimization: CandidateOptimizationConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        legacy_fields = {
            "representation_id",
            "preprocessor",
            "numeric_preprocessor",
            "categorical_preprocessor",
            "feature_recipe_id",
        }
        found_legacy = sorted(legacy_fields.intersection(values))
        if found_legacy:
            raise ValueError(
                f"Legacy candidate fields {found_legacy} are no longer supported. "
                "Use candidate.representation."
            )
        return values

    @model_validator(mode="after")
    def validate_model_candidate(self) -> "ModelCandidateConfig":
        if self.model_params and self.optimization is not None:
            raise ValueError(
                "candidate.model_params and candidate.optimization are mutually exclusive. "
                "Use model_params for fixed training or optimization for tuning, not both."
            )
        self.representation_id = self.representation.to_runtime_spec().representation_id
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
                    representation_spec = candidate.representation.to_runtime_spec()
                    representation_contract = build_representation_contract(representation_spec)
                    resolved_model_registry_key = self.resolve_model_registry_key_for_index(candidate_index - 1)
                    validate_model_representation_compatibility(
                        task_type=competition.task_type,
                        model_id=resolved_model_registry_key,
                        representation_contract=representation_contract,
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
            if screening.optimization is not None:
                optimization = screening.optimization
                if optimization.n_trials is None and optimization.timeout_seconds is None:
                    raise ValueError(
                        "At least one screening.optimization stopping condition is required. "
                        "Set screening.optimization.n_trials or screening.optimization.timeout_seconds."
                    )
                for candidate_index, candidate in enumerate(self.experiment.candidates, start=1):
                    if not isinstance(candidate, ModelCandidateConfig):
                        continue
                    if candidate.optimization is not None:
                        continue
                    model_registry_key = resolve_candidate_model_id(
                        task_type=competition.task_type,
                        model_family=candidate.model_family,
                    )
                    if not is_model_tunable(competition.task_type, model_registry_key):
                        supported_tunable_model_families = get_tunable_model_ids(competition.task_type)
                        raise ValueError(
                            f"screening.optimization would apply to experiment.candidates[{candidate_index}] "
                            f"(model_family='{candidate.model_family}'), but this model does not support "
                            f"optimization for task_type '{competition.task_type}'. "
                            f"Supported tunable model families: {supported_tunable_model_families}. "
                            "Either remove screening.optimization, add per-candidate optimization "
                            "to override it, or remove the untunable candidate."
                        )
            has_model_candidate = any(
                isinstance(c, ModelCandidateConfig) for c in self.experiment.candidates
            )
            if not has_model_candidate:
                raise ValueError(
                    "screening requires at least one model candidate in experiment.candidates. "
                    "Blend candidates are not supported in screening mode."
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
    def active_candidate_index(self) -> int:
        return self.experiment.active_candidate_index

    @property
    def configured_candidate_ids(self) -> list[str]:
        return [self.resolve_candidate_id_for_index(candidate_index) for candidate_index in range(self.candidate_count)]

    def get_candidate(self, candidate_index: int | None = None) -> ExperimentCandidateConfig:
        if candidate_index is None:
            return self.experiment.candidate
        return self.experiment.candidates[candidate_index]

    def with_candidate_index(self, candidate_index: int, screening: bool = False) -> "AppConfig":
        if candidate_index < 0 or candidate_index >= self.candidate_count:
            raise ValueError(
                f"Candidate index must be between 0 and {self.candidate_count - 1}. Got {candidate_index}."
            )
        copied_config = self.model_copy(deep=True)
        copied_config.experiment.active_candidate_index = candidate_index
        if screening:
            copied_config.active_run_stage = "screening"
            if self.screening is not None:
                copied_config.competition.cv = self.screening.cv.model_copy(deep=True)
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

    def resolve_model_registry_key_for_index(self, candidate_index: int | None = None) -> str:
        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            raise ValueError("resolved_model_registry_key is only available for model candidates.")
        return resolve_candidate_model_id(
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
        )

    def resolve_representation_spec_for_index(self, candidate_index: int | None = None):
        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            raise ValueError("representation spec is only available for model candidates.")
        return candidate.representation.to_runtime_spec()

    def resolve_representation_contract_for_index(self, candidate_index: int | None = None):
        return build_representation_contract(self.resolve_representation_spec_for_index(candidate_index))


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

        representation_contract = self.resolve_representation_contract_for_index(candidate_index)
        resolved_runtime_execution_context = resolve_model_candidate_runtime_execution(
            requested_compute_target=self.experiment.runtime.compute_target,
            requested_gpu_backend=self.experiment.runtime.gpu_backend,
            capabilities=runtime_execution_context.capabilities,
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
            has_native_categorical=representation_contract.has_native_categorical,
            has_sparse_numeric=representation_contract.has_sparse_numeric,
        )
        return resolved_runtime_execution_context.__class__(
            requested_compute_target=resolved_runtime_execution_context.requested_compute_target,
            resolved_compute_target=resolved_runtime_execution_context.resolved_compute_target,
            requested_gpu_backend=resolved_runtime_execution_context.requested_gpu_backend,
            resolved_gpu_backend=resolved_runtime_execution_context.resolved_gpu_backend,
            capabilities=resolved_runtime_execution_context.capabilities,
            fallback_reason=resolved_runtime_execution_context.fallback_reason,
            rapids_hooks_installed=rapids_hooks_installed,
            sparse_to_dense_coercion=resolved_runtime_execution_context.sparse_to_dense_coercion,
        )

    def preprocessing_execution_plan_for_index(self, candidate_index: int | None = None):
        candidate = self.get_candidate(candidate_index)
        if not isinstance(candidate, ModelCandidateConfig):
            raise ValueError("preprocessing_execution_plan is only available for model candidates.")

        representation_contract = self.resolve_representation_contract_for_index(candidate_index)
        runtime_execution_context = self.runtime_execution_context_for_index(candidate_index)
        return resolve_preprocessing_execution_plan(
            runtime_execution_context=runtime_execution_context,
            representation_contract=representation_contract,
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
                    "representation": candidate.representation.to_payload(),
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
