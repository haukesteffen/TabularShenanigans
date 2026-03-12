import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric
from tabular_shenanigans.feature_recipes import IDENTITY_FEATURE_RECIPE_ID, resolve_feature_recipe_id
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
from tabular_shenanigans.preprocess import (
    CATEGORICAL_PREPROCESSOR_IDS,
    NUMERIC_PREPROCESSOR_IDS,
    build_preprocessing_scheme_id,
)


class ConfigError(ValueError):
    pass


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


class CandidateOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["optuna"] = "optuna"
    n_trials: int | None = Field(default=None, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    random_state: int = 42


class BaseCandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCandidateConfig(BaseCandidateConfig):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["model"] = "model"
    feature_recipe_id: str = Field(default=IDENTITY_FEATURE_RECIPE_ID, min_length=1)
    numeric_preprocessor: str | None = None
    categorical_preprocessor: str | None = None
    model_family: Literal[
        "ridge",
        "elasticnet",
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "lightgbm",
        "catboost",
        "xgboost",
    ]
    model_params: dict[str, object] = Field(default_factory=dict)
    optimization: CandidateOptimizationConfig = Field(default_factory=CandidateOptimizationConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_preprocessor(cls, values: object) -> object:
        if isinstance(values, dict) and "preprocessor" in values:
            raise ValueError(
                "experiment.candidate.preprocessor is no longer supported. "
                "Use experiment.candidate.numeric_preprocessor and "
                "experiment.candidate.categorical_preprocessor."
            )
        return values

    @model_validator(mode="after")
    def validate_model_candidate(self) -> "ModelCandidateConfig":
        if self.model_params and self.optimization.enabled:
            raise ValueError(
                "The current runtime does not support combining experiment.candidate.model_params "
                "with enabled experiment.candidate.optimization."
            )

        if self.numeric_preprocessor is None or self.categorical_preprocessor is None:
            raise ValueError(
                "Model candidates require experiment.candidate.numeric_preprocessor and "
                "experiment.candidate.categorical_preprocessor."
            )

        if self.numeric_preprocessor not in NUMERIC_PREPROCESSOR_IDS:
            raise ValueError(
                "Unsupported experiment.candidate.numeric_preprocessor "
                f"'{self.numeric_preprocessor}'. Supported values: {sorted(NUMERIC_PREPROCESSOR_IDS)}"
            )

        if self.categorical_preprocessor not in CATEGORICAL_PREPROCESSOR_IDS:
            raise ValueError(
                "Unsupported experiment.candidate.categorical_preprocessor "
                f"'{self.categorical_preprocessor}'. Supported values: {sorted(CATEGORICAL_PREPROCESSOR_IDS)}"
            )
        return self

    @property
    def preprocessing_scheme_id(self) -> str:
        if self.numeric_preprocessor is None or self.categorical_preprocessor is None:
            raise ValueError(
                "preprocessing_scheme_id requires experiment.candidate.numeric_preprocessor and "
                "experiment.candidate.categorical_preprocessor."
            )
        return build_preprocessing_scheme_id(
            numeric_preprocessor_id=self.numeric_preprocessor,
            categorical_preprocessor_id=self.categorical_preprocessor,
        )


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


class ExperimentSubmitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    message_prefix: str | None = None


class ExperimentTrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str = Field(min_length=1)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: ExperimentCandidateConfig
    submit: ExperimentSubmitConfig = Field(default_factory=ExperimentSubmitConfig)
    tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition: CompetitionConfig
    experiment: ExperimentConfig

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

        if self.is_model_candidate:
            candidate = self.experiment.candidate
            candidate.feature_recipe_id = resolve_feature_recipe_id(candidate.feature_recipe_id)
            resolved_model_registry_key = self.resolved_model_registry_key
            validate_model_preprocessing_compatibility(
                task_type=competition.task_type,
                model_id=resolved_model_registry_key,
                categorical_preprocessor_id=candidate.categorical_preprocessor,
            )
            validate_model_parameter_overrides(
                task_type=competition.task_type,
                model_id=resolved_model_registry_key,
                parameter_overrides=candidate.model_params,
            )
            optimization = candidate.optimization
            if optimization.enabled:
                if optimization.n_trials is None and optimization.timeout_seconds is None:
                    raise ValueError(
                        "At least one experiment.candidate.optimization stopping condition is required. "
                        "Set experiment.candidate.optimization.n_trials or "
                        "experiment.candidate.optimization.timeout_seconds."
                    )
                if not is_model_tunable(competition.task_type, resolved_model_registry_key):
                    supported_tunable_model_families = get_tunable_model_ids(competition.task_type)
                    raise ValueError(
                        f"Configured model_family '{candidate.model_family}' does not support optimization for "
                        f"task_type '{competition.task_type}'. Supported tunable model families: "
                        f"{supported_tunable_model_families}"
                    )
        return self

    @property
    def is_model_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, ModelCandidateConfig)

    @property
    def is_blend_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, BlendCandidateConfig)

    @property
    def resolved_model_registry_key(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("resolved_model_registry_key is only available for model candidates.")
        candidate = self.experiment.candidate
        return resolve_candidate_model_id(
            task_type=self.competition.task_type,
            model_family=candidate.model_family,
        )

    @property
    def resolved_blend_weights(self) -> list[float]:
        if not self.is_blend_candidate:
            raise ValueError("resolved_blend_weights is only available for blend candidates.")
        candidate = self.experiment.candidate
        return normalize_blend_weights(
            base_candidate_ids=candidate.base_candidate_ids,
            configured_weights=candidate.weights,
        )

    @property
    def resolved_candidate_id(self) -> str:
        competition = self.competition
        candidate = self.experiment.candidate
        if self.is_model_candidate:
            optimization_payload: dict[str, object] = {"enabled": False}
            if candidate.optimization.enabled:
                optimization_payload = candidate.optimization.model_dump(mode="python")
            fingerprint_payload = {
                "competition": {
                    "task_type": competition.task_type,
                    "primary_metric": competition.primary_metric,
                    "cv": competition.cv.model_dump(mode="python"),
                    "features": competition.features.model_dump(mode="python"),
                    "positive_label": competition.positive_label,
                },
                "candidate": {
                    "feature_recipe_id": candidate.feature_recipe_id,
                    "numeric_preprocessor": candidate.numeric_preprocessor,
                    "categorical_preprocessor": candidate.categorical_preprocessor,
                    "preprocessing_scheme_id": candidate.preprocessing_scheme_id,
                    "model_family": candidate.model_family,
                    "model_registry_key": self.resolved_model_registry_key,
                    "model_params": candidate.model_params,
                    "optimization": optimization_payload,
                },
            }
            return build_model_candidate_id(
                feature_recipe_id=candidate.feature_recipe_id,
                preprocessing_scheme_id=candidate.preprocessing_scheme_id,
                model_registry_key=self.resolved_model_registry_key,
                fingerprint_payload=fingerprint_payload,
            )

        normalized_weights = self.resolved_blend_weights
        fingerprint_payload = {
            "competition": {
                "task_type": competition.task_type,
                "primary_metric": competition.primary_metric,
                "cv": competition.cv.model_dump(mode="python"),
                "features": competition.features.model_dump(mode="python"),
            },
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
