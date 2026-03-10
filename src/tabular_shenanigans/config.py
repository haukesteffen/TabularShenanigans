import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric
from tabular_shenanigans.feature_recipes import resolve_feature_recipe_id
from tabular_shenanigans.models import (
    get_tunable_candidate_model_specs,
    is_model_tunable,
    resolve_candidate_model_id,
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

    candidate_id: str = Field(min_length=1)


class ModelCandidateConfig(BaseCandidateConfig):
    model_config = ConfigDict(extra="forbid")

    candidate_type: Literal["model"] = "model"
    feature_recipe_id: str = Field(default="identity", min_length=1)
    preprocessor: Literal["onehot", "ordinal", "native", "frequency"]
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

    @model_validator(mode="after")
    def validate_model_candidate(self) -> "ModelCandidateConfig":
        if self.model_params and self.optimization.enabled:
            raise ValueError(
                "The current runtime does not support combining experiment.candidate.model_params "
                "with enabled experiment.candidate.optimization."
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


class ExperimentSubmitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    message_prefix: str | None = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    notes: str | None = None
    candidate: ExperimentCandidateConfig
    submit: ExperimentSubmitConfig = Field(default_factory=ExperimentSubmitConfig)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition: CompetitionConfig
    experiment: ExperimentConfig

    @model_validator(mode="after")
    def validate_config(self) -> "AppConfig":
        normalized_primary_metric = normalize_primary_metric(self.competition.primary_metric)
        if normalized_primary_metric is None:
            raise ValueError(
                "Configured competition.primary_metric is not supported. "
                f"Supported values: {sorted(set(SUPPORTED_PRIMARY_METRICS.values()))}"
            )
        if not is_metric_valid_for_task(self.competition.task_type, normalized_primary_metric):
            raise ValueError(
                "Configured competition.primary_metric "
                f"'{normalized_primary_metric}' is not valid for task_type '{self.competition.task_type}'."
            )
        self.competition.primary_metric = normalized_primary_metric

        if self.competition.task_type != "binary" and self.competition.positive_label is not None:
            raise ValueError("competition.positive_label is only supported for binary task_type.")

        if self.is_model_candidate:
            self.experiment.candidate.feature_recipe_id = resolve_feature_recipe_id(
                self.experiment.candidate.feature_recipe_id
            )

            resolved_model_id = self.resolved_model_id
            optimization = self.experiment.candidate.optimization
            if optimization.enabled:
                if optimization.n_trials is None and optimization.timeout_seconds is None:
                    raise ValueError(
                        "At least one experiment.candidate.optimization stopping condition is required. "
                        "Set experiment.candidate.optimization.n_trials or "
                        "experiment.candidate.optimization.timeout_seconds."
                    )
                if not is_model_tunable(self.task_type, resolved_model_id):
                    supported_tunable_combinations = [
                        f"{model_family}+{preprocessor}"
                        for model_family, preprocessor, _ in get_tunable_candidate_model_specs(self.task_type)
                    ]
                    raise ValueError(
                        "Configured experiment.candidate does not support optimization for task_type "
                        f"'{self.task_type}'. Supported tunable combinations: {supported_tunable_combinations}"
                    )

        return self

    @property
    def competition_slug(self) -> str:
        return self.competition.slug

    @property
    def task_type(self) -> Literal["regression", "binary"]:
        return self.competition.task_type

    @property
    def primary_metric(self) -> str:
        return self.competition.primary_metric

    @property
    def positive_label(self) -> str | int | bool | None:
        return self.competition.positive_label

    @property
    def id_column(self) -> str | None:
        return self.competition.id_column

    @property
    def label_column(self) -> str | None:
        return self.competition.label_column

    @property
    def force_categorical(self) -> list[str]:
        return self.competition.features.force_categorical

    @property
    def force_numeric(self) -> list[str]:
        return self.competition.features.force_numeric

    @property
    def drop_columns(self) -> list[str]:
        return self.competition.features.drop_columns

    @property
    def low_cardinality_int_threshold(self) -> int | None:
        return self.competition.features.low_cardinality_int_threshold

    @property
    def cv_n_splits(self) -> int:
        return self.competition.cv.n_splits

    @property
    def cv_shuffle(self) -> bool:
        return self.competition.cv.shuffle

    @property
    def cv_random_state(self) -> int:
        return self.competition.cv.random_state

    @property
    def candidate_id(self) -> str:
        return self.experiment.candidate.candidate_id

    @property
    def candidate_type(self) -> str:
        return self.experiment.candidate.candidate_type

    @property
    def is_model_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, ModelCandidateConfig)

    @property
    def is_blend_candidate(self) -> bool:
        return isinstance(self.experiment.candidate, BlendCandidateConfig)

    @property
    def model_family(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("model_family is only available for model candidates.")
        return self.experiment.candidate.model_family

    @property
    def feature_recipe_id(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("feature_recipe_id is only available for model candidates.")
        return self.experiment.candidate.feature_recipe_id

    @property
    def preprocessor(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("preprocessor is only available for model candidates.")
        return self.experiment.candidate.preprocessor

    @property
    def resolved_model_id(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("resolved_model_id is only available for model candidates.")
        return resolve_candidate_model_id(
            task_type=self.task_type,
            model_family=self.model_family,
            preprocessor=self.preprocessor,
        )

    @property
    def model_parameter_overrides(self) -> dict[str, object] | None:
        if not self.is_model_candidate:
            raise ValueError("model_parameter_overrides is only available for model candidates.")
        if not self.experiment.candidate.model_params:
            return None
        return dict(self.experiment.candidate.model_params)

    @property
    def base_candidate_ids(self) -> list[str]:
        if not self.is_blend_candidate:
            raise ValueError("base_candidate_ids is only available for blend candidates.")
        return list(self.experiment.candidate.base_candidate_ids)

    @property
    def blend_weights(self) -> list[float] | None:
        if not self.is_blend_candidate:
            raise ValueError("blend_weights is only available for blend candidates.")
        if self.experiment.candidate.weights is None:
            return None
        return list(self.experiment.candidate.weights)

    @property
    def submit_enabled(self) -> bool:
        return self.experiment.submit.enabled

    @property
    def submit_message_prefix(self) -> str | None:
        return self.experiment.submit.message_prefix


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
