import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric
from tabular_shenanigans.feature_recipes import resolve_feature_recipe_id
from tabular_shenanigans.models import (
    get_tunable_model_ids,
    is_model_tunable,
    resolve_candidate_model_id,
    validate_model_preprocessing_compatibility,
)
from tabular_shenanigans.preprocess import (
    CATEGORICAL_PREPROCESSOR_IDS,
    LEGACY_PREPROCESSOR_MAPPING,
    NUMERIC_PREPROCESSOR_IDS,
    build_preprocessing_scheme_id,
    resolve_legacy_preprocessor_selection,
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
    preprocessor: str | None = None
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

    @model_validator(mode="after")
    def validate_model_candidate(self) -> "ModelCandidateConfig":
        if self.model_params and self.optimization.enabled:
            raise ValueError(
                "The current runtime does not support combining experiment.candidate.model_params "
                "with enabled experiment.candidate.optimization."
            )

        if self.preprocessor is not None:
            if self.numeric_preprocessor is not None or self.categorical_preprocessor is not None:
                raise ValueError(
                    "Configure either experiment.candidate.preprocessor or the split "
                    "experiment.candidate.numeric_preprocessor + experiment.candidate.categorical_preprocessor, "
                    "not both."
                )
            resolved_numeric_preprocessor, resolved_categorical_preprocessor = resolve_legacy_preprocessor_selection(
                self.preprocessor
            )
            self.numeric_preprocessor = resolved_numeric_preprocessor
            self.categorical_preprocessor = resolved_categorical_preprocessor

        if self.numeric_preprocessor is None or self.categorical_preprocessor is None:
            legacy_preprocessor_ids = sorted(LEGACY_PREPROCESSOR_MAPPING)
            raise ValueError(
                "Model candidates require experiment.candidate.numeric_preprocessor and "
                "experiment.candidate.categorical_preprocessor. "
                f"Legacy experiment.candidate.preprocessor values remain temporarily supported: {legacy_preprocessor_ids}"
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

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None

    @model_validator(mode="after")
    def validate_tracking_config(self) -> "ExperimentTrackingConfig":
        if not self.enabled:
            return self

        if not self.tracking_uri:
            raise ValueError(
                "experiment.tracking.tracking_uri is required when experiment.tracking.enabled=true."
            )
        if not self.experiment_name:
            raise ValueError(
                "experiment.tracking.experiment_name is required when experiment.tracking.enabled=true."
            )

        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    notes: str | None = None
    candidate: ExperimentCandidateConfig
    submit: ExperimentSubmitConfig = Field(default_factory=ExperimentSubmitConfig)
    tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)


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
            validate_model_preprocessing_compatibility(
                task_type=self.task_type,
                model_id=resolved_model_id,
                categorical_preprocessor_id=self.categorical_preprocessor,
            )
            optimization = self.experiment.candidate.optimization
            if optimization.enabled:
                if optimization.n_trials is None and optimization.timeout_seconds is None:
                    raise ValueError(
                        "At least one experiment.candidate.optimization stopping condition is required. "
                        "Set experiment.candidate.optimization.n_trials or "
                        "experiment.candidate.optimization.timeout_seconds."
                    )
                if not is_model_tunable(self.task_type, resolved_model_id):
                    supported_tunable_model_families = get_tunable_model_ids(self.task_type)
                    raise ValueError(
                        f"Configured model_family '{self.model_family}' does not support optimization for task_type "
                        f"'{self.task_type}'. Supported tunable model families: {supported_tunable_model_families}"
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
    def numeric_preprocessor(self) -> str:
        if not self.is_model_candidate or self.experiment.candidate.numeric_preprocessor is None:
            raise ValueError("numeric_preprocessor is only available for model candidates.")
        return self.experiment.candidate.numeric_preprocessor

    @property
    def categorical_preprocessor(self) -> str:
        if not self.is_model_candidate or self.experiment.candidate.categorical_preprocessor is None:
            raise ValueError("categorical_preprocessor is only available for model candidates.")
        return self.experiment.candidate.categorical_preprocessor

    @property
    def preprocessing_scheme_id(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("preprocessing_scheme_id is only available for model candidates.")
        return build_preprocessing_scheme_id(
            numeric_preprocessor_id=self.numeric_preprocessor,
            categorical_preprocessor_id=self.categorical_preprocessor,
        )

    @property
    def resolved_model_id(self) -> str:
        if not self.is_model_candidate:
            raise ValueError("resolved_model_id is only available for model candidates.")
        return resolve_candidate_model_id(
            task_type=self.task_type,
            model_family=self.model_family,
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

    @property
    def tracking_enabled(self) -> bool:
        return self.experiment.tracking.enabled

    @property
    def tracking_uri(self) -> str:
        if not self.tracking_enabled or self.experiment.tracking.tracking_uri is None:
            raise ValueError("tracking_uri is only available when experiment.tracking.enabled=true.")
        return self.experiment.tracking.tracking_uri

    @property
    def tracking_experiment_name(self) -> str:
        if not self.tracking_enabled or self.experiment.tracking.experiment_name is None:
            raise ValueError(
                "tracking_experiment_name is only available when experiment.tracking.enabled=true."
            )
        return self.experiment.tracking.experiment_name


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
