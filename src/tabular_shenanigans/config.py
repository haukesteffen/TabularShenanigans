from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric
from tabular_shenanigans.models import (
    get_default_model_id,
    get_supported_model_ids,
    is_model_id_valid_for_task,
    resolve_model_id,
)


class ConfigError(ValueError):
    pass


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_slug: str = Field(min_length=1)
    task_type: Literal["regression", "binary"]
    primary_metric: str
    model_id: str | None = None
    model_ids: list[str] | None = None
    positive_label: str | int | bool | None = None
    id_column: str | None = None
    label_column: str | None = None
    force_categorical: list[str] = Field(default_factory=list)
    force_numeric: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    low_cardinality_int_threshold: int | None = Field(default=None, ge=1)
    cv_n_splits: int = Field(default=7, ge=2)
    cv_shuffle: bool = True
    cv_random_state: int = 42
    submit_enabled: bool = False
    submit_message_prefix: str | None = None

    @model_validator(mode="after")
    def validate_task_and_metric(self) -> "AppConfig":
        normalized_primary_metric = normalize_primary_metric(self.primary_metric)
        if normalized_primary_metric is None:
            raise ValueError(
                "Configured primary_metric is not supported. "
                f"Supported values: {sorted(set(SUPPORTED_PRIMARY_METRICS.values()))}"
            )
        if not is_metric_valid_for_task(self.task_type, normalized_primary_metric):
            raise ValueError(
                f"Configured primary_metric '{normalized_primary_metric}' is not valid for task_type '{self.task_type}'."
            )
        self.primary_metric = normalized_primary_metric

        if self.model_id is not None and self.model_ids is not None:
            raise ValueError("model_id and model_ids are mutually exclusive. Use only one of them.")

        if self.model_ids is not None:
            if not self.model_ids:
                raise ValueError("model_ids must contain at least one model_id.")
            resolved_model_ids = self.model_ids
        elif self.model_id is not None:
            resolved_model_ids = [self.model_id]
        else:
            resolved_model_ids = [get_default_model_id(self.task_type)]

        supported_model_ids = get_supported_model_ids(self.task_type, include_aliases=True)
        invalid_model_ids = [
            model_id for model_id in resolved_model_ids if not is_model_id_valid_for_task(self.task_type, model_id)
        ]
        if invalid_model_ids:
            raise ValueError(
                f"Configured model_ids {invalid_model_ids} are not valid for task_type '{self.task_type}'. "
                f"Supported model_ids: {supported_model_ids}"
            )

        canonical_model_ids = [resolve_model_id(self.task_type, model_id) for model_id in resolved_model_ids]
        if len(set(canonical_model_ids)) != len(canonical_model_ids):
            raise ValueError(f"model_ids must not contain duplicates after alias resolution: {canonical_model_ids}")

        self.model_ids = canonical_model_ids
        self.model_id = canonical_model_ids[0] if len(canonical_model_ids) == 1 else None

        if self.task_type != "binary" and self.positive_label is not None:
            raise ValueError("positive_label is only supported for binary task_type.")
        return self


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
