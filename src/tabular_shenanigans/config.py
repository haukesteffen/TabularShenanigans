from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabular_shenanigans.data import SUPPORTED_PRIMARY_METRICS, is_metric_valid_for_task, normalize_primary_metric


class ConfigError(ValueError):
    pass


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_slug: str = Field(min_length=1)
    task_type: Literal["regression", "binary"]
    primary_metric: str
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
        if self.task_type != "binary" and self.positive_label is not None:
            raise ValueError("positive_label is only supported for binary task_type.")
        return self


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError("Config must be a top-level mapping.")

    return AppConfig.model_validate(raw_data)
