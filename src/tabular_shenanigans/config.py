from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigError(ValueError):
    pass


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_slug: str = Field(min_length=1)


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
