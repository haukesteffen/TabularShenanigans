from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BootstrapRuntimeConfig:
    compute_target: str = "auto"
    gpu_backend: str = "auto"


def _validate_compute_target(value: object) -> str:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError("experiment.runtime.compute_target must be a string when provided.")

    normalized_value = value.strip().lower()
    if normalized_value in {"auto", "cpu", "gpu"}:
        return normalized_value

    raise ValueError(
        "experiment.runtime.compute_target must be one of ['auto', 'cpu', 'gpu']."
    )


def _validate_gpu_backend(value: object) -> str:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise ValueError("experiment.runtime.gpu_backend must be a string when provided.")

    normalized_value = value.strip().lower()
    if normalized_value in {"auto", "patch", "native"}:
        return normalized_value

    raise ValueError(
        "experiment.runtime.gpu_backend must be one of ['auto', 'patch', 'native']."
    )


def load_bootstrap_runtime_config(path: str | Path = "config.yaml") -> BootstrapRuntimeConfig:
    config_path = Path(path)

    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Config file not found: {config_path}. "
            "Create repository-root config.yaml from config.binary.example.yaml "
            "or config.regression.example.yaml."
        ) from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {exc}") from exc

    if raw_data is None:
        return BootstrapRuntimeConfig()
    if not isinstance(raw_data, dict):
        raise ValueError("Config must be a top-level mapping.")

    experiment = raw_data.get("experiment")
    if experiment is None:
        return BootstrapRuntimeConfig()
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a mapping when provided.")

    runtime = experiment.get("runtime")
    if runtime is None:
        return BootstrapRuntimeConfig()
    if not isinstance(runtime, dict):
        raise ValueError("experiment.runtime must be a mapping when provided.")

    return BootstrapRuntimeConfig(
        compute_target=_validate_compute_target(runtime.get("compute_target")),
        gpu_backend=_validate_gpu_backend(runtime.get("gpu_backend")),
    )
