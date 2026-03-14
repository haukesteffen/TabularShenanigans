from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BootstrapRuntimeConfig:
    compute_target: str = "auto"
    gpu_backend: str = "auto"
    task_type: str | None = None
    candidate_type: str | None = None
    model_family: str | None = None
    numeric_preprocessor: str | None = None
    categorical_preprocessor: str | None = None


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
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("experiment.runtime must be a mapping when provided.")
    candidate = experiment.get("candidate")
    if candidate is not None and not isinstance(candidate, dict):
        raise ValueError("experiment.candidate must be a mapping when provided.")
    competition = raw_data.get("competition")
    if competition is not None and not isinstance(competition, dict):
        raise ValueError("competition must be a mapping when provided.")

    return BootstrapRuntimeConfig(
        compute_target=_validate_compute_target(None if runtime is None else runtime.get("compute_target")),
        gpu_backend=_validate_gpu_backend(None if runtime is None else runtime.get("gpu_backend")),
        task_type=None if competition is None else competition.get("task_type"),
        candidate_type=None if candidate is None else candidate.get("candidate_type"),
        model_family=None if candidate is None else candidate.get("model_family"),
        numeric_preprocessor=None if candidate is None else candidate.get("numeric_preprocessor"),
        categorical_preprocessor=None if candidate is None else candidate.get("categorical_preprocessor"),
    )
