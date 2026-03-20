from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class BootstrapCandidateRuntimeConfig:
    candidate_type: str | None = None
    model_family: str | None = None
    representation_id: str | None = None


@dataclass(frozen=True)
class BootstrapRuntimeConfig:
    compute_target: str = "auto"
    gpu_backend: str = "auto"
    task_type: str | None = None
    experiment_candidates: tuple[BootstrapCandidateRuntimeConfig, ...] = ()
    screening_candidates: tuple[BootstrapCandidateRuntimeConfig, ...] = ()


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


def _coerce_bootstrap_candidate(candidate: object) -> BootstrapCandidateRuntimeConfig:
    if candidate is None:
        return BootstrapCandidateRuntimeConfig()
    if not isinstance(candidate, dict):
        raise ValueError("experiment.candidates items must be mappings when provided.")
    return BootstrapCandidateRuntimeConfig(
        candidate_type=candidate.get("candidate_type"),
        model_family=candidate.get("model_family"),
        representation_id=candidate.get("representation_id"),
    )


def _coerce_bootstrap_candidate_list(candidates: object, field_name: str) -> list[BootstrapCandidateRuntimeConfig]:
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        raise ValueError(f"{field_name} must be a list when provided.")
    return [_coerce_bootstrap_candidate(candidate_item) for candidate_item in candidates]


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
    candidates = experiment.get("candidates")
    if candidate is not None:
        raise ValueError(
            "Legacy experiment.candidate is no longer supported. "
            "Use experiment.candidates."
        )
    competition = raw_data.get("competition")
    if competition is not None and not isinstance(competition, dict):
        raise ValueError("competition must be a mapping when provided.")
    screening = raw_data.get("screening")
    if screening is not None and not isinstance(screening, dict):
        raise ValueError("screening must be a mapping when provided.")

    experiment_candidate_list: list[BootstrapCandidateRuntimeConfig] = []
    if candidates is not None:
        experiment_candidate_list = _coerce_bootstrap_candidate_list(candidates, "experiment.candidates")

    screening_candidate_list: list[BootstrapCandidateRuntimeConfig] = []
    if screening is not None:
        screening_repr_ids = screening.get("representation_ids")
        screening_model_families = screening.get("model_families")
        if isinstance(screening_repr_ids, list) and isinstance(screening_model_families, list):
            for repr_id in screening_repr_ids:
                for model_family in screening_model_families:
                    screening_candidate_list.append(
                        BootstrapCandidateRuntimeConfig(
                            candidate_type="model",
                            model_family=model_family if isinstance(model_family, str) else None,
                            representation_id=repr_id if isinstance(repr_id, str) else None,
                        )
                    )

    return BootstrapRuntimeConfig(
        compute_target=_validate_compute_target(None if runtime is None else runtime.get("compute_target")),
        gpu_backend=_validate_gpu_backend(None if runtime is None else runtime.get("gpu_backend")),
        task_type=None if competition is None else competition.get("task_type"),
        experiment_candidates=tuple(experiment_candidate_list),
        screening_candidates=tuple(screening_candidate_list),
    )
