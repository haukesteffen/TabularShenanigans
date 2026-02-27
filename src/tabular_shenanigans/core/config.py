from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid YAML content in {path}")
    return data


def _merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_competition_config(competition: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "competitions" / f"{competition}.yaml")


def load_profile_config(profile: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "profiles" / f"{profile}.yaml")


def load_runtime_config(competition: str, profile: str) -> dict[str, Any]:
    global_cfg = _read_yaml(CONFIG_DIR / "global.yaml")
    comp_cfg = load_competition_config(competition)
    prof_cfg = load_profile_config(profile)
    return _merge_dicts(_merge_dicts(global_cfg, comp_cfg), prof_cfg)


def list_competitions() -> list[str]:
    comp_dir = CONFIG_DIR / "competitions"
    if not comp_dir.exists():
        return []
    return sorted(path.stem for path in comp_dir.glob("*.yaml"))


def list_profiles() -> list[str]:
    profile_dir = CONFIG_DIR / "profiles"
    if not profile_dir.exists():
        return []
    return sorted(path.stem for path in profile_dir.glob("*.yaml"))


def write_yaml_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)
