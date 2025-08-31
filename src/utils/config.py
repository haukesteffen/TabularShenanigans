from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """General parameters.

    Attributes:
        competition_name (str): The name of the Kaggle competition.
        db (str): The database file.
        id_column (str): The column name for the unique identifier in the dataset.
        target_column (str): The column name for the target variable in the dataset.
        root (Path): Absolute path to the project root directory.
            This is always auto-detected from this module upward based on
            repository markers (e.g., `pyproject.toml`, `.git`).
    """

    # Pydantic v2 model configuration: strict types, no extra keys, immutable
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    competition_name: str
    db: str
    id_column: str
    target_column: str
    root: Path


def _convert_paths(data: dict) -> dict:
    """Recursively convert values likely representing filesystem locations to Path.

    Any key containing 'path' or 'dir' is interpreted as a filesystem
    location and converted to `pathlib.Path`.
    """
    for key, value in list(data.items()):
        if isinstance(value, dict):
            data[key] = _convert_paths(value)
        elif isinstance(value, str):
            key_l = key.lower()
            if ("path" in key_l) or ("dir" in key_l):
                data[key] = Path(value)
    return data


def _find_repo_root(start: Optional[Path] = None) -> Path:
    """Best-effort detection of the repository root directory.

    Looks upward from `start` (or this file) for common project markers.
    Falls back to the current working directory if nothing is found.
    """
    markers = {"pyproject.toml", ".git", "uv.lock", "config.yaml"}
    here = start or Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    # Fallback
    return Path.cwd()


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file and resolve project root.

    - Always auto-detects the repository root based on this module's location.
    - Does not read or honor any root-related value from YAML.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Convert likely path-like keys to Path objects first
    cfg_dict = _convert_paths(raw)

    # Auto-detect based on repository markers, searching upward from this module
    cfg_dict["root"] = _find_repo_root()

    # Instantiate Config class
    model = Config(**cfg_dict)
    print(f"Config loaded from {path}")
    return model
