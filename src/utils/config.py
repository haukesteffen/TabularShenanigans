from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

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

    # Optional URIs for storages (resolved to absolute during load_config)
    db_uri: Optional[str] = None
    mlflow_backend_uri: Optional[str] = None
    mlflow_artifact_uri: Optional[str] = None
    optuna_storage_uri: Optional[str] = None


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


def _resolve_uri_against_root(root: Path, uri: str) -> str:
    """Resolve local file-based URIs (sqlite/file) to absolute URIs under root.

    - sqlite: converts relative forms (e.g., sqlite:///db/data.db) to absolute
      with four slashes (sqlite:////abs/path.db). Leaves :memory: untouched.
    - file: converts relative forms (file:artifacts, file://artifacts) to
      absolute file URIs (file:///abs/path). Absolute file URIs are preserved.
    - Other schemes: returned unchanged.
    """
    try:
        parsed = urlparse(uri)
    except Exception:
        # Not a URI; treat as file path relative to root
        return (root / uri).resolve().as_uri()

    scheme = (parsed.scheme or "").lower()

    # Handle sqlite URIs
    if scheme.startswith("sqlite"):
        # Special in-memory database
        if parsed.path == ":memory:" or uri.endswith(":memory:"):
            return uri

        # Absolute path already (sqlite:////...)
        if uri.startswith("sqlite:////"):
            return uri

        # Relative path (sqlite:///relative or sqlite://relative)
        # Combine netloc+path if netloc present (rare for sqlite)
        relative = (parsed.netloc + parsed.path).lstrip("/")
        abs_path = (root / relative).resolve()
        return f"sqlite:////{abs_path.as_posix()}"

    # Handle file URIs
    if scheme == "file":
        # Fully-qualified absolute file URI
        if uri.startswith("file:///") and (parsed.path or "").startswith("/"):
            return uri
        # Treat anything else as relative to root (file:artifacts or file://artifacts)
        relative = (parsed.netloc + parsed.path).lstrip("/")
        abs_path = (root / relative).resolve()
        return abs_path.as_uri()

    # Non file-based schemes: leave unchanged
    return uri


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
    root = _find_repo_root()
    cfg_dict["root"] = root

    # Resolve storage URIs against root, with sensible defaults if omitted
    # Database (prefer explicit db_uri; otherwise build from `db` file name)
    db_uri_base = cfg_dict.get("db_uri")
    if not isinstance(db_uri_base, str) or not db_uri_base:
        # Fall back to conventional location under db/
        db_file = cfg_dict.get("db") or "data.db"
        db_uri_base = f"sqlite:///db/{db_file}"
    cfg_dict["db_uri"] = _resolve_uri_against_root(root, db_uri_base)

    # MLflow backend and artifacts
    mf_backend_base = cfg_dict.get("mlflow_backend_uri") or "sqlite:///db/mlflow.db"
    cfg_dict["mlflow_backend_uri"] = _resolve_uri_against_root(root, mf_backend_base)

    mf_artifacts_base = cfg_dict.get("mlflow_artifact_uri") or "file://artifacts"
    cfg_dict["mlflow_artifact_uri"] = _resolve_uri_against_root(root, mf_artifacts_base)

    # Optuna storage
    optuna_base = cfg_dict.get("optuna_storage_uri") or "sqlite:///db/optuna.db"
    cfg_dict["optuna_storage_uri"] = _resolve_uri_against_root(root, optuna_base)

    # Instantiate Config class
    model = Config(**cfg_dict)
    print(f"Config loaded from {path}")
    return model
