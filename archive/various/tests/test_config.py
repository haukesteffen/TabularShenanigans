import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
from pydantic import ValidationError

# Ensure `src/` is importable for `utils.*`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config  # noqa: E402


def _expected_repo_root() -> Path:
    """Replicate repo-root detection for assertions in tests."""
    cur = Path(__file__).resolve().parent
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists():
            return parent.resolve()
    # Fallback: repository root by layout assumption
    return REPO_ROOT.resolve()


def _load_with(monkeypatch: pytest.MonkeyPatch, mapping: dict) -> "Config":  # noqa: F821
    """Load config by mocking yaml.safe_load to return `mapping`.

    Uses the repository's existing `config.yaml` path to satisfy file opening,
    but the contents are ignored due to the mock.
    """
    import utils.config as cfg_mod

    monkeypatch.setattr(cfg_mod.yaml, "safe_load", lambda stream: mapping)
    # Reuse existing config path to avoid creating files
    cfg_path = REPO_ROOT / "config.yaml"
    return load_config(cfg_path)


def test_root_auto_detection_and_yaml_root_ignored(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "root": "/definitely/not/the/repo/root",
        },
    )
    assert cfg.root == _expected_repo_root()


def test_sqlite_db_uri_default_from_db(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
        },
    )
    # Expect sqlite absolute form with four slashes
    assert cfg.db_uri.startswith("sqlite:////")
    parsed = urlparse(cfg.db_uri)
    # Path of sqlite URI should point to <repo>/db/data.db
    expected = (_expected_repo_root() / "db" / "data.db").resolve()
    assert Path(parsed.path).resolve() == expected


def test_sqlite_relative_uri_resolves_absolute(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "db_uri": "sqlite:///db/alt.db",
        },
    )
    assert cfg.db_uri.startswith("sqlite:////")
    parsed = urlparse(cfg.db_uri)
    expected = (_expected_repo_root() / "db" / "alt.db").resolve()
    assert Path(parsed.path).resolve() == expected


def test_sqlite_absolute_uri_preserved(monkeypatch: pytest.MonkeyPatch):
    abs_path = (_expected_repo_root() / "db" / "abs.db").resolve()
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "db_uri": f"sqlite:////{abs_path.as_posix()}",
        },
    )
    assert cfg.db_uri == f"sqlite:////{abs_path.as_posix()}"


def test_sqlite_memory_uri_preserved(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "db_uri": "sqlite:///:memory:",
        },
    )
    assert cfg.db_uri == "sqlite:///:memory:"


def test_file_artifact_uri_resolves_absolute(monkeypatch: pytest.MonkeyPatch):
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "mlflow_artifact_uri": "file://artifacts",
        },
    )
    parsed = urlparse(cfg.mlflow_artifact_uri)
    assert parsed.scheme == "file"
    expected = (_expected_repo_root() / "artifacts").resolve()
    assert Path(parsed.path).resolve() == expected


def test_non_file_uris_passthrough(monkeypatch: pytest.MonkeyPatch):
    pg_uri = "postgresql://user@localhost:5432/optuna"
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
            "optuna_storage_uri": pg_uri,
        },
    )
    assert cfg.optuna_storage_uri == pg_uri


def test_extra_keys_forbidden(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(Exception):
        _ = _load_with(
            monkeypatch,
            {
                "competition_name": "playground-series-s5e8",
                "db": "data.db",
                "id_column": "id",
                "target_column": "y",
                "unexpected": "value",
            },
        )


def test_strict_types_and_immutable(monkeypatch: pytest.MonkeyPatch):
    # Wrong type for id_column should fail (strict types)
    with pytest.raises(Exception):
        _ = _load_with(
            monkeypatch,
            {
                "competition_name": "playground-series-s5e8",
                "db": "data.db",
                "id_column": 123,  # wrong type
                "target_column": "y",
            },
        )

    # Correct types load fine; then mutation should fail (frozen)
    cfg = _load_with(
        monkeypatch,
        {
            "competition_name": "playground-series-s5e8",
            "db": "data.db",
            "id_column": "id",
            "target_column": "y",
        },
    )
    with pytest.raises(ValidationError):
        # Attempt mutation; should raise due to frozen model
        cfg.db = "other.db"
