from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "run"


def build_run_id(model_family: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_safe_tag(model_family)}"


def competition_root(artifacts_dir: Path, competition: str) -> Path:
    return artifacts_dir / competition


def runs_dir(artifacts_dir: Path, competition: str) -> Path:
    return competition_root(artifacts_dir, competition) / "runs"


def latest_dir(artifacts_dir: Path, competition: str) -> Path:
    return competition_root(artifacts_dir, competition) / "latest"


def latest_run_file(artifacts_dir: Path, competition: str) -> Path:
    return competition_root(artifacts_dir, competition) / "latest_run.txt"


def get_latest_run_id(artifacts_dir: Path, competition: str) -> str | None:
    latest_id_file = latest_run_file(artifacts_dir, competition)
    if not latest_id_file.exists():
        return None
    value = latest_id_file.read_text(encoding="utf-8").strip()
    return value or None


def list_run_ids(artifacts_dir: Path, competition: str) -> list[str]:
    root = runs_dir(artifacts_dir, competition)
    if not root.exists():
        return []
    run_ids = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(run_ids, reverse=True)


def prepare_new_run(artifacts_dir: Path, competition: str, model_family: str) -> tuple[str, Path]:
    run_id = build_run_id(model_family)
    run_path = runs_dir(artifacts_dir, competition) / run_id
    run_path.mkdir(parents=True, exist_ok=False)
    return run_id, run_path


def resolve_existing_run(
    artifacts_dir: Path,
    competition: str,
    run_id: str | None,
) -> tuple[str, Path]:
    root = competition_root(artifacts_dir, competition)
    run_root = runs_dir(artifacts_dir, competition)
    latest_path = latest_dir(artifacts_dir, competition)
    latest_id_file = latest_run_file(artifacts_dir, competition)

    if run_id and run_id != "latest":
        candidate = run_root / run_id
        if candidate.exists():
            return run_id, candidate
        raise FileNotFoundError(f"Run '{run_id}' not found at {candidate}")

    if latest_id_file.exists():
        latest_id = latest_id_file.read_text(encoding="utf-8").strip()
        if latest_id:
            candidate = run_root / latest_id
            if candidate.exists():
                return latest_id, candidate

    if latest_path.exists():
        # Backward-compatible fallback for old single-run layout.
        return "latest", latest_path

    raise FileNotFoundError(
        f"No existing run found under {root}. Run training first to create one."
    )


def promote_run_to_latest(
    artifacts_dir: Path,
    competition: str,
    run_id: str,
    run_path: Path,
) -> Path:
    latest_id_file = latest_run_file(artifacts_dir, competition)
    latest_path = latest_dir(artifacts_dir, competition)

    latest_id_file.parent.mkdir(parents=True, exist_ok=True)
    latest_id_file.write_text(run_id, encoding="utf-8")

    if latest_path.exists():
        shutil.rmtree(latest_path)
    shutil.copytree(run_path, latest_path)
    return latest_path
