from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def _resolve_kaggle_bin() -> str | None:
    # Prefer PATH, but also support venv-local binary when PATH isn't activated.
    found = shutil.which("kaggle")
    if found is not None:
        return found
    candidate = Path(sys.executable).parent / "kaggle"
    if candidate.exists():
        return str(candidate)
    return None


def fetch_competition_data(competition_slug: str, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_bin = _resolve_kaggle_bin()
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not in PATH. Install it with `uv pip install kaggle`."
        )

    zip_path = output_dir / f"{competition_slug}.zip"
    cmd = [
        kaggle_bin,
        "competitions",
        "download",
        "-c",
        competition_slug,
        "-p",
        str(output_dir),
    ]
    if force:
        cmd.append("--force")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = stderr if stderr != "Unknown error" else (stdout or stderr)
        raise RuntimeError(
            "Kaggle download failed. Verify ~/.kaggle/kaggle.json and competition slug. "
            f"Details: {details}"
        ) from exc

    if zip_path.exists():
        with ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)

    for extra_zip in output_dir.glob("*.zip"):
        if extra_zip.name != zip_path.name:
            with ZipFile(extra_zip, "r") as archive:
                archive.extractall(output_dir)

    return output_dir
