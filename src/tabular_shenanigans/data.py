from pathlib import Path
import subprocess


def fetch_competition_data(competition_slug: str) -> Path:
    target_dir = Path("data") / competition_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.glob("*.zip")):
        return target_dir

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            competition_slug,
            "-p",
            str(target_dir),
        ],
        check=True,
    )
    return target_dir
