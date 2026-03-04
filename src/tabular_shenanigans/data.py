from pathlib import Path
import subprocess
import zipfile

import pandas as pd


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


def find_competition_zip(competition_slug: str) -> Path:
    data_dir = Path("data") / competition_slug
    zip_files = sorted(data_dir.glob("*.zip"))
    if not zip_files:
        raise ValueError(f"No competition zip found in {data_dir}")
    return zip_files[0]


def read_csv_from_zip(zip_path: Path, member_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise ValueError(f"Missing {member_name} in {zip_path}")
        with archive.open(member_name) as f:
            return pd.read_csv(f)


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    candidate_columns = [col for col in train_df.columns if col not in test_df.columns]
    if len(candidate_columns) != 1:
        raise ValueError(f"Could not infer a single target column. Candidates: {candidate_columns}")
    return candidate_columns[0]
