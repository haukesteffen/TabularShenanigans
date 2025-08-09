import os
import time
import zipfile
from typing import Tuple

import kaggle
import pandas as pd


def download_competition_data(
    competition_name: str, data_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads the competition data from Kaggle, extracts it to the specified directory, returns a tuple of train and test DataFrames.

    Args:
        competition_name (str): The name of the Kaggle competition.
        data_dir (str): The directory where the data will be saved.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    # Ensure data directory exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Check if the competition data is already downloaded, else download it
    zip_file_path = os.path.join(data_path, f"{competition_name}.zip")
    if os.path.exists(zip_file_path):
        print(
            f"Data for competition '{competition_name}' already exists at '{zip_file_path}'."
        )
    else:
        print(f"Downloading data for competition '{competition_name}'...")
        kaggle.api.competition_download_files(
            competition=competition_name, path=data_path, quiet=True
        )

    # Extract the zip file
    zf = zipfile.ZipFile(zip_file_path)
    with zf.open("train.csv") as f:
        train = pd.read_csv(f).convert_dtypes()
    with zf.open("test.csv") as f:
        test = pd.read_csv(f).convert_dtypes()
    return train, test


def upload_submission(
    competition_name: str, data_dir: str, submission_file: str
) -> None:
    """
    Uploads the submission file to the specified Kaggle competition. After successfully submitting, it renames the file to avoid re-uploading.

    Args:
        competition_name (str): The name of the Kaggle competition.
        data_dir (str): The directory where the submission file is located.
        submission_file (str): The path to the submission file.

    Returns:
        None
    """
    # Ensure the submission file exists
    submission_path = os.path.join(data_dir, submission_file)
    if not os.path.exists(submission_path):
        raise FileNotFoundError(
            f"Submission file '{submission_file}' does not exist in '{data_dir}'."
        )

    # Upload the submission file
    print(f"Uploading '{submission_file}' to competition '{competition_name}'...")
    kaggle.api.competition_submit(
        file_name=submission_path,
        message="Initial submission",
        competition=competition_name,
    )
