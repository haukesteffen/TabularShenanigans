import os
import time
import zipfile
import kaggle
import pandas as pd
from typing import Tuple


def download_competition_data(
        competition_name: str, 
        data_dir: str
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
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if the competition data is already downloaded, else download it
    zip_file_path = os.path.join(data_dir, f"{competition_name}.zip")
    if os.path.exists(zip_file_path):
        print(f"Data for competition '{competition_name}' already exists at '{zip_file_path}'.")
    else:
        print(f"Downloading data for competition '{competition_name}'...")
        kaggle.api.competition_download_files(competition=competition_name, path=data_dir, quiet=True)
    
    # Extract the zip file
    zf = zipfile.ZipFile(zip_file_path)
    test = pd.read_csv(zf.open('test.csv'))
    train = pd.read_csv(zf.open('train.csv'))
    zf.close()

    return train, test
    
    
def upload_submission(
        competition_name: str, 
        data_dir: str,
        submission_file: str
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
        raise FileNotFoundError(f"Submission file '{submission_file}' does not exist in '{data_dir}'.")

    # Upload the submission file
    print(f"Uploading '{submission_file}' to competition '{competition_name}'...")
    kaggle.api.competition_submit(file_name=submission_path, message="Initial submission", competition=competition_name)

    # Check if the submission was successful
    _check_submission_status(competition_name=competition_name)

def _check_submission_status(
        poll_interval_seconds: int = 10,
        max_attempts: int = 20,
        competition_name: str = "playground-series-s5e6"
    ) -> None:
    """
    Polls Kaggle to check the status of the most recent submission.
    
    Args:
        poll_interval_seconds (int): How many seconds to wait between checks.
        max_attempts (int): Maximum number of times to check before timing out.
        competition_name (str): The name of the Kaggle competition.
    """
    print("\nChecking submission status. This may take a few minutes...")
    attempts = 0
    while attempts < max_attempts:
        try:
            submissions = kaggle.api.competition_submissions(competition_name)
            if not submissions:
                print("No submissions found yet. Waiting...")
                time.sleep(poll_interval_seconds)
                attempts += 1
                continue
            latest_submission = submissions[0]
            status = str(latest_submission.status)

            if status == 'SubmissionStatus.COMPLETE':
                print(f"\nSubmission scoring is complete. Score: {latest_submission.public_score}")
                return
            elif status == 'SubmissionStatus.ERROR':
                print(f"Submission resulted in an error: {latest_submission.error_description}")
                return False
            elif status in ['SubmissionStatus.PENDING', 'SubmissionStatus.RUNNING']:
                time.sleep(poll_interval_seconds)
                attempts += 1
            else:
                print(f"Encountered an unknown status: '{status}'. Aborting check.")
                return

        except Exception as e:
            print(f"An error occurred while checking status: {e}. Retrying in {poll_interval_seconds}s...")
            time.sleep(poll_interval_seconds)
            attempts += 1
            
    print("\nTimed out waiting for submission to complete. Please check the Kaggle website manually.")
