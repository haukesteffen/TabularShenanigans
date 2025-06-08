from utils.data_util import download_competition_data, upload_submission

COMPETITION_NAME = "playground-series-s5e6"
DATA_DIR = "./data"
SUBMISSION_FILE = "submission.csv"

def main():
    print("Hello from tabularshenanigans!")
    upload_submission(
        competition_name=COMPETITION_NAME,
        data_dir='.',
        submission_file='sample_submission.csv'
    )
    

if __name__ == "__main__":
    main()
