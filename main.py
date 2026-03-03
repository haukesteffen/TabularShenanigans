import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config
from tabular_shenanigans.data import fetch_competition_data
from tabular_shenanigans.eda import run_eda
from tabular_shenanigans.preprocess import run_preprocessing


def main() -> None:
    config = load_config()
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")
    report_dir = run_eda(config.competition_slug)
    print(f"EDA reports ready: {report_dir}")
    artifact_dir = run_preprocessing(config.competition_slug)
    print(f"Preprocessing artifacts ready: {artifact_dir}")


if __name__ == "__main__":
    main()
