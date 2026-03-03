import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config
from tabular_shenanigans.data import fetch_competition_data


def main() -> None:
    config = load_config()
    data_dir = fetch_competition_data(config.competition_slug)
    print(f"Data ready: {data_dir}")


if __name__ == "__main__":
    main()
