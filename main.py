import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.config import load_config


def main() -> None:
    config = load_config()
    print(f"Config loaded: {config.competition_slug}")


if __name__ == "__main__":
    main()
