import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tabular_shenanigans.bootstrap import main


if __name__ == "__main__":
    main(sys.argv[1:])
