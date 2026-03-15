import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tabular_shenanigans.lightgbm_cuda_backend import probe_lightgbm_cuda_build


def main() -> int:
    try:
        validation_result = probe_lightgbm_cuda_build()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(validation_result.to_dict(), indent=2, sort_keys=True))
    if validation_result.validated:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
