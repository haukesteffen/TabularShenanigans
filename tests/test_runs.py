from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tabular_shenanigans.core.runs import (
    get_latest_run_id,
    list_run_ids,
    promote_run_to_latest,
    resolve_existing_run,
)


class RunsTests(unittest.TestCase):
    def test_resolve_existing_run_uses_latest_run_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts_dir = Path(tmp) / "artifacts"
            competition = "demo"
            run_id = "20260101_000000_model"
            run_path = artifacts_dir / competition / "runs" / run_id
            run_path.mkdir(parents=True, exist_ok=True)
            (run_path / "marker.txt").write_text("ok", encoding="utf-8")

            promote_run_to_latest(
                artifacts_dir=artifacts_dir,
                competition=competition,
                run_id=run_id,
                run_path=run_path,
            )

            latest_id = get_latest_run_id(artifacts_dir, competition)
            self.assertEqual(latest_id, run_id)

            resolved_id, resolved_path = resolve_existing_run(
                artifacts_dir=artifacts_dir,
                competition=competition,
                run_id=None,
            )
            self.assertEqual(resolved_id, run_id)
            self.assertEqual(resolved_path, run_path)
            self.assertEqual(list_run_ids(artifacts_dir, competition), [run_id])


if __name__ == "__main__":
    unittest.main()
