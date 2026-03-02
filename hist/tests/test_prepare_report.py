from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tabular_shenanigans.data.prepare import prepare_competition_data


class PrepareReportTests(unittest.TestCase):
    def test_prepare_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            raw_dir = data_dir / "demo" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "PassengerId": [1, 2],
                    "Feature": [10, 10],
                    "Survived": [0, 1],
                }
            ).to_csv(raw_dir / "train.csv", index=False)
            pd.DataFrame({"PassengerId": [3], "Feature": [None]}).to_csv(
                raw_dir / "test.csv", index=False
            )

            cfg = {
                "schema": {"target": "Survived", "id_column": "PassengerId", "task_type": "classification"}
            }
            prepare_competition_data(runtime_cfg=cfg, competition="demo", data_dir=data_dir)

            report_path = data_dir / "demo" / "processed" / "prepare_report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["target_column"], "Survived")
            self.assertIn("constant_columns_train", report)


if __name__ == "__main__":
    unittest.main()
