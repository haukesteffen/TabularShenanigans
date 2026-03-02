from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tabular_shenanigans.submission.validate import validate_submission_file


class SubmissionValidationTests(unittest.TestCase):
    def test_validate_submission_rejects_invalid_label_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids_path = root / "ids.csv"
            sub_path = root / "submission.csv"

            pd.DataFrame({"PassengerId": [1, 2, 3]}).to_csv(ids_path, index=False)
            pd.DataFrame({"PassengerId": [1, 2, 3], "Survived": [0, 2, 1]}).to_csv(
                sub_path, index=False
            )

            with self.assertRaises(ValueError):
                validate_submission_file(
                    submission_path=sub_path,
                    ids_path=ids_path,
                    id_col="PassengerId",
                    target_col="Survived",
                    prediction_type="label",
                    positive_label=1,
                    negative_label=0,
                )


if __name__ == "__main__":
    unittest.main()
