from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tabular_shenanigans.cli.main import app


class CleanCliTests(unittest.TestCase):
    def test_clean_dry_run_does_not_delete(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts" / "demo").mkdir(parents=True)
            (root / "data" / "demo" / "raw").mkdir(parents=True)
            (root / "data" / "demo" / "processed").mkdir(parents=True)

            with patch("tabular_shenanigans.cli.main.PROJECT_ROOT", root):
                result = runner.invoke(
                    app,
                    [
                        "clean",
                        "--all-competitions",
                        "--scope",
                        "all",
                        "--dry-run",
                    ],
                )

            self.assertEqual(result.exit_code, 0)
            self.assertTrue((root / "artifacts" / "demo").exists())
            self.assertIn("Dry-run only", result.stdout)


if __name__ == "__main__":
    unittest.main()
