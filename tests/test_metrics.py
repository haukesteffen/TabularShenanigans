from __future__ import annotations

import unittest

import numpy as np

from tabular_shenanigans.core.metrics import (
    compute_metrics_bundle,
    validate_metric_compatibility,
)


class MetricsTests(unittest.TestCase):
    def test_validate_metric_compatibility_rejects_regression_accuracy(self) -> None:
        with self.assertRaises(ValueError):
            validate_metric_compatibility("accuracy", "regression", None)

    def test_compute_metrics_bundle_binary_with_scores(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_scores = np.array([0.1, 0.9, 0.8, 0.3])
        y_labels = (y_scores >= 0.5).astype(int)

        metrics = compute_metrics_bundle(
            task_type="classification",
            n_classes=2,
            y_true=y_true,
            y_pred_labels=y_labels,
            y_pred_scores=y_scores,
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("logloss", metrics)


if __name__ == "__main__":
    unittest.main()
