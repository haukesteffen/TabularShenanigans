from __future__ import annotations

import random
import unittest

from tabular_shenanigans.models.tuning import sample_params


class TuningTests(unittest.TestCase):
    def test_sample_params_supports_multiple_spec_types(self) -> None:
        rng = random.Random(42)
        params = sample_params(
            {
                "a": [1, 2, 3],
                "b": {"type": "int", "low": 1, "high": 5, "step": 2},
                "c": {"type": "float", "low": 0.1, "high": 0.2},
                "d": {"type": "categorical", "choices": ["x", "y"]},
                "e": 7,
            },
            rng,
        )
        self.assertIn(params["a"], {1, 2, 3})
        self.assertIn(params["b"], {1, 3, 5})
        self.assertGreaterEqual(params["c"], 0.1)
        self.assertLessEqual(params["c"], 0.2)
        self.assertIn(params["d"], {"x", "y"})
        self.assertEqual(params["e"], 7)


if __name__ == "__main__":
    unittest.main()
