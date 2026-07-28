from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_chest_xrv.py"
_SPEC = importlib.util.spec_from_file_location("eval_chest_for_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)


class ThresholdMetricTests(unittest.TestCase):
    def test_operational_metrics_surface_normal_false_positives(self) -> None:
        labels = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        probabilities = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.1],
                [0.7, 0.8],
            ]
        )

        metrics = _EVAL.compute_threshold_metrics(
            probabilities, labels, threshold=0.5
        )

        self.assertEqual(metrics["macro_sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["macro_specificity"], 0.5)
        self.assertEqual(metrics["normal_any_false_positive"], 1.0)
        self.assertEqual(metrics["normal_mean_findings"], 1.5)


if __name__ == "__main__":
    unittest.main()
