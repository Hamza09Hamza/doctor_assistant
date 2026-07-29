import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.select_chest_thresholds import select_from_benchmark


class SelectThresholdScriptTests(unittest.TestCase):
    def _artifact(self, directory: Path, purpose: str = "development_validation"):
        prediction_path = directory / "valid.predictions.npz"
        np.savez_compressed(
            prediction_path,
            probabilities=np.asarray(
                [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
            ),
            labels=np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]),
            class_names=np.asarray(["a", "b"]),
        )
        benchmark_path = directory / "valid.json"
        benchmark_path.write_text(
            json.dumps(
                {
                    "purpose": purpose,
                    "dataset": {"split": "valid"},
                    "model": {"name": "example"},
                    "predictions": str(prediction_path),
                }
            )
        )
        return benchmark_path

    def test_mirror_selection_is_diagnostic_and_not_pipeline_loadable(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = select_from_benchmark(
                self._artifact(Path(raw)),
                sensitivity_target=1.0,
                specificity_floor=1.0,
                min_positives=2,
                min_negatives=2,
            )
        self.assertTrue(artifact["diagnostic_selection_complete"])
        self.assertEqual(
            set(artifact["diagnostic_candidate_thresholds"]),
            {"a", "b"},
        )
        self.assertFalse(artifact["threshold_export_eligible"])
        self.assertFalse(artifact["thresholds_complete"])
        self.assertIsNone(artifact["pipeline_thresholds"])

    def test_rejects_held_out_test_artifact(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "validation"):
                select_from_benchmark(
                    self._artifact(Path(raw), purpose="frozen_held_out_evaluation"),
                    sensitivity_target=0.8,
                    specificity_floor=0.5,
                    min_positives=1,
                    min_negatives=1,
                )


if __name__ == "__main__":
    unittest.main()
