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

    def test_selects_from_validation_predictions(self):
        with tempfile.TemporaryDirectory() as raw:
            artifact = select_from_benchmark(
                self._artifact(Path(raw)),
                sensitivity_target=1.0,
                specificity_floor=1.0,
                min_positives=2,
                min_negatives=2,
            )
        self.assertTrue(artifact["thresholds_complete"])
        self.assertEqual(set(artifact["pipeline_thresholds"]), {"a", "b"})

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
