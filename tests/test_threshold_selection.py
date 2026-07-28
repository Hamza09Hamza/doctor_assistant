from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from evaluation.thresholds import (
    calibrated_threshold_dict,
    evaluate_frozen_thresholds,
    load_calibrated_thresholds,
    select_per_label_thresholds,
)


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_most_specific_threshold_meeting_sensitivity(self) -> None:
        labels = np.array([[1], [1], [1], [0], [0], [0]])
        probabilities = np.array([[0.9], [0.8], [0.7], [0.6], [0.2], [0.1]])

        selection = select_per_label_thresholds(
            probabilities,
            labels,
            ["Finding"],
            sensitivity_target=2 / 3,
            min_positives=1,
            min_negatives=1,
        )[0]

        self.assertAlmostEqual(selection.threshold, 0.7)
        self.assertEqual(selection.sensitivity, 1.0)
        self.assertEqual(selection.specificity, 1.0)
        self.assertTrue(selection.meets_constraints)

    def test_rare_label_is_explicitly_unsupported(self) -> None:
        labels = np.array([[1], [0], [0], [0]])
        probabilities = np.array([[0.9], [0.2], [0.1], [0.3]])
        selection = select_per_label_thresholds(
            probabilities,
            labels,
            ["Rare"],
            min_positives=2,
        )[0]
        self.assertFalse(selection.supported)
        with self.assertRaisesRegex(ValueError, "Rare"):
            calibrated_threshold_dict([selection])

    def test_frozen_thresholds_are_evaluated_without_reselection(self) -> None:
        labels = np.array([[1], [0], [1], [0]])
        probabilities = np.array([[0.8], [0.7], [0.6], [0.2]])
        result = evaluate_frozen_thresholds(
            probabilities,
            labels,
            ["Finding"],
            {"Finding": 0.65},
        )["Finding"]
        self.assertEqual(result["threshold"], 0.65)
        self.assertEqual(result["sensitivity"], 0.5)
        self.assertEqual(result["specificity"], 0.5)

    def test_incomplete_artifact_cannot_be_loaded_into_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            path.write_text(
                json.dumps(
                    {
                        "thresholds_complete": False,
                        "pipeline_thresholds": None,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_calibrated_thresholds(path)


if __name__ == "__main__":
    unittest.main()
