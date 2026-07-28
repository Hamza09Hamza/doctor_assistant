import unittest

import numpy as np

from evaluation.multilabel_analysis import (
    evaluate_multilabel_classifier,
    evaluate_subgroups,
    rank_classification_errors,
)


class MultilabelAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.names = ["common", "rare"]
        self.labels = np.asarray(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )
        self.probabilities = np.asarray(
            [
                [0.90, 0.10],
                [0.80, 0.20],
                [0.40, 0.85],
                [0.30, 0.05],
                [0.20, 0.40],
                [0.10, 0.45],
            ]
        )

    def test_scorecard_reports_ranking_calibration_and_operating_metrics(self):
        result = evaluate_multilabel_classifier(
            self.probabilities,
            self.labels,
            self.names,
            patient_ids=["a", "a", "b", "c", "d", "e"],
            bootstrap_samples=10,
        )
        self.assertEqual(result["images"], 6)
        self.assertEqual(result["patients"], 5)
        self.assertEqual(result["scoreable_labels"], 2)
        self.assertGreater(result["macro"]["auroc"], 0.5)
        self.assertIn("auprc", result["per_label"]["rare"])
        self.assertIn("brier", result["per_label"]["rare"])
        self.assertIn("ece", result["per_label"]["rare"])
        self.assertEqual(
            result["confidence_intervals"]["method"],
            "patient_clustered_percentile_bootstrap",
        )

    def test_ranked_errors_surface_most_confident_cases(self):
        errors = rank_classification_errors(
            self.probabilities,
            self.labels,
            self.names,
            sample_ids=[f"study-{index}" for index in range(6)],
        )
        self.assertEqual(
            errors["common"]["false_negatives"][0]["sample_id"], "study-4"
        )
        self.assertEqual(
            errors["rare"]["false_negatives"][0]["sample_id"], "study-5"
        )

    def test_subgroups_do_not_reselect_thresholds(self):
        result = evaluate_subgroups(
            self.probabilities,
            self.labels,
            self.names,
            {"view": ["PA", "PA", "PA", "AP", "AP", "AP"]},
            thresholds={"common": 0.75, "rare": 0.8},
            min_images=3,
        )
        self.assertEqual(
            result["view"]["PA"]["per_label"]["common"]["threshold"], 0.75
        )
        self.assertEqual(
            result["view"]["AP"]["per_label"]["rare"]["threshold"], 0.8
        )

    def test_invalid_threshold_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing thresholds"):
            evaluate_multilabel_classifier(
                self.probabilities,
                self.labels,
                self.names,
                thresholds={"common": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
