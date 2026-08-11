"""Offline checks for the lung-nodule detector's pure logic: turning raw detection dicts
into `Finding`s. No MONAI network, bundle download, DICOM conversion, or GPU is touched
here -- mirrors tests/test_mri_brats.py's approach of unit-testing the seams around the
expensive model call, not the model call itself.
"""

from __future__ import annotations

import math
import unittest

from experts.ct_lung_nodule import (
    LUNG_NODULE_REPORTED_MAP,
    LUNG_NODULE_REPORTED_MAR,
    LUNG_NODULE_SCORE_THRESHOLD,
    findings_from_detections,
)


def _detection(center_lps_mm=(1.0, 2.0, 3.0), size_whd_mm=(10.0, 8.0, 6.0), score=0.5) -> dict:
    return {
        "center_lps_mm": center_lps_mm,
        "size_whd_mm": size_whd_mm,
        "score": score,
    }


class FindingsFromDetectionsTests(unittest.TestCase):
    def test_single_detection_above_threshold_becomes_one_finding(self) -> None:
        findings = findings_from_detections([_detection(score=0.6)])

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.label, "pulmonary nodule")
        self.assertTrue(finding.present)
        self.assertEqual(finding.canonical_label, "nodule")
        self.assertEqual(finding.source, "ct-lung-nodule")

    def test_score_maps_to_probability_not_confidence(self) -> None:
        # The detector's own per-box score is a model output, analogous to a
        # classification score -- it belongs in `probability`; `confidence` is left
        # None because no separate calibrated reliability estimate exists (see
        # ct_totalsegmentator.py / mri_brats.py, which leave it None for the same reason).
        findings = findings_from_detections([_detection(score=0.73)])

        self.assertAlmostEqual(findings[0].probability, 0.73)
        self.assertIsNone(findings[0].confidence)

    def test_default_min_score_matches_the_validated_operating_point(self) -> None:
        self.assertEqual(LUNG_NODULE_SCORE_THRESHOLD, 0.3)

        below = findings_from_detections([_detection(score=0.29)])
        at = findings_from_detections([_detection(score=0.3)])

        self.assertEqual(below, [])
        self.assertEqual(len(at), 1)

    def test_custom_min_score_overrides_default(self) -> None:
        findings = findings_from_detections([_detection(score=0.4)], min_score=0.5)
        self.assertEqual(findings, [])

        findings = findings_from_detections([_detection(score=0.6)], min_score=0.5)
        self.assertEqual(len(findings), 1)

    def test_size_mm_is_the_max_box_edge(self) -> None:
        findings = findings_from_detections(
            [_detection(size_whd_mm=(10.0, 25.0, 6.0), score=0.9)]
        )
        self.assertAlmostEqual(findings[0].size_mm, 25.0)

    def test_volume_ml_uses_ellipsoid_not_bounding_box(self) -> None:
        # A 10x10x10 mm box: bounding-box volume would be 1000 mm^3 = 1.0 mL; the
        # ellipsoid inscribed in that box (radius 5mm on each axis) is (4/3)*pi*5^3
        # mm^3 =~ 523.6 mm^3 =~ 0.5236 mL -- about half, as documented in the module.
        findings = findings_from_detections(
            [_detection(size_whd_mm=(10.0, 10.0, 10.0), score=0.9)]
        )
        expected_ml = (4.0 / 3.0) * math.pi * 5.0 * 5.0 * 5.0 / 1000.0
        self.assertAlmostEqual(findings[0].volume_ml, expected_ml, places=6)
        self.assertLess(findings[0].volume_ml, 1.0)  # strictly less than the box volume

    def test_center_and_size_carried_into_extra(self) -> None:
        findings = findings_from_detections(
            [_detection(center_lps_mm=(12.5, -3.0, 40.0), size_whd_mm=(4.0, 5.0, 6.0), score=0.9)]
        )
        extra = findings[0].extra
        self.assertEqual(extra["center_lps_mm"], (12.5, -3.0, 40.0))
        self.assertEqual(extra["size_whd_mm"], (4.0, 5.0, 6.0))

    def test_carries_reported_map_and_mar_on_every_finding(self) -> None:
        findings = findings_from_detections(
            [_detection(score=0.9), _detection(score=0.8)]
        )
        self.assertTrue(findings)
        for finding in findings:
            self.assertEqual(finding.extra["reported_map"], LUNG_NODULE_REPORTED_MAP)
            self.assertEqual(finding.extra["reported_mar"], LUNG_NODULE_REPORTED_MAR)

    def test_sorted_by_descending_score(self) -> None:
        findings = findings_from_detections(
            [_detection(score=0.4), _detection(score=0.9), _detection(score=0.6)]
        )
        scores = [f.probability for f in findings]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_detections_returns_empty_list(self) -> None:
        self.assertEqual(findings_from_detections([]), [])

    def test_all_detections_below_threshold_returns_empty_list(self) -> None:
        findings = findings_from_detections(
            [_detection(score=0.05), _detection(score=0.1)]
        )
        self.assertEqual(findings, [])


class ExpertModuleContractTests(unittest.TestCase):
    """The module must stay importable without torch/monai/SimpleITK installed -- those
    are lazy imports, only touched inside predict()/_load_detector()/_run_detector()."""

    def test_expert_class_exposes_the_expertmodel_contract_without_heavy_imports(self) -> None:
        from experts.ct_lung_nodule import LungNoduleDetectorExpert

        expert = LungNoduleDetectorExpert(bundle_root="/nonexistent/bundle/root")

        self.assertEqual(expert.name, "ct_lung_nodule")
        from core.enums import BodyPart, Modality

        self.assertEqual(expert.modality, Modality.CT)
        self.assertEqual(expert.body_part, BodyPart.CHEST)
        self.assertTrue(expert.version)
        self.assertIn("pulmonary nodule", expert.class_names)


if __name__ == "__main__":
    unittest.main()
