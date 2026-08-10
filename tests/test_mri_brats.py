"""Offline checks for the BraTS expert's pure logic: sequence-name resolution and
mask-to-Finding conversion. No MONAI network, bundle download, or GPU is touched here --
mirrors tests/test_totalsegmentator_demo.py's approach of unit-testing the seams around
the expensive model call, not the model call itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from experts.mri_brats import (
    BUNDLE_MODALITY_ORDER,
    canonical_sequence_name,
    findings_from_tc_wt_et_masks,
    resolve_sequence_paths,
)


class CanonicalSequenceNameTests(unittest.TestCase):
    def test_recognises_common_aliases(self) -> None:
        self.assertEqual(canonical_sequence_name("FLAIR"), "flair")
        self.assertEqual(canonical_sequence_name("t2_flair"), "flair")
        self.assertEqual(canonical_sequence_name("T1"), "t1")
        self.assertEqual(canonical_sequence_name("t1w"), "t1")
        self.assertEqual(canonical_sequence_name("T1c"), "t1c")
        self.assertEqual(canonical_sequence_name("t1ce"), "t1c")
        self.assertEqual(canonical_sequence_name("t1gd"), "t1c")
        self.assertEqual(canonical_sequence_name("T2"), "t2")

    def test_t1gd_is_not_confused_with_plain_t1(self) -> None:
        # The exact bug this project already hit once with MSD data: 't1gd' starts with
        # 't1' but is the contrast-enhanced sequence -- prefix matching would silently
        # destroy the enhancing-tumour channel.
        self.assertEqual(canonical_sequence_name("t1gd"), "t1c")
        self.assertNotEqual(canonical_sequence_name("t1gd"), canonical_sequence_name("t1"))

    def test_unrecognised_sequence_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            canonical_sequence_name("dwi")


class ResolveSequencePathsTests(unittest.TestCase):
    def test_orders_into_bundle_channel_order(self) -> None:
        paths = {
            "flair": "/data/flair.nii.gz",
            "t1": "/data/t1.nii.gz",
            "t1ce": "/data/t1c.nii.gz",
            "t2": "/data/t2.nii.gz",
        }

        ordered = resolve_sequence_paths(paths)

        self.assertEqual(BUNDLE_MODALITY_ORDER, ("t1c", "t1", "t2", "flair"))
        self.assertEqual(
            ordered,
            ["/data/t1c.nii.gz", "/data/t1.nii.gz", "/data/t2.nii.gz", "/data/flair.nii.gz"],
        )

    def test_missing_sequence_raises(self) -> None:
        paths = {"t1c": "a", "t1": "b", "t2": "c"}  # no flair

        with self.assertRaises(ValueError):
            resolve_sequence_paths(paths)


class FindingsFromMasksTests(unittest.TestCase):
    def test_volume_and_size_use_physical_spacing(self) -> None:
        tc = np.zeros((4, 4, 4), dtype=bool)
        tc[1:3, 1:3, 1:3] = True  # 2x2x2 = 8 voxels
        wt = np.zeros((4, 4, 4), dtype=bool)
        wt[0:4, 0:4, 0:4] = True  # 64 voxels, whole volume
        et = np.zeros((4, 4, 4), dtype=bool)  # empty

        findings = findings_from_tc_wt_et_masks(
            [tc, wt, et], spacing=(1.0, 1.0, 1.0), min_volume_ml=0.0
        )

        by_code = {f.canonical_label: f for f in findings}
        self.assertAlmostEqual(by_code["TC"].volume_ml, 8 / 1000.0)
        self.assertAlmostEqual(by_code["TC"].size_mm, 2.0)
        self.assertAlmostEqual(by_code["WT"].volume_ml, 64 / 1000.0)
        self.assertFalse(by_code["ET"].present)
        self.assertIsNone(by_code["ET"].size_mm)

    def test_without_spacing_reports_no_physical_units(self) -> None:
        tc = np.ones((2, 2, 2), dtype=bool)

        findings = findings_from_tc_wt_et_masks([tc, tc, tc], spacing=None)

        for finding in findings:
            self.assertIsNone(finding.volume_ml)
            self.assertIsNone(finding.size_mm)
            self.assertEqual(finding.extra["voxels"], 8)

    def test_carries_contamination_caveat_on_every_finding(self) -> None:
        tc = np.ones((2, 2, 2), dtype=bool)

        findings = findings_from_tc_wt_et_masks(
            [tc, tc, tc], spacing=(1.0, 1.0, 1.0), contamination_status="UNVERIFIED"
        )

        self.assertTrue(findings)
        for finding in findings:
            self.assertEqual(finding.extra["contamination_status"], "UNVERIFIED")
            self.assertIn("reference_dice", finding.extra)

    def test_min_volume_ml_filters_small_regions(self) -> None:
        tiny = np.zeros((10, 10, 10), dtype=bool)
        tiny[0, 0, 0] = True  # 1 voxel
        big = np.ones((10, 10, 10), dtype=bool)

        findings = findings_from_tc_wt_et_masks(
            [tiny, big, big], spacing=(1.0, 1.0, 1.0), min_volume_ml=0.5
        )

        labels = {f.canonical_label for f in findings}
        self.assertNotIn("TC", labels)
        self.assertIn("WT", labels)
        self.assertIn("ET", labels)

    def test_source_is_stamped_mri_brats(self) -> None:
        tc = np.ones((2, 2, 2), dtype=bool)

        findings = findings_from_tc_wt_et_masks([tc, tc, tc], spacing=(1.0, 1.0, 1.0))

        for finding in findings:
            self.assertEqual(finding.source, "mri-brats")


if __name__ == "__main__":
    unittest.main()
