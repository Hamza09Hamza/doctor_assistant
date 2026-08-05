"""Regression tests for scripts/build_lung_nodule_seg_bundle.py.

Covers the two pieces that are new, untested-until-now code: rasterizing a sphere into
a real CT's voxel grid via SimpleITK (verified against the known analytic sphere-volume
formula, not just "it ran without crashing"), and building a multi-object SEG that
correctly references its source series.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import highdicom  # noqa: F401
    import SimpleITK as sitk

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

if DEPS_AVAILABLE:
    import pydicom

    from scripts.build_lung_nodule_seg_bundle import _build_seg, _rasterize_ellipsoid
    from scripts.nifti_to_dicom import build_dicom_series


@unittest.skipUnless(DEPS_AVAILABLE, "highdicom/SimpleITK not installed in this environment")
class RasterizeEllipsoidTests(unittest.TestCase):
    def test_sphere_voxel_count_matches_analytic_volume(self):
        size = (30, 30, 30)
        img = sitk.Image(size, sitk.sitkInt16)
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        radius = 5.0
        mask = _rasterize_ellipsoid(img, (15.0, 15.0, 15.0), (radius, radius, radius))

        self.assertEqual(mask.shape, (30, 30, 30))
        expected_volume = (4.0 / 3.0) * np.pi * radius**3
        relative_error = abs(mask.sum() - expected_volume) / expected_volume
        self.assertLess(relative_error, 0.25, "rasterized sphere volume too far from analytic value")

    def test_center_included_far_point_excluded(self):
        size = (30, 30, 30)
        img = sitk.Image(size, sitk.sitkInt16)
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        mask = _rasterize_ellipsoid(img, (15.0, 15.0, 15.0), (5.0, 5.0, 5.0))
        self.assertTrue(mask[15, 15, 15])  # (z, y, x)
        self.assertFalse(mask[0, 0, 0])


@unittest.skipUnless(DEPS_AVAILABLE, "highdicom/SimpleITK not installed in this environment")
class BuildSegTests(unittest.TestCase):
    def test_seg_references_real_ct_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            volume = np.random.rand(20, 20, 20).astype(np.float32) * 100
            ct_dir = tmp_path / "ct"
            build_dicom_series(
                volume, np.eye(4), ct_dir, series_description="synthetic CT", modality="CT"
            )

            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
            files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0], False, True)
            reader.SetFileNames(files)
            image = reader.Execute()
            source_datasets = [pydicom.dcmread(f) for f in files]

            gt_mask = _rasterize_ellipsoid(image, (-10.0, -10.0, 10.0), (2.5, 2.5, 2.5))
            seg_path = tmp_path / "seg.dcm"
            _build_seg([gt_mask], ["ground truth"], source_datasets, "test", seg_path)

            seg = pydicom.dcmread(str(seg_path))
            self.assertEqual(str(seg.Modality), "SEG")
            self.assertEqual(str(seg.StudyInstanceUID), source_datasets[0].StudyInstanceUID)
            referenced = {
                str(item.SeriesInstanceUID) for item in seg.ReferencedSeriesSequence
            }
            self.assertIn(source_datasets[0].SeriesInstanceUID, referenced)
            self.assertEqual(len(seg.SegmentSequence), 1)


if __name__ == "__main__":
    unittest.main()
