"""Regression tests for scripts/nifti_to_dicom.py.

Verifies the RAS+ -> LPS geometry conversion against an independent computation
(done fresh in this file, not by importing the module's own _ras_to_lps helper) and
confirms pixel content lands on the geometrically correct slice/position -- the same
verify-against-an-independent-computation discipline used for the DICOM SEG row/col
fix earlier this session, applied here before this module ever touches real data.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom

from scripts.nifti_to_dicom import build_dicom_series


class BuildDicomSeriesTests(unittest.TestCase):
    def setUp(self):
        # Deliberately anisotropic spacing and a nonzero origin, so a row/col or
        # axis mixup would be numerically detectable (an isotropic, origin-at-zero
        # fixture would hide exactly this class of bug -- the same gap that let the
        # SEG row/col swap slip past its first round of tests).
        self.affine_ras = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, -20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.volume = np.zeros((4, 5, 3), dtype=np.float32)
        self.volume[1, 2, 1] = 500.0  # marker at voxel (i=1, j=2, k=1)

    def test_geometry_matches_independent_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build_dicom_series(self.volume, self.affine_ras, tmp_path, series_description="test")

            for k in range(3):
                ds = pydicom.dcmread(str(tmp_path / f"slice_{k:04d}.dcm"), force=True)
                # world = affine_ras @ [0, 0, k, 1]; LPS = (-x, -y, z). Computed fresh
                # here, not via the module under test.
                world_ras = self.affine_ras @ np.array([0.0, 0.0, float(k), 1.0])
                expected_lps = [-world_ras[0], -world_ras[1], world_ras[2]]
                got = [float(v) for v in ds.ImagePositionPatient]
                self.assertTrue(
                    np.allclose(got, expected_lps, atol=1e-3),
                    f"slice {k}: expected {expected_lps}, got {got}",
                )

    def test_pixel_content_lands_on_correct_slice_and_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            build_dicom_series(self.volume, self.affine_ras, tmp_path, series_description="test")

            brightest_slice = None
            for k in range(3):
                ds = pydicom.dcmread(str(tmp_path / f"slice_{k:04d}.dcm"), force=True)
                pixels = ds.pixel_array
                if pixels.max() > 0:
                    brightest_slice = (k, np.unravel_index(pixels.argmax(), pixels.shape))

            self.assertIsNotNone(brightest_slice, "marker voxel not found in any slice")
            k, (row, col) = brightest_slice
            # Marker was placed at volume[i=1, j=2, k=1]; this module's declared
            # convention is axis 0 (i) -> row, axis 1 (j) -> column, axis 2 (k) -> slice.
            self.assertEqual((k, row, col), (1, 1, 2))

    def test_rejects_non_3d_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_dicom_series(
                    np.zeros((4, 5)), self.affine_ras, Path(tmp), series_description="test"
                )


if __name__ == "__main__":
    unittest.main()
