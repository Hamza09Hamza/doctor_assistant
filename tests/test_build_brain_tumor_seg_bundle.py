"""Regression tests for scripts/build_brain_tumor_seg_bundle.py's SEG construction.

Verifies two things found necessary only by actually running this against highdicom,
not by reading its docs: 1) SegmentDescription requires an AlgorithmIdentificationSequence
whenever algorithm_type is not MANUAL (highdicom raises TypeError without it), and
2) the resulting SEG passes the same checks
scripts/run_totalsegmentator_dicom_seg.py's validate_dicom_seg enforces before this
project's existing Orthanc publish pipeline will accept a bundle.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import highdicom  # noqa: F401

    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False

import pydicom

from scripts.nifti_to_dicom import build_dicom_series

if HIGHDICOM_AVAILABLE:
    from scripts.build_brain_tumor_seg_bundle import _build_seg


@unittest.skipUnless(HIGHDICOM_AVAILABLE, "highdicom not installed in this environment")
class BuildPredictionSegTests(unittest.TestCase):
    def setUp(self):
        self.affine_ras = np.eye(4)
        self.ni, self.nj, self.nk = 8, 8, 6

    def _build_source_series(self, tmp_path: Path):
        volume = np.random.rand(self.ni, self.nj, self.nk).astype(np.float32)
        return build_dicom_series(
            volume, self.affine_ras, tmp_path / "source_dicom", series_description="test"
        )

    def test_empty_segment_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_datasets = self._build_source_series(tmp_path)

            mask = np.zeros((3, self.ni, self.nj, self.nk), dtype=bool)
            mask[0, 3:5, 3:5, 2:4] = True  # TC
            mask[1, 2:6, 2:6, 1:5] = True  # WT
            # ET left empty on purpose.
            mask_frames = np.transpose(mask, (3, 1, 2, 0))

            seg_path = tmp_path / "seg.dcm"
            _build_seg(
                mask_frames, source_datasets, seg_path,
                series_description="test", series_number=10, algorithm_type="AUTOMATIC",
            )

            seg = pydicom.dcmread(str(seg_path), force=True)
            labels = {str(s.SegmentLabel) for s in seg.SegmentSequence}
            self.assertIn("Tumour core (TC)", labels)
            self.assertIn("Whole tumour (WT)", labels)
            self.assertNotIn("Enhancing tumour (ET)", labels)

    def test_seg_references_source_series_correctly(self):
        """Matches exactly what scripts/run_totalsegmentator_dicom_seg.py's
        validate_dicom_seg checks -- this is what has to pass for the existing (unmodified)
        publish pipeline to accept the bundle this script produces."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_datasets = self._build_source_series(tmp_path)

            mask = np.zeros((3, self.ni, self.nj, self.nk), dtype=bool)
            mask[1, 2:6, 2:6, 1:5] = True
            mask_frames = np.transpose(mask, (3, 1, 2, 0))

            seg_path = tmp_path / "seg.dcm"
            _build_seg(
                mask_frames, source_datasets, seg_path,
                series_description="test", series_number=10, algorithm_type="AUTOMATIC",
            )

            seg = pydicom.dcmread(str(seg_path), force=True)
            self.assertEqual(str(seg.SOPClassUID), "1.2.840.10008.5.1.4.1.1.66.4")
            self.assertEqual(str(seg.Modality), "SEG")
            self.assertEqual(str(seg.StudyInstanceUID), source_datasets[0].StudyInstanceUID)
            referenced = {
                str(item.SeriesInstanceUID)
                for item in getattr(seg, "ReferencedSeriesSequence", [])
            }
            self.assertIn(source_datasets[0].SeriesInstanceUID, referenced)

    def test_all_segments_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_datasets = self._build_source_series(tmp_path)
            mask_frames = np.zeros((self.nk, self.ni, self.nj, 3), dtype=bool)
            seg_path = tmp_path / "seg.dcm"
            with self.assertRaises(RuntimeError):
                _build_seg(
                    mask_frames, source_datasets, seg_path,
                    series_description="test", series_number=10, algorithm_type="AUTOMATIC",
                )


if __name__ == "__main__":
    unittest.main()
