"""Regression tests for scripts/lidc_seg_ground_truth.py.

These exist specifically because the module shipped with a real bug that its first
round of tests did not catch: ImageOrientationPatient's two direction-cosine triplets
were assigned to the wrong local variable names (row_dir/col_dir swapped). That bug was
invisible to isotropic-spacing, identity-orientation test fixtures -- a row/col swap on
a symmetric-in-X/Y setup produces the same centroid either way. It was only caught by a
real visual check (a ground-truth marker plotting outside the patient's body on an
actual Colab run). The anisotropic-spacing fixture below is the regression guard that
gap needed: with row_spacing != col_spacing, a row/col swap changes the result.

Requires highdicom to build synthetic fixture files; not a runtime dependency of the
module under test (highdicom lives in the TotalSegmentator .venv-ct environment per
docs/DEVELOPMENT.md, not the main requirements.txt), so this whole module is skipped,
not failed, when it is unavailable.
"""

from __future__ import annotations

import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

try:
    import highdicom as hd
    from highdicom.seg.content import SegmentDescription
    from highdicom.sr.coding import CodedConcept

    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False

from scripts.lidc_seg_ground_truth import consensus_nodules, extract_reader_nodules

_COMMON_PATIENT_FIELDS = dict(
    PatientBirthDate="",
    PatientSex="",
    AccessionNumber="",
    StudyID="",
    ReferringPhysicianName="",
    StudyDate="20200101",
    StudyTime="000000",
)


def _make_source_dataset(
    z_index: int,
    rows: int,
    cols: int,
    pixel_spacing: list[float],
    slice_spacing_mm: float,
    series_uid: str,
    study_uid: str,
    frame_of_ref_uid: str,
) -> Dataset:
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_of_ref_uid
    ds.Modality = "CT"
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = pixel_spacing
    ds.SliceThickness = slice_spacing_mm
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, float(z_index) * slice_spacing_mm]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PatientID = "TEST"
    ds.PatientName = "Test^Test"
    ds.SeriesNumber = 1
    ds.InstanceNumber = z_index + 1
    for key, value in _COMMON_PATIENT_FIELDS.items():
        setattr(ds, key, value)
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PixelData = np.zeros((rows, cols), dtype=np.int16).tobytes()
    return ds


def _build_seg(path, mask, pixel_spacing, rows=10, cols=10, slice_spacing_mm=1.0):
    n_frames = mask.shape[0]
    series_uid = generate_uid()
    study_uid = generate_uid()
    frame_of_ref_uid = generate_uid()
    sources = [
        _make_source_dataset(
            z, rows, cols, pixel_spacing, slice_spacing_mm, series_uid, study_uid, frame_of_ref_uid
        )
        for z in range(n_frames)
    ]
    segment_description = SegmentDescription(
        segment_number=1,
        segment_label="test",
        segmented_property_category=CodedConcept("91723000", "SCT", "Anatomical Structure"),
        segmented_property_type=CodedConcept("108369006", "SCT", "Neoplasm"),
        algorithm_type="MANUAL",
    )
    seg = hd.seg.Segmentation(
        source_images=sources,
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[segment_description],
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="test",
        manufacturer_model_name="test",
        software_versions="0.0",
        device_serial_number="0",
    )
    seg.save_as(str(path))


@unittest.skipUnless(HIGHDICOM_AVAILABLE, "highdicom not installed in this environment")
class ExtractReaderNodulesTests(unittest.TestCase):
    def test_anisotropic_spacing_catches_row_col_swap(self):
        """The regression guard: row_spacing != col_spacing makes a row/col direction
        swap produce a detectably different (and wrong) centroid, unlike an isotropic
        fixture where the swap is invisible."""
        import tempfile
        from pathlib import Path

        row_spacing_mm, col_spacing_mm = 2.0, 0.5
        mask = np.zeros((1, 10, 10), dtype=bool)
        # 5 pixels wide along the column axis, 1 pixel tall along the row axis.
        mask[0, 5, 2:7] = True

        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "seg.dcm"
            _build_seg(seg_path, mask, [row_spacing_mm, col_spacing_mm])

            nodules = extract_reader_nodules(seg_path, slice_spacing_mm=1.0)

        self.assertEqual(len(nodules), 1)
        cx, cy, cz = nodules[0]["center_lps_mm"]
        # 5 pixels at col indices 2..6 (mean index 4) * col_spacing 0.5mm -> x ~= 2.0mm.
        self.assertAlmostEqual(cx, 2.0, delta=0.3)
        # 1 pixel at row index 5 * row_spacing 2.0mm -> y ~= 10.0mm.
        self.assertAlmostEqual(cy, 10.0, delta=0.3)

    def test_single_segment_uses_shared_functional_group(self):
        """DICOM SEG may put SegmentIdentificationSequence in
        SharedFunctionalGroupsSequence (not per-frame) when only one segment is used
        throughout -- confirmed against a real encoder (highdicom) that this happens."""
        import tempfile
        from pathlib import Path

        mask = np.zeros((3, 10, 10), dtype=bool)
        mask[0, 3:6, 3:6] = True
        mask[1, 3:6, 3:6] = True

        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "seg.dcm"
            _build_seg(seg_path, mask, [1.0, 1.0])

            ds = pydicom.dcmread(str(seg_path))
            self.assertIn("SegmentIdentificationSequence", ds.SharedFunctionalGroupsSequence[0])

            nodules = extract_reader_nodules(seg_path, slice_spacing_mm=1.0)

        self.assertEqual(len(nodules), 1)
        self.assertEqual(nodules[0]["voxel_count"], 18)


class ConsensusNodulesTests(unittest.TestCase):
    def test_three_of_four_readers_forms_consensus(self):
        nodules_by_reader = {
            "reader1": [{"center_lps_mm": (0.0, 0.0, 0.0), "diameter_mm": 6.0}],
            "reader2": [{"center_lps_mm": (1.0, 0.5, 0.0), "diameter_mm": 5.5}],
            "reader3": [{"center_lps_mm": (0.5, -0.5, 1.0), "diameter_mm": 6.5}],
            "reader4": [{"center_lps_mm": (100.0, 100.0, 100.0), "diameter_mm": 4.0}],
        }
        result = consensus_nodules(nodules_by_reader, min_readers=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["reader_count"], 3)

    def test_two_readers_do_not_form_consensus(self):
        nodules_by_reader = {
            "reader1": [{"center_lps_mm": (0.0, 0.0, 0.0), "diameter_mm": 6.0}],
            "reader2": [{"center_lps_mm": (1.0, 0.0, 0.0), "diameter_mm": 5.5}],
        }
        result = consensus_nodules(nodules_by_reader, min_readers=3)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
