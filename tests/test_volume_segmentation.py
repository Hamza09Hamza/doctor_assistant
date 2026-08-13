"""Geometry, measurement, and DICOM SEG tests for the MedSAM2 volume bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from api.volume_segmentation import (
    PROMPTED_STRUCTURE_CATEGORY_CODE,
    PROMPTED_STRUCTURE_TYPE_CODE,
    build_interactive_dicom_seg,
    load_dicom_volume,
    measure_volume,
)
from scripts.nifti_to_dicom import build_dicom_series

try:
    import highdicom  # noqa: F401

    HIGHDICOM_AVAILABLE = True
except ImportError:
    HIGHDICOM_AVAILABLE = False


class VolumePreparationTests(unittest.TestCase):
    def test_prompted_structure_codes_do_not_assert_lesion_or_abnormality(self) -> None:
        self.assertEqual(PROMPTED_STRUCTURE_CATEGORY_CODE, ("85756007", "SCT", "Tissue"))
        self.assertEqual(PROMPTED_STRUCTURE_TYPE_CODE, ("85756007", "SCT", "Tissue"))
        self.assertNotEqual(PROMPTED_STRUCTURE_CATEGORY_CODE[0], "49755003")
        self.assertNotEqual(PROMPTED_STRUCTURE_TYPE_CODE[0], "52988006")

    def test_load_tracks_seed_identity_and_physical_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_uid = generate_uid()
            datasets = build_dicom_series(
                np.random.default_rng(4).normal(size=(12, 10, 4)).astype(np.float32),
                np.diag([0.5, 0.75, 2.5, 1.0]),
                root,
                series_description="volume helper test",
                modality="CT",
                series_instance_uid=series_uid,
            )
            seed_uid = str(datasets[2].SOPInstanceUID)
            loaded = load_dicom_volume(
                root,
                series_instance_uid=series_uid,
                seed_sop_instance_uid=seed_uid,
            )

            self.assertEqual(loaded.display_volume.shape, (4, 12, 10))
            self.assertEqual(loaded.sop_instance_uids[loaded.seed_index], seed_uid)
            self.assertAlmostEqual(loaded.row_spacing_mm, 0.5)
            self.assertAlmostEqual(loaded.column_spacing_mm, 0.75)
            self.assertAlmostEqual(loaded.slice_spacing_mm, 2.5)

            mask = np.zeros_like(loaded.display_volume, dtype=bool)
            mask[1:3, 2:6, 3:8] = True
            measured = measure_volume(mask, loaded)
            self.assertEqual(measured.voxel_count, 40)
            self.assertEqual(measured.segmented_slice_count, 2)
            self.assertAlmostEqual(measured.volume_ml, 40 * 0.5 * 0.75 * 2.5 / 1000.0)
            self.assertAlmostEqual(
                measured.axial_bbox_diagonal_mm,
                np.hypot(4 * 0.5, 5 * 0.75),
            )
            self.assertAlmostEqual(measured.craniocaudal_extent_mm, 2 * 2.5)

    def test_unknown_seed_uid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_uid = generate_uid()
            build_dicom_series(
                np.zeros((8, 8, 2), dtype=np.float32),
                np.eye(4),
                root,
                series_description="missing seed test",
                modality="CT",
                series_instance_uid=series_uid,
            )
            with self.assertRaises(FileNotFoundError):
                load_dicom_volume(
                    root,
                    series_instance_uid=series_uid,
                    seed_sop_instance_uid=generate_uid(),
                )

    def test_explicit_viewer_window_overrides_stored_dicom_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_uid = generate_uid()
            datasets = build_dicom_series(
                np.linspace(-1000, 500, 8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2),
                np.eye(4),
                root,
                series_description="viewer window override",
                modality="CT",
                series_instance_uid=series_uid,
            )
            for index, dataset in enumerate(datasets):
                dataset.WindowCenter = 40
                dataset.WindowWidth = 350
                pydicom.dcmwrite(
                    root / f"slice_{index:04d}.dcm",
                    dataset,
                    enforce_file_format=True,
                )

            stored = load_dicom_volume(
                root,
                series_instance_uid=series_uid,
                seed_sop_instance_uid=str(datasets[0].SOPInstanceUID),
            )
            viewer = load_dicom_volume(
                root,
                series_instance_uid=series_uid,
                seed_sop_instance_uid=str(datasets[0].SOPInstanceUID),
                window_center=-600,
                window_width=1500,
            )

            self.assertFalse(np.array_equal(stored.display_volume, viewer.display_volume))


@unittest.skipUnless(HIGHDICOM_AVAILABLE, "highdicom not installed in this environment")
class InteractiveDicomSegTests(unittest.TestCase):
    def test_generated_seg_references_source_and_has_nonempty_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_uid = generate_uid()
            datasets = build_dicom_series(
                np.random.default_rng(9).normal(size=(12, 10, 4)).astype(np.float32),
                np.diag([0.5, 0.75, 2.5, 1.0]),
                root / "source",
                series_description="DICOM SEG source",
                modality="CT",
                series_instance_uid=series_uid,
            )
            loaded = load_dicom_volume(
                root / "source",
                series_instance_uid=series_uid,
                seed_sop_instance_uid=str(datasets[1].SOPInstanceUID),
            )
            mask = np.zeros_like(loaded.display_volume, dtype=bool)
            mask[1:3, 3:7, 2:6] = True
            output = root / "derived" / "medsam2_seg.dcm"

            seg_series_uid, seg_sop_uid = build_interactive_dicom_seg(
                mask,
                loaded,
                output,
                segment_label="Prompted structure",
                model_version="medsam2:test",
            )

            seg = pydicom.dcmread(str(output))
            self.assertEqual(str(seg.Modality), "SEG")
            self.assertEqual(str(seg.SeriesInstanceUID), seg_series_uid)
            self.assertEqual(str(seg.SOPInstanceUID), seg_sop_uid)
            self.assertEqual(str(seg.StudyInstanceUID), str(loaded.datasets[0].StudyInstanceUID))
            referenced = {
                str(item.SeriesInstanceUID)
                for item in getattr(seg, "ReferencedSeriesSequence", [])
            }
            self.assertIn(series_uid, referenced)
            segment = seg.SegmentSequence[0]
            self.assertEqual(str(segment.SegmentAlgorithmType), "SEMIAUTOMATIC")
            category = segment.SegmentedPropertyCategoryCodeSequence[0]
            property_type = segment.SegmentedPropertyTypeCodeSequence[0]
            self.assertEqual(
                (str(category.CodeValue), str(category.CodingSchemeDesignator), str(category.CodeMeaning)),
                PROMPTED_STRUCTURE_CATEGORY_CODE,
            )
            self.assertEqual(
                (
                    str(property_type.CodeValue),
                    str(property_type.CodingSchemeDesignator),
                    str(property_type.CodeMeaning),
                ),
                PROMPTED_STRUCTURE_TYPE_CODE,
            )
            self.assertGreater(int(seg.NumberOfFrames), 0)
            self.assertTrue(seg.PixelData)


if __name__ == "__main__":
    unittest.main()
