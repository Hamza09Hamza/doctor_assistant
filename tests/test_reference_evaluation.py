"""Synthetic tests for prompt-anchored reader DICOM SEG comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    SegmentationStorage,
    generate_uid,
)

from api.reference_evaluation import (
    ReferenceEvaluationError,
    compare_prediction,
    select_prompt_matched_references,
)


def _write_seg(
    path: Path,
    *,
    source_uids: tuple[str, ...],
    segment_volumes: dict[int, np.ndarray],
) -> None:
    """Write a small uncompressed SEG-like fixture with standard frame references."""
    frames: list[np.ndarray] = []
    frame_groups: list[Dataset] = []
    for segment_number, volume in sorted(segment_volumes.items()):
        for source_index, frame in enumerate(volume):
            if not frame.any():
                continue
            frames.append(frame.astype(np.uint8))

            source_image = Dataset()
            source_image.ReferencedSOPClassUID = CTImageStorage
            source_image.ReferencedSOPInstanceUID = source_uids[source_index]
            derivation = Dataset()
            derivation.SourceImageSequence = [source_image]
            segment_identification = Dataset()
            segment_identification.ReferencedSegmentNumber = segment_number
            functional_group = Dataset()
            functional_group.DerivationImageSequence = [derivation]
            functional_group.SegmentIdentificationSequence = [segment_identification]
            frame_groups.append(functional_group)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SegmentationStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = Dataset()
    dataset.file_meta = file_meta
    dataset.SOPClassUID = SegmentationStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "SEG"
    dataset.SegmentationType = "BINARY"
    dataset.Rows = frames[0].shape[0]
    dataset.Columns = frames[0].shape[1]
    dataset.NumberOfFrames = len(frames)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    # Eight-bit fixture pixels keep the test writer simple.  The evaluator treats any
    # decoded nonzero value as segmented and also supports standards-valid 1-bit SEGs.
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.PerFrameFunctionalGroupsSequence = frame_groups
    dataset.SegmentSequence = []
    for segment_number in sorted(segment_volumes):
        segment = Dataset()
        segment.SegmentNumber = segment_number
        segment.SegmentLabel = f"Reader nodule {segment_number}"
        dataset.SegmentSequence.append(segment)
    dataset.PixelData = np.stack(frames, axis=0).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _volume(*, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    result = np.zeros((3, 16, 16), dtype=bool)
    result[:, y0:y1, x0:x1] = True
    return result


class ReferenceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.reference_dir = Path(self._tmp.name) / "reference"
        self.reference_dir.mkdir()
        self.source_uids = tuple(generate_uid() for _ in range(3))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_four_readers_produce_three_of_four_consensus_and_physical_volumes(self) -> None:
        references = (
            _volume(x0=5, x1=10, y0=5, y1=10),
            _volume(x0=5, x1=11, y0=5, y1=10),
            _volume(x0=5, x1=10, y0=4, y1=10),
            _volume(x0=6, x1=10, y0=5, y1=10),
        )
        for index, reference in enumerate(references, start=1):
            _write_seg(
                self.reference_dir / f"lidc_reader_{index}.dcm",
                source_uids=self.source_uids,
                segment_volumes={1: reference},
            )

        selection = select_prompt_matched_references(
            self.reference_dir,
            source_sop_instance_uids=self.source_uids,
            source_shape=(3, 16, 16),
            seed_sop_instance_uid=self.source_uids[1],
            box_xyxy=(4.0, 4.0, 12.0, 12.0),
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.reader_count, 4)
        self.assertEqual(selection.matched_reader_count, 4)
        self.assertEqual(selection.consensus_reader_threshold, 3)

        prediction = references[0]
        result = compare_prediction(prediction, selection, voxel_volume_mm3=0.6)
        expected_consensus = np.stack(references, axis=0).sum(axis=0) >= 3
        expected_dice = (
            2
            * np.logical_and(prediction, expected_consensus).sum()
            / (prediction.sum() + expected_consensus.sum())
        )
        self.assertTrue(result.consensus_available)
        self.assertEqual(result.consensus_rule, "at_least_3_of_4_readers")
        self.assertAlmostEqual(result.consensus_dice, expected_dice)
        self.assertEqual(result.consensus_voxel_count, int(expected_consensus.sum()))
        self.assertAlmostEqual(
            result.consensus_volume_ml,
            expected_consensus.sum() * 0.6 / 1000.0,
        )
        self.assertEqual(len(result.readers), 4)
        self.assertTrue(all(reader.matched for reader in result.readers))
        self.assertAlmostEqual(
            result.readers[0].reference_volume_ml,
            references[0].sum() * 0.6 / 1000.0,
        )

    def test_segment_is_selected_from_prompt_not_highest_prediction_dice(self) -> None:
        prompt_target = _volume(x0=3, x1=6, y0=3, y1=6)
        elsewhere = _volume(x0=11, x1=14, y0=11, y1=14)
        _write_seg(
            self.reference_dir / "reader_with_two_nodules.dcm",
            source_uids=self.source_uids,
            segment_volumes={1: prompt_target, 2: elsewhere},
        )

        selection = select_prompt_matched_references(
            self.reference_dir,
            source_sop_instance_uids=self.source_uids,
            source_shape=(3, 16, 16),
            seed_sop_instance_uid=self.source_uids[1],
            box_xyxy=(2.0, 2.0, 7.0, 7.0),
        )
        assert selection is not None
        self.assertEqual(selection.readers[0].segment_number, 1)

        # If the evaluator selected post hoc, this prediction would score 1.0 against
        # segment 2.  Prompt-first selection correctly keeps the score at zero.
        result = compare_prediction(elsewhere, selection, voxel_volume_mm3=1.0)
        self.assertEqual(result.readers[0].dice, 0.0)
        self.assertEqual(result.consensus_dice, 0.0)

    def test_unrelated_prompt_does_not_force_a_reader_match(self) -> None:
        _write_seg(
            self.reference_dir / "reader.dcm",
            source_uids=self.source_uids,
            segment_volumes={1: _volume(x0=3, x1=6, y0=3, y1=6)},
        )
        selection = select_prompt_matched_references(
            self.reference_dir,
            source_sop_instance_uids=self.source_uids,
            source_shape=(3, 16, 16),
            seed_sop_instance_uid=self.source_uids[1],
            box_xyxy=(11.0, 11.0, 15.0, 15.0),
        )
        assert selection is not None
        self.assertEqual(selection.matched_reader_count, 0)
        result = compare_prediction(
            _volume(x0=11, x1=14, y0=11, y1=14),
            selection,
            voxel_volume_mm3=1.0,
        )
        self.assertFalse(result.consensus_available)
        self.assertIsNone(result.consensus_dice)
        self.assertFalse(result.readers[0].matched)
        self.assertIsNone(result.readers[0].dice)

    def test_unknown_source_sop_reference_is_rejected_not_aligned_by_frame_order(self) -> None:
        invalid_uids = (self.source_uids[0], generate_uid(), self.source_uids[2])
        _write_seg(
            self.reference_dir / "wrong_source.dcm",
            source_uids=invalid_uids,
            segment_volumes={1: _volume(x0=3, x1=6, y0=3, y1=6)},
        )
        with self.assertRaisesRegex(
            ReferenceEvaluationError,
            "not in the evaluated series",
        ):
            select_prompt_matched_references(
                self.reference_dir,
                source_sop_instance_uids=self.source_uids,
                source_shape=(3, 16, 16),
                seed_sop_instance_uid=self.source_uids[1],
                box_xyxy=(2.0, 2.0, 7.0, 7.0),
            )


class ReferenceComparisonRouteTests(unittest.TestCase):
    """The optional comparison contract is emitted by the real segment-volume route."""

    def test_segment_volume_reports_staged_four_reader_consensus(self) -> None:
        from fastapi.testclient import TestClient

        from api.db import build_engine, build_session_factory
        from api.main import create_app
        from api.models import Series, Study
        from routing import ExpertRegistry
        from scripts.nifti_to_dicom import build_dicom_series

        class FakeVolumeSegmenter:
            version = "fake-medsam2:reference-comparison"

            def segment_volume(self, volume, seed_index, box_xyxy):
                result = np.zeros_like(volume, dtype=bool)
                result[:, 8:12, 9:13] = True
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series_dir = root / "series"
            series_uid = generate_uid()
            source_datasets = build_dicom_series(
                np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3),
                np.diag([0.7, 0.8, 2.0, 1.0]),
                series_dir,
                series_description="Reference comparison route test",
                modality="CT",
                series_instance_uid=series_uid,
            )
            source_uids = tuple(str(item.SOPInstanceUID) for item in source_datasets)
            reference_dir = series_dir / "reference"
            reference_dir.mkdir()
            reference = np.zeros((3, 32, 32), dtype=bool)
            reference[:, 8:12, 9:13] = True
            for reader_index in range(1, 5):
                _write_seg(
                    reference_dir / f"lidc_reader_{reader_index}.dcm",
                    source_uids=source_uids,
                    segment_volumes={1: reference},
                )

            database_url = f"sqlite:///{root / 'test.db'}"
            app = create_app(
                database_url=database_url,
                storage_dir=root / "uploads",
                registry=ExpertRegistry(),
                ohif_origin="http://testserver",
                medsam2=FakeVolumeSegmenter(),
            )
            engine = build_engine(database_url)
            try:
                with build_session_factory(engine)() as db:
                    study = Study(
                        modality="ct",
                        body_part="chest",
                        source_filename="synthetic",
                        source="test",
                    )
                    db.add(study)
                    db.flush()
                    series = Series(
                        study_id=study.id,
                        dicom_series_uid=series_uid,
                        dicom_modality="CT",
                        modality="ct",
                        body_part="chest",
                        instance_count=3,
                        storage_dir=str(series_dir),
                        analysis_eligible=True,
                    )
                    db.add(series)
                    db.commit()
                    series_id = series.id

                with TestClient(app) as client:
                    response = client.post(
                        f"/v1/series/{series_id}/segment-volume",
                        json={
                            "sop_instance_uid": source_uids[1],
                            "box_xyxy": [7, 7, 14, 14],
                            "publish_to_orthanc": False,
                        },
                    )

                self.assertEqual(response.status_code, 200, response.text)
                comparison = response.json()["reference_comparison"]
                self.assertEqual(comparison["reader_count"], 4)
                self.assertEqual(comparison["matched_reader_count"], 4)
                self.assertEqual(
                    comparison["consensus_rule"],
                    "at_least_3_of_4_readers",
                )
                self.assertTrue(comparison["consensus_available"])
                self.assertEqual(comparison["consensus_dice"], 1.0)
                self.assertEqual(comparison["consensus_voxel_count"], 48)
                self.assertEqual(len(comparison["readers"]), 4)
                self.assertTrue(
                    all(item["dice"] == 1.0 for item in comparison["readers"])
                )
            finally:
                engine.dispose()
                app.state.engine.dispose()


if __name__ == "__main__":
    unittest.main()
