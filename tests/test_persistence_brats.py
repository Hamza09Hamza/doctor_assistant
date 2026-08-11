"""Offline tests for the BraTS multi-sequence wiring in `api/persistence.py`.

`_match_brats_sequences` and `_build_brats_scan` are the pure-ish logic this module
adds to bridge the gap `experts/mri_brats.py`'s docstring used to flag: nothing in the
live API populated `scan.meta.extra["sequence_paths"]`. These tests build synthetic
Study/Series rows (mirroring `tests/test_dicom_ingest.py`'s in-memory DICOM pattern) and
a fake BraTS-shaped expert (mirroring `tests/test_api.py`'s `WorkingExpert`), so no
MONAI bundle, network, or GPU is touched -- only `SeriesDescription` tag reading and the
grouping/`Scan`-construction logic run for real.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from api import persistence
from api.db import Base, build_engine, build_session_factory
from api.models import Analysis, Series, Study
from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from experts.mri_brats import BUNDLE_MODALITY_ORDER
from routing import ExpertRegistry


def _write_dicom_instance(directory: Path, *, series_description: str | None) -> Path:
    """One minimal DICOM file on disk with (optionally) a SeriesDescription tag --
    enough for `pydicom.dcmread(..., stop_before_pixels=True)`, no pixel data needed."""
    directory.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "MR"
    ds.PatientID = "TESTPAT"
    if series_description is not None:
        ds.SeriesDescription = series_description

    path = directory / "instance_0000.dcm"
    ds.save_as(path, enforce_file_format=True, little_endian=True, implicit_vr=False)
    return path


class BraTSPersistenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        engine = build_engine(f"sqlite:///{self.tmp_path / 'test.db'}")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        self.session_factory = build_session_factory(engine)
        self.db = self.session_factory()
        self.addCleanup(self.db.close)

    def _make_study(self, *, storage_path: str | None = None) -> Study:
        study = Study(
            modality=Modality.UNKNOWN.value,
            body_part=BodyPart.UNKNOWN.value,
            source_filename="orthanc-study-test",
            source="orthanc" if storage_path is None else "upload",
            storage_path=storage_path,
        )
        self.db.add(study)
        self.db.flush()
        return study

    def _add_series(
        self,
        study: Study,
        *,
        series_description: str | None,
        modality: Modality = Modality.MRI,
        body_part: BodyPart = BodyPart.BRAIN,
    ) -> Series:
        series_dir = self.tmp_path / "series" / generate_uid().replace(".", "_")
        _write_dicom_instance(series_dir, series_description=series_description)
        series = Series(
            study_id=study.id,
            dicom_series_uid=generate_uid(),
            dicom_modality="MR",
            modality=modality.value,
            body_part=body_part.value,
            storage_dir=str(series_dir),
            analysis_eligible=True,
        )
        self.db.add(series)
        self.db.flush()
        return series

    # -- _series_description ------------------------------------------------

    def test_series_description_reads_the_dicom_tag(self) -> None:
        series_dir = self.tmp_path / "one_series"
        _write_dicom_instance(series_dir, series_description="T1c")

        self.assertEqual(
            persistence._series_description(str(series_dir)), "T1c"
        )

    def test_series_description_missing_tag_returns_none(self) -> None:
        series_dir = self.tmp_path / "no_description"
        _write_dicom_instance(series_dir, series_description=None)

        self.assertIsNone(persistence._series_description(str(series_dir)))

    def test_series_description_empty_directory_returns_none(self) -> None:
        empty_dir = self.tmp_path / "empty"
        empty_dir.mkdir()

        self.assertIsNone(persistence._series_description(str(empty_dir)))

    # -- _match_brats_sequences ----------------------------------------------

    def test_matches_all_four_canonical_sequences(self) -> None:
        study = self._make_study()
        t1c = self._add_series(study, series_description="T1c")  # -> t1c
        t1 = self._add_series(study, series_description="T1")
        t2 = self._add_series(study, series_description="T2")
        flair = self._add_series(study, series_description="FLAIR")
        self.db.commit()

        result = persistence._match_brats_sequences(self.db, study.id)

        self.assertIsNotNone(result)
        self.assertEqual(set(result), set(BUNDLE_MODALITY_ORDER))
        self.assertEqual(result["t1c"], t1c.storage_dir)
        self.assertEqual(result["t1"], t1.storage_dir)
        self.assertEqual(result["t2"], t2.storage_dir)
        self.assertEqual(result["flair"], flair.storage_dir)

    def test_missing_one_sequence_returns_none(self) -> None:
        study = self._make_study()
        self._add_series(study, series_description="T1c")
        self._add_series(study, series_description="T1")
        self._add_series(study, series_description="T2")
        # no FLAIR series
        self.db.commit()

        self.assertIsNone(persistence._match_brats_sequences(self.db, study.id))

    def test_scout_and_unrecognised_series_are_skipped_not_fatal(self) -> None:
        study = self._make_study()
        self._add_series(study, series_description="T1c")
        self._add_series(study, series_description="T1")
        self._add_series(study, series_description="T2")
        self._add_series(study, series_description="FLAIR")
        # A scout/localizer series that isn't part of the 4-sequence protocol --
        # canonical_sequence_name raises ValueError internally; must not blow up the
        # whole match, and must not appear in the result.
        self._add_series(study, series_description="SCOUT")
        self.db.commit()

        result = persistence._match_brats_sequences(self.db, study.id)

        self.assertIsNotNone(result)
        self.assertEqual(set(result), set(BUNDLE_MODALITY_ORDER))

    def test_non_brain_mri_series_are_ignored(self) -> None:
        study = self._make_study()
        self._add_series(study, series_description="T1c")
        self._add_series(study, series_description="T1")
        self._add_series(study, series_description="T2")
        # Same description, but body part isn't brain -- must not count toward the match.
        self._add_series(
            study, series_description="FLAIR", body_part=BodyPart.SPINE
        )
        self.db.commit()

        self.assertIsNone(persistence._match_brats_sequences(self.db, study.id))

    def test_duplicate_sequence_keeps_first_and_does_not_raise(self) -> None:
        study = self._make_study()
        first = self._add_series(study, series_description="T1c")
        self._add_series(study, series_description="T1c")  # duplicate t1c
        self._add_series(study, series_description="T1")
        self._add_series(study, series_description="T2")
        self._add_series(study, series_description="FLAIR")
        self.db.commit()

        result = persistence._match_brats_sequences(self.db, study.id)

        self.assertIsNotNone(result)
        self.assertEqual(result["t1c"], first.storage_dir)

    def test_no_series_at_all_returns_none(self) -> None:
        study = self._make_study()
        self.db.commit()

        self.assertIsNone(persistence._match_brats_sequences(self.db, study.id))

    # -- _build_brats_scan -----------------------------------------------------

    def test_build_brats_scan_stamps_sequence_paths_and_niche(self) -> None:
        study = self._make_study()
        sequence_paths = {name: f"/data/{name}" for name in BUNDLE_MODALITY_ORDER}

        scan = persistence._build_brats_scan(study, sequence_paths)

        self.assertIsInstance(scan, Scan)
        self.assertEqual(scan.meta.modality, Modality.MRI)
        self.assertEqual(scan.meta.body_part, BodyPart.BRAIN)
        self.assertEqual(scan.meta.extra["sequence_paths"], sequence_paths)


class FakeBraTSExpert:
    """Stands in for the real `BraTSExpert` -- records what `sequence_paths` it was
    given rather than running any MONAI bundle, mirroring `tests/test_api.py`'s
    `WorkingExpert` pattern for a lightweight expert double."""

    name = "fake_mri_brats"
    modality = Modality.MRI
    body_part = BodyPart.BRAIN
    class_names: list[str] = []
    version = "fake-brats:v1"

    def __init__(self) -> None:
        self.seen_sequence_paths: dict[str, str] | None = None

    def predict(self, scan) -> Prediction:
        self.seen_sequence_paths = (scan.meta.extra or {}).get("sequence_paths")
        return Prediction(expert=self.name, class_probs={}, meta=scan.meta)


class RunAnalysisBraTSRoutingTestCase(unittest.TestCase):
    """End-to-end through `persistence.run_analysis` with a fake expert, proving the
    grouped `Scan` actually reaches `Pipeline.analyze_scan` -> the routed expert with
    the right sequence_paths -- not just that the helper functions compute the right
    dict in isolation."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

        engine = build_engine(f"sqlite:///{self.tmp_path / 'test.db'}")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        self.session_factory = build_session_factory(engine)
        self.db = self.session_factory()
        self.addCleanup(self.db.close)

    def _series_dir(self, description: str) -> str:
        series_dir = self.tmp_path / "series" / generate_uid().replace(".", "_")
        _write_dicom_instance(series_dir, series_description=description)
        return str(series_dir)

    def test_study_level_analysis_routes_to_brats_with_all_four_sequences(self) -> None:
        study = Study(
            modality=Modality.UNKNOWN.value,
            body_part=BodyPart.UNKNOWN.value,
            source_filename="orthanc-study-test",
            source="orthanc",
            storage_path=None,
        )
        self.db.add(study)
        self.db.flush()
        for description in ("T1c", "T1", "T2", "FLAIR"):
            self.db.add(
                Series(
                    study_id=study.id,
                    dicom_series_uid=generate_uid(),
                    dicom_modality="MR",
                    modality=Modality.MRI.value,
                    body_part=BodyPart.BRAIN.value,
                    storage_dir=self._series_dir(description),
                    analysis_eligible=True,
                )
            )
        self.db.commit()

        registry = ExpertRegistry()
        fake_expert = FakeBraTSExpert()
        registry.register(fake_expert)

        analysis = persistence.create_queued_analysis(self.db, study_id=study.id)
        persistence.run_analysis(self.session_factory, registry, analysis.id)

        self.db.expire_all()
        refreshed = self.db.get(Analysis, analysis.id)
        self.assertEqual(refreshed.status, "complete", refreshed.error)
        self.assertIsNotNone(fake_expert.seen_sequence_paths)
        self.assertEqual(set(fake_expert.seen_sequence_paths), set(BUNDLE_MODALITY_ORDER))

    def test_study_level_analysis_with_incomplete_sequences_fails_closed(self) -> None:
        study = Study(
            modality=Modality.UNKNOWN.value,
            body_part=BodyPart.UNKNOWN.value,
            source_filename="orthanc-study-test",
            source="orthanc",
            storage_path=None,
        )
        self.db.add(study)
        self.db.flush()
        # Only 3 of the 4 required sequences -- no fallback single path exists either
        # (storage_path is None), so this must fail closed rather than silently
        # proceeding with a partial channel set.
        for description in ("T1c", "T1", "T2"):
            self.db.add(
                Series(
                    study_id=study.id,
                    dicom_series_uid=generate_uid(),
                    dicom_modality="MR",
                    modality=Modality.MRI.value,
                    body_part=BodyPart.BRAIN.value,
                    storage_dir=self._series_dir(description),
                    analysis_eligible=True,
                )
            )
        self.db.commit()

        registry = ExpertRegistry()
        registry.register(FakeBraTSExpert())

        analysis = persistence.create_queued_analysis(self.db, study_id=study.id)
        persistence.run_analysis(self.session_factory, registry, analysis.id)

        self.db.expire_all()
        refreshed = self.db.get(Analysis, analysis.id)
        self.assertEqual(refreshed.status, "failed")
        self.assertIn("no analyzable storage path", refreshed.error)


if __name__ == "__main__":
    unittest.main()
