"""Offline tests for `api/dicom_ingest.py` — no Orthanc, Docker, or network required.

A synthetic DICOM file is built in-memory with `pydicom.dataset.FileDataset` (verified
by hand to round-trip through both `pydicom.dcmread` header extraction and
`ingest.loaders.load_scan`, exactly the two consumers `api/dicom_ingest.py` and
`api/persistence.py` exercise). A `FakeOrthancClient` stands in for
`api.orthanc_client.OrthancClient` — same public method surface
(`query_study`/`query_series`/`find_instance_ids`/`download_instance`), no HTTP —
mirroring `tests/test_pipeline.py`'s fixture-expert pattern.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from api import dicom_ingest, persistence
from api.db import Base, build_engine, build_session_factory
from api.models import Analysis, Series, Study
from core.enums import BodyPart, Modality
from core.types import Prediction
from routing import ExpertRegistry


def _build_dicom_bytes(
    *,
    study_uid: str,
    series_uid: str,
    patient_id: str = "TESTPAT1",
    modality: str = "CR",
    body_part_examined: str | None = "CHEST",
    view_position: str | None = "AP",
    rows: int = 8,
    cols: int = 8,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = modality
    ds.PatientID = patient_id
    # Deliberately included, exactly what tests below assert never reaches our DB.
    ds.PatientName = "Test^Patient"
    if body_part_examined is not None:
        ds.BodyPartExamined = body_part_examined
    if view_position is not None:
        ds.ViewPosition = view_position
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((rows, cols), dtype=np.uint16).tobytes()

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True, little_endian=True, implicit_vr=False)
    return buffer.getvalue()


def _dicom_json_element(value: str) -> dict:
    return {"Value": [value]}


class FakeOrthancClient:
    """Stands in for `api.orthanc_client.OrthancClient` — same method surface, no HTTP.

    `series` maps SeriesInstanceUID -> (dicom_modality, list of instance DICOM bytes).
    """

    def __init__(self, *, study_uid: str, patient_id: str, series: dict[str, tuple[str, list[bytes]]]) -> None:
        self.study_uid = study_uid
        self.patient_id = patient_id
        self.series = series

    def query_study(self, study_instance_uid: str) -> dict:
        assert study_instance_uid == self.study_uid
        return {"00100020": _dicom_json_element(self.patient_id)}

    def query_series(self, study_instance_uid: str) -> list[dict]:
        assert study_instance_uid == self.study_uid
        return [
            {
                "0020000E": _dicom_json_element(series_uid),
                "00080060": _dicom_json_element(dicom_modality),
            }
            for series_uid, (dicom_modality, _instances) in self.series.items()
        ]

    def find_instance_ids(self, series_instance_uid: str) -> list[str]:
        _modality, instances = self.series[series_instance_uid]
        return [f"orthanc-instance-{i}" for i in range(len(instances))]

    def download_instance(self, orthanc_instance_id: str) -> bytes:
        index = int(orthanc_instance_id.rsplit("-", 1)[-1])
        for _modality, instances in self.series.values():
            if index < len(instances):
                return instances[index]
        raise AssertionError(f"unknown instance id {orthanc_instance_id!r}")


class WorkingExpert:
    name = "test_chest_expert"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Effusion"]

    def predict(self, scan):
        return Prediction(expert=self.name, class_probs={"Effusion": 0.9}, meta=scan.meta)


class DicomIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage_dir = Path(self._tmp.name) / "storage"
        self.storage_dir.mkdir()

        engine = build_engine(f"sqlite:///{Path(self._tmp.name) / 'test.db'}")
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        self.session_factory = build_session_factory(engine)
        self.db = self.session_factory()
        self.addCleanup(self.db.close)

    def test_import_creates_study_and_eligible_series(self) -> None:
        study_uid = generate_uid()
        series_uid = generate_uid()
        client = FakeOrthancClient(
            study_uid=study_uid,
            patient_id="TESTPAT1",
            series={series_uid: ("CR", [_build_dicom_bytes(study_uid=study_uid, series_uid=series_uid)])},
        )

        study = dicom_ingest.import_study_from_orthanc(
            self.db, storage_dir=self.storage_dir, orthanc_client=client, study_instance_uid=study_uid
        )

        self.assertEqual(study.source, "orthanc")
        self.assertEqual(study.dicom_study_uid, study_uid)
        self.assertEqual(len(study.series), 1)
        series = study.series[0]
        self.assertEqual(series.dicom_series_uid, series_uid)
        self.assertEqual(series.modality, Modality.XRAY.value)
        self.assertEqual(series.body_part, BodyPart.CHEST.value)
        self.assertEqual(series.view_position, "AP")
        self.assertTrue(series.analysis_eligible)
        self.assertIsNone(series.ineligible_reason)
        self.assertTrue(Path(series.storage_dir).is_dir())
        self.assertEqual(len(list(Path(series.storage_dir).iterdir())), 1)

    def test_patient_reference_is_hashed_never_raw(self) -> None:
        study_uid = generate_uid()
        series_uid = generate_uid()
        client = FakeOrthancClient(
            study_uid=study_uid,
            patient_id="TESTPAT1",
            series={series_uid: ("CR", [_build_dicom_bytes(study_uid=study_uid, series_uid=series_uid)])},
        )

        study = dicom_ingest.import_study_from_orthanc(
            self.db, storage_dir=self.storage_dir, orthanc_client=client, study_instance_uid=study_uid
        )

        self.assertIsNotNone(study.patient_reference)
        self.assertNotEqual(study.patient_reference, "TESTPAT1")
        self.assertEqual(
            study.patient_reference,
            dicom_ingest._pseudonymous_patient_reference("TESTPAT1"),
        )

    def test_unmapped_modality_is_recorded_but_ineligible(self) -> None:
        study_uid = generate_uid()
        series_uid = generate_uid()
        client = FakeOrthancClient(
            study_uid=study_uid,
            patient_id="TESTPAT1",
            series={
                series_uid: (
                    "XA",  # not in _DICOM_MODALITY_MAP
                    [_build_dicom_bytes(study_uid=study_uid, series_uid=series_uid, modality="XA")],
                )
            },
        )

        study = dicom_ingest.import_study_from_orthanc(
            self.db, storage_dir=self.storage_dir, orthanc_client=client, study_instance_uid=study_uid
        )

        series = study.series[0]
        self.assertFalse(series.analysis_eligible)
        self.assertIn("XA", series.ineligible_reason)

    def test_missing_body_part_examined_is_ineligible(self) -> None:
        study_uid = generate_uid()
        series_uid = generate_uid()
        client = FakeOrthancClient(
            study_uid=study_uid,
            patient_id="TESTPAT1",
            series={
                series_uid: (
                    "CR",
                    [
                        _build_dicom_bytes(
                            study_uid=study_uid, series_uid=series_uid, body_part_examined=None
                        )
                    ],
                )
            },
        )

        study = dicom_ingest.import_study_from_orthanc(
            self.db, storage_dir=self.storage_dir, orthanc_client=client, study_instance_uid=study_uid
        )

        series = study.series[0]
        self.assertFalse(series.analysis_eligible)
        self.assertIn("BodyPartExamined", series.ineligible_reason)

    def test_series_analysis_runs_through_the_existing_pipeline(self) -> None:
        study_uid = generate_uid()
        series_uid = generate_uid()
        client = FakeOrthancClient(
            study_uid=study_uid,
            patient_id="TESTPAT1",
            series={series_uid: ("CR", [_build_dicom_bytes(study_uid=study_uid, series_uid=series_uid)])},
        )
        study = dicom_ingest.import_study_from_orthanc(
            self.db, storage_dir=self.storage_dir, orthanc_client=client, study_instance_uid=study_uid
        )
        series = study.series[0]

        registry = ExpertRegistry()
        registry.register(WorkingExpert())
        analysis = persistence.create_queued_analysis(self.db, study_id=study.id, series_id=series.id)

        persistence.run_analysis(self.session_factory, registry, analysis.id)

        # `run_analysis` commits through a *different* session; this test's own
        # session still has the pre-update row cached in its identity map, so a plain
        # `db.get()` would silently return stale data without this.
        self.db.expire_all()
        refreshed = self.db.get(Analysis, analysis.id)
        self.assertEqual(refreshed.status, "complete", refreshed.error)
        self.assertTrue(refreshed.verification_ok)
        self.assertTrue(any(f.label == "Effusion" for f in refreshed.findings))


if __name__ == "__main__":
    unittest.main()
