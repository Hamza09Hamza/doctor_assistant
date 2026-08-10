"""End-to-end tests for `POST /v1/series/{id}/segment-box` (api/routes/segmentation.py).

Mirrors tests/test_api.py's approach: a disposable SQLite DB, a real FastAPI app, and a
test-double segmenter (no real MedSAM checkpoint/torch/GPU) — offline and fast.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from api.db import build_engine, build_session_factory
from api.main import create_app
from api.models import Series, Study
from experts.medsam_interactive import decode_binary_mask_rle
from routing import ExpertRegistry


class FakeSegmenter:
    """Returns a deterministic box-shaped mask instead of running a real model."""

    version = "fake-medsam:test"

    def segment_box(self, image, box_xyxy):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask


class RaisingSegmenter:
    def segment_box(self, image, box_xyxy):
        raise ValueError("box does not overlap a usable region")


def _write_synthetic_dicom(directory: Path, sop_instance_uid: str) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPInstanceUID = sop_instance_uid
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "OT"
    ds.Rows, ds.Columns = 32, 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    arr = np.linspace(0, 1000, 32 * 32, dtype=np.uint16).reshape(32, 32)
    ds.PixelData = arr.tobytes()

    path = directory / f"instance_{sop_instance_uid}.dcm"
    ds.save_as(str(path), enforce_file_format=True, little_endian=True, implicit_vr=False)
    return path


class SegmentationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.storage_dir = tmp_path / "series"
        self.storage_dir.mkdir()
        self.sop_uid = generate_uid()
        _write_synthetic_dicom(self.storage_dir, self.sop_uid)

        database_url = f"sqlite:///{tmp_path / 'test.db'}"

        self.app = create_app(
            database_url=database_url,
            storage_dir=tmp_path / "uploads",
            registry=ExpertRegistry(),
            ohif_origin="http://testserver",
            medsam=FakeSegmenter(),
        )
        self.client = TestClient(self.app)

        # create_app() already ran Base.metadata.create_all against this database_url;
        # insert the fixture rows afterward (mirrors tests/test_api.py's
        # `_insert_series`), using a fresh engine on the same URL rather than the app's
        # own so this doesn't depend on internal app.state layout.
        engine = build_engine(database_url)
        self.addCleanup(engine.dispose)
        session_factory = build_session_factory(engine)
        with session_factory() as db:
            study = Study(modality="ct", body_part="abdomen", source_filename="x", source="orthanc")
            db.add(study)
            db.flush()
            series = Series(
                study_id=study.id,
                dicom_series_uid="1.2.3",
                dicom_modality="CT",
                modality="ct",
                body_part="abdomen",
                instance_count=1,
                storage_dir=str(self.storage_dir),
                analysis_eligible=True,
            )
            db.add(series)
            db.commit()
            self.series_id = series.id

    def tearDown(self) -> None:
        self.app.state.engine.dispose()
        self._tmp.cleanup()

    def test_segment_box_returns_decodable_mask(self) -> None:
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-box",
            json={"sop_instance_uid": self.sop_uid, "box_xyxy": [4, 4, 20, 20]},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sop_instance_uid"], self.sop_uid)
        self.assertEqual(body["model_version"], "fake-medsam:test")

        mask = decode_binary_mask_rle(body["mask_rle"])
        self.assertEqual(mask.shape, (32, 32))
        self.assertTrue(mask[10, 10])  # inside the requested box
        self.assertFalse(mask[0, 0])  # outside it

    def test_unknown_series_is_404(self) -> None:
        response = self.client.post(
            "/v1/series/does-not-exist/segment-box",
            json={"sop_instance_uid": self.sop_uid, "box_xyxy": [0, 0, 10, 10]},
        )
        self.assertEqual(response.status_code, 404)

    def test_unknown_sop_instance_is_404(self) -> None:
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-box",
            json={"sop_instance_uid": "1.2.3.4.5.6.does.not.exist", "box_xyxy": [0, 0, 10, 10]},
        )
        self.assertEqual(response.status_code, 404)

    def test_segmenter_value_error_is_422(self) -> None:
        self.app.state.medsam = RaisingSegmenter()
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-box",
            json={"sop_instance_uid": self.sop_uid, "box_xyxy": [0, 0, 1, 1]},
        )
        self.assertEqual(response.status_code, 422)

    def test_unconfigured_medsam_is_503(self) -> None:
        self.app.state.medsam = None
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-box",
            json={"sop_instance_uid": self.sop_uid, "box_xyxy": [0, 0, 10, 10]},
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
