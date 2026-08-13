"""End-to-end tests for `POST /v1/series/{id}/segment-box` (api/routes/segmentation.py).

Mirrors tests/test_api.py's approach: a disposable SQLite DB, a real FastAPI app, and a
test-double segmenter (no real MedSAM checkpoint/torch/GPU) — offline and fast.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
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


class FakeVolumeSegmenter:
    version = "fake-medsam2:test"

    def segment_volume(self, volume, seed_index, box_xyxy):
        self.last_seed_index = seed_index
        self.last_box = box_xyxy
        mask = np.zeros_like(volume, dtype=bool)
        mask[:, 8:12, 9:13] = True
        return mask


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


class VolumeSegmentationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.nifti_to_dicom import build_dicom_series

        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.storage_dir = tmp_path / "series"
        self.series_uid = generate_uid()
        datasets = build_dicom_series(
            np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3),
            np.diag([0.7, 0.8, 2.0, 1.0]),
            self.storage_dir,
            series_description="MedSAM2 API test",
            modality="CT",
            series_instance_uid=self.series_uid,
        )
        self.sop_uids = [str(dataset.SOPInstanceUID) for dataset in datasets]
        self.study_uid = str(datasets[0].StudyInstanceUID)
        self.segmenter = FakeVolumeSegmenter()
        database_url = f"sqlite:///{tmp_path / 'test.db'}"
        self.app = create_app(
            database_url=database_url,
            storage_dir=tmp_path / "uploads",
            registry=ExpertRegistry(),
            ohif_origin="http://testserver",
            medsam=FakeSegmenter(),
            medsam2=self.segmenter,
        )
        self.client = TestClient(self.app)

        engine = build_engine(database_url)
        self.addCleanup(engine.dispose)
        session_factory = build_session_factory(engine)
        self.session_factory = session_factory
        with session_factory() as db:
            study = Study(modality="ct", body_part="chest", source_filename="x", source="orthanc")
            db.add(study)
            db.flush()
            series = Series(
                study_id=study.id,
                dicom_series_uid=self.series_uid,
                dicom_modality="CT",
                modality="ct",
                body_part="chest",
                instance_count=3,
                storage_dir=str(self.storage_dir),
                analysis_eligible=True,
            )
            db.add(series)
            db.commit()
            self.series_id = series.id

    def tearDown(self) -> None:
        self.app.state.engine.dispose()
        self._tmp.cleanup()

    def test_full_volume_masks_measurements_and_dicom_seg_contract(self) -> None:
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-volume",
            json={
                "sop_instance_uid": self.sop_uids[1],
                "box_xyxy": [7, 7, 14, 14],
                "publish_to_orthanc": False,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["seed_sop_instance_uid"], self.sop_uids[1])
        self.assertEqual(body["source_slice_count"], 3)
        self.assertEqual(body["segmented_slice_count"], 3)
        self.assertEqual(body["voxel_count"], 48)
        self.assertAlmostEqual(body["volume_ml"], 48 * 0.7 * 0.8 * 2.0 / 1000.0)
        self.assertEqual({item["sop_instance_uid"] for item in body["masks"]}, set(self.sop_uids))
        self.assertEqual(body["orthanc_status"], "disabled")
        self.assertEqual(self.segmenter.last_seed_index, 1)

        seg_files = list((self.storage_dir / "derived").glob("prompted3d_*.dcm"))
        self.assertEqual(len(seg_files), 1)
        seg = pydicom.dcmread(str(seg_files[0]))
        self.assertEqual(str(seg.Modality), "SEG")
        self.assertEqual(str(seg.SeriesInstanceUID), body["dicom_seg_series_instance_uid"])
        self.assertEqual(str(seg.SOPInstanceUID), body["dicom_seg_sop_instance_uid"])
        self.assertEqual(str(seg.StudyInstanceUID), self.study_uid)
        self.assertEqual(str(seg.ReferencedSeriesSequence[0].SeriesInstanceUID), self.series_uid)

        artifact = body["dicom_seg_artifact"]
        self.assertEqual(
            artifact["download_path"],
            (
                f"/v1/series/{self.series_id}/segmentations/"
                f"{body['dicom_seg_sop_instance_uid']}/dicom"
            ),
        )
        self.assertEqual(
            artifact["series_instance_uid"], body["dicom_seg_series_instance_uid"]
        )
        self.assertEqual(
            artifact["sop_instance_uid"], body["dicom_seg_sop_instance_uid"]
        )
        self.assertEqual(artifact["byte_length"], seg_files[0].stat().st_size)
        self.assertEqual(
            artifact["sha256"], hashlib.sha256(seg_files[0].read_bytes()).hexdigest()
        )

    def test_dicom_seg_artifact_download_preserves_bytes_and_content_type(self) -> None:
        created = self.client.post(
            f"/v1/series/{self.series_id}/segment-volume",
            json={
                "sop_instance_uid": self.sop_uids[1],
                "box_xyxy": [7, 7, 14, 14],
                "publish_to_orthanc": False,
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        artifact = created.json()["dicom_seg_artifact"]

        downloaded = self.client.get(artifact["download_path"])

        self.assertEqual(downloaded.status_code, 200, downloaded.text)
        self.assertEqual(downloaded.headers["content-type"], "application/dicom")
        self.assertEqual(int(downloaded.headers["content-length"]), artifact["byte_length"])
        self.assertEqual(downloaded.headers["x-content-sha256"], artifact["sha256"])
        self.assertEqual(hashlib.sha256(downloaded.content).hexdigest(), artifact["sha256"])
        dataset = pydicom.dcmread(BytesIO(downloaded.content), stop_before_pixels=True)
        self.assertEqual(str(dataset.Modality), "SEG")
        self.assertEqual(str(dataset.SOPInstanceUID), artifact["sop_instance_uid"])

    def test_dicom_seg_artifact_download_is_scoped_to_its_source_series(self) -> None:
        created = self.client.post(
            f"/v1/series/{self.series_id}/segment-volume",
            json={
                "sop_instance_uid": self.sop_uids[1],
                "box_xyxy": [7, 7, 14, 14],
                "publish_to_orthanc": False,
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        seg_sop_uid = created.json()["dicom_seg_sop_instance_uid"]

        # Point a second API series row at the same directory. The requested SEG file
        # exists there, but its DICOM reference identifies the original source series,
        # so the download route must still reject it.
        with self.session_factory() as db:
            study = Study(
                modality="ct",
                body_part="chest",
                source_filename="other",
                source="orthanc",
            )
            db.add(study)
            db.flush()
            other = Series(
                study_id=study.id,
                dicom_series_uid=generate_uid(),
                dicom_modality="CT",
                modality="ct",
                body_part="chest",
                instance_count=3,
                storage_dir=str(self.storage_dir),
                analysis_eligible=True,
            )
            db.add(other)
            db.commit()
            other_series_id = other.id

        wrong_uid = self.client.get(
            f"/v1/series/{self.series_id}/segmentations/{generate_uid()}/dicom"
        )
        wrong_series = self.client.get(
            f"/v1/series/{other_series_id}/segmentations/{seg_sop_uid}/dicom"
        )

        self.assertEqual(wrong_uid.status_code, 404)
        self.assertEqual(wrong_series.status_code, 404)

    def test_unconfigured_medsam2_is_503(self) -> None:
        self.app.state.medsam2 = None
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-volume",
            json={"sop_instance_uid": self.sop_uids[1], "box_xyxy": [7, 7, 14, 14]},
        )
        self.assertEqual(response.status_code, 503)

    def test_overlapping_volume_inference_is_rejected_before_model_work(self) -> None:
        self.app.state.medsam2_lock.acquire()
        try:
            response = self.client.post(
                f"/v1/series/{self.series_id}/segment-volume",
                json={
                    "sop_instance_uid": self.sop_uids[1],
                    "box_xyxy": [7, 7, 14, 14],
                    "publish_to_orthanc": False,
                },
            )
        finally:
            self.app.state.medsam2_lock.release()

        self.assertEqual(response.status_code, 429)
        self.assertIn("already running", response.json()["detail"])

    def test_colab_mode_forces_orthanc_publication_off(self) -> None:
        self.app.state.disable_orthanc_publication = True
        response = self.client.post(
            f"/v1/series/{self.series_id}/segment-volume",
            json={
                "sop_instance_uid": self.sop_uids[1],
                "box_xyxy": [7, 7, 14, 14],
                # The normal OHIF client requests publication. A remote Colab API must
                # not interpret localhost:8042 as the Mac's Orthanc.
                "publish_to_orthanc": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["orthanc_status"], "disabled")
        self.assertIsNone(response.json()["warning"])


if __name__ == "__main__":
    unittest.main()
