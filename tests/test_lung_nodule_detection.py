"""Pure geometry and fake-expert API tests for lung-nodule candidate prompts."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from api.db import build_engine, build_session_factory
from api.lung_nodule_detection import (
    DicomSliceGeometry,
    project_lps_aabb_to_nearest_slice,
)
from api.main import create_app
from api.models import Series, Study
from core.enums import BodyPart, Modality
from routing import ExpertRegistry


class FakeLungNoduleDetector:
    name = "ct_lung_nodule"
    modality = Modality.CT
    body_part = BodyPart.CHEST
    min_score = 0.3
    version = "fake-lung-detector:test"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def detect(self, source_path: Path) -> list[dict]:
        self.calls.append(source_path)
        return [
            {
                "score": 0.60,
                "center_lps_mm": (-1.0, 30.0, 9.8),
                "size_whd_mm": (6.0, 8.0, 4.0),
            },
            {
                "score": 0.90,
                "center_lps_mm": (10.0, 12.0, 5.2),
                "size_whd_mm": (6.0, 4.0, 4.0),
            },
            {
                "score": 0.29,
                "center_lps_mm": (15.0, 15.0, 0.0),
                "size_whd_mm": (5.0, 5.0, 5.0),
            },
        ]


def _write_geometry_dicom(
    directory: Path,
    *,
    series_uid: str,
    frame_uid: str,
    sop_uid: str,
    z_position: float,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = directory / f"instance_{z_position:04.1f}.dcm"
    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.FrameOfReferenceUID = frame_uid
    dataset.Modality = "CT"
    dataset.Rows = 32
    dataset.Columns = 32
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.save_as(str(path), enforce_file_format=True)


class LungNoduleGeometryTests(unittest.TestCase):
    def test_world_aabb_is_projected_on_oblique_axes_and_nearest_slice(self) -> None:
        root_half = math.sqrt(0.5)
        first_direction = (root_half, 0.0, root_half)
        second_direction = (0.0, 1.0, 0.0)
        normal = (-root_half, 0.0, root_half)

        def position(distance: float) -> tuple[float, float, float]:
            return tuple(distance * value for value in normal)

        slices = tuple(
            DicomSliceGeometry(
                sop_instance_uid=f"slice-{distance:g}",
                image_position_lps_mm=position(distance),
                image_orientation_patient=(*first_direction, *second_direction),
                pixel_spacing_mm=(2.0, 1.0),
                rows=64,
                columns=64,
                frame_of_reference_uid="frame-1",
            )
            for distance in (0.0, 5.0, 10.0)
        )
        # Pixel center (column=10,row=20), 1.1 mm from the middle source plane.
        center = tuple(
            10.0 * first_direction[index]
            + 40.0 * second_direction[index]
            + 6.1 * normal[index]
            for index in range(3)
        )

        projected = project_lps_aabb_to_nearest_slice(
            slices,
            center_lps_mm=center,
            size_whd_mm=(4.0, 2.0, 6.0),
        )

        self.assertIsNotNone(projected)
        self.assertEqual(projected.seed_sop_instance_uid, "slice-5")
        # The LPS X and Z half-extents both contribute along this tilted image's
        # column axis. A naive LPS-x/spacing conversion would not produce this box.
        self.assertEqual(projected.box_xyxy, (6, 19, 14, 21))

    def test_projection_clips_to_image_bounds(self) -> None:
        geometry = DicomSliceGeometry(
            sop_instance_uid="edge-slice",
            image_position_lps_mm=(0.0, 0.0, 0.0),
            image_orientation_patient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            pixel_spacing_mm=(1.0, 1.0),
            rows=10,
            columns=10,
        )
        projected = project_lps_aabb_to_nearest_slice(
            [geometry],
            center_lps_mm=(-1.0, 9.0, 0.0),
            size_whd_mm=(6.0, 8.0, 2.0),
        )

        self.assertIsNotNone(projected)
        self.assertEqual(projected.box_xyxy, (0, 5, 2, 10))


class LungNoduleDetectionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.series_dir = root / "series"
        self.series_dir.mkdir()
        self.series_uid = generate_uid()
        frame_uid = generate_uid()
        self.sop_uids = [generate_uid() for _ in range(3)]
        for uid, position in zip(self.sop_uids, (0.0, 5.0, 10.0), strict=True):
            _write_geometry_dicom(
                self.series_dir,
                series_uid=self.series_uid,
                frame_uid=frame_uid,
                sop_uid=uid,
                z_position=position,
            )

        self.detector = FakeLungNoduleDetector()
        registry = ExpertRegistry()
        registry.register(self.detector)
        database_url = f"sqlite:///{root / 'test.db'}"
        self.app = create_app(
            database_url=database_url,
            storage_dir=root / "uploads",
            registry=registry,
            ohif_origin="http://testserver",
            medsam2=object(),
        )
        self.client = TestClient(self.app)

        engine = build_engine(database_url)
        self.addCleanup(engine.dispose)
        session_factory = build_session_factory(engine)
        with session_factory() as db:
            study = Study(
                modality="ct",
                body_part="chest",
                source_filename="synthetic",
                source="orthanc",
            )
            db.add(study)
            db.flush()
            series = Series(
                study_id=study.id,
                dicom_series_uid=self.series_uid,
                dicom_modality="CT",
                modality="ct",
                body_part="chest",
                instance_count=3,
                storage_dir=str(self.series_dir),
                analysis_eligible=True,
            )
            db.add(series)
            db.commit()
            self.series_id = series.id

    def tearDown(self) -> None:
        self.app.state.engine.dispose()
        self._tmp.cleanup()

    def test_returns_only_configured_threshold_candidates_with_viewer_prompts(self) -> None:
        response = self.client.post(
            f"/v1/series/{self.series_id}/detect-lung-nodules",
            json={},
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["series_id"], self.series_id)
        self.assertEqual(body["model_version"], self.detector.version)
        self.assertEqual(body["min_score"], 0.3)
        self.assertEqual(body["source_slice_count"], 3)
        self.assertEqual([item["score"] for item in body["detections"]], [0.9, 0.6])
        self.assertEqual(
            body["detections"][0]["seed_sop_instance_uid"], self.sop_uids[1]
        )
        self.assertEqual(body["detections"][0]["box_xyxy"], [7, 10, 13, 14])
        self.assertEqual(
            body["detections"][1]["seed_sop_instance_uid"], self.sop_uids[2]
        )
        self.assertEqual(body["detections"][1]["box_xyxy"], [0, 26, 2, 32])
        self.assertEqual(self.detector.calls, [self.series_dir])

    def test_request_can_only_tighten_configured_threshold(self) -> None:
        response = self.client.post(
            f"/v1/series/{self.series_id}/detect-lung-nodules",
            json={"min_score": 0.7},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["min_score"], 0.7)
        self.assertEqual(
            [item["score"] for item in response.json()["detections"]], [0.9]
        )

        response = self.client.post(
            f"/v1/series/{self.series_id}/detect-lung-nodules",
            json={"min_score": 0.1},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["min_score"], 0.3)
        self.assertEqual(
            [item["score"] for item in response.json()["detections"]], [0.9, 0.6]
        )

    def test_shared_inference_slot_rejects_overlap_before_detector_work(self) -> None:
        self.assertIs(self.app.state.inference_lock, self.app.state.medsam2_lock)
        self.app.state.medsam2_lock.acquire()
        try:
            response = self.client.post(
                f"/v1/series/{self.series_id}/detect-lung-nodules",
                json={},
            )
        finally:
            self.app.state.medsam2_lock.release()

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.detector.calls, [])

    def test_unconfigured_detector_is_503(self) -> None:
        self.app.state.registry = ExpertRegistry()
        response = self.client.post(
            f"/v1/series/{self.series_id}/detect-lung-nodules",
            json={},
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
