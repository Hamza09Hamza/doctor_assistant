"""End-to-end tests for the Phase 1 API (`api/`).

Runs entirely offline: `DATABASE_URL` is overridden per-test to a disposable SQLite
file under a `tempfile.TemporaryDirectory` (see `api/db.py` — only portable SQLAlchemy
column types are used in `api/models.py`, so this is safe), and the expert registry is
a test double (mirroring `tests/test_pipeline.py`'s `MutatingExpert`/`FailingExpert`
fixtures) rather than the real GPU-backed adapters. No live Postgres, network, or GPU
is required to run this file.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api.db import build_engine, build_session_factory
from api.main import create_app
from api.models import Series, Study
from core.enums import BodyPart, Modality
from core.types import Prediction
from routing import ExpertRegistry


class WorkingExpert:
    name = "test_chest_expert"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Effusion"]
    version = "test-expert:v1"

    def predict(self, scan):
        return Prediction(expert=self.name, class_probs={"Effusion": 0.9}, meta=scan.meta)


class AlwaysFailingExpert:
    name = "test_broken_expert"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names: list[str] = []

    def predict(self, scan):
        raise RuntimeError("simulated failure")


def _make_test_image(directory: Path) -> Path:
    path = directory / "sample.png"
    Image.new("L", (32, 32), color=128).save(path)
    return path


def _wait_for_terminal_status(client: TestClient, analysis_id: str, *, timeout: float = 5.0) -> dict:
    """TestClient runs BackgroundTasks synchronously before a request call returns, so
    this loop should resolve on its first iteration in practice — kept as a real
    polling loop anyway so the test exercises the same status-endpoint contract a real
    client would use, and doesn't silently depend on that TestClient implementation
    detail."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/analyses/{analysis_id}/status")
        response.raise_for_status()
        body = response.json()
        if body["status"] in ("complete", "failed"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"analysis {analysis_id} never reached a terminal status")


def _wait_for_series_analyses(client: TestClient, series_id: str, *, timeout: float = 5.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/series/{series_id}/analyses")
        response.raise_for_status()
        body = response.json()
        if body and body[0]["status"] in ("complete", "failed"):
            return body
        time.sleep(0.05)
    raise TimeoutError(f"series {series_id} analyses never reached a terminal status")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.db_path = self.tmp_path / "test.db"
        self.storage_dir = self.tmp_path / "storage"

    def _build_client(self, registry: ExpertRegistry) -> TestClient:
        app = create_app(
            database_url=f"sqlite:///{self.db_path}",
            storage_dir=self.storage_dir,
            registry=registry,
        )
        self.addCleanup(app.state.engine.dispose)
        return TestClient(app)

    def _upload_study(self, client: TestClient) -> str:
        image_path = _make_test_image(self.tmp_path)
        with image_path.open("rb") as handle:
            response = client.post(
                "/v1/studies",
                files={"file": ("sample.png", handle, "image/png")},
                data={"modality": "xray", "body_part": "chest"},
            )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def test_upload_creates_a_study(self) -> None:
        client = self._build_client(ExpertRegistry())

        study_id = self._upload_study(client)

        response = client.get(f"/v1/studies/{study_id}")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["modality"], "xray")
        self.assertEqual(body["body_part"], "chest")
        self.assertEqual(body["source_filename"], "sample.png")

    def test_health_describes_full_api_without_loading_optional_models(self) -> None:
        client = self._build_client(ExpertRegistry())

        response = client.get("/health")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "mode": "full-api",
                "ready": True,
                "medsam2_configured": False,
                "medsam2_loaded": False,
                "model_version": None,
                "lung_nodule_detector_configured": False,
                "lung_nodule_detector_loaded": False,
                "lung_nodule_detector_version": None,
                "max_concurrent_inferences": 1,
                "max_concurrent_volume_inferences": 1,
                "orthanc_publication": "enabled",
            },
        )

    def test_analysis_completes_and_matches_pipeline_output(self) -> None:
        registry = ExpertRegistry()
        registry.register(WorkingExpert())
        client = self._build_client(registry)
        study_id = self._upload_study(client)

        submit = client.post(f"/v1/studies/{study_id}/analyses")
        self.assertEqual(submit.status_code, 202)
        self.assertEqual(submit.json()["status"], "queued")
        analysis_id = submit.json()["analysis_id"]

        status = _wait_for_terminal_status(client, analysis_id)
        self.assertEqual(status["status"], "complete")

        result = client.get(f"/v1/analyses/{analysis_id}").json()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["verification_ok"])
        self.assertIsNotNone(result["report_text"])
        self.assertTrue(any(f["label"] == "Effusion" for f in result["findings"]))
        self.assertTrue(any(r["label"] == "Effusion" for r in result["recommendations"]))
        executions = result["expert_executions"]
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["status"], "completed")
        self.assertEqual(executions[0]["expert_version"], "test-expert:v1")

    def test_all_experts_failing_reaches_a_terminal_failed_status(self) -> None:
        registry = ExpertRegistry()
        registry.register(AlwaysFailingExpert())
        client = self._build_client(registry)
        study_id = self._upload_study(client)

        submit = client.post(f"/v1/studies/{study_id}/analyses")
        analysis_id = submit.json()["analysis_id"]

        status = _wait_for_terminal_status(client, analysis_id)
        self.assertEqual(status["status"], "failed")
        self.assertIsNotNone(status["error"])

    def test_unknown_study_or_analysis_returns_404(self) -> None:
        client = self._build_client(ExpertRegistry())

        self.assertEqual(client.get("/v1/studies/does-not-exist").status_code, 404)
        self.assertEqual(client.post("/v1/studies/does-not-exist/analyses").status_code, 404)
        self.assertEqual(client.get("/v1/analyses/does-not-exist").status_code, 404)

    def _insert_series(self, *, dicom_series_uid: str) -> Series:
        """Insert a Study+Series directly (bypassing dicom_ingest/Orthanc — that
        pipeline is covered separately in tests/test_dicom_ingest.py). These two
        endpoints only need a row to exist, not a loadable DICOM file behind it."""
        engine = build_engine(f"sqlite:///{self.db_path}")
        self.addCleanup(engine.dispose)
        session = build_session_factory(engine)()
        study = Study(
            modality=Modality.UNKNOWN.value,
            body_part=BodyPart.UNKNOWN.value,
            source_filename="orthanc-study-test",
            source="orthanc",
        )
        session.add(study)
        session.flush()
        series = Series(
            study_id=study.id,
            dicom_series_uid=dicom_series_uid,
            dicom_modality="CR",
            modality=Modality.XRAY.value,
            body_part=BodyPart.CHEST.value,
            storage_dir="/tmp/unused-in-this-test",
            analysis_eligible=True,
        )
        session.add(series)
        session.commit()
        session.refresh(series)
        session.close()
        return series

    def test_series_lookup_by_dicom_uid(self) -> None:
        client = self._build_client(ExpertRegistry())
        series = self._insert_series(dicom_series_uid="1.2.3.4.5")

        found = client.get("/v1/series", params={"dicom_series_uid": "1.2.3.4.5"})
        self.assertEqual(found.status_code, 200, found.text)
        self.assertEqual(found.json()["id"], series.id)

        missing = client.get("/v1/series", params={"dicom_series_uid": "no.such.uid"})
        self.assertEqual(missing.status_code, 404)

    def test_series_analyses_list_reflects_submitted_runs(self) -> None:
        registry = ExpertRegistry()
        registry.register(WorkingExpert())
        client = self._build_client(registry)
        series = self._insert_series(dicom_series_uid="1.2.3.4.6")

        empty = client.get(f"/v1/series/{series.id}/analyses")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json(), [])

        # Submitting against this series' placeholder storage_dir will fail (no real
        # DICOM file there) — that's fine, this test only checks the list endpoint
        # reflects whatever run happened, not that it succeeded.
        submit = client.post(f"/v1/series/{series.id}/analyses")
        self.assertEqual(submit.status_code, 202)

        listed = _wait_for_series_analyses(client, series.id)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], submit.json()["analysis_id"])


if __name__ == "__main__":
    unittest.main()
