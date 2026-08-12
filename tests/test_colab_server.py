"""Lightweight checks for the inference-only Colab app factory."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.colab_server import create_app
from core.enums import BodyPart, Modality


class FakeLungNoduleDetector:
    name = "ct_lung_nodule"
    modality = Modality.CT
    body_part = BodyPart.CHEST
    min_score = 0.3
    version = "fake-lung-detector:colab"

    def __init__(self, *, bundle_root: str) -> None:
        self.bundle_root = bundle_root

    @property
    def is_loaded(self) -> bool:
        return False


class FakePreloadableModel:
    version = "fake-preloadable:model"

    def __init__(self) -> None:
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def preload(self) -> None:
        self.loaded = True


class FakePreloadableDetector(FakeLungNoduleDetector):
    def __init__(self, *, bundle_root: str) -> None:
        super().__init__(bundle_root=bundle_root)
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def preload(self) -> None:
        self.loaded = True


class ColabServerTests(unittest.TestCase):
    def test_health_and_resource_guards_without_loading_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "Orthanc-macOS-26.4.2.zip"
            archive.write_bytes(b"test-archive")
            env = {
                "DATABASE_URL": f"sqlite:///{root / 'colab.db'}",
                "STORAGE_DIR": str(root / "storage"),
                "MEDSAM2_BACKEND": "disabled",
                "LUNG_NODULE_BUNDLE_ROOT": "",
                "COLAB_PRELOAD_MODELS": "",
                "COLAB_DOWNLOAD_DIRECTORY": str(root),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                app = create_app()
            self.addCleanup(app.state.engine.dispose)

            with TestClient(app) as client:
                response = client.get("/health")
                archive_response = client.get("/downloads/orthanc-macos")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
            self.assertEqual(response.json()["mode"], "colab-inference-only")
            self.assertFalse(response.json()["ready"])
            self.assertFalse(response.json()["medsam2_configured"])
            self.assertFalse(response.json()["medsam2_loaded"])
            self.assertFalse(response.json()["lung_nodule_detector_configured"])
            self.assertFalse(response.json()["lung_nodule_detector_loaded"])
            self.assertIsNone(response.json()["lung_nodule_detector_version"])
            self.assertEqual(response.json()["max_concurrent_inferences"], 1)
            self.assertEqual(response.json()["max_concurrent_volume_inferences"], 1)
            self.assertEqual(response.json()["orthanc_publication"], "disabled")
            self.assertTrue(app.state.disable_orthanc_publication)
            self.assertEqual(app.state.registry.experts(), [])
            self.assertEqual(archive_response.status_code, 200)
            self.assertEqual(archive_response.content, b"test-archive")
            self.assertEqual(archive_response.headers["content-type"], "application/zip")

    def test_registers_only_lung_detector_when_bundle_root_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "DATABASE_URL": f"sqlite:///{root / 'colab.db'}",
                "STORAGE_DIR": str(root / "storage"),
                "MEDSAM2_BACKEND": "disabled",
                "LUNG_NODULE_BUNDLE_ROOT": str(root / "bundles"),
                "COLAB_PRELOAD_MODELS": "",
                "COLAB_DOWNLOAD_DIRECTORY": "",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "experts.ct_lung_nodule.LungNoduleDetectorExpert",
                    FakeLungNoduleDetector,
                ),
            ):
                app = create_app()
            self.addCleanup(app.state.engine.dispose)

            experts = app.state.registry.experts()
            self.assertEqual(len(experts), 1)
            self.assertIs(experts[0], app.state.lung_nodule_detector)
            self.assertIsInstance(experts[0], FakeLungNoduleDetector)
            self.assertEqual(experts[0].bundle_root, str(root / "bundles"))

            with TestClient(app) as client:
                health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["lung_nodule_detector_configured"])
            self.assertFalse(health.json()["lung_nodule_detector_loaded"])
            self.assertEqual(
                health.json()["lung_nodule_detector_version"],
                "fake-lung-detector:colab",
            )

    def test_preloads_both_models_inside_the_colab_app_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmenter = FakePreloadableModel()
            env = {
                "DATABASE_URL": f"sqlite:///{root / 'colab.db'}",
                "STORAGE_DIR": str(root / "storage"),
                "MEDSAM2_BACKEND": "disabled",
                "LUNG_NODULE_BUNDLE_ROOT": str(root / "bundles"),
                "COLAB_PRELOAD_MODELS": "1",
                "COLAB_DOWNLOAD_DIRECTORY": "",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "experts.ct_lung_nodule.LungNoduleDetectorExpert",
                    FakePreloadableDetector,
                ),
                mock.patch("api.colab_server.create_base_app") as create_base_app,
            ):
                from api.main import create_app as real_create_app

                create_base_app.side_effect = lambda registry: real_create_app(
                    database_url=env["DATABASE_URL"],
                    storage_dir=root / "storage",
                    registry=registry,
                    medsam2=segmenter,
                )
                app = create_app()
            self.addCleanup(app.state.engine.dispose)

            with TestClient(app) as client:
                health = client.get("/health").json()
            self.assertTrue(segmenter.loaded)
            self.assertTrue(app.state.lung_nodule_detector.loaded)
            self.assertTrue(health["ready"])


if __name__ == "__main__":
    unittest.main()
