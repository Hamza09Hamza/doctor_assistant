"""Lightweight checks for the inference-only Colab app factory."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.colab_server import create_app


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
            self.assertFalse(response.json()["medsam2_configured"])
            self.assertEqual(response.json()["max_concurrent_volume_inferences"], 1)
            self.assertEqual(response.json()["orthanc_publication"], "disabled")
            self.assertTrue(app.state.disable_orthanc_publication)
            self.assertEqual(app.state.registry.experts(), [])
            self.assertEqual(archive_response.status_code, 200)
            self.assertEqual(archive_response.content, b"test-archive")
            self.assertEqual(archive_response.headers["content-type"], "application/zip")


if __name__ == "__main__":
    unittest.main()
