"""Resource-bounded app factory for the temporary Colab GPU inference server.

The normal API builds every configured expert. Colab registers only the dedicated lung
nodule detector (when its bundle root is configured) alongside prompted MedSAM2, and
disables attempts to publish DICOM SEG into ``localhost`` Orthanc (which would refer to
the Colab VM, not the Mac). Both GPU paths share the base app's single inference slot.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from routing import ExpertRegistry

from .main import create_app as create_base_app


def _build_colab_registry() -> tuple[ExpertRegistry, object | None]:
    registry = ExpertRegistry()
    bundle_root = os.getenv("LUNG_NODULE_BUNDLE_ROOT")
    if not bundle_root:
        return registry, None

    from experts.ct_lung_nodule import LungNoduleDetectorExpert

    detector = LungNoduleDetectorExpert(bundle_root=bundle_root)
    registry.register(detector)
    return registry, detector


def create_app() -> FastAPI:
    registry, lung_nodule_detector = _build_colab_registry()
    app = create_base_app(registry=registry)
    app.state.runtime_mode = "colab-inference-only"
    app.state.lung_nodule_detector = lung_nodule_detector
    app.state.disable_orthanc_publication = True
    if os.getenv("COLAB_PRELOAD_MODELS", "").strip().lower() in {"1", "true", "yes"}:
        segmenter = getattr(app.state, "medsam2", None)
        if segmenter is None or lung_nodule_detector is None:
            raise RuntimeError(
                "COLAB_PRELOAD_MODELS requires both MedSAM2 and the lung-nodule detector"
            )
        # This runs inside the single Uvicorn subprocess. Preloading in the notebook
        # process would duplicate both GPU models and defeat the resource guard.
        segmenter.preload()
        lung_nodule_detector.preload()
    download_directory = os.getenv("COLAB_DOWNLOAD_DIRECTORY")
    archive_path = (
        Path(download_directory).expanduser() / "Orthanc-macOS-26.4.2.zip"
        if download_directory
        else None
    )

    def orthanc_macos_archive() -> Path | None:
        return archive_path if archive_path and archive_path.is_file() else None

    @app.get("/downloads/orthanc-macos", include_in_schema=False)
    def download_orthanc_macos() -> FileResponse:
        archive = orthanc_macos_archive()
        if archive is None:
            raise HTTPException(status_code=404, detail="Orthanc macOS archive is unavailable")
        return FileResponse(
            archive,
            media_type="application/zip",
            filename="Orthanc-macOS-26.4.2.zip",
        )

    return app
