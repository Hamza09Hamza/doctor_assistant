"""Resource-bounded app factory for the temporary Colab GPU inference server.

The normal API builds the complete expert registry. Colab only needs the prompted 3D
segmentation route, so this factory injects an empty registry and disables attempts to
publish DICOM SEG into ``localhost`` Orthanc (which would refer to the Colab VM, not the
Mac). The returned masks remain source-SOP keyed and can be painted by local OHIF.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from routing import ExpertRegistry

from .main import create_app as create_base_app


def create_app() -> FastAPI:
    app = create_base_app(registry=ExpertRegistry())
    app.state.disable_orthanc_publication = True
    download_directory = os.getenv("COLAB_DOWNLOAD_DIRECTORY")
    archive_path = (
        Path(download_directory).expanduser() / "Orthanc-macOS-26.4.2.zip"
        if download_directory
        else None
    )

    def orthanc_macos_archive() -> Path | None:
        return archive_path if archive_path and archive_path.is_file() else None

    @app.get("/health", tags=["runtime"])
    def health() -> dict:
        segmenter = getattr(app.state, "medsam2", None)
        return {
            "status": "ok",
            "mode": "colab-inference-only",
            "medsam2_configured": segmenter is not None,
            "model_version": getattr(segmenter, "version", None),
            "max_concurrent_volume_inferences": 1,
            "orthanc_publication": "disabled",
        }

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
