"""Resource-bounded app factory for the temporary Colab GPU inference server.

The normal API builds the complete expert registry. Colab only needs the prompted 3D
segmentation route, so this factory injects an empty registry and disables attempts to
publish DICOM SEG into ``localhost`` Orthanc (which would refer to the Colab VM, not the
Mac). The returned masks remain source-SOP keyed and can be painted by local OHIF.
"""

from __future__ import annotations

from fastapi import FastAPI

from routing import ExpertRegistry

from .main import create_app as create_base_app


def create_app() -> FastAPI:
    app = create_base_app(registry=ExpertRegistry())
    app.state.disable_orthanc_publication = True

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

    return app
