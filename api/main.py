"""FastAPI app factory.

`create_app()` takes explicit overrides rather than reading `get_settings()` directly
inside route handlers, so `tests/test_api.py` can build a fully isolated app (its own
disposable SQLite file, its own temp storage directory, its own registry) without any
production state — global module-level app/engine objects would make that impossible.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from routing import ExpertRegistry

from .config import get_settings
from .db import Base, build_engine, build_session_factory
from .orthanc_client import OrthancConfig
from .registry import build_default_registry
from .routes.analyses import router as analyses_router
from .routes.lung_nodule_detection import router as lung_nodule_detection_router
from .routes.segmentation import router as segmentation_router
from .routes.series import router as series_router
from .routes.studies import router as studies_router

logger = logging.getLogger(__name__)


def _build_medsam(*, checkpoint_path: str | None, model_id: str | None):
    """Env-var-gated, failure-isolated the same way `api/registry.py::_try_register`
    handles every expert — a missing/misconfigured checkpoint must not stop the API
    from serving everything else.

    `checkpoint_path` (MEDSAM_CHECKPOINT_PATH) takes precedence when set — a local
    directory with a `transformers`-format checkpoint (config.json/pytorch_model.bin),
    for a fully offline deployment. `model_id` (MEDSAM_MODEL_ID) is the Hub fallback —
    `transformers` downloads/caches it on first real use, not at startup. Both being
    unset means the feature stays off by default, matching every other expert here.
    """
    if not checkpoint_path and not model_id:
        logger.info(
            "api.main: interactive segmentation skipped — set MEDSAM_CHECKPOINT_PATH "
            "(local dir) or MEDSAM_MODEL_ID (HF Hub id) to enable "
            "POST /v1/series/{id}/segment-box."
        )
        return None
    from experts.medsam_interactive import MedSAMBoxSegmenter

    try:
        if checkpoint_path:
            return MedSAMBoxSegmenter(checkpoint_path=checkpoint_path)
        return MedSAMBoxSegmenter(model_id=model_id)
    except Exception as exc:  # noqa: BLE001 — isolate from the rest of app startup
        logger.warning("api.main: MedSAM unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def _build_medsam2(
    *,
    checkpoint_path: str | None,
    model_config: str | None,
    device: str | None,
    backend: str | None,
    mlx_model: str | None,
):
    """Build CUDA MedSAM2 or Apple-Silicon MLX SAM2 behind one volume contract."""
    requested_backend = (backend or "auto").strip().lower()
    if requested_backend not in {"auto", "torch", "mlx", "disabled"}:
        logger.warning("api.main: unsupported MEDSAM2_BACKEND=%r", requested_backend)
        return None
    if requested_backend == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            requested_backend = "mlx"
        elif checkpoint_path:
            requested_backend = "torch"
        else:
            requested_backend = "disabled"

    if requested_backend == "disabled":
        logger.info(
            "api.main: 3D interactive segmentation skipped — set "
            "MEDSAM2_BACKEND=mlx on Apple Silicon or MEDSAM2_CHECKPOINT_PATH for "
            "the Torch/CUDA backend."
        )
        return None

    try:
        if requested_backend == "mlx":
            from experts.medsam2_volume import SAM2MLXVolumeSegmenter

            return SAM2MLXVolumeSegmenter(
                model=mlx_model or "avbiswas/sam2.1-hiera-small-mlx-16bit",
            )
        if not checkpoint_path:
            raise ValueError("MEDSAM2_CHECKPOINT_PATH is required for the Torch backend")
        from experts.medsam2_volume import MedSAM2VolumeSegmenter

        return MedSAM2VolumeSegmenter(
            checkpoint_path=checkpoint_path,
            model_config=model_config or "configs/sam2.1_hiera_t512.yaml",
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - isolate optional runtime at startup
        logger.warning("api.main: MedSAM2 unavailable (%s: %s)", type(exc).__name__, exc)
        return None


def create_app(
    *,
    database_url: str | None = None,
    storage_dir: Path | None = None,
    registry: ExpertRegistry | None = None,
    orthanc_config: OrthancConfig | None = None,
    ohif_origin: str | None = None,
    medsam=None,
    medsam2=None,
) -> FastAPI:
    settings = get_settings()
    resolved_database_url = database_url or settings.database_url
    resolved_storage_dir = storage_dir or settings.storage_dir
    resolved_storage_dir.mkdir(parents=True, exist_ok=True)
    resolved_orthanc_config = orthanc_config or OrthancConfig(
        base_url=settings.orthanc_url,
        username=settings.orthanc_username,
        password=settings.orthanc_password,
    )
    resolved_ohif_origin = ohif_origin or settings.ohif_origin

    engine = build_engine(resolved_database_url)
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)

    app = FastAPI(title="Doctor Assistant API", version="0.1.0")
    app.state.engine = engine  # exposed so callers (tests) can dispose it on teardown
    app.state.session_factory = session_factory
    app.state.storage_dir = resolved_storage_dir
    app.state.orthanc_config = resolved_orthanc_config
    # Building the real registry touches every expert's constructor (see
    # api/registry.py) — allow a caller (tests, or a future multi-process deployment)
    # to inject one instead of paying that cost, or to inject a fake for testing.
    app.state.registry = registry if registry is not None else build_default_registry()
    # Mirrors `registry` above: a caller (tests) can inject a fake `medsam` segmenter
    # directly instead of paying for a real checkpoint load, or to exercise the 404/503
    # paths deterministically.
    app.state.medsam = (
        medsam
        if medsam is not None
        else _build_medsam(
            checkpoint_path=os.environ.get("MEDSAM_CHECKPOINT_PATH"),
            model_id=os.environ.get("MEDSAM_MODEL_ID"),
        )
    )
    app.state.medsam2 = (
        medsam2
        if medsam2 is not None
        else _build_medsam2(
            checkpoint_path=os.environ.get("MEDSAM2_CHECKPOINT_PATH"),
            model_config=os.environ.get("MEDSAM2_MODEL_CONFIG"),
            device=os.environ.get("MEDSAM2_DEVICE"),
            backend=os.environ.get("MEDSAM2_BACKEND"),
            mlx_model=os.environ.get("MEDSAM2_MLX_MODEL"),
        )
    )
    # MedSAM2 and the 3D lung-nodule detector can each consume most of a GPU. Even one
    # Uvicorn worker may execute several synchronous requests concurrently in its
    # thread pool, so they share one nonblocking inference slot. Keep the legacy state
    # name as an alias for callers/tests that predate the detector endpoint.
    app.state.inference_lock = threading.Lock()
    app.state.medsam2_lock = app.state.inference_lock
    app.state.runtime_mode = "full-api"
    app.state.lung_nodule_detector = None
    app.state.disable_orthanc_publication = False

    # The OHIF findings panel (viewer/ohif/extensions/extension-doctor-assistant)
    # calls this API cross-origin from OHIF's own dev server — scoped to one
    # configurable origin rather than "*", matching this project's explicit-not-
    # permissive-by-default posture elsewhere.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved_ohif_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id, request.method, request.url.path, response.status_code, duration_ms,
        )
        return response

    @app.get("/health", tags=["runtime"])
    def health() -> dict:
        """Expose optional inference capabilities through one base-app contract.

        ``api.colab_server`` changes only app state after constructing this app. Reading
        that state at request time keeps full and Colab deployments on the same route
        instead of registering two competing ``GET /health`` handlers.
        """

        segmenter = getattr(app.state, "medsam2", None)
        detector = getattr(app.state, "lung_nodule_detector", None)
        if detector is None:
            detector = next(
                (
                    expert
                    for expert in app.state.registry.experts()
                    if getattr(expert, "name", None) == "ct_lung_nodule"
                ),
                None,
            )
        publication_disabled = bool(
            getattr(app.state, "disable_orthanc_publication", False)
        )
        medsam2_configured = segmenter is not None
        detector_configured = detector is not None
        medsam2_loaded = bool(getattr(segmenter, "is_loaded", False))
        detector_loaded = bool(getattr(detector, "is_loaded", False))
        runtime_mode = getattr(app.state, "runtime_mode", "full-api")
        ready = (
            medsam2_configured
            and detector_configured
            and medsam2_loaded
            and detector_loaded
            if runtime_mode == "colab-inference-only"
            else True
        )
        return {
            "status": "ok",
            "mode": runtime_mode,
            "ready": ready,
            "medsam2_configured": medsam2_configured,
            "medsam2_loaded": medsam2_loaded,
            "model_version": getattr(segmenter, "version", None),
            "lung_nodule_detector_configured": detector_configured,
            "lung_nodule_detector_loaded": detector_loaded,
            "lung_nodule_detector_version": getattr(detector, "version", None),
            "max_concurrent_inferences": 1,
            "max_concurrent_volume_inferences": 1,
            "orthanc_publication": "disabled" if publication_disabled else "enabled",
        }

    app.include_router(studies_router)
    app.include_router(analyses_router)
    app.include_router(series_router)
    app.include_router(segmentation_router)
    app.include_router(lung_nodule_detection_router)
    return app


# Run with:  uvicorn api.main:create_app --factory
#
# Deliberately no module-level `app = create_app()` instance here — that would make
# merely *importing* this module (which `tests/test_api.py` needs to do to reach
# `create_app`) try to connect to the production DATABASE_URL and build every real
# expert as an import-time side effect. uvicorn's `--factory` flag calls `create_app()`
# itself instead of importing a pre-built instance, which is what avoids that.
