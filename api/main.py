"""FastAPI app factory.

`create_app()` takes explicit overrides rather than reading `get_settings()` directly
inside route handlers, so `tests/test_api.py` can build a fully isolated app (its own
disposable SQLite file, its own temp storage directory, its own registry) without any
production state — global module-level app/engine objects would make that impossible.
"""

from __future__ import annotations

import logging
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
from .routes.series import router as series_router
from .routes.studies import router as studies_router

logger = logging.getLogger(__name__)


def create_app(
    *,
    database_url: str | None = None,
    storage_dir: Path | None = None,
    registry: ExpertRegistry | None = None,
    orthanc_config: OrthancConfig | None = None,
    ohif_origin: str | None = None,
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

    app.include_router(studies_router)
    app.include_router(analyses_router)
    app.include_router(series_router)
    return app


# Run with:  uvicorn api.main:create_app --factory
#
# Deliberately no module-level `app = create_app()` instance here — that would make
# merely *importing* this module (which `tests/test_api.py` needs to do to reach
# `create_app`) try to connect to the production DATABASE_URL and build every real
# expert as an import-time side effect. uvicorn's `--factory` flag calls `create_app()`
# itself instead of importing a pre-built instance, which is what avoids that.
