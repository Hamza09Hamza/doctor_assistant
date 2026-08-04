"""Shared FastAPI dependencies for the route modules."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session


def get_db(request: Request) -> Iterator[Session]:
    """A request-scoped session, built from the session factory `main.py` puts on
    `app.state` at startup — never a module-level global, so tests can point a
    freshly-built app at a disposable database without touching production state."""
    session_factory = request.app.state.session_factory
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
