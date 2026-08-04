"""SQLAlchemy engine/session wiring.

`build_engine`/`build_session_factory` are functions rather than a module-level
singleton so tests can point them at a disposable SQLite file without the production
`DATABASE_URL` ever being touched — the app factory in `api/main.py` wires these
together per-app-instance instead of relying on import-time global state.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str):
    # SQLite needs this flag to allow use across the request thread and the
    # BackgroundTasks thread FastAPI/Starlette runs jobs on; Postgres ignores it via
    # connect_args being empty.
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def build_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A request-scoped session — used as a FastAPI dependency via `Depends`."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
