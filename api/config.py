"""Runtime settings, read directly from the environment.

No settings library — matches how the rest of this repo reads configuration (scripts
take CLI args, Colab notebooks read `userdata.get(...)`); a plain dataclass is enough
for the handful of values this service needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_storage_dir() -> Path:
    return Path(os.environ.get("STORAGE_DIR", "./api_storage")).resolve()


@dataclass
class Settings:
    # Defaults to a local Postgres matching deployments/docker-compose.yml; the test
    # suite overrides this to a disposable SQLite file (see tests/test_api.py).
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://doctor_assistant:doctor_assistant@localhost:5432/doctor_assistant",
        )
    )
    storage_dir: Path = field(default_factory=_default_storage_dir)
    # Matches the `orthanc` service in deployments/docker-compose.yml.
    orthanc_url: str = field(
        default_factory=lambda: os.environ.get("ORTHANC_URL", "http://localhost:8042")
    )
    orthanc_username: str = field(
        default_factory=lambda: os.environ.get("ORTHANC_USERNAME", "doctor_assistant")
    )
    orthanc_password: str = field(
        default_factory=lambda: os.environ.get("ORTHANC_PASSWORD", "doctor_assistant")
    )
    # The OHIF dev server's origin (see viewer/ohif/), allowed via CORS to call this API.
    ohif_origin: str = field(
        default_factory=lambda: os.environ.get("OHIF_ORIGIN", "http://localhost:3000")
    )


def get_settings() -> Settings:
    """Read settings fresh from the environment on every call.

    Deliberately not memoized: tests set `DATABASE_URL`/`STORAGE_DIR` per-test via
    `monkeypatch`-style env overrides before building the app, and a cached singleton
    would silently keep serving the first test's values to every test after it.
    """
    return Settings()
