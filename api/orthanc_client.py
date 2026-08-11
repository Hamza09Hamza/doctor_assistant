"""A thin client for talking to Orthanc.

Two different Orthanc APIs are used deliberately, not interchangeably:

- **QIDO-RS** (`/dicom-web/studies...`) for querying study/series metadata — this is
  what actually exercises DICOMweb, the vision doc's explicit Phase 2 ask, and is the
  portable, standard interface that would work unmodified against a non-Orthanc PACS.
- **Orthanc's own proprietary REST API** (`/tools/find`, `/instances/{id}/file`) for
  resolving a DICOM UID to Orthanc's internal IDs and downloading raw instance bytes.
  The DICOMweb-standard equivalent (WADO-RS) returns `multipart/related` responses,
  which need their own parser for no behavioral benefit here — a deliberate, contained
  tradeoff (isolated to `download_instance`), not a silent shortcut around DICOMweb.

Config values (`ORTHANC_URL`/`ORTHANC_USERNAME`/`ORTHANC_PASSWORD`) verified against
current upstream Orthanc docs, not guessed — see the Phase 2 plan for sources. This
sandbox has no Docker, so the exact behavior against a real running Orthanc is unverified
end-to-end; `OrthancError` messages are written to make that gap easy to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class OrthancError(RuntimeError):
    """Orthanc returned something this client doesn't know how to interpret. Fails
    closed rather than guessing at a partial or malformed response."""


@dataclass
class OrthancConfig:
    base_url: str
    username: str
    password: str


class OrthancClient:
    def __init__(self, config: OrthancConfig, *, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            auth=(config.username, config.password),
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OrthancClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def query_study(self, study_instance_uid: str) -> dict:
        """QIDO-RS study-level query. Raises `OrthancError` if the study isn't found —
        never returns an empty/partial result silently."""
        response = self._client.get(
            "/dicom-web/studies", params={"StudyInstanceUID": study_instance_uid}
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            raise OrthancError(f"no study found for StudyInstanceUID={study_instance_uid!r}")
        return results[0]

    def query_series(self, study_instance_uid: str) -> list[dict]:
        """QIDO-RS series-level query for every series under a study."""
        response = self._client.get(f"/dicom-web/studies/{study_instance_uid}/series")
        response.raise_for_status()
        return response.json()

    def find_instance_ids(self, series_instance_uid: str) -> list[str]:
        """Orthanc-internal instance IDs for a DICOM series, via Orthanc's own
        `/tools/find` (used only to resolve IDs for `download_instance`, see module
        docstring)."""
        response = self._client.post(
            "/tools/find",
            json={
                "Level": "Series",
                "Expand": True,
                "Query": {"SeriesInstanceUID": series_instance_uid},
            },
        )
        response.raise_for_status()
        matches = response.json()
        if not matches:
            raise OrthancError(f"no series found for SeriesInstanceUID={series_instance_uid!r}")
        instances = matches[0].get("Instances")
        if not instances:
            raise OrthancError(f"series {series_instance_uid!r} has no instances in Orthanc")
        return instances

    def download_instance(self, orthanc_instance_id: str) -> bytes:
        """Raw DICOM file bytes for one instance, via Orthanc's own REST API."""
        response = self._client.get(f"/instances/{orthanc_instance_id}/file")
        response.raise_for_status()
        return response.content

    def upload_instance(self, dicom_bytes: bytes) -> dict:
        """Store one DICOM object (for example an AI-produced SEG) in Orthanc."""
        response = self._client.post(
            "/instances",
            content=dicom_bytes,
            headers={"Content-Type": "application/dicom"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise OrthancError("Orthanc instance upload returned a non-object response")
        return payload
