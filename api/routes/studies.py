"""`POST /v1/studies`, `GET /v1/studies/{id}` — upload a scan, look it up.

Accepts any file extension `ingest.loaders.load_scan` already supports (2D images or
volumetric formats) — this endpoint doesn't restrict that further, so no new format
handling was needed to support both.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from core.enums import BodyPart, Modality

from .. import dicom_ingest, persistence
from ..models import Study
from ..orthanc_client import OrthancClient, OrthancError
from ..schemas import StudyImportRequest, StudyResponse
from .deps import get_db

router = APIRouter(prefix="/v1/studies", tags=["studies"])


@router.post("", response_model=StudyResponse, status_code=201)
async def upload_study(
    request: Request,
    file: UploadFile,
    modality: str = Form(...),
    body_part: str = Form(...),
    db: Session = Depends(get_db),
) -> StudyResponse:
    try:
        Modality(modality)
        BodyPart(body_part)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not file.filename:
        raise HTTPException(status_code=422, detail="uploaded file has no filename")

    storage_dir: Path = request.app.state.storage_dir
    study_id = uuid.uuid4().hex
    study_dir = storage_dir / "studies" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    destination = study_dir / file.filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    study = persistence.create_study(
        db,
        study_id=study_id,
        modality=modality,
        body_part=body_part,
        source_filename=file.filename,
        storage_path=str(destination),
    )
    return StudyResponse.model_validate(study)


@router.post("/import", response_model=StudyResponse, status_code=201)
def import_study(
    body: StudyImportRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> StudyResponse:
    with OrthancClient(request.app.state.orthanc_config) as orthanc_client:
        try:
            study = dicom_ingest.import_study_from_orthanc(
                db,
                storage_dir=request.app.state.storage_dir,
                orthanc_client=orthanc_client,
                study_instance_uid=body.study_instance_uid,
            )
        except OrthancError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StudyResponse.model_validate(study)


@router.get("/{study_id}", response_model=StudyResponse)
def get_study(study_id: str, db: Session = Depends(get_db)) -> StudyResponse:
    study = db.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"no study {study_id!r}")
    return StudyResponse.model_validate(study)
