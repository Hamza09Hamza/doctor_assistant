"""`GET /v1/series/{id}`, `POST /v1/series/{id}/analyses` — analysis submission for a
DICOM-imported series (the study-level equivalent, `POST /v1/studies/{id}/analyses`,
still exists unchanged for plain single-file uploads — see `api/routes/analyses.py`).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import persistence
from ..models import Analysis, Series
from ..schemas import AnalysisStatusResponse, AnalysisSubmitResponse, SeriesResponse
from .deps import get_db

router = APIRouter(prefix="/v1/series", tags=["series"])


@router.get("", response_model=SeriesResponse)
def find_series_by_dicom_uid(dicom_series_uid: str, db: Session = Depends(get_db)) -> SeriesResponse:
    """Resolve a DICOM SeriesInstanceUID (what the OHIF panel actually has on hand) to
    our internal series row."""
    series = db.execute(
        select(Series).where(Series.dicom_series_uid == dicom_series_uid)
    ).scalar_one_or_none()
    if series is None:
        raise HTTPException(
            status_code=404, detail=f"no series with dicom_series_uid={dicom_series_uid!r}"
        )
    return SeriesResponse.model_validate(series)


@router.get("/{series_id}", response_model=SeriesResponse)
def get_series(series_id: str, db: Session = Depends(get_db)) -> SeriesResponse:
    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    return SeriesResponse.model_validate(series)


@router.get("/{series_id}/analyses", response_model=list[AnalysisStatusResponse])
def list_series_analyses(series_id: str, db: Session = Depends(get_db)) -> list[AnalysisStatusResponse]:
    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    analyses = db.execute(
        select(Analysis).where(Analysis.series_id == series_id).order_by(Analysis.created_at.desc())
    ).scalars().all()
    return [AnalysisStatusResponse.model_validate(a) for a in analyses]


@router.post("/{series_id}/analyses", response_model=AnalysisSubmitResponse, status_code=202)
def submit_series_analysis(
    series_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnalysisSubmitResponse:
    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    if not series.analysis_eligible:
        raise HTTPException(status_code=422, detail=series.ineligible_reason)

    analysis = persistence.create_queued_analysis(db, study_id=series.study_id, series_id=series.id)
    background_tasks.add_task(
        persistence.run_analysis,
        request.app.state.session_factory,
        request.app.state.registry,
        analysis.id,
    )
    return AnalysisSubmitResponse(analysis_id=analysis.id, status=analysis.status)
