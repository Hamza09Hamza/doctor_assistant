"""`POST /v1/studies/{id}/analyses`, `GET /v1/analyses/{id}[/status]`.

Submitting an analysis returns immediately (`status: "queued"`) — the actual
`Pipeline.analyze` call happens in a `BackgroundTasks` job
(`api.persistence.run_analysis`) after the response has already gone out, which is the
whole point of Phase 1: the request thread never blocks on GPU inference.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .. import persistence
from ..models import Analysis, Study
from ..schemas import AnalysisResultResponse, AnalysisStatusResponse, AnalysisSubmitResponse
from .deps import get_db

router = APIRouter(tags=["analyses"])


@router.post(
    "/v1/studies/{study_id}/analyses",
    response_model=AnalysisSubmitResponse,
    status_code=202,
)
def submit_analysis(
    study_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnalysisSubmitResponse:
    study = db.get(Study, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail=f"no study {study_id!r}")

    analysis = persistence.create_queued_analysis(db, study_id=study_id)
    background_tasks.add_task(
        persistence.run_analysis,
        request.app.state.session_factory,
        request.app.state.registry,
        analysis.id,
    )
    return AnalysisSubmitResponse(analysis_id=analysis.id, status=analysis.status)


@router.get("/v1/analyses/{analysis_id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisStatusResponse:
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"no analysis {analysis_id!r}")
    return AnalysisStatusResponse.model_validate(analysis)


@router.get("/v1/analyses/{analysis_id}", response_model=AnalysisResultResponse)
def get_analysis_result(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisResultResponse:
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"no analysis {analysis_id!r}")
    return AnalysisResultResponse.model_validate(analysis)
