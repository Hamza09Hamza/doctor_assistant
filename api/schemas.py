"""Pydantic response models — the HTTP-facing shape of the ORM rows in `api/models.py`.

`model_config = {"from_attributes": True}` lets every response model be built directly
from an ORM instance (`Response.model_validate(row)`), so `api/routes/*.py` never
hand-assembles dicts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SeriesResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    study_id: str
    dicom_series_uid: str
    dicom_modality: str
    modality: str
    body_part: str
    view_position: str | None
    instance_count: int
    analysis_eligible: bool
    ineligible_reason: str | None
    created_at: datetime


class StudyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    modality: str
    body_part: str
    source_filename: str
    source: str
    dicom_study_uid: str | None
    patient_reference: str | None
    created_at: datetime
    series: list[SeriesResponse] = []


class StudyImportRequest(BaseModel):
    study_instance_uid: str


class AnalysisSubmitResponse(BaseModel):
    analysis_id: str
    status: str


class AnalysisStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class FindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    canonical_label: str | None
    probability: float | None
    present: bool
    confidence: float | None
    laterality: str | None
    location: str | None
    size_mm: float | None
    volume_ml: float | None
    count: int
    source: str
    execution_id: str | None


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    text: str
    urgency: str


class ExpertExecutionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    execution_id: str
    expert: str
    status: str
    expert_version: str | None
    preprocessing_version: str | None
    error: str | None


class SegmentBoxRequest(BaseModel):
    """One on-demand interactive-segmentation request: a single 2D slice + a box drawn
    around the structure of interest. Not tied to any `Analysis` row — this is stateless
    per-call, the viewer owns accept/reject of the returned mask."""

    sop_instance_uid: str
    box_xyxy: tuple[float, float, float, float]


class SegmentBoxResponse(BaseModel):
    sop_instance_uid: str
    mask_rle: dict
    model_version: str


class SegmentVolumeRequest(BaseModel):
    """A box on one source slice, propagated through its complete DICOM series."""

    sop_instance_uid: str
    box_xyxy: tuple[float, float, float, float]
    segment_label: str = "AI prompted lesion"
    publish_to_orthanc: bool = True
    # Optional active OHIF VOI. This is essential for structures such as lung nodules:
    # a CT may store a mediastinal default even while the user is viewing a lung window.
    window_center: float | None = None
    window_width: float | None = None


class SegmentVolumeSlice(BaseModel):
    sop_instance_uid: str
    mask_rle: dict


class SegmentVolumeResponse(BaseModel):
    seed_sop_instance_uid: str
    masks: list[SegmentVolumeSlice]
    source_slice_count: int
    segmented_slice_count: int
    voxel_count: int
    volume_ml: float
    axial_bbox_diagonal_mm: float
    model_version: str
    dicom_seg_series_instance_uid: str
    dicom_seg_sop_instance_uid: str
    orthanc_status: str
    warning: str | None = None


class AnalysisResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    study_id: str
    series_id: str | None
    status: str
    pipeline_analysis_id: str | None
    error: str | None
    report_text: str | None
    verification_ok: bool | None
    triage_urgency: str | None
    created_at: datetime
    updated_at: datetime
    findings: list[FindingResponse]
    recommendations: list[RecommendationResponse]
    expert_executions: list[ExpertExecutionResponse]
