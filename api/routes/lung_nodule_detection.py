"""Synchronous, viewer-oriented lung-nodule candidate detection.

Unlike the generic analysis pipeline, this endpoint preserves the detector's spatial
output and maps every candidate directly to a source DICOM slice and clipped pixel box.
The model remains a candidate detector, not a diagnosis: an empty list only means no
candidate cleared the configured operating threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from core.enums import BodyPart, Modality

from ..lung_nodule_detection import (
    load_dicom_slice_geometries,
    project_lps_aabb_to_nearest_slice,
)
from ..models import Series
from ..schemas import (
    DetectLungNodulesRequest,
    DetectLungNodulesResponse,
    LungNoduleDetectionResponse,
)
from .deps import get_db

router = APIRouter(prefix="/v1/series", tags=["lung-nodule-detection"])
logger = logging.getLogger(__name__)


def _configured_detector(request: Request):
    detector = getattr(request.app.state, "lung_nodule_detector", None)
    if detector is not None:
        return detector
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return None
    return next(
        (
            expert
            for expert in registry.experts()
            if getattr(expert, "name", None) == "ct_lung_nodule"
        ),
        None,
    )


def _three_finite_numbers(value, *, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detector {name} must contain three numeric values") from exc
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"detector {name} must contain three finite numeric values")
    return values[0], values[1], values[2]


def _effective_threshold(detector, requested: float | None) -> float:
    try:
        configured = float(detector.min_score)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="lung nodule detector has no valid configured min_score",
        ) from exc
    if not math.isfinite(configured) or not 0.0 <= configured <= 1.0:
        raise HTTPException(
            status_code=503,
            detail="lung nodule detector configured min_score must be between 0 and 1",
        )
    return configured if requested is None else max(configured, requested)


@router.post(
    "/{series_id}/detect-lung-nodules",
    response_model=DetectLungNodulesResponse,
)
def detect_lung_nodules(
    series_id: str,
    payload: DetectLungNodulesRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> DetectLungNodulesResponse:
    detector = _configured_detector(request)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="lung nodule detection is not configured (set LUNG_NODULE_BUNDLE_ROOT)",
        )

    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    if series.modality != Modality.CT.value or series.body_part != BodyPart.CHEST.value:
        raise HTTPException(
            status_code=422,
            detail="lung nodule detection requires a CT chest series",
        )

    min_score = _effective_threshold(detector, payload.min_score)
    try:
        slices = load_dicom_slice_geometries(
            Path(series.storage_dir),
            series_instance_uid=series.dicom_series_uid,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    inference_lock = getattr(
        request.app.state,
        "inference_lock",
        getattr(request.app.state, "medsam2_lock", None),
    )
    if inference_lock is not None and not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="another inference is already running; retry after it finishes",
        )
    try:
        raw_detections = detector.detect(Path(series.storage_dir))
    except Exception as exc:  # noqa: BLE001 - return a JSON error to the remote OHIF client
        logger.exception("Lung nodule detection failed for series %s", series_id)
        raise HTTPException(
            status_code=500,
            detail=f"lung nodule detection failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if inference_lock is not None:
            inference_lock.release()

    if not isinstance(raw_detections, Sequence) or isinstance(
        raw_detections, (str, bytes)
    ):
        raise HTTPException(status_code=500, detail="lung nodule detector returned invalid output")

    results: list[LungNoduleDetectionResponse] = []
    try:
        for detection in raw_detections:
            if not isinstance(detection, Mapping):
                raise ValueError("each detector candidate must be an object")
            score = float(detection["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("detector score must be finite and between 0 and 1")
            if score < min_score:
                continue
            center = _three_finite_numbers(
                detection["center_lps_mm"], name="center_lps_mm"
            )
            size = _three_finite_numbers(detection["size_whd_mm"], name="size_whd_mm")
            if any(value <= 0 for value in size):
                raise ValueError("detector size_whd_mm values must be positive")
            projected = project_lps_aabb_to_nearest_slice(
                slices,
                center_lps_mm=center,
                size_whd_mm=size,
            )
            if projected is None:
                continue
            results.append(
                LungNoduleDetectionResponse(
                    score=score,
                    center_lps_mm=center,
                    size_whd_mm=size,
                    seed_sop_instance_uid=projected.seed_sop_instance_uid,
                    box_xyxy=projected.box_xyxy,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"lung nodule detector returned invalid candidate geometry: {exc}",
        ) from exc

    results.sort(key=lambda item: item.score, reverse=True)
    return DetectLungNodulesResponse(
        series_id=series.id,
        model_version=str(getattr(detector, "version", "unknown")),
        min_score=min_score,
        source_slice_count=len(slices),
        detections=results,
    )
