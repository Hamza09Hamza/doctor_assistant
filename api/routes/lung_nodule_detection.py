"""Synchronous, viewer-oriented lung-nodule candidate detection.

Unlike the generic analysis pipeline, this endpoint preserves the detector's spatial
output and maps every candidate directly to a source DICOM slice and clipped pixel box.
The model remains a candidate detector, not a diagnosis: an empty list only means no
candidate cleared the configured operating threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from core.enums import BodyPart, Modality

from ..lung_nodule_detection import (
    DicomSliceGeometry,
    load_dicom_slice_geometries,
    project_lps_aabb_to_nearest_slice,
)
from ..lung_nodule_cache import (
    CACHE_SCHEMA_VERSION,
    cache_directory,
    read_entry,
    read_latest,
    write_entry,
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


def _sha256_json(value) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _source_fingerprint(
    *,
    series_instance_uid: str,
    slices: Sequence[DicomSliceGeometry],
) -> str:
    """Fingerprint source identity and geometry without reading CT pixel data."""

    geometry = [
        {
            "sop_instance_uid": item.sop_instance_uid,
            "image_position_lps_mm": item.image_position_lps_mm,
            "image_orientation_patient": item.image_orientation_patient,
            "pixel_spacing_mm": item.pixel_spacing_mm,
            "rows": item.rows,
            "columns": item.columns,
            "frame_of_reference_uid": item.frame_of_reference_uid,
        }
        for item in sorted(slices, key=lambda value: value.sop_instance_uid)
    ]
    source_identity = {
        "series_instance_uid": series_instance_uid,
        "slices": geometry,
    }
    return f"sha256:{_sha256_json(source_identity)}"


def _model_identity(detector, *, model_version: str) -> dict:
    """Stable detector/preprocess identity without hashing multi-GB checkpoint bytes.

    Every expert exposes its version; a more specialized detector may additionally
    expose an explicit checkpoint fingerprint or preprocessing version.  Version is
    the documented fallback when it exposes no stronger identity.  Values are
    normalized to strings because only identity, never a filesystem read or secret
    value, is needed in the cache contract.
    """

    identity = {"model_version": model_version}
    for attribute in (
        "model_fingerprint",
        "preprocessing_version",
        "preprocess_version",
        "checkpoint_sha256",
        "bundle_version",
    ):
        value = getattr(detector, attribute, None)
        if value is not None:
            identity[attribute] = str(value)
    return identity


def _model_fingerprint(detector, *, model_version: str) -> str:
    return f"sha256:{_sha256_json(_model_identity(detector, model_version=model_version))}"


def _cache_key(*, source_fingerprint: str, model_fingerprint: str) -> str:
    return _sha256_json(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "model_fingerprint": model_fingerprint,
        }
    )


def _candidate_id(
    *,
    source_fingerprint: str,
    model_fingerprint: str,
    candidate: Mapping,
    ordinal: int,
) -> str:
    digest = _sha256_json(
        {
            "source_fingerprint": source_fingerprint,
            "model_fingerprint": model_fingerprint,
            "score": candidate["score"],
            "center_lps_mm": candidate["center_lps_mm"],
            "size_whd_mm": candidate["size_whd_mm"],
            "seed_sop_instance_uid": candidate["seed_sop_instance_uid"],
            "box_xyxy": candidate["box_xyxy"],
            "ordinal": ordinal,
        }
    )
    return f"candidate-{digest[:24]}"


def _results_from_raw(
    *,
    raw_detections: Sequence,
    slices: Sequence[DicomSliceGeometry],
    source_fingerprint: str,
    model_fingerprint: str,
    min_score: float,
) -> list[LungNoduleDetectionResponse]:
    projected_results: list[dict] = []
    for detection in raw_detections:
        if not isinstance(detection, Mapping):
            raise ValueError("each detector candidate must be an object")
        score = float(detection["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("detector score must be finite and between 0 and 1")
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
        projected_results.append(
            {
                "score": score,
                "center_lps_mm": center,
                "size_whd_mm": size,
                "seed_sop_instance_uid": projected.seed_sop_instance_uid,
                "box_xyxy": projected.box_xyxy,
            }
        )

    projected_results.sort(key=lambda item: item["score"], reverse=True)
    return [
        LungNoduleDetectionResponse(
            candidate_id=_candidate_id(
                source_fingerprint=source_fingerprint,
                model_fingerprint=model_fingerprint,
                candidate=item,
                ordinal=index,
            ),
            score=item["score"],
            center_lps_mm=item["center_lps_mm"],
            size_whd_mm=item["size_whd_mm"],
            seed_sop_instance_uid=item["seed_sop_instance_uid"],
            box_xyxy=item["box_xyxy"],
        )
        for index, item in enumerate(projected_results)
        if item["score"] >= min_score
    ]


def _response_from_raw_entry(
    entry: Mapping,
    *,
    series: Series,
    slices: Sequence[DicomSliceGeometry],
    source_fingerprint: str,
    model_fingerprint: str,
    cache_key: str,
    model_version: str,
    min_score: float,
) -> DetectLungNodulesResponse | None:
    raw_detections = entry.get("raw_detections")
    response = entry.get("response")
    if (
        not isinstance(raw_detections, Sequence)
        or isinstance(raw_detections, (str, bytes))
        or not isinstance(response, Mapping)
    ):
        return None
    try:
        results = _results_from_raw(
            raw_detections=raw_detections,
            slices=slices,
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            min_score=min_score,
        )
        return DetectLungNodulesResponse(
            series_id=series.id,
            model_version=model_version,
            run_id=str(response["run_id"]),
            cache_status="hit",
            elapsed_ms=int(response["elapsed_ms"]),
            generated_at=response["generated_at"],
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            cache_key=cache_key,
            min_score=min_score,
            source_slice_count=len(slices),
            detections=results,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_series_and_slices(series_id: str, db: Session):
    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    if series.modality != Modality.CT.value or series.body_part != BodyPart.CHEST.value:
        raise HTTPException(
            status_code=422,
            detail="lung nodule detection requires a CT chest series",
        )
    try:
        slices = load_dicom_slice_geometries(
            Path(series.storage_dir),
            series_instance_uid=series.dicom_series_uid,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return series, slices


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

    series, slices = _load_series_and_slices(series_id, db)
    min_score = _effective_threshold(detector, payload.min_score)
    model_version = str(getattr(detector, "version", "unknown"))
    source_fingerprint = _source_fingerprint(
        series_instance_uid=series.dicom_series_uid,
        slices=slices,
    )
    model_fingerprint = _model_fingerprint(detector, model_version=model_version)
    cache_key = _cache_key(
        source_fingerprint=source_fingerprint,
        model_fingerprint=model_fingerprint,
    )
    cache_dir = cache_directory(Path(series.storage_dir))
    if not payload.force:
        cached = read_entry(cache_dir, cache_key)
        if (
            cached is not None
            and cached.get("series_id") == series.id
            and cached.get("source_fingerprint") == source_fingerprint
            and cached.get("model_fingerprint") == model_fingerprint
        ):
            response = _response_from_raw_entry(
                cached,
                series=series,
                slices=slices,
                source_fingerprint=source_fingerprint,
                model_fingerprint=model_fingerprint,
                cache_key=cache_key,
                model_version=model_version,
                min_score=min_score,
            )
            if response is not None:
                return response

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
    started_at = time.perf_counter()
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

    try:
        results = _results_from_raw(
            raw_detections=raw_detections,
            slices=slices,
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            min_score=min_score,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"lung nodule detector returned invalid candidate geometry: {exc}",
        ) from exc

    generated_at = datetime.now(timezone.utc)
    response = DetectLungNodulesResponse(
        series_id=series.id,
        model_version=model_version,
        run_id=uuid.uuid4().hex,
        cache_status="miss",
        elapsed_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
        generated_at=generated_at,
        source_fingerprint=source_fingerprint,
        model_fingerprint=model_fingerprint,
        cache_key=cache_key,
        min_score=min_score,
        source_slice_count=len(slices),
        detections=results,
    )
    normalized_raw = [
        {
            "score": float(item["score"]),
            "center_lps_mm": list(
                _three_finite_numbers(item["center_lps_mm"], name="center_lps_mm")
            ),
            "size_whd_mm": list(
                _three_finite_numbers(item["size_whd_mm"], name="size_whd_mm")
            ),
        }
        for item in raw_detections
    ]
    entry = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "series_id": series.id,
        "source_fingerprint": source_fingerprint,
        "model_fingerprint": model_fingerprint,
        "raw_detections": normalized_raw,
        "response": response.model_dump(mode="json"),
    }
    try:
        write_entry(cache_dir, cache_key, entry)
    except OSError as exc:
        logger.warning("Could not cache lung nodule run %s: %s", response.run_id, exc)
    return response


@router.get(
    "/{series_id}/detect-lung-nodules/latest",
    response_model=DetectLungNodulesResponse,
)
def latest_lung_nodule_detection(
    series_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> DetectLungNodulesResponse:
    """Return the newest durable run for the current source and detector."""

    detector = _configured_detector(request)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="lung nodule detection is not configured (set LUNG_NODULE_BUNDLE_ROOT)",
        )
    series, slices = _load_series_and_slices(series_id, db)
    model_version = str(getattr(detector, "version", "unknown"))
    model_fingerprint = _model_fingerprint(detector, model_version=model_version)
    min_score = _effective_threshold(detector, None)
    source_fingerprint = _source_fingerprint(
        series_instance_uid=series.dicom_series_uid,
        slices=slices,
    )
    cached = read_latest(
        cache_directory(Path(series.storage_dir)),
        series_id=series.id,
        source_fingerprint=source_fingerprint,
        model_fingerprint=model_fingerprint,
    )
    if cached is not None:
        response = _response_from_raw_entry(
            cached,
            series=series,
            slices=slices,
            source_fingerprint=source_fingerprint,
            model_fingerprint=model_fingerprint,
            cache_key=_cache_key(
                source_fingerprint=source_fingerprint,
                model_fingerprint=model_fingerprint,
            ),
            model_version=model_version,
            min_score=min_score,
        )
        if response is not None:
            return response
    raise HTTPException(
        status_code=404,
        detail="no cached lung nodule detection exists for the current source series",
    )
