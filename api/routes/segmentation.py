"""On-demand box-prompted 2D and full-volume segmentation routes.

Deliberately not the `BackgroundTasks`-queued `Pipeline.analyze` pattern: the viewer is
waiting mid-interaction for masks it can accept/reject immediately, and these prompted
results do not become automatic `Analysis`/`Finding` rows. The 2D route is transient;
the full-volume route additionally persists its accepted model output as DICOM SEG.
"""

from __future__ import annotations

import logging
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from experts.medsam_interactive import encode_binary_mask_rle

from ..orthanc_client import OrthancClient
from ..models import Series
from ..schemas import (
    SegmentBoxRequest,
    SegmentBoxResponse,
    SegmentVolumeRequest,
    SegmentVolumeResponse,
    SegmentVolumeSlice,
)
from ..volume_segmentation import (
    build_interactive_dicom_seg,
    load_dicom_volume,
    measure_volume,
)
from .deps import get_db

router = APIRouter(prefix="/v1/series", tags=["segmentation"])
logger = logging.getLogger(__name__)


def find_instance_file(storage_dir: Path, sop_instance_uid: str) -> Path:
    """Locate the one file in a series' storage directory carrying this SOPInstanceUID.

    Series are stored as plain `instance_NNNN.dcm` files (see `api/dicom_ingest.py`) with
    no filename-to-UID index kept on disk, so this reads headers only (`stop_before_pixels`)
    until it finds a match rather than loading full pixel data for every candidate.
    """
    import pydicom

    for path in sorted(storage_dir.glob("*.dcm")):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:  # noqa: BLE001 — a non-DICOM/corrupt file just isn't a match
            continue
        if getattr(header, "SOPInstanceUID", None) == sop_instance_uid:
            return path
    raise FileNotFoundError(
        f"no instance with SOPInstanceUID={sop_instance_uid!r} under {storage_dir}"
    )


def displayable_slice(dataset) -> "object":
    """A DICOM instance's pixel data as a uint8 (H,W) image, windowed the same way the
    viewer would show it: real-world values via Modality LUT, then a 1st/99th-percentile
    stretch to 0-255.

    Percentile rather than literal min/max, and applied after RescaleSlope/Intercept, not
    before — both are lessons this project already paid for once in `nifti_to_dicom.py`
    (see notebooks/HANDOFF.md section 3.2): stored-pixel-unit windows and literal-min/max
    stretches both produced visibly wrong images against real data.
    """
    import numpy as np

    arr = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    real = arr * slope + intercept

    p_lo, p_hi = np.percentile(real, (1.0, 99.0))
    if p_hi <= p_lo:
        p_lo, p_hi = float(real.min()), float(real.max() + 1.0)
    stretched = np.clip((real - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


@router.post("/{series_id}/segment-box", response_model=SegmentBoxResponse)
def segment_box(
    series_id: str,
    payload: SegmentBoxRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> SegmentBoxResponse:
    segmenter = getattr(request.app.state, "medsam", None)
    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="interactive segmentation is not configured on this deployment "
            "(set MEDSAM_CHECKPOINT_PATH)",
        )

    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")

    try:
        instance_path = find_instance_file(Path(series.storage_dir), payload.sop_instance_uid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    import pydicom

    dataset = pydicom.dcmread(str(instance_path))
    image = displayable_slice(dataset)

    try:
        mask = segmenter.segment_box(image, payload.box_xyxy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SegmentBoxResponse(
        sop_instance_uid=payload.sop_instance_uid,
        mask_rle=encode_binary_mask_rle(mask),
        model_version=segmenter.version,
    )


@router.post("/{series_id}/segment-volume", response_model=SegmentVolumeResponse)
def segment_volume(
    series_id: str,
    payload: SegmentVolumeRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> SegmentVolumeResponse:
    """Propagate one slice box through a DICOM stack and persist a DICOM SEG.

    Sparse RLE masks are returned for immediate OHIF painting.  The standards-based
    DICOM SEG is always stored under the imported series' ``derived`` directory and is
    also sent to Orthanc unless the caller explicitly disables publication.
    """
    segmenter = getattr(request.app.state, "medsam2", None)
    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="3D interactive segmentation is not configured (use "
            "MEDSAM2_BACKEND=mlx on Apple Silicon, or configure the Torch MedSAM2 checkpoint)",
        )
    try:
        import highdicom  # noqa: F401 - preflight before expensive model inference
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="highdicom is required to persist the MedSAM2 result as DICOM SEG",
        ) from exc

    series = db.get(Series, series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series {series_id!r}")
    label = payload.segment_label.strip()
    if not label:
        raise HTTPException(status_code=422, detail="segment_label cannot be empty")

    inference_lock = getattr(request.app.state, "medsam2_lock", None)
    if inference_lock is not None and not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="another full-volume inference is already running; retry after it finishes",
        )
    try:
        return _segment_volume_unlocked(
            series=series,
            payload=payload,
            request=request,
            segmenter=segmenter,
            label=label,
        )
    finally:
        if inference_lock is not None:
            inference_lock.release()


def _segment_volume_unlocked(
    *,
    series: Series,
    payload: SegmentVolumeRequest,
    request: Request,
    segmenter,
    label: str,
) -> SegmentVolumeResponse:
    """Execute one resource-heavy request after the caller owns the inference slot."""
    try:
        source = load_dicom_volume(
            Path(series.storage_dir),
            series_instance_uid=series.dicom_series_uid,
            seed_sop_instance_uid=payload.sop_instance_uid,
            window_center=payload.window_center,
            window_width=payload.window_width,
        )
        mask = segmenter.segment_volume(
            source.display_volume,
            source.seed_index,
            payload.box_xyxy,
        )
        measurements = measure_volume(mask, source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    derived_dir = Path(series.storage_dir) / "derived"
    # Keep the artifact name backend-neutral: on Apple Silicon this may come from
    # stock SAM2 MLX or a converted MedSAM2 checkpoint, while CUDA uses MedSAM2.
    seg_path = derived_dir / f"prompted3d_{uuid.uuid4().hex}.dcm"
    try:
        seg_series_uid, seg_sop_uid = build_interactive_dicom_seg(
            mask,
            source,
            seg_path,
            segment_label=label,
            model_version=segmenter.version,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"DICOM SEG creation failed: {exc}") from exc

    orthanc_status = "disabled"
    warning = None
    publication_disabled = bool(
        getattr(request.app.state, "disable_orthanc_publication", False)
    )
    if payload.publish_to_orthanc and not publication_disabled:
        try:
            with OrthancClient(request.app.state.orthanc_config) as orthanc:
                orthanc.upload_instance(seg_path.read_bytes())
            orthanc_status = "published"
        except Exception as exc:  # noqa: BLE001 - preserve the completed local result
            orthanc_status = "failed"
            warning = (
                "The 3D mask and DICOM SEG were created, but Orthanc publication failed: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning("Prompted 3D DICOM SEG publication failed: %s", warning)

    masks = [
        SegmentVolumeSlice(
            sop_instance_uid=sop_uid,
            mask_rle=encode_binary_mask_rle(frame),
        )
        for sop_uid, frame in zip(source.sop_instance_uids, mask, strict=True)
        if frame.any()
    ]
    return SegmentVolumeResponse(
        seed_sop_instance_uid=payload.sop_instance_uid,
        masks=masks,
        source_slice_count=len(source.sop_instance_uids),
        segmented_slice_count=measurements.segmented_slice_count,
        voxel_count=measurements.voxel_count,
        volume_ml=measurements.volume_ml,
        axial_bbox_diagonal_mm=measurements.axial_bbox_diagonal_mm,
        model_version=segmenter.version,
        dicom_seg_series_instance_uid=seg_series_uid,
        dicom_seg_sop_instance_uid=seg_sop_uid,
        orthanc_status=orthanc_status,
        warning=warning,
    )
