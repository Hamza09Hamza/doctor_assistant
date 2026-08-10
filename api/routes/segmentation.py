"""`POST /v1/series/{id}/segment-box` — on-demand interactive 2D segmentation.

Deliberately NOT the `BackgroundTasks`-queued pattern `analyses.py`/`series.py` use for
`Pipeline.analyze`: MedSAM inference on one already-loaded 2D slice is sub-second, the
caller (the viewer, mid-interaction) needs the mask back in the same request, and no
`Analysis`/`Finding` DB rows make sense for a single exploratory click — the frontend
owns accept/reject of the result, nothing is persisted here.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from experts.medsam_interactive import encode_binary_mask_rle

from ..models import Series
from ..schemas import SegmentBoxRequest, SegmentBoxResponse
from .deps import get_db

router = APIRouter(prefix="/v1/series", tags=["segmentation"])


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
