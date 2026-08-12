"""DICOM geometry helpers for viewer-ready lung-nodule candidates.

The MONAI detector reports a world-aligned bounding box in patient LPS millimetres.
OHIF's prompted-segmentation API needs a source SOP Instance UID and an ``xyxy`` box
in that source image's pixels.  Keeping this conversion separate from the model and
HTTP route makes the clinically important coordinate math deterministic and testable
without Torch, MONAI, or pixel-data decoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class DicomSliceGeometry:
    """The DICOM attributes required to map LPS coordinates onto one source image."""

    sop_instance_uid: str
    image_position_lps_mm: Vector3
    image_orientation_patient: tuple[float, float, float, float, float, float]
    pixel_spacing_mm: tuple[float, float]
    rows: int
    columns: int
    frame_of_reference_uid: str | None = None


@dataclass(frozen=True)
class ProjectedDetection:
    """A source-image prompt suitable for OHIF and the MedSAM2 volume endpoint."""

    seed_sop_instance_uid: str
    box_xyxy: tuple[int, int, int, int]


def _numbers(value, *, count: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DICOM {name} must contain {count} numeric values") from exc
    if len(result) != count or not all(math.isfinite(item) for item in result):
        raise ValueError(f"DICOM {name} must contain {count} finite numeric values")
    return result


def dicom_slice_geometry(dataset) -> DicomSliceGeometry:
    """Extract and validate geometry from a single-frame DICOM image dataset."""

    sop_uid = str(getattr(dataset, "SOPInstanceUID", "")).strip()
    if not sop_uid:
        raise ValueError("DICOM image is missing SOPInstanceUID")
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("DICOM image is missing valid Rows or Columns") from exc
    if rows <= 0 or columns <= 0:
        raise ValueError("DICOM Rows and Columns must be positive")

    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1))
    if number_of_frames != 1:
        raise ValueError("enhanced multi-frame DICOM geometry is not supported")

    position = _numbers(
        getattr(dataset, "ImagePositionPatient", None),
        count=3,
        name="ImagePositionPatient",
    )
    orientation = _numbers(
        getattr(dataset, "ImageOrientationPatient", None),
        count=6,
        name="ImageOrientationPatient",
    )
    spacing = _numbers(
        getattr(dataset, "PixelSpacing", None),
        count=2,
        name="PixelSpacing",
    )
    if spacing[0] <= 0 or spacing[1] <= 0:
        raise ValueError("DICOM PixelSpacing values must be positive")

    # Validate the orientation here so malformed slices fail before model inference.
    column_index_direction = _unit(
        orientation[:3], name="first ImageOrientationPatient vector"
    )
    row_index_direction = _unit(
        orientation[3:], name="second ImageOrientationPatient vector"
    )
    if abs(_dot(column_index_direction, row_index_direction)) > 1e-3:
        raise ValueError("DICOM ImageOrientationPatient vectors must be orthogonal")

    frame_uid = str(getattr(dataset, "FrameOfReferenceUID", "")).strip() or None
    return DicomSliceGeometry(
        sop_instance_uid=sop_uid,
        image_position_lps_mm=(position[0], position[1], position[2]),
        image_orientation_patient=(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
            orientation[4],
            orientation[5],
        ),
        pixel_spacing_mm=(spacing[0], spacing[1]),
        rows=rows,
        columns=columns,
        frame_of_reference_uid=frame_uid,
    )


def load_dicom_slice_geometries(
    storage_dir: Path,
    *,
    series_instance_uid: str,
) -> tuple[DicomSliceGeometry, ...]:
    """Read only source headers and return complete single-frame slice geometry."""

    import pydicom

    geometries: list[DicomSliceGeometry] = []
    matching_instances = 0
    for path in sorted(storage_dir.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:  # noqa: BLE001 - unrelated/corrupt files are not source images
            continue
        if str(getattr(dataset, "SeriesInstanceUID", "")) != series_instance_uid:
            continue
        matching_instances += 1
        try:
            geometries.append(dicom_slice_geometry(dataset))
        except ValueError as exc:
            raise ValueError(f"invalid geometry in {path.name}: {exc}") from exc

    if matching_instances == 0:
        raise FileNotFoundError(
            f"no image instances for DICOM series {series_instance_uid!r}"
        )
    frame_uids = {
        item.frame_of_reference_uid
        for item in geometries
        if item.frame_of_reference_uid is not None
    }
    if len(frame_uids) > 1:
        raise ValueError("DICOM series contains multiple FrameOfReferenceUID values")
    return tuple(geometries)


def project_lps_aabb_to_nearest_slice(
    slices: Sequence[DicomSliceGeometry],
    *,
    center_lps_mm: Sequence[float],
    size_whd_mm: Sequence[float],
) -> ProjectedDetection | None:
    """Project a world-aligned LPS AABB onto the nearest DICOM source slice.

    DICOM's first ImageOrientationPatient vector points along increasing *column*
    index (the direction of a displayed image row), while its second vector points
    along increasing *row* index.  Projecting all eight world-AABB corners onto both
    vectors is required for oblique acquisitions: simply dividing the LPS X/Y extents
    by PixelSpacing is only correct for unrotated axial images.

    The returned ``x1``/``y1`` bounds are exclusive, matching the existing MedSAM box
    contract.  ``None`` means the projected box does not overlap a usable two-pixel
    region of the selected image.
    """

    if not slices:
        raise ValueError("at least one DICOM slice is required")
    center_values = _numbers(center_lps_mm, count=3, name="detection center_lps_mm")
    size_values = _numbers(size_whd_mm, count=3, name="detection size_whd_mm")
    if any(value <= 0 for value in size_values):
        raise ValueError("detection size_whd_mm values must be positive")
    center: Vector3 = (center_values[0], center_values[1], center_values[2])

    def plane_distance(item: DicomSliceGeometry) -> float:
        _column_direction, _row_direction, normal = _slice_axes(item)
        return abs(_dot(_subtract(center, item.image_position_lps_mm), normal))

    selected = min(slices, key=plane_distance)
    column_index_direction, row_index_direction, _normal = _slice_axes(selected)
    row_spacing, column_spacing = selected.pixel_spacing_mm
    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("DICOM PixelSpacing values must be positive")
    if selected.rows <= 0 or selected.columns <= 0:
        raise ValueError("DICOM Rows and Columns must be positive")

    half = tuple(value / 2.0 for value in size_values)
    corners = [
        (
            center[0] + sign_x * half[0],
            center[1] + sign_y * half[1],
            center[2] + sign_z * half[2],
        )
        for sign_x, sign_y, sign_z in product((-1.0, 1.0), repeat=3)
    ]
    projected_columns = []
    projected_rows = []
    for corner in corners:
        offset = _subtract(corner, selected.image_position_lps_mm)
        projected_columns.append(_dot(offset, column_index_direction) / column_spacing)
        projected_rows.append(_dot(offset, row_index_direction) / row_spacing)

    x0 = max(0, min(math.floor(min(projected_columns)), selected.columns - 1))
    x1 = max(0, min(math.ceil(max(projected_columns)), selected.columns))
    y0 = max(0, min(math.floor(min(projected_rows)), selected.rows - 1))
    y1 = max(0, min(math.ceil(max(projected_rows)), selected.rows))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return ProjectedDetection(
        seed_sop_instance_uid=selected.sop_instance_uid,
        box_xyxy=(x0, y0, x1, y1),
    )


def _slice_axes(geometry: DicomSliceGeometry) -> tuple[Vector3, Vector3, Vector3]:
    orientation = geometry.image_orientation_patient
    column_index_direction = _unit(
        orientation[:3], name="first ImageOrientationPatient vector"
    )
    row_index_direction = _unit(
        orientation[3:], name="second ImageOrientationPatient vector"
    )
    if abs(_dot(column_index_direction, row_index_direction)) > 1e-3:
        raise ValueError("DICOM ImageOrientationPatient vectors must be orthogonal")
    normal = _unit(
        _cross(column_index_direction, row_index_direction),
        name="ImageOrientationPatient slice normal",
    )
    return column_index_direction, row_index_direction, normal


def _subtract(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[0]) - float(right[0]),
        float(left[1]) - float(right[1]),
        float(left[2]) - float(right[2]),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _cross(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _unit(vector: Sequence[float], *, name: str) -> Vector3:
    norm = math.sqrt(_dot(vector, vector))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"{name} must be non-zero and finite")
    return (
        float(vector[0]) / norm,
        float(vector[1]) / norm,
        float(vector[2]) / norm,
    )
