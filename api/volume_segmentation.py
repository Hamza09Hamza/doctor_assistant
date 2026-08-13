"""DICOM-series preparation and output for interactive 3D segmentation.

Every array in this module is ordered by an explicit list of source datasets.  That
identity-preserving rule matters more than any assumed ascending/descending stack
direction: the API response maps masks back by SOP Instance UID and highdicom receives
the mask frames in the exact same order as ``source_images``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# A free-form box prompt says only "segment the selected structure". It does not
# establish pathology. DICOM CID 7150 permits Tissue as a generic segmentation
# category; its linked current property-type group CID 7191 includes CID 7166, where
# Tissue is the corresponding generic type.
# Keep these explicit and test-visible so this path can never silently regress to the
# clinically stronger Morphologically Abnormal Structure / Lesion assertion.
PROMPTED_STRUCTURE_CATEGORY_CODE = ("85756007", "SCT", "Tissue")
PROMPTED_STRUCTURE_TYPE_CODE = ("85756007", "SCT", "Tissue")


@dataclass(frozen=True)
class LoadedDicomVolume:
    datasets: tuple[object, ...]
    display_volume: object
    seed_index: int
    sop_instance_uids: tuple[str, ...]
    row_spacing_mm: float
    column_spacing_mm: float
    slice_spacing_mm: float
    slice_coordinates_mm: tuple[float, ...]


@dataclass(frozen=True)
class VolumeMeasurements:
    voxel_count: int
    volume_ml: float
    # Maximum diagonal of an axis-aligned 2D bounding box on any axial slice.
    # This is deliberately not called a lesion long axis or "axial span".
    axial_bbox_diagonal_mm: float
    craniocaudal_extent_mm: float
    segmented_slice_count: int


def _first_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        candidate = value
    else:
        try:
            candidate = value[0]
        except (IndexError, KeyError, TypeError):
            candidate = value
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _spatial_coordinate(dataset) -> float | None:
    import numpy as np

    orientation = getattr(dataset, "ImageOrientationPatient", None)
    position = getattr(dataset, "ImagePositionPatient", None)
    if orientation is None or position is None or len(orientation) != 6 or len(position) != 3:
        return None
    row_direction = np.asarray(orientation[:3], dtype=float)
    column_direction = np.asarray(orientation[3:], dtype=float)
    normal = np.cross(row_direction, column_direction)
    if float(np.linalg.norm(normal)) < 1e-6:
        return None
    return float(np.dot(np.asarray(position, dtype=float), normal))


def _sort_datasets(datasets: list) -> list:
    spatial = [_spatial_coordinate(dataset) for dataset in datasets]
    if all(value is not None for value in spatial) and len(set(round(v, 5) for v in spatial)) == len(spatial):
        return [
            dataset
            for _position, dataset in sorted(
                zip(spatial, datasets, strict=True), key=lambda pair: pair[0]
            )
        ]

    def fallback_key(dataset) -> tuple[int, str]:
        try:
            instance_number = int(getattr(dataset, "InstanceNumber", 0))
        except (TypeError, ValueError):
            instance_number = 0
        return instance_number, str(getattr(dataset, "SOPInstanceUID", ""))

    return sorted(datasets, key=fallback_key)


def _slice_spacing(datasets: list) -> float:
    import numpy as np

    positions = [_spatial_coordinate(dataset) for dataset in datasets]
    if len(positions) > 1 and all(position is not None for position in positions):
        diffs = np.abs(np.diff(np.asarray(positions, dtype=float)))
        positive = diffs[diffs > 1e-6]
        if positive.size:
            return float(np.median(positive))

    for attribute in ("SpacingBetweenSlices", "SliceThickness"):
        value = _first_number(getattr(datasets[0], attribute, None))
        if value is not None and value > 0:
            return abs(value)
    return 1.0


def _display_volume(
    datasets: list,
    *,
    window_center: float | None = None,
    window_width: float | None = None,
):
    """Convert stored pixels to a consistently-windowed uint8 ``(z,y,x)`` volume."""
    import numpy as np

    frames = []
    for dataset in datasets:
        pixels = dataset.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError("interactive 3D segmentation currently requires single-frame slices")
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        frames.append(pixels * slope + intercept)
    volume = np.stack(frames, axis=0)

    center = window_center
    width = window_width
    if center is None or width is None:
        center = _first_number(getattr(datasets[0], "WindowCenter", None))
        width = _first_number(getattr(datasets[0], "WindowWidth", None))
    if center is not None and width is not None and width > 1:
        low, high = center - width / 2.0, center + width / 2.0
    else:
        low, high = (float(value) for value in np.percentile(volume, (1.0, 99.0)))
    if high <= low:
        low, high = float(volume.min()), float(volume.max())
    if high <= low:
        high = low + 1.0
    return (np.clip((volume - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def load_dicom_volume(
    storage_dir: Path,
    *,
    series_instance_uid: str,
    seed_sop_instance_uid: str,
    window_center: float | None = None,
    window_width: float | None = None,
) -> LoadedDicomVolume:
    """Load one imported single-frame DICOM series and locate the prompt slice."""
    import pydicom

    datasets = []
    for path in sorted(storage_dir.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(str(path))
        except Exception:  # noqa: BLE001 - ignore unrelated/corrupt files in storage
            continue
        if str(getattr(dataset, "SeriesInstanceUID", "")) != series_instance_uid:
            continue
        if not getattr(dataset, "SOPInstanceUID", None) or not hasattr(dataset, "PixelData"):
            continue
        datasets.append(dataset)

    if not datasets:
        raise FileNotFoundError(f"no image instances for DICOM series {series_instance_uid!r}")
    datasets = _sort_datasets(datasets)

    rows_columns = {
        (int(getattr(dataset, "Rows", 0)), int(getattr(dataset, "Columns", 0)))
        for dataset in datasets
    }
    if len(rows_columns) != 1 or (0, 0) in rows_columns:
        raise ValueError("all DICOM slices must have the same non-zero Rows and Columns")
    frame_uids = {
        str(getattr(dataset, "FrameOfReferenceUID", "")) for dataset in datasets
        if getattr(dataset, "FrameOfReferenceUID", None)
    }
    if len(frame_uids) > 1:
        raise ValueError("DICOM series contains multiple FrameOfReferenceUID values")

    sop_uids = tuple(str(dataset.SOPInstanceUID) for dataset in datasets)
    try:
        seed_index = sop_uids.index(seed_sop_instance_uid)
    except ValueError as exc:
        raise FileNotFoundError(
            f"no instance with SOPInstanceUID={seed_sop_instance_uid!r} in series"
        ) from exc

    spacing = getattr(datasets[0], "PixelSpacing", (1.0, 1.0))
    if len(spacing) != 2:
        raise ValueError("DICOM PixelSpacing must contain row and column spacing")
    row_spacing, column_spacing = float(spacing[0]), float(spacing[1])
    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("DICOM PixelSpacing values must be positive")

    return LoadedDicomVolume(
        datasets=tuple(datasets),
        display_volume=_display_volume(
            datasets,
            window_center=window_center,
            window_width=window_width,
        ),
        seed_index=seed_index,
        sop_instance_uids=sop_uids,
        row_spacing_mm=row_spacing,
        column_spacing_mm=column_spacing,
        slice_spacing_mm=_slice_spacing(datasets),
        slice_coordinates_mm=tuple(
            coordinate
            for dataset in datasets
            if (coordinate := _spatial_coordinate(dataset)) is not None
        ),
    )


def measure_volume(mask, source: LoadedDicomVolume) -> VolumeMeasurements:
    import numpy as np

    arr = np.asarray(mask, dtype=bool)
    if arr.shape != source.display_volume.shape:
        raise ValueError(
            f"mask shape {arr.shape} does not match source volume {source.display_volume.shape}"
        )
    voxel_count = int(arr.sum())
    volume_ml = (
        voxel_count
        * source.row_spacing_mm
        * source.column_spacing_mm
        * source.slice_spacing_mm
        / 1000.0
    )
    max_diagonal = 0.0
    for frame in arr:
        rows, columns = np.where(frame)
        if not rows.size:
            continue
        row_span = (int(rows.max()) - int(rows.min()) + 1) * source.row_spacing_mm
        column_span = (int(columns.max()) - int(columns.min()) + 1) * source.column_spacing_mm
        max_diagonal = max(max_diagonal, float(np.hypot(row_span, column_span)))
    occupied_slice_indices = np.flatnonzero(arr.any(axis=(1, 2)))
    if not occupied_slice_indices.size:
        craniocaudal_extent_mm = 0.0
    elif len(source.slice_coordinates_mm) == arr.shape[0]:
        first = source.slice_coordinates_mm[int(occupied_slice_indices[0])]
        last = source.slice_coordinates_mm[int(occupied_slice_indices[-1])]
        # Add one representative slice thickness: coordinate difference measures
        # centre-to-centre extent, while occupied-mask extent includes both end slices.
        craniocaudal_extent_mm = abs(last - first) + source.slice_spacing_mm
    else:
        craniocaudal_extent_mm = (
            float(occupied_slice_indices[-1] - occupied_slice_indices[0] + 1)
            * source.slice_spacing_mm
        )
    return VolumeMeasurements(
        voxel_count=voxel_count,
        volume_ml=volume_ml,
        axial_bbox_diagonal_mm=max_diagonal,
        craniocaudal_extent_mm=craniocaudal_extent_mm,
        segmented_slice_count=int(np.count_nonzero(arr.any(axis=(1, 2)))),
    )


def build_interactive_dicom_seg(
    mask,
    source: LoadedDicomVolume,
    output_path: Path,
    *,
    segment_label: str,
    model_version: str,
) -> tuple[str, str]:
    """Encode a one-segment, source-referenced DICOM SEG and return its UIDs."""
    import highdicom as hd
    import numpy as np
    from highdicom import AlgorithmIdentificationSequence
    from highdicom.seg.content import SegmentDescription
    from highdicom.sr.coding import CodedConcept
    from pydicom.uid import generate_uid

    arr = np.asarray(mask, dtype=bool)
    if arr.shape != source.display_volume.shape or not arr.any():
        raise ValueError("cannot encode an empty or geometry-mismatched DICOM SEG mask")

    series_uid = generate_uid()
    sop_uid = generate_uid()
    algorithm_name = "SAM2 MLX" if model_version.startswith("sam2-mlx:") else "MedSAM2"
    algorithm = AlgorithmIdentificationSequence(
        name=algorithm_name,
        family=CodedConcept("113092", "DCM", "Deep Learning"),
        version=model_version[:64],
        source="doctor_assistant prompted 3D segmentation",
    )
    description = SegmentDescription(
        segment_number=1,
        segment_label=segment_label[:64],
        segmented_property_category=CodedConcept(*PROMPTED_STRUCTURE_CATEGORY_CODE),
        segmented_property_type=CodedConcept(*PROMPTED_STRUCTURE_TYPE_CODE),
        algorithm_type="SEMIAUTOMATIC",
        algorithm_identification=algorithm,
    )
    segmentation = hd.seg.Segmentation(
        source_images=list(source.datasets),
        pixel_array=arr,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[description],
        series_instance_uid=series_uid,
        series_number=910,
        sop_instance_uid=sop_uid,
        instance_number=1,
        series_description=f"{algorithm_name} prompted 3D segmentation",
        manufacturer="doctor_assistant",
        manufacturer_model_name=algorithm_name,
        software_versions=model_version[:64],
        device_serial_number="0",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation.save_as(str(output_path))
    return str(series_uid), str(sop_uid)
