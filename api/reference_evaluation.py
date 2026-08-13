"""Prompt-anchored comparison against staged reader DICOM SEG objects.

This module is intentionally independent of FastAPI and model code.  Reference
selection happens from the source-image identity and the user's seed prompt only;
the generated prediction is not accepted until :func:`compare_prediction` is called.
That ordering prevents the tempting but invalid practice of choosing whichever
radiologist annotation gives a model its highest Dice score.

All SEG frames are aligned by their explicitly referenced source SOP Instance UID.
Frame order, InstanceNumber, and spatial proximity are never used as substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


class ReferenceEvaluationError(ValueError):
    """A staged reference cannot be evaluated without weakening identity checks."""


@dataclass(frozen=True)
class PromptMatchedReader:
    reader_id: str
    segment_number: int | None
    segment_label: str | None
    prompt_overlap_voxels: int
    volume: Any | None = field(repr=False)

    @property
    def matched(self) -> bool:
        return self.volume is not None


@dataclass(frozen=True)
class PromptReferenceSelection:
    """Reader targets selected before model output is available."""

    readers: tuple[PromptMatchedReader, ...]
    source_shape: tuple[int, int, int]
    matching_method: str = "seed_slice_prompt_center_then_box_overlap"
    reference_set: str = "staged_reader_dicom_seg"

    @property
    def reader_count(self) -> int:
        return len(self.readers)

    @property
    def matched_reader_count(self) -> int:
        return sum(reader.matched for reader in self.readers)

    @property
    def consensus_reader_threshold(self) -> int:
        # LIDC's consensus reference requires at least 3 of 4 readers.  Using the
        # same 75% rule keeps one-reader development fixtures useful without
        # relabeling a 2-of-4 vote as consensus.
        return max(1, math.ceil(self.reader_count * 0.75))


@dataclass(frozen=True)
class ReaderComparisonResult:
    reader_id: str
    matched: bool
    segment_number: int | None
    segment_label: str | None
    prompt_overlap_voxels: int
    dice: float | None
    reference_voxel_count: int | None
    reference_volume_ml: float | None
    reference_segmented_slice_count: int | None


@dataclass(frozen=True)
class ReferenceComparisonResult:
    reference_set: str
    matching_method: str
    reader_count: int
    matched_reader_count: int
    consensus_rule: str
    consensus_reader_threshold: int
    consensus_available: bool
    consensus_dice: float | None
    consensus_voxel_count: int | None
    consensus_volume_ml: float | None
    consensus_segmented_slice_count: int | None
    readers: tuple[ReaderComparisonResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON/Pydantic-ready contract without exposing mask arrays."""
        return {
            "reference_set": self.reference_set,
            "matching_method": self.matching_method,
            "reader_count": self.reader_count,
            "matched_reader_count": self.matched_reader_count,
            "consensus_rule": self.consensus_rule,
            "consensus_reader_threshold": self.consensus_reader_threshold,
            "consensus_available": self.consensus_available,
            "consensus_dice": self.consensus_dice,
            "consensus_voxel_count": self.consensus_voxel_count,
            "consensus_volume_ml": self.consensus_volume_ml,
            "consensus_segmented_slice_count": self.consensus_segmented_slice_count,
            "readers": [
                {
                    "reader_id": item.reader_id,
                    "matched": item.matched,
                    "segment_number": item.segment_number,
                    "segment_label": item.segment_label,
                    "prompt_overlap_voxels": item.prompt_overlap_voxels,
                    "dice": item.dice,
                    "reference_voxel_count": item.reference_voxel_count,
                    "reference_volume_ml": item.reference_volume_ml,
                    "reference_segmented_slice_count": (
                        item.reference_segmented_slice_count
                    ),
                }
                for item in self.readers
            ],
        }


def _source_uid_for_frame(functional_group, *, path: Path, frame_index: int) -> str:
    """Read the standard per-frame derivation reference, with no positional fallback."""
    try:
        derivations = functional_group.DerivationImageSequence
        source_images = derivations[0].SourceImageSequence
        source_uids = {
            str(item.ReferencedSOPInstanceUID)
            for item in source_images
            if getattr(item, "ReferencedSOPInstanceUID", None)
        }
    except (AttributeError, IndexError, TypeError) as exc:
        raise ReferenceEvaluationError(
            f"{path.name} frame {frame_index} has no source SOP Instance UID reference"
        ) from exc
    if len(source_uids) != 1:
        raise ReferenceEvaluationError(
            f"{path.name} frame {frame_index} must reference exactly one source SOP "
            f"Instance UID, found {len(source_uids)}"
        )
    return next(iter(source_uids))


def _segment_number_for_frame(functional_group, *, path: Path, frame_index: int) -> int:
    try:
        return int(
            functional_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ReferenceEvaluationError(
            f"{path.name} frame {frame_index} has no valid ReferencedSegmentNumber"
        ) from exc


def _load_segment_volumes(
    path: Path,
    *,
    uid_to_index: dict[str, int],
    source_shape: tuple[int, int, int],
) -> tuple[dict[int, Any], dict[int, str | None]]:
    import numpy as np
    import pydicom
    from pydicom.uid import SegmentationStorage

    try:
        dataset = pydicom.dcmread(str(path))
    except Exception as exc:  # noqa: BLE001 - convert decoder errors to domain context
        raise ReferenceEvaluationError(
            f"could not read staged reference {path.name}: {exc}"
        ) from exc
    if str(getattr(dataset, "SOPClassUID", "")) != str(SegmentationStorage):
        raise ReferenceEvaluationError(f"{path.name} is not a DICOM Segmentation object")
    if str(getattr(dataset, "Modality", "")) != "SEG":
        raise ReferenceEvaluationError(f"{path.name} does not declare Modality=SEG")

    try:
        frames = np.asarray(dataset.pixel_array)
    except Exception as exc:  # noqa: BLE001 - compressed-pixel errors need file context
        raise ReferenceEvaluationError(
            f"could not decode pixel data from staged reference {path.name}: {exc}"
        ) from exc
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]
    if frames.ndim != 3 or tuple(frames.shape[1:]) != tuple(source_shape[1:]):
        raise ReferenceEvaluationError(
            f"{path.name} frame shape {tuple(frames.shape)} does not match source "
            f"shape {source_shape}"
        )

    functional_groups = getattr(dataset, "PerFrameFunctionalGroupsSequence", None)
    if functional_groups is None or len(functional_groups) != frames.shape[0]:
        raise ReferenceEvaluationError(
            f"{path.name} does not have one functional group per SEG frame"
        )

    labels = {
        int(item.SegmentNumber): str(getattr(item, "SegmentLabel", "")) or None
        for item in getattr(dataset, "SegmentSequence", [])
        if getattr(item, "SegmentNumber", None) is not None
    }
    if not labels:
        raise ReferenceEvaluationError(
            f"{path.name} has no valid segment definitions in SegmentSequence"
        )
    volumes: dict[int, Any] = {}
    occupied: set[tuple[int, int]] = set()
    for frame_index, (frame, functional_group) in enumerate(
        zip(frames, functional_groups, strict=True)
    ):
        segment_number = _segment_number_for_frame(
            functional_group,
            path=path,
            frame_index=frame_index,
        )
        if segment_number not in labels:
            raise ReferenceEvaluationError(
                f"{path.name} frame {frame_index} references undefined segment "
                f"{segment_number}"
            )
        source_uid = _source_uid_for_frame(
            functional_group,
            path=path,
            frame_index=frame_index,
        )
        if source_uid not in uid_to_index:
            raise ReferenceEvaluationError(
                f"{path.name} frame {frame_index} references source SOP "
                f"{source_uid!r}, which is not in the evaluated series"
            )
        source_index = uid_to_index[source_uid]
        identity = (segment_number, source_index)
        if identity in occupied:
            raise ReferenceEvaluationError(
                f"{path.name} contains duplicate frames for segment {segment_number} "
                f"and source SOP {source_uid!r}"
            )
        occupied.add(identity)
        volume = volumes.setdefault(
            segment_number,
            np.zeros(source_shape, dtype=bool),
        )
        volume[source_index] = np.asarray(frame) > 0

    if not volumes:
        raise ReferenceEvaluationError(f"{path.name} contains no SEG frames")
    return volumes, labels


def _prompt_bounds(
    box_xyxy: tuple[float, float, float, float],
    *,
    rows: int,
    columns: int,
) -> tuple[int, int, int, int, int, int]:
    import numpy as np

    values = np.asarray(box_xyxy, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ReferenceEvaluationError("prompt box must contain four finite coordinates")
    x0_float, x1_float = sorted((float(values[0]), float(values[2])))
    y0_float, y1_float = sorted((float(values[1]), float(values[3])))
    x0 = max(0, min(columns, math.floor(x0_float)))
    x1 = max(0, min(columns, math.ceil(x1_float)))
    y0 = max(0, min(rows, math.floor(y0_float)))
    y1 = max(0, min(rows, math.ceil(y1_float)))
    if x1 <= x0 or y1 <= y0:
        raise ReferenceEvaluationError("prompt box does not cover a source-image pixel")
    center_x = max(0, min(columns - 1, int(round((x0_float + x1_float) / 2.0))))
    center_y = max(0, min(rows - 1, int(round((y0_float + y1_float) / 2.0))))
    return x0, y0, x1, y1, center_x, center_y


def select_prompt_matched_references(
    reference_dir: Path,
    *,
    source_sop_instance_uids: tuple[str, ...],
    source_shape: tuple[int, int, int],
    seed_sop_instance_uid: str,
    box_xyxy: tuple[float, float, float, float],
) -> PromptReferenceSelection | None:
    """Select one target per reader using the seed slice and prompt, not prediction.

    The prompt center is preferred.  If reader contour boundaries exclude that exact
    pixel, any positive contour overlap with the prompt box is used.  A tie is resolved
    by segment number, which is deterministic and model-independent.
    """
    import numpy as np

    paths = sorted(path for path in reference_dir.glob("*.dcm") if path.is_file())
    if not paths:
        return None
    if len(source_shape) != 3 or any(int(size) <= 0 for size in source_shape):
        raise ReferenceEvaluationError(f"invalid source volume shape {source_shape}")
    if len(source_sop_instance_uids) != source_shape[0]:
        raise ReferenceEvaluationError(
            "source SOP Instance UID count does not match source volume depth"
        )
    if len(set(source_sop_instance_uids)) != len(source_sop_instance_uids):
        raise ReferenceEvaluationError("source SOP Instance UIDs must be unique")
    uid_to_index = {
        uid: index for index, uid in enumerate(source_sop_instance_uids)
    }
    if seed_sop_instance_uid not in uid_to_index:
        raise ReferenceEvaluationError(
            f"seed SOP Instance UID {seed_sop_instance_uid!r} is not in the source series"
        )
    seed_index = uid_to_index[seed_sop_instance_uid]
    x0, y0, x1, y1, center_x, center_y = _prompt_bounds(
        box_xyxy,
        rows=source_shape[1],
        columns=source_shape[2],
    )

    readers: list[PromptMatchedReader] = []
    for path in paths:
        volumes, labels = _load_segment_volumes(
            path,
            uid_to_index=uid_to_index,
            source_shape=source_shape,
        )
        candidates = []
        for segment_number, volume in volumes.items():
            seed_frame = np.asarray(volume[seed_index], dtype=bool)
            center_contains = bool(seed_frame[center_y, center_x])
            overlap = int(seed_frame[y0:y1, x0:x1].sum())
            if center_contains or overlap:
                candidates.append(
                    (int(center_contains), overlap, -segment_number, segment_number, volume)
                )

        if candidates:
            _center, overlap, _tie_break, segment_number, volume = max(candidates)
            readers.append(
                PromptMatchedReader(
                    reader_id=path.stem,
                    segment_number=segment_number,
                    segment_label=labels.get(segment_number),
                    prompt_overlap_voxels=overlap,
                    volume=volume,
                )
            )
        else:
            readers.append(
                PromptMatchedReader(
                    reader_id=path.stem,
                    segment_number=None,
                    segment_label=None,
                    prompt_overlap_voxels=0,
                    volume=None,
                )
            )

    return PromptReferenceSelection(
        readers=tuple(readers),
        source_shape=tuple(int(size) for size in source_shape),
    )


def _dice(first, second) -> float:
    import numpy as np

    intersection = int(np.logical_and(first, second).sum())
    denominator = int(np.asarray(first, dtype=bool).sum()) + int(
        np.asarray(second, dtype=bool).sum()
    )
    return 2.0 * intersection / denominator if denominator else 1.0


def compare_prediction(
    prediction,
    selection: PromptReferenceSelection,
    *,
    voxel_volume_mm3: float,
) -> ReferenceComparisonResult:
    """Score a prediction against the targets already fixed by the prompt."""
    import numpy as np

    predicted = np.asarray(prediction, dtype=bool)
    if predicted.shape != selection.source_shape:
        raise ReferenceEvaluationError(
            f"prediction shape {predicted.shape} does not match reference source shape "
            f"{selection.source_shape}"
        )
    if not math.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ReferenceEvaluationError("voxel_volume_mm3 must be finite and positive")

    threshold = selection.consensus_reader_threshold
    votes = np.zeros(selection.source_shape, dtype=np.uint16)
    reader_results: list[ReaderComparisonResult] = []
    for reader in selection.readers:
        if not reader.matched:
            reader_results.append(
                ReaderComparisonResult(
                    reader_id=reader.reader_id,
                    matched=False,
                    segment_number=None,
                    segment_label=None,
                    prompt_overlap_voxels=0,
                    dice=None,
                    reference_voxel_count=None,
                    reference_volume_ml=None,
                    reference_segmented_slice_count=None,
                )
            )
            continue
        reference = np.asarray(reader.volume, dtype=bool)
        votes += reference.astype(np.uint16)
        voxel_count = int(reference.sum())
        reader_results.append(
            ReaderComparisonResult(
                reader_id=reader.reader_id,
                matched=True,
                segment_number=reader.segment_number,
                segment_label=reader.segment_label,
                prompt_overlap_voxels=reader.prompt_overlap_voxels,
                dice=_dice(predicted, reference),
                reference_voxel_count=voxel_count,
                reference_volume_ml=voxel_count * voxel_volume_mm3 / 1000.0,
                reference_segmented_slice_count=int(
                    np.count_nonzero(reference.any(axis=(1, 2)))
                ),
            )
        )

    consensus = votes >= threshold
    consensus_voxel_count = int(consensus.sum())
    consensus_available = (
        selection.matched_reader_count >= threshold and consensus_voxel_count > 0
    )
    return ReferenceComparisonResult(
        reference_set=selection.reference_set,
        matching_method=selection.matching_method,
        reader_count=selection.reader_count,
        matched_reader_count=selection.matched_reader_count,
        consensus_rule=f"at_least_{threshold}_of_{selection.reader_count}_readers",
        consensus_reader_threshold=threshold,
        consensus_available=consensus_available,
        consensus_dice=_dice(predicted, consensus) if consensus_available else None,
        consensus_voxel_count=consensus_voxel_count if consensus_available else None,
        consensus_volume_ml=(
            consensus_voxel_count * voxel_volume_mm3 / 1000.0
            if consensus_available
            else None
        ),
        consensus_segmented_slice_count=(
            int(np.count_nonzero(consensus.any(axis=(1, 2))))
            if consensus_available
            else None
        ),
        readers=tuple(reader_results),
    )
