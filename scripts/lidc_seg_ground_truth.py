"""Derive nodule ground truth directly from LIDC-IDRI multi-reader DICOM SEG files.

Used specifically for LIDC cases that are NOT part of LUNA16 (see
LUNG_NODULE_REPLACEMENT_CASE in run_monai_pathology_experts.py), so LUNA16's own
annotations.csv cannot be looked up -- it simply has no row for these scans by
construction. Ground truth instead comes from the four radiologists' own DICOM SEG
segmentations, published alongside every LIDC-IDRI case as the "DICOM-LIDC-IDRI-Nodules"
collection.

This is a good-faith reconstruction of LUNA16's stated reference-standard rule (a lesion
counts only if at least 3 of 4 readers marked overlapping segmentations), not a run of
LUNA16's own reference implementation -- stated plainly rather than implied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom


def _seg_frames(ds: "pydicom.Dataset"):
    """Yield (segment_number, image_position_patient_mm, 2D bool mask) per SEG frame.

    The DICOM SEG standard allows Segment Identification Sequence to live in
    SharedFunctionalGroupsSequence instead of being repeated in every per-frame group,
    whenever the whole series only ever uses one segment. Confirmed against a real
    encoder (highdicom): a single-segment, multi-frame SEG puts it in Shared, not
    per-frame -- so both locations must be checked, not just per-frame.
    """
    bits_allocated = int(ds.BitsAllocated)
    if bits_allocated != 1:
        raise ValueError(f"expected a 1-bit packed SEG, got BitsAllocated={bits_allocated}")
    pixel_array = ds.pixel_array
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[None, ...]
    frame_groups = ds.PerFrameFunctionalGroupsSequence
    if len(frame_groups) != pixel_array.shape[0]:
        raise ValueError(
            f"frame count mismatch: {pixel_array.shape[0]} pixel frames vs "
            f"{len(frame_groups)} functional groups"
        )
    shared = ds.SharedFunctionalGroupsSequence[0]
    shared_segment_number = (
        int(shared.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        if "SegmentIdentificationSequence" in shared
        else None
    )
    for frame_index, group in enumerate(frame_groups):
        if "SegmentIdentificationSequence" in group:
            segment_number = int(group.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        elif shared_segment_number is not None:
            segment_number = shared_segment_number
        else:
            raise ValueError(
                f"frame {frame_index}: no Segment Identification Sequence in either the "
                "per-frame or shared functional groups"
            )
        position = tuple(float(v) for v in group.PlanePositionSequence[0].ImagePositionPatient)
        yield segment_number, position, pixel_array[frame_index].astype(bool)


def extract_reader_nodules(seg_path: Path, slice_spacing_mm: float) -> list[dict]:
    """Return one entry per distinct marked segment (nodule candidate) in one reader's SEG.

    slice_spacing_mm is passed in from the parent CT series rather than re-derived from
    the SEG's own frame positions, because a reader may mark only 1-2 slices for a small
    nodule -- too few points to safely infer spacing from.
    """
    ds = pydicom.dcmread(str(seg_path))
    if str(getattr(ds, "Modality", "")) != "SEG":
        raise ValueError(f"{seg_path} is not a DICOM SEG object")

    shared = ds.SharedFunctionalGroupsSequence[0]
    row_mm, col_mm = (float(v) for v in shared.PixelMeasuresSequence[0].PixelSpacing)
    orientation = [float(v) for v in shared.PlaneOrientationSequence[0].ImageOrientationPatient]
    row_dir = np.array(orientation[0:3])
    col_dir = np.array(orientation[3:6])
    voxel_volume_mm3 = row_mm * col_mm * slice_spacing_mm

    points_by_segment: dict[int, list[np.ndarray]] = {}
    for segment_number, position, frame in _seg_frames(ds):
        ys, xs = np.nonzero(frame)
        if len(xs) == 0:
            continue
        origin = np.array(position)
        points = origin + np.outer(xs, col_dir) * col_mm + np.outer(ys, row_dir) * row_mm
        points_by_segment.setdefault(segment_number, []).append(points)

    nodules = []
    for segment_number, chunks in points_by_segment.items():
        all_points = np.concatenate(chunks, axis=0)
        voxel_count = len(all_points)
        volume_mm3 = voxel_count * voxel_volume_mm3
        # LUNA16 itself reports nodule size as the diameter of the equivalent sphere of
        # the marked volume, not a bounding-box extent -- matched here for comparability.
        diameter_mm = 2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
        nodules.append(
            {
                "segment_number": segment_number,
                "center_lps_mm": tuple(float(v) for v in all_points.mean(axis=0)),
                "diameter_mm": float(diameter_mm),
                "voxel_count": int(voxel_count),
            }
        )
    return nodules


def consensus_nodules(nodules_by_reader: dict[str, list[dict]], min_readers: int = 3) -> list[dict]:
    """Cluster nodule candidates across readers; keep clusters with >= min_readers support.

    Join rule: two candidates from different readers are the same lesion if their centers
    are closer than the sum of their two radii (their equivalent spheres overlap). This is
    a standard, easily-inspectable merge rule -- not LUNA16's own unpublished clustering
    code -- documented here rather than presented as if it were.
    """
    flat: list[tuple[str, dict]] = [
        (reader, nodule) for reader, nodules in nodules_by_reader.items() for nodule in nodules
    ]
    n = len(flat)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        reader_i, nodule_i = flat[i]
        center_i = np.array(nodule_i["center_lps_mm"])
        for j in range(i + 1, n):
            reader_j, nodule_j = flat[j]
            if reader_j == reader_i:
                continue
            center_j = np.array(nodule_j["center_lps_mm"])
            distance = float(np.linalg.norm(center_i - center_j))
            if distance <= (nodule_i["diameter_mm"] + nodule_j["diameter_mm"]) / 2.0:
                union(i, j)

    clusters: dict[int, list[tuple[str, dict]]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(flat[i])

    consensus = []
    for members in clusters.values():
        readers_present = {reader for reader, _ in members}
        if len(readers_present) < min_readers:
            continue
        centers = np.array([n["center_lps_mm"] for _, n in members])
        diameters = [n["diameter_mm"] for _, n in members]
        consensus.append(
            {
                "center_lps_mm": tuple(float(v) for v in centers.mean(axis=0)),
                "diameter_mm": float(np.mean(diameters)),
                "supporting_readers": sorted(readers_present),
                "reader_count": len(readers_present),
            }
        )
    return consensus
