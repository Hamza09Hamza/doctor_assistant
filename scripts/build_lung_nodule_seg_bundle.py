#!/usr/bin/env python3
"""Build an OHIF-viewable bundle for the lung-nodule case: the real CT, an "AI
detections" DICOM SEG (each kept detection as its own segment), and a "ground truth
(4-reader consensus)" DICOM SEG -- two independently toggleable overlays on real DICOM,
through the existing bundle/publish path (scripts/publish_dicom_seg_to_orthanc.py,
unmodified, run locally afterward).

Unlike the brain-tumour case, this needs no synthetic DICOM step: LIDC-IDRI-0672 is a
real DICOM CT, already staged by run_monai_pathology_experts.py. This script only
rasterizes the manifest's detection boxes and ground-truth nodule (points + sizes) into
per-slice boolean masks aligned to that real CT's own voxel grid, using SimpleITK's
TransformPhysicalPointToContinuousIndex/TransformIndexToPhysicalPoint -- the same
independent, already-verified library used by scripts/plot_lung_nodule_detections.py,
deliberately not the hand-rolled DICOM orientation math that caused two real bugs
earlier this session.

Usage (Colab, after run_monai_pathology_experts.py --expert lung_nodule has completed):

    python scripts/build_lung_nodule_seg_bundle.py \
        --ct-dir /content/monai_scratch/lung_nodule_case/ct \
        --manifest /content/drive/MyDrive/doctor_assistant/monai_experts/results/monai_experts_manifest.json \
        --output-dir /content/drive/MyDrive/doctor_assistant/monai_experts/results \
        --top-n 5 --score-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_ct(ct_dir: Path, series_uid: str):
    """Returns (sitk.Image, [pydicom.Dataset, ...]) in IDENTICAL slice order -- both
    built from the exact same GetGDCMSeriesFileNames() file-path list, so a mask frame
    index always corresponds to the same source_images entry with no risk of the two
    orderings silently disagreeing."""
    import pydicom
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_uid, False, True)
    if not files:
        raise RuntimeError(f"no DICOM files found for series {series_uid} under {ct_dir}")
    reader.SetFileNames(files)
    image = reader.Execute()
    datasets = [pydicom.dcmread(f) for f in files]
    return image, datasets


def _rasterize_ellipsoid(image, center_lps_mm, radii_mm) -> "np.ndarray":
    """Boolean mask, shape matching sitk.GetArrayFromImage(image) (z,y,x), True inside
    the ellipsoid. Only iterates a local bounding box around the object (a few voxels
    across for a ~5mm nodule), not the whole CT volume, for speed."""
    import numpy as np

    size = image.GetSize()  # (nx, ny, nz)
    spacing = image.GetSpacing()  # (sx, sy, sz) mm
    center_idx = image.TransformPhysicalPointToContinuousIndex(tuple(center_lps_mm))

    pad = [int(np.ceil(radii_mm[axis] / spacing[axis])) + 1 for axis in range(3)]
    lo = [max(0, int(round(center_idx[axis])) - pad[axis]) for axis in range(3)]
    hi = [min(size[axis] - 1, int(round(center_idx[axis])) + pad[axis]) for axis in range(3)]

    mask = np.zeros((size[2], size[1], size[0]), dtype=bool)  # (z, y, x)
    center = np.array(center_lps_mm)
    radii = np.array(radii_mm)
    for iz in range(lo[2], hi[2] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for ix in range(lo[0], hi[0] + 1):
                point = np.array(image.TransformIndexToPhysicalPoint((ix, iy, iz)))
                if np.sum(((point - center) / radii) ** 2) <= 1.0:
                    mask[iz, iy, ix] = True
    return mask


def _build_seg(
    mask_frames_zyx,  # dict[int position -> (z,y,x) bool array] OR a single (n,z,y,x)
    labels: list[str],
    source_datasets: list,
    series_description: str,
    output_path: Path,
) -> None:
    import highdicom as hd
    import numpy as np
    from highdicom import AlgorithmIdentificationSequence
    from highdicom.seg.content import SegmentDescription
    from highdicom.sr.coding import CodedConcept
    from pydicom.uid import generate_uid

    # (segment, z, y, x) -> (frame=z, row=y, col=x, segment)
    stacked = np.stack(mask_frames_zyx, axis=-1)  # (z, y, x, n_segments)

    algorithm_identification = AlgorithmIdentificationSequence(
        name="lung_nodule_ct_detection",
        family=CodedConcept("113092", "DCM", "Deep Learning"),
        version="MONAI Model Zoo",
        source="doctor_assistant / scripts/run_monai_pathology_experts.py",
    )
    segment_descriptions = [
        SegmentDescription(
            segment_number=i + 1,
            segment_label=label,
            segmented_property_category=CodedConcept("91723000", "SCT", "Anatomical Structure"),
            segmented_property_type=CodedConcept("108369006", "SCT", "Neoplasm"),
            algorithm_type="AUTOMATIC",
            algorithm_identification=algorithm_identification,
        )
        for i, label in enumerate(labels)
    ]
    seg = hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=stacked,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=generate_uid(),
        series_number=20,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        series_description=series_description,
        manufacturer="doctor_assistant",
        manufacturer_model_name="lung_nodule_ct_detection (MONAI Model Zoo)",
        software_versions="run_monai_pathology_experts.py",
        device_serial_number="0",
    )
    seg.save_as(str(output_path))


def build(args) -> Path:
    from scripts.run_monai_pathology_experts import LUNG_NODULE_CT_SERIES_UID

    manifest = json.loads(args.manifest.read_text())
    result = next(r for r in manifest["results"] if r["expert"] == "lung_nodule")
    ground_truth = result["ground_truth"]
    kept_detections = sorted(
        (d for d in result["detections"] if d["score"] >= args.score_threshold),
        key=lambda d: -d["score"],
    )[: args.top_n]
    if not kept_detections:
        raise RuntimeError(
            f"no detections at score >= {args.score_threshold}; lower --score-threshold"
        )

    print(f"Loading CT and rasterizing {len(ground_truth)} ground-truth nodule(s) "
          f"and {len(kept_detections)} detection(s) ...", flush=True)
    image, source_datasets = _load_ct(args.ct_dir, args.series_uid or LUNG_NODULE_CT_SERIES_UID)

    gt_masks = []
    gt_labels = []
    for i, gt in enumerate(ground_truth):
        radius = gt["diameter_mm"] / 2.0
        gt_masks.append(_rasterize_ellipsoid(image, gt["center_lps_mm"], (radius, radius, radius)))
        gt_labels.append(f"GT nodule {i + 1} ({gt['reader_count']}/4 readers, {gt['diameter_mm']:.1f}mm)")

    det_masks = []
    det_labels = []
    for i, det in enumerate(kept_detections):
        w, h, d = det["size_whd_mm"]
        det_masks.append(_rasterize_ellipsoid(image, det["center_lps_mm"], (w / 2, h / 2, d / 2)))
        det_labels.append(f"AI detection #{i + 1} score={det['score']:.2f}")

    work_dir = args.scratch_dir / "lung_nodule_seg_bundle"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    dicom_dir = work_dir / "source_dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)
    for ds in source_datasets:
        ds.save_as(str(dicom_dir / f"{ds.SOPInstanceUID}.dcm"))

    ai_seg_path = work_dir / "totalsegmentator_seg.dcm"
    _build_seg(
        det_masks, det_labels, source_datasets,
        "AI lung-nodule detections -- experimental, not a diagnosis", ai_seg_path,
    )

    expert_dir = work_dir / "expert_reference"
    expert_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_path = expert_dir / "ground_truth_consensus_seg.dcm"
    _build_seg(
        gt_masks, gt_labels, source_datasets,
        "Ground truth: >=3/4 LIDC radiologist DICOM SEG consensus", gt_seg_path,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "lung_nodule_ohif_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dcm_path in dicom_dir.glob("*.dcm"):
            archive.write(dcm_path, f"source_dicom/{dcm_path.name}")
        archive.write(ai_seg_path, "totalsegmentator_seg.dcm")
        archive.write(gt_seg_path, f"expert_reference/{gt_seg_path.name}")

    print(f"Bundle written: {bundle_path}", flush=True)
    print(
        "Locally: python scripts/publish_dicom_seg_to_orthanc.py "
        f"--bundle <downloaded {bundle_path.name}>",
        flush=True,
    )
    return bundle_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content/monai_scratch"))
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--series-uid", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
