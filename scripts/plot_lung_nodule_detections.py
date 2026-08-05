#!/usr/bin/env python3
"""Render CT slices with the ground-truth nodule and top detections marked on them.

A direct diagnostic for scripts/run_monai_pathology_experts.py's lung_nodule result:
does the detector's top-scoring detection sit on real, plausible anatomy near the true
nodule, or somewhere that looks like a coordinate-conversion bug?

Uses SimpleITK's own TransformPhysicalPointToContinuousIndex for the world-mm -> pixel
conversion, deliberately NOT the hand-rolled ImageOrientationPatient/PixelSpacing math
this script used in its first version. That first version shared its row/col convention
with scripts/lidc_seg_ground_truth.py, and both had the identical bug (confirmed and
fixed) -- which meant this script could not actually catch that class of error: fixing
the same mistake in both the forward (ground-truth extraction) and inverse (this
script's pixel projection) mapping cancels out, leaving the rendered image unchanged
even after the "fix" landed. That happened for real on this project's first fix attempt.
SimpleITK is a mature, independently-implemented library with no shared code path with
lidc_seg_ground_truth.py, so agreement between the two now means something.

Usage (Colab, after run_monai_pathology_experts.py --expert lung_nodule has completed):

    python scripts/plot_lung_nodule_detections.py \
        --ct-dir /content/monai_scratch/lung_nodule_case/ct \
        --manifest /content/drive/MyDrive/doctor_assistant/monai_experts/results/monai_experts_manifest.json \
        --output-dir /content/drive/MyDrive/doctor_assistant/monai_experts/results \
        --top-n 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_ct_image(ct_dir: Path):
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"no DICOM series found under {ct_dir}")
    # recursive=True: mirrors the same defensive choice made in
    # run_monai_pathology_experts.py's _ct_dicom_to_nifti -- idc-index's on-disk layout
    # is not guaranteed flat.
    files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0], False, True)
    reader.SetFileNames(files)
    return reader.Execute()


def render(ct_dir: Path, manifest_path: Path, output_dir: Path, top_n: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import SimpleITK as sitk

    manifest = json.loads(manifest_path.read_text())
    result = next(r for r in manifest["results"] if r["expert"] == "lung_nodule")
    ground_truth = result["ground_truth"]
    detections = sorted(result["detections"], key=lambda d: -d["score"])[:top_n]

    image = _load_ct_image(ct_dir)
    volume = sitk.GetArrayFromImage(image)  # numpy order: (slice, row, col) = (z, y, x)
    spacing = image.GetSpacing()  # (x, y, z) mm

    objects = [("ground truth", gt, "lime") for gt in ground_truth] + [
        (f"det #{i + 1} score={d['score']:.2f}", d, "red") for i, d in enumerate(detections)
    ]

    fig, axes = plt.subplots(1, len(objects), figsize=(5 * len(objects), 5))
    if len(objects) == 1:
        axes = [axes]

    for ax, (label, obj, color) in zip(axes, objects):
        center_lps_mm = tuple(obj["center_lps_mm"])
        # SimpleITK convention (verified against a synthetic image before ever touching
        # a real DICOM file): index order is (x, y, z), matching image.GetSize(), while
        # GetArrayFromImage returns numpy order (z, y, x). So col=idx_x, row=idx_y,
        # slice=idx_z.
        idx_x, idx_y, idx_z = image.TransformPhysicalPointToContinuousIndex(center_lps_mm)
        slice_index = int(round(idx_z))
        slice_index = max(0, min(slice_index, volume.shape[0] - 1))

        pixels = volume[slice_index].astype(np.int16)
        # Standard lung window (level -600, width 1500) for visibility of both airway
        # and soft tissue structures a nodule or false positive would sit against.
        windowed = np.clip(pixels, -1350, 150)

        diameter_mm = obj.get("diameter_mm")
        if diameter_mm is None:
            diameter_mm = max(obj.get("size_whd_mm", (5.0, 5.0, 5.0)))
        radius_px = max(diameter_mm / 2.0 / spacing[0], 3.0)

        ax.imshow(windowed, cmap="gray")
        circle = plt.Circle((idx_x, idx_y), radius_px, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.set_title(f"{label}\nslice {slice_index}, z={center_lps_mm[2]:.0f}mm")
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "lung_nodule_detections_overlay.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5, help="how many top-score detections to render")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = render(args.ct_dir, args.manifest, args.output_dir, args.top_n)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
