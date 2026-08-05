#!/usr/bin/env python3
"""Render CT slices with the ground-truth nodule and top detections marked on them.

A direct diagnostic for scripts/run_monai_pathology_experts.py's lung_nodule result:
does the detector's top-scoring detection sit on real, plausible anatomy near the true
nodule, or somewhere that looks like a coordinate-conversion bug? This draws directly on
the actual CT pixels rather than building a new DICOM SEG / Orthanc / OHIF pipeline for
one diagnostic look -- fewer moving parts to get wrong for a question that just needs a
picture.

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
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CtSlice:
    path: Path
    instance_number: int
    position_lps_mm: tuple[float, float, float]
    row_dir: tuple[float, float, float]
    col_dir: tuple[float, float, float]
    row_spacing_mm: float
    col_spacing_mm: float


def _load_ct_slices(ct_dir: Path) -> list[CtSlice]:
    import pydicom

    slices: list[CtSlice] = []
    for path in sorted(item for item in ct_dir.rglob("*") if item.is_file()):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue
        row_mm, col_mm = (float(v) for v in ds.PixelSpacing)
        iop = [float(v) for v in ds.ImageOrientationPatient]
        # DICOM PS3.3 C.7.6.2.1.1: first triplet = direction of increasing COLUMN index
        # (the direction you move traversing along "the first row"), second triplet =
        # direction of increasing ROW index. Was backwards here (matching the same bug
        # already fixed in lidc_seg_ground_truth.py) -- caught by this script's own
        # output: the ground-truth marker plotted outside the patient's body.
        slices.append(
            CtSlice(
                path=path,
                instance_number=int(ds.InstanceNumber),
                position_lps_mm=tuple(float(v) for v in ds.ImagePositionPatient),
                col_dir=tuple(iop[0:3]),
                row_dir=tuple(iop[3:6]),
                row_spacing_mm=row_mm,
                col_spacing_mm=col_mm,
            )
        )
    if not slices:
        raise RuntimeError(f"no CT slices found under {ct_dir}")
    slices.sort(key=lambda s: s.instance_number)
    return slices


def _nearest_slice(slices: list[CtSlice], z_mm: float) -> CtSlice:
    return min(slices, key=lambda s: abs(s.position_lps_mm[2] - z_mm))


def _world_to_pixel(slice_: CtSlice, point_lps_mm: tuple[float, float, float]) -> tuple[float, float]:
    """Inverse of the forward mapping used in scripts/lidc_seg_ground_truth.py:
    world = origin + col*col_dir*col_spacing + row*row_dir*row_spacing.
    row_dir/col_dir are guaranteed orthonormal by the DICOM standard, so projecting the
    offset onto each axis directly (dot product) inverts that mapping exactly.
    """
    import numpy as np

    offset = np.array(point_lps_mm) - np.array(slice_.position_lps_mm)
    row = float(np.dot(offset, slice_.row_dir)) / slice_.row_spacing_mm
    col = float(np.dot(offset, slice_.col_dir)) / slice_.col_spacing_mm
    return row, col


def render(ct_dir: Path, manifest_path: Path, output_dir: Path, top_n: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pydicom

    manifest = json.loads(manifest_path.read_text())
    result = next(r for r in manifest["results"] if r["expert"] == "lung_nodule")
    ground_truth = result["ground_truth"]
    detections = sorted(result["detections"], key=lambda d: -d["score"])[:top_n]

    slices = _load_ct_slices(ct_dir)

    objects = [("ground truth", gt, "lime") for gt in ground_truth] + [
        (f"det #{i + 1} score={d['score']:.2f}", d, "red") for i, d in enumerate(detections)
    ]

    fig, axes = plt.subplots(1, len(objects), figsize=(5 * len(objects), 5))
    if len(objects) == 1:
        axes = [axes]

    for ax, (label, obj, color) in zip(axes, objects):
        center = obj["center_lps_mm"]
        slice_ = _nearest_slice(slices, center[2])
        pixels = pydicom.dcmread(str(slice_.path)).pixel_array.astype(np.int16)
        # Standard lung window (level -600, width 1500) for visibility of both airway
        # and soft tissue structures a nodule or false positive would sit against.
        windowed = np.clip(pixels, -1350, 150)
        row, col = _world_to_pixel(slice_, center)
        radius_px = max(obj.get("diameter_mm", max(obj.get("size_whd_mm", (5, 5, 5)))) / 2.0 / slice_.col_spacing_mm, 3.0)

        ax.imshow(windowed, cmap="gray")
        circle = plt.Circle((col, row), radius_px, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.set_title(f"{label}\nslice {slice_.instance_number}, z={center[2]:.0f}mm")
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
