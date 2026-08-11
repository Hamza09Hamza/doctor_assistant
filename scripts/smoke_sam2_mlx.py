#!/usr/bin/env python3
"""Run a small, deterministic native-MLX volume-propagation smoke test.

This validates model loading, Metal execution, a box prompt, and propagation in both
directions. It is a runtime check, not a medical-accuracy evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experts.medsam2_volume import SAM2MLXVolumeSegmenter


def synthetic_volume() -> np.ndarray:
    depth = 5
    height = width = 128
    y, x = np.ogrid[:height, :width]
    volume = np.full((depth, height, width), 22, dtype=np.uint8)
    for z in range(depth):
        center_x = 62 + z
        radius_x = 20 - abs(z - 2)
        radius_y = 24 - abs(z - 2)
        object_mask = (
            ((x - center_x) / radius_x) ** 2
            + ((y - 64) / radius_y) ** 2
            <= 1
        )
        volume[z, object_mask] = 225
    return volume


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="avbiswas/sam2.1-hiera-tiny-mlx-16bit",
        help="Hugging Face model ID or local MLX safetensors checkpoint",
    )
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    volume = synthetic_volume()
    segmenter = SAM2MLXVolumeSegmenter(
        model=args.model,
        image_size=args.image_size,
        keep_prompt_component=False,
    )
    started = time.perf_counter()
    mask = segmenter.segment_volume(
        volume,
        seed_index=2,
        box_xyxy=(38, 36, 88, 92),
    )
    elapsed = time.perf_counter() - started
    nonempty_slices = np.flatnonzero(mask.any(axis=(1, 2))).tolist()
    if nonempty_slices != list(range(volume.shape[0])):
        raise SystemExit(f"propagation did not cover every slice: {nonempty_slices}")

    print(f"backend={segmenter.version}")
    print(f"shape={mask.shape} nonempty_slices={nonempty_slices}")
    print(f"voxels={int(mask.sum())} elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
