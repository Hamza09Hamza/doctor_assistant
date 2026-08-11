#!/usr/bin/env python3
"""Extract the raw Hiera windowed position embedding from an official MedSAM2/SAM2
Torch checkpoint, for ``experts.medsam2_volume.SAM2MLXVolumeSegmenter`` to recompute a
correctly-sized position embedding at MLX inference time.

Why this is needed: ``mlx-sam-convert`` bakes the trunk's small learned
``pos_embed`` (7x7) and ``pos_embed_window`` (8x8) tensors into a single precomposed
256x256 ``pos_embed_full`` grid -- the patch grid for a 1024px image at stride 4 -- and
the MLX runtime just slices that grid's top-left corner to fit whatever ``image_size``
is actually configured. That slice is only correct when ``image_size`` is 1024. MedSAM2's
CT-lesion checkpoint runs at 512px (a 128x128 patch grid), so the sliced corner is the
position embedding for the wrong quadrant of the image -- silently, since the shapes
still line up and nothing errors. This was root-caused by comparing image-encoder
output between the official Torch predictor and mlx-sam on the same LIDC-IDRI-0686 CT
slice: cosine similarity ~0.91 (vs. ~1.0 for a correctly patched embedding).

This script requires PyTorch (run it from ``.venv-torch`` or any environment that can
``torch.load`` the checkpoint); the MLX runtime that consumes its output stays
torch-free. Run once per converted checkpoint:

    source .venv-torch/bin/activate
    python scripts/extract_hiera_pos_embed.py \\
        --checkpoint checkpoints/MedSAM2_CTLesion.pt \\
        --mlx-checkpoint checkpoints/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors

The output sidecar path is derived from ``--mlx-checkpoint`` by
``experts.medsam2_volume._hiera_pos_embed_sidecar_path`` (``<stem>.pos_embed_raw.npz``
next to the converted checkpoint), so no separate ``--output`` is needed for the normal
case; pass it explicitly to write somewhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Official Torch checkpoint (.pt)")
    parser.add_argument(
        "--mlx-checkpoint",
        type=Path,
        required=True,
        help="Converted MLX checkpoint (.safetensors) whose sidecar path this derives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Sidecar .npz path; defaults next to --mlx-checkpoint as <stem>.pos_embed_raw.npz",
    )
    parser.add_argument(
        "--state-dict-key",
        default="image_encoder.trunk",
        help="Prefix of the Hiera trunk's keys in the checkpoint's state dict",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experts.medsam2_volume import _hiera_pos_embed_sidecar_path

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    pos_embed_key = f"{args.state_dict_key}.pos_embed"
    pos_embed_window_key = f"{args.state_dict_key}.pos_embed_window"
    if pos_embed_key not in sd or pos_embed_window_key not in sd:
        raise KeyError(
            f"{args.checkpoint} has no {pos_embed_key!r}/{pos_embed_window_key!r}; "
            "pass --state-dict-key if this checkpoint uses a different prefix"
        )

    pos_embed = sd[pos_embed_key].detach().cpu().numpy()
    pos_embed_window = sd[pos_embed_window_key].detach().cpu().numpy()

    output = args.output or _hiera_pos_embed_sidecar_path(args.mlx_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, pos_embed=pos_embed, pos_embed_window=pos_embed_window)
    print(f"pos_embed {pos_embed.shape}, pos_embed_window {pos_embed_window.shape}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
