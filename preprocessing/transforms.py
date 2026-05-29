"""Config-driven preprocessing built on MONAI transforms.

One builder serves 2D and 3D: the length of `spatial_size` selects the rank, and
MONAI's transforms adapt automatically. Intensity handling is explicit because it
is modality-specific and a common source of silent bugs (a CT windowed like an
X-ray looks like noise to the model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class PreprocessConfig:
    """Knobs for turning a raw `Scan.data` tensor into a model input."""

    # Target spatial size: (H, W) for 2D, (D, H, W) for 3D.
    spatial_size: tuple[int, ...] = (224, 224)
    # Channels the backbone expects (1 for medical grayscale, 3 for ImageNet nets).
    in_channels: int = 3
    # "scale" -> min-max to [0,1]; "zscore" -> per-image standardize;
    # "ct_window" -> clamp to a Hounsfield window then [0,1].
    intensity: str = "scale"
    ct_window: tuple[float, float] = (-1000.0, 400.0)
    augment: bool = True
    # Probabilities for train-time spatial/intensity augmentation.
    aug_prob: float = 0.3
    extra_meta: dict = field(default_factory=dict)


class AdaptChannels:
    """Force a fixed channel count by repeating or trimming the channel axis."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[0]
        if c == self.n:
            return x
        tail = [1] * (x.ndim - 1)
        if c > self.n:
            return x[: self.n]
        reps = (self.n + c - 1) // c
        return x.repeat(reps, *tail)[: self.n]


def build_preprocess(cfg: PreprocessConfig, train: bool = False) -> Callable:
    """Compose a transform: raw channels-first tensor -> normalized model input.

    `train=True` appends light augmentation. The same builder is reused at
    inference with `train=False` so eval-time preprocessing can never drift from
    what the model was trained on.
    """
    from monai.transforms import (
        Compose,
        EnsureType,
        NormalizeIntensity,
        RandAdjustContrast,
        RandFlip,
        RandGaussianNoise,
        Resize,
        ScaleIntensity,
        ScaleIntensityRange,
    )

    steps: list = [EnsureType(data_type="tensor", dtype=torch.float32)]

    if cfg.intensity == "ct_window":
        lo, hi = cfg.ct_window
        steps.append(ScaleIntensityRange(a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True))
    elif cfg.intensity == "zscore":
        steps.append(NormalizeIntensity(nonzero=True, channel_wise=True))
    else:  # "scale"
        steps.append(ScaleIntensity(minv=0.0, maxv=1.0))

    steps.append(Resize(spatial_size=cfg.spatial_size))
    steps.append(AdaptChannels(cfg.in_channels))

    if train and cfg.augment:
        steps += [
            RandFlip(prob=cfg.aug_prob, spatial_axis=None),
            RandGaussianNoise(prob=cfg.aug_prob, std=0.02),
            RandAdjustContrast(prob=cfg.aug_prob, gamma=(0.8, 1.2)),
        ]

    steps.append(EnsureType(data_type="tensor", dtype=torch.float32))
    return Compose(steps)
