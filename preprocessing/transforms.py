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
    # Optional model-owned channel normalization, applied after [0,1]-space
    # augmentation. Pretrained timm backbones populate these from their resolved
    # data config rather than relying on hard-coded ImageNet constants.
    channel_mean: tuple[float, ...] | None = None
    channel_std: tuple[float, ...] | None = None
    interpolation: str | None = None
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


class NormalizeChannels:
    """Apply a fixed per-channel ``(x - mean) / std`` model input contract."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        if len(mean) != len(std) or not mean:
            raise ValueError("channel_mean and channel_std must have equal non-zero length")
        if any(value <= 0 for value in std):
            raise ValueError("channel_std values must be positive")
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != len(self.mean):
            raise ValueError(
                f"normalization has {len(self.mean)} channels but input has {x.shape[0]}"
            )
        shape = (len(self.mean),) + (1,) * (x.ndim - 1)
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).reshape(shape)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).reshape(shape)
        return (x - mean) / std


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

    resize_kwargs: dict = {"spatial_size": cfg.spatial_size}
    if cfg.interpolation is not None:
        resize_kwargs["mode"] = cfg.interpolation
    steps.append(Resize(**resize_kwargs))
    steps.append(AdaptChannels(cfg.in_channels))

    if train and cfg.augment:
        steps += [
            # MONAI interprets ``None`` as all spatial axes, which turns a 2-D
            # image by 180 degrees. Chest radiographs may be mirrored left/right,
            # but must never be trained upside down.
            RandFlip(prob=cfg.aug_prob, spatial_axis=len(cfg.spatial_size) - 1),
            RandGaussianNoise(prob=cfg.aug_prob, std=0.02),
            RandAdjustContrast(prob=cfg.aug_prob, gamma=(0.8, 1.2)),
        ]

    if (cfg.channel_mean is None) != (cfg.channel_std is None):
        raise ValueError("channel_mean and channel_std must be provided together")
    if cfg.channel_mean is not None and cfg.channel_std is not None:
        steps.append(NormalizeChannels(cfg.channel_mean, cfg.channel_std))

    steps.append(EnsureType(data_type="tensor", dtype=torch.float32))
    return Compose(steps)
