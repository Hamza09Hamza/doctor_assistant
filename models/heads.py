"""Task heads — the parts that read the shared feature map and produce outputs.

Each head is small and single-purpose. They share one backbone, so training them
together is multi-task learning: the segmentation head forces the encoder to learn
*where* a lesion is, which sharpens the classification head and curbs reliance on
confounders. All heads are 2D/3D-agnostic, parameterized by `spatial_dims`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from core.enums import TaskType


def _conv(spatial_dims: int):
    return nn.Conv2d if spatial_dims == 2 else nn.Conv3d


def _adaptive_avg_pool(spatial_dims: int):
    return nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)


class Head(nn.Module):
    """Base head. Carries the `task` so outputs can be decoded generically."""

    task: TaskType


class ClassificationHead(Head):
    """Global-pool the feature map, then a linear classifier -> class logits.

    `multilabel` flips the *meaning* of the logits, not the layer: when True the
    classes are independent (sigmoid + BCE) so several can fire at once — the chest
    X-ray case, where one study is e.g. both Cardiomegaly and Effusion. When False
    the classes are mutually exclusive (softmax + cross-entropy) — the brain-tumour
    case. Decoders and losses read this flag to pick the right activation.
    """

    task = TaskType.CLASSIFICATION

    def __init__(self, in_channels: int, num_classes: int, spatial_dims: int = 2,
                 dropout: float = 0.2, multilabel: bool = False) -> None:
        super().__init__()
        self.multilabel = multilabel
        self.pool = _adaptive_avg_pool(spatial_dims)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat).flatten(1)
        return self.fc(self.drop(x))


class SegmentationHead(Head):
    """Project features to per-class logits and upsample to the input resolution.

    A deliberately light decoder (1x1 conv): enough to localize and to feed the
    report/explainability layers without a full U-Net's parameter cost. Swap in a
    richer decoder per-expert when a task needs crisp boundaries.
    """

    task = TaskType.SEGMENTATION

    def __init__(self, in_channels: int, num_classes: int, spatial_dims: int = 2) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.classifier = _conv(spatial_dims)(in_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor, out_size: tuple[int, ...] | None = None) -> torch.Tensor:
        logits = self.classifier(feat)
        if out_size is not None and tuple(logits.shape[2:]) != tuple(out_size):
            mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
            logits = F.interpolate(logits, size=out_size, mode=mode, align_corners=False)
        return logits


class ConfidenceHead(Head):
    """Predict a single scalar reliability for the whole input (pre-sigmoid).

    This is calibrated certainty about *this* prediction, distinct from any class
    probability — the value the triage layer reads to decide what a human sees first.
    """

    task = TaskType.CONFIDENCE

    def __init__(self, in_channels: int, spatial_dims: int = 2) -> None:
        super().__init__()
        self.pool = _adaptive_avg_pool(spatial_dims)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.pool(feat).flatten(1)
        return self.fc(x).squeeze(1)  # logit; caller applies sigmoid
