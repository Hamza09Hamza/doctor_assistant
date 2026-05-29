"""The data contracts that flow between pipeline stages.

A `Scan` enters at ingestion and is carried (with its metadata) through routing
and preprocessing. Each expert turns a `Scan` into a `Prediction`. Keeping these
shapes fixed is what lets stages be developed and swapped independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .enums import BodyPart, Modality, TaskType

if TYPE_CHECKING:  # avoid importing torch at module load for cheap syntax checks
    import torch


@dataclass
class ScanMetadata:
    """Everything we know about a scan that isn't pixel data."""

    modality: Modality = Modality.UNKNOWN
    body_part: BodyPart = BodyPart.UNKNOWN
    # Physical voxel spacing in mm, ordered to match the data axes (e.g. (z,y,x)).
    spacing: tuple[float, ...] | None = None
    original_shape: tuple[int, ...] | None = None
    source_path: str | None = None
    # Modality-specific extras (DICOM tags, window center/width, patient id, ...).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scan:
    """A loaded, model-ready image.

    `data` is a float tensor shaped (C, H, W) for 2D or (C, D, H, W) for 3D.
    Channels-first and batch-free; the trainer/inference layer adds the batch dim.
    """

    data: "torch.Tensor"
    meta: ScanMetadata = field(default_factory=ScanMetadata)

    @property
    def spatial_dims(self) -> int:
        """2 for planar images, 3 for volumes (data is channels-first)."""
        return self.data.ndim - 1


@dataclass
class HeadOutput:
    """Raw output of a single head before it is decoded into a Prediction."""

    task: TaskType
    name: str
    tensor: "torch.Tensor"


@dataclass
class Prediction:
    """The unified result every expert returns, regardless of internal design.

    Downstream stages (ensemble, explainability, reporting) only ever read this
    shape — they never need to know how a given expert was built.
    """

    expert: str
    # class label -> probability, e.g. {"glioma": 0.89, "meningioma": 0.07, ...}
    class_probs: dict[str, float] = field(default_factory=dict)
    # Optional lesion mask, shaped like the input minus the channel dim.
    segmentation: "torch.Tensor | None" = None
    # Calibrated reliability of THIS prediction in [0, 1] (not a class score).
    confidence: float | None = None
    # Explainability heatmap (e.g. Grad-CAM), same spatial shape as the input.
    heatmap: "torch.Tensor | None" = None
    meta: ScanMetadata = field(default_factory=ScanMetadata)

    @property
    def top_label(self) -> str | None:
        if not self.class_probs:
            return None
        return max(self.class_probs, key=self.class_probs.get)

    @property
    def top_score(self) -> float | None:
        label = self.top_label
        return None if label is None else self.class_probs[label]
