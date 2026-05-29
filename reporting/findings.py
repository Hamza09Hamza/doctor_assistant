"""From raw model output to *structured findings* — the layer the reporter speaks from.

This is the bridge most pipelines miss. A classifier emits a *kind*
("glioma 0.89"); a segmentation head emits a *mask*; a confidence head emits a
*reliability*. None of those is a report. A report needs measured details:
size, location, laterality, how many lesions.

A `Finding` is that structured, audited detail. It is produced **deterministically**
here — by thresholding scores and by running classical geometry on masks/heatmaps —
*never* by a language model. The reporter then only verbalizes these facts, so every
sentence in the final report traces back to a number computed in this file.

Two extraction paths, because findings come from different places per modality:

  - `findings_from_classification`  multi-label scores (+ optional heatmap location).
      Chest X-ray lives here: 14 findings can co-occur, there are no masks, so
      *where* comes from Grad-CAM via a `Localizer`.
  - `findings_from_mask`            connected-component geometry on a segmentation mask.
      Brain MRI lives here: size in mm and volume in mL are measured off the mask
      using the scan's voxel `spacing`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # keep torch out of import path for cheap/offline use
    import torch

# A Localizer turns a 2D heatmap into a coarse anatomical location.
# Returns (laterality, region), either of which may be None.
Localizer = Callable[[np.ndarray], "tuple[str | None, str | None]"]


@dataclass
class Finding:
    """One audited observation. Categorical fields always set; quantitative ones
    only when a mask/heatmap let us *measure* them — absent stays None (never guessed)."""

    label: str                              # e.g. "Cardiomegaly", "glioma"
    probability: float                      # model score for this finding, [0, 1]
    present: bool = True                    # passed the decision threshold
    confidence: float | None = None         # calibrated reliability of the prediction

    # --- quantitative detail (filled only when measurable) ---
    size_mm: float | None = None            # largest diameter, mm
    volume_ml: float | None = None          # lesion volume, mL
    laterality: str | None = None           # "left" | "right" | "midline" | "bilateral"
    location: str | None = None             # region / zone / lobe label
    count: int = 1                          # number of discrete lesions for this label

    # --- provenance, so a report sentence can be traced to its source ---
    source: str = ""                        # "classification+gradcam", "segmentation", ...
    extra: dict[str, Any] = field(default_factory=dict)

    def to_facts(self) -> dict[str, Any]:
        """Compact, JSON-safe dict of *only the known* facts — the reporter's input.

        Omitting None keeps unmeasured details out of the prompt entirely, so the
        language model is never tempted to fill a blank it was handed.
        """
        facts: dict[str, Any] = {"label": self.label, "probability": round(self.probability, 3)}
        if self.confidence is not None:
            facts["confidence"] = round(self.confidence, 3)
        if self.size_mm is not None:
            facts["size_mm"] = round(self.size_mm, 1)
        if self.volume_ml is not None:
            facts["volume_ml"] = round(self.volume_ml, 2)
        if self.laterality is not None:
            facts["laterality"] = self.laterality
        if self.location is not None:
            facts["location"] = self.location
        if self.count != 1:
            facts["count"] = self.count
        return facts


def _resolve_threshold(label: str, thresholds: float | Mapping[str, float]) -> float:
    if isinstance(thresholds, Mapping):
        return float(thresholds.get(label, thresholds.get("__default__", 0.5)))
    return float(thresholds)


def findings_from_classification(
    class_probs: Mapping[str, float],
    *,
    thresholds: float | Mapping[str, float] = 0.5,
    confidence: float | None = None,
    heatmaps: Mapping[str, np.ndarray] | None = None,
    localizer: Localizer | None = None,
    normal_labels: Sequence[str] = ("No Finding", "Normal", "none", "no_tumor"),
    include_absent: bool = False,
) -> list[Finding]:
    """Turn (multi-label) class scores into findings.

    This is the chest-X-ray path: scores are treated as *independent* (a study can be
    both "Cardiomegaly" and "Effusion"), so each label is thresholded on its own.
    Per-label `thresholds` let you tune sensitivity per finding (clinically you bias
    toward catching effusions, etc.). If a `localizer` and per-label `heatmaps` are
    given, each present finding gets a coarse laterality/region from its Grad-CAM map.
    """
    normal = {n.lower() for n in normal_labels}
    findings: list[Finding] = []
    for label, prob in class_probs.items():
        if label.lower() in normal:
            continue
        present = float(prob) >= _resolve_threshold(label, thresholds)
        if not present and not include_absent:
            continue

        laterality = location = None
        if present and localizer is not None and heatmaps is not None and label in heatmaps:
            laterality, location = localizer(np.asarray(heatmaps[label], dtype=np.float32))

        findings.append(
            Finding(
                label=label,
                probability=float(prob),
                present=present,
                confidence=confidence,
                laterality=laterality,
                location=location,
                source="classification+gradcam" if laterality or location else "classification",
            )
        )
    # most clinically salient first
    findings.sort(key=lambda f: f.probability, reverse=True)
    return findings


def _as_numpy(mask: "np.ndarray | torch.Tensor") -> np.ndarray:
    if hasattr(mask, "detach"):  # torch tensor
        return mask.detach().cpu().numpy()
    return np.asarray(mask)


def findings_from_mask(
    mask: "np.ndarray | torch.Tensor",
    label: str,
    *,
    spacing: tuple[float, ...] | None = None,
    confidence: float | None = None,
    probability: float = 1.0,
    min_voxels: int = 10,
    laterality_axis: int = -1,
    region_namer: Callable[[tuple[float, ...]], str | None] | None = None,
) -> list[Finding]:
    """Measure findings off a binary/integer segmentation mask via connected components.

    This is the brain-MRI path. Each connected blob becomes one `Finding`, with:
      - `volume_ml`  = voxel count x voxel volume (from `spacing`, mm) / 1000
      - `size_mm`    = largest bounding-box extent in mm (an upper-bound diameter proxy)
      - `laterality` = centroid vs. the mid-plane along `laterality_axis`
      - `location`   = optional, via a `region_namer` (e.g. an atlas lookup on the centroid)

    `spacing` is voxel size in mm ordered to match the mask axes. Without it, sizes are
    reported in voxels inside `extra` rather than mm, so we never fake physical units.
    """
    from scipy import ndimage  # local import: scipy is heavy and only needed here

    arr = _as_numpy(mask)
    binary = arr > 0
    if not binary.any():
        return []

    labeled, n = ndimage.label(binary)
    ndim = binary.ndim
    spacing = tuple(spacing) if spacing is not None else None
    voxel_vol_mm3 = float(np.prod(spacing)) if spacing else None
    axis = laterality_axis % ndim
    midline = binary.shape[axis] / 2.0

    findings: list[Finding] = []
    for comp in range(1, n + 1):
        coords = np.argwhere(labeled == comp)
        voxel_count = int(coords.shape[0])
        if voxel_count < min_voxels:
            continue

        extent_vox = coords.max(axis=0) - coords.min(axis=0) + 1  # bbox side lengths
        centroid = coords.mean(axis=0)

        size_mm = volume_ml = None
        extra: dict[str, Any] = {"voxels": voxel_count}
        if spacing is not None and len(spacing) == ndim:
            extent_mm = extent_vox * np.asarray(spacing)
            size_mm = float(extent_mm.max())
            volume_ml = voxel_count * voxel_vol_mm3 / 1000.0
        else:
            extra["bbox_voxels"] = extent_vox.tolist()

        side = centroid[axis]
        laterality = "left" if side < midline else "right"

        location = region_namer(tuple(centroid)) if region_namer is not None else None

        findings.append(
            Finding(
                label=label,
                probability=float(probability),
                present=True,
                confidence=confidence,
                size_mm=size_mm,
                volume_ml=volume_ml,
                laterality=laterality,
                location=location,
                source="segmentation",
                extra=extra,
            )
        )

    # collapse multiplicity into a count on the largest finding when several blobs share a label
    findings.sort(key=lambda f: (f.volume_ml or f.extra.get("voxels", 0)), reverse=True)
    for f in findings:
        f.count = len(findings)
    return findings


class GridZoneLocalizer:
    """Map a heatmap's center of mass to a coarse anatomical zone.

    For chest X-ray there is no mask, so 'where' is inferred from a per-finding
    Grad-CAM map: split the frame into lateral halves and vertical bands and report
    which one carries the activation mass. Deliberately coarse — it states "right
    lower zone", not pixel coordinates, which is the honest resolution of a CAM.

    Radiographic convention: on a frontal chest film the patient's right is on the
    *image left*. Set `radiological=True` (default) to report patient-side laterality.
    """

    def __init__(
        self,
        vertical_bands: Sequence[str] = ("upper zone", "mid zone", "lower zone"),
        radiological: bool = True,
        min_mass: float = 1e-6,
    ) -> None:
        self.vertical_bands = tuple(vertical_bands)
        self.radiological = radiological
        self.min_mass = min_mass

    def __call__(self, heatmap: np.ndarray) -> tuple[str | None, str | None]:
        hm = np.asarray(heatmap, dtype=np.float32)
        hm = np.clip(hm - hm.min(), 0, None)  # shift to non-negative weights
        total = hm.sum()
        if hm.ndim != 2 or total <= self.min_mass:
            return None, None

        rows, cols = hm.shape
        r_idx = np.arange(rows)[:, None]
        c_idx = np.arange(cols)[None, :]
        row_cm = float((hm * r_idx).sum() / total)
        col_cm = float((hm * c_idx).sum() / total)

        band = self.vertical_bands[min(int(row_cm / rows * len(self.vertical_bands)),
                                       len(self.vertical_bands) - 1)]

        image_left = col_cm < cols / 2.0
        # image-left == patient-right under radiographic convention
        laterality = ("right" if image_left else "left") if self.radiological else (
            "left" if image_left else "right")
        return laterality, band
