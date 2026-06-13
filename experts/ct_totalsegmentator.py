"""CT organ-segmentation expert wrapping TotalSegmentator — no training required.

TotalSegmentator (Wasserthal et al.) segments 100+ anatomical structures from a CT
volume with a pretrained nnU-Net, at Dice scores most labs can't beat from scratch. We
wrap it as an `ExpertModel` so the router/orchestrator treat it like any other expert:

  * `predict(scan)` runs TotalSegmentator on the scan's *on-disk* volume — it needs the
    spatially-resolved NIfTI/DICOM (real voxel spacing), not the resized training tensor —
    and returns the multilabel mask plus the id→structure map in a `Prediction`.
  * `findings_from_prediction(scan, pred)` (the optional hook the pipeline looks for)
    turns each segmented structure into a measured `Finding`: volume in mL from the voxel
    count × spacing, and a bounding-box extent in mm. These are real measurements, so the
    reporter verbalizes them and the verifier can ground every number against them.

Nothing here is trained; it is a deploy-and-go expert. The heavy deps (`totalsegmentator`,
`nibabel`, `numpy`) are imported lazily so importing this module stays cheap offline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from reporting.findings import Finding


def findings_from_label_counts(
    counts: dict[int, int],
    class_map: dict[int, str],
    spacing: tuple[float, ...] | None,
    extents_vox: dict[int, Sequence[int]] | None = None,
    *,
    min_volume_ml: float = 0.0,
    confidence: float | None = None,
) -> list[Finding]:
    """Build measured `Finding`s from per-label voxel counts (the pure, testable core).

    `counts` maps a structure's integer label to its voxel count; `class_map` maps the
    same ids to names. With `spacing` (mm per axis) we convert to physical units:
    `volume_ml = voxels × ∏spacing / 1000` and `size_mm = max(bbox_extent × spacing)`.
    Without spacing we report neither (never fake physical units) — only the count, in
    `extra`. Structures below `min_volume_ml` are dropped as noise.
    """
    voxel_vol_ml = (math.prod(spacing) / 1000.0) if spacing else None
    extents_vox = extents_vox or {}
    findings: list[Finding] = []
    for label_id, n in counts.items():
        if label_id == 0:  # background
            continue
        name = class_map.get(label_id, f"structure_{label_id}")
        volume_ml = (n * voxel_vol_ml) if voxel_vol_ml is not None else None
        if volume_ml is not None and volume_ml < min_volume_ml:
            continue

        size_mm = None
        ext = extents_vox.get(label_id)
        if ext is not None and spacing is not None and len(spacing) == len(ext):
            size_mm = float(max(e * s for e, s in zip(ext, spacing)))

        findings.append(
            Finding(
                label=name,
                probability=1.0,  # segmentation is a hard assignment, not a score
                present=True,
                confidence=confidence,
                volume_ml=volume_ml,
                size_mm=size_mm,
                source="ct-totalsegmentator",
                extra={"voxels": int(n)} if volume_ml is None else {},
            )
        )
    findings.sort(key=lambda f: (f.volume_ml or f.extra.get("voxels", 0)), reverse=True)
    return findings


class TotalSegmentatorExpert:
    """Pretrained CT organ segmentation as a routable expert.

    `roi_subset` restricts inference to named structures (much faster, e.g.
    `["liver", "spleen", "kidney_left", "kidney_right"]`); `fast=True` uses the 3 mm
    model. Register one instance under every CT niche it should serve via
    `ExpertRegistry.register_niche` — the same weights read chest and abdominal CT.
    """

    def __init__(
        self,
        *,
        name: str = "ct_totalsegmentator",
        body_part: BodyPart = BodyPart.ABDOMEN,
        fast: bool = True,
        roi_subset: Sequence[str] | None = None,
        min_volume_ml: float = 1.0,
        quiet: bool = True,
    ) -> None:
        self.name = name
        self.modality = Modality.CT
        self.body_part = body_part
        self.fast = fast
        self.roi_subset = list(roi_subset) if roi_subset else None
        self.min_volume_ml = float(min_volume_ml)
        self.quiet = quiet
        # Advertised vocabulary (used by the verifier's "named-but-not-present" check).
        self.class_names: list[str] = list(self.roi_subset) if self.roi_subset else []

    def predict(self, scan: Scan) -> Prediction:
        import nibabel as nib  # noqa: F401  (kept explicit so a missing dep is obvious)
        import numpy as np
        from totalsegmentator.python_api import totalsegmentator

        src = scan.meta.source_path
        if not src:
            raise ValueError(
                "TotalSegmentatorExpert needs scan.meta.source_path — the on-disk CT "
                "(NIfTI file or DICOM directory) with real voxel spacing."
            )

        seg_img = totalsegmentator(
            src,
            output=None,
            ml=True,                 # one multilabel volume, not per-structure files
            fast=self.fast,
            roi_subset=self.roi_subset,
            quiet=self.quiet,
        )
        arr = np.asarray(seg_img.dataobj)
        spacing = tuple(float(z) for z in seg_img.header.get_zooms()[:arr.ndim])

        pred = Prediction(expert=self.name, meta=scan.meta)
        import torch

        pred.segmentation = torch.as_tensor(arr.astype("int64"))
        pred.confidence = 0.9  # nnU-Net is strong, but this is not a calibrated score
        pred.meta.extra = dict(pred.meta.extra or {})
        pred.meta.extra["seg_spacing"] = spacing
        pred.meta.extra["label_map"] = self._class_map()
        return pred

    def findings_from_prediction(self, scan: Scan, pred: Prediction) -> list[Finding]:
        import numpy as np

        if pred.segmentation is None:
            return []
        seg = pred.segmentation
        arr = seg.detach().cpu().numpy() if hasattr(seg, "detach") else np.asarray(seg)
        spacing = pred.meta.extra.get("seg_spacing")
        class_map = pred.meta.extra.get("label_map") or self._class_map()

        labels, counts = np.unique(arr, return_counts=True)
        count_by_id = {int(l): int(c) for l, c in zip(labels, counts) if int(l) != 0}
        extents = self._bbox_extents(arr, count_by_id.keys())
        return findings_from_label_counts(
            count_by_id,
            class_map,
            spacing,
            extents,
            min_volume_ml=self.min_volume_ml,
            confidence=pred.confidence,
        )

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _bbox_extents(arr, label_ids) -> dict[int, list[int]]:
        """Per-label bounding-box side lengths in voxels (for a size_mm proxy)."""
        import numpy as np

        extents: dict[int, list[int]] = {}
        for label_id in label_ids:
            coords = np.argwhere(arr == label_id)
            if coords.size == 0:
                continue
            extents[int(label_id)] = (coords.max(0) - coords.min(0) + 1).tolist()
        return extents

    @staticmethod
    def _class_map() -> dict[int, str]:
        """Integer label → structure name, from TotalSegmentator's own mapping."""
        from totalsegmentator.map_to_binary import class_map

        return {int(k): v for k, v in class_map["total"].items()}
