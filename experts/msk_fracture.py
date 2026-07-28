"""MSK fracture detection — pretrained YOLOv8 wrapped as an expert.

Why this exists: MURA (the usual academic fracture-detection benchmark) never publishes
model weights by design — its leaderboard scores against a hidden test set, so there is
nothing to actually download and run. GRAZPEDWRI-DX (Nagy et al.) — 20,327 real pediatric
wrist trauma X-rays — has an actively maintained, MIT-licensed, weights-included
alternative instead: RuiyangJu et al.'s YOLOv8 fracture detector (Sci Rep 2023). We wrap
its release checkpoint as an `ExpertModel`, same contract as every other expert here:

  * `predict(scan)` runs the pretrained YOLOv8 net and stores its raw detections
    (class id, confidence, bounding box) in `Prediction.meta.extra["detections"]`.
  * `findings_from_prediction(scan, pred)` (the hook the pipeline looks for) turns each
    detection into a `Finding` — no custom classification/segmentation decoder applies
    here since this expert's output is bounding boxes, not per-label scores or a mask.

A more accurate variant exists (YOLOv8-ResCBAM, +2.2 AP50 per the paper) but it requires
the authors' modified `ultralytics` fork to load its custom attention modules — real
integration risk for a research repo of unknown ongoing maintenance. This plain-YOLOv8
checkpoint loads through the stock `ultralytics` package, so it's the reliable choice for
a working prototype; swapping in ResCBAM later is a drop-in weights change if the extra
accuracy is ever worth that integration cost.

Nothing here is trained; it is a deploy-and-go expert. `ultralytics` imports lazily so
importing this module stays cheap offline. The checkpoint isn't on PyPI/HF, so unlike the
other adapters we download it ourselves (once, cached locally) from its GitHub release —
no API, no network at inference.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Sequence

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan
from reporting.findings import Finding

# GRAZPEDWRI-DX's 9-class vocabulary, in the index order the checkpoint was trained on
# (from the repo's meta.yaml). "fracture", "boneanomaly", "bonelesion" are the clinically
# meaningful findings; the rest are dataset artifacts/signs (metal hardware, text markers
# burned into the film, a specific positioning sign) kept for completeness/traceability.
GRAZPEDWRI_LABELS: tuple[str, ...] = (
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text",
)

_WEIGHTS_URL = (
    "https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8/"
    "releases/download/Trained_model/best.pt"
)
_WEIGHTS_CACHE = os.path.expanduser("~/.cache/doctor_assistant_weights/yolov8_fracture_best.pt")


def _default_weights_path() -> str:
    if not os.path.isfile(_WEIGHTS_CACHE):
        os.makedirs(os.path.dirname(_WEIGHTS_CACHE), exist_ok=True)
        urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_CACHE)
    return _WEIGHTS_CACHE


def findings_from_detections(
    detections: Sequence[tuple[int, float, Sequence[float]]],
    class_names: Sequence[str],
) -> list[Finding]:
    """Pure, testable core: (cls_id, confidence, xyxy-box) tuples -> `Finding`s.

    Deliberately reports no laterality or anatomical zone: unlike chest X-ray, an
    isolated limb film carries no fixed radiographic convention to read a side off a
    bounding box from (it could be either wrist, in any rotation), so we'd be guessing.
    The box goes in `extra` instead — a real, traceable fact rather than a fabricated one.
    """
    counts: dict[str, int] = {}
    for cls_id, _, _ in detections:
        label = class_names[int(cls_id)]
        counts[label] = counts.get(label, 0) + 1

    findings = [
        Finding(
            label=(label := class_names[int(cls_id)]),
            probability=float(conf),
            present=True,
            count=counts[label],
            source="yolo-detection",
            extra={"bbox_xyxy": [float(v) for v in xyxy]},
        )
        for cls_id, conf, xyxy in detections
    ]
    findings.sort(key=lambda f: f.probability, reverse=True)
    return findings


def _scan_to_rgb_uint8(data):
    """(C,H,W) or (H,W) tensor, any range -> (H,W,3) uint8 array, what ultralytics expects."""
    import numpy as np

    x = data.as_tensor() if hasattr(data, "as_tensor") else data
    x = x.detach().float()
    if x.ndim == 2:                # (H, W) -> (1, H, W)
        x = x.unsqueeze(0)
    if x.shape[0] == 1:             # grayscale -> 3-channel
        x = x.repeat(3, 1, 1)
    elif x.shape[0] > 3:
        x = x[:3]
    lo, hi = float(x.min()), float(x.max())
    x = (x - lo) / (hi - lo + 1e-8)  # -> [0, 1]
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # (H, W, 3)
    return arr


class MSKFractureExpert:
    """Pretrained YOLOv8 wrist-fracture detector as a routable (XRAY, BONE) expert.

    Trained on pediatric wrist trauma X-rays; register it under (XRAY, BONE). `confidence`
    is YOLO's own detection threshold (not the pipeline's finding threshold — this expert's
    findings are pre-filtered at detection time, so `Pipeline(thresholds=...)` mostly acts
    as a secondary floor here). The model loads lazily on the first `predict`.
    """

    def __init__(
        self,
        *,
        name: str = "msk_fracture_yolov8",
        weights_path: str | None = None,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.BONE
        self.weights_path = weights_path
        self.confidence = float(confidence)
        self.device = device
        self.class_names: list[str] = list(GRAZPEDWRI_LABELS)
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.weights_path or _default_weights_path())

    def predict(self, scan: Scan) -> Prediction:
        self._ensure_loaded()
        img = _scan_to_rgb_uint8(scan.data)
        results = self._model.predict(
            img, conf=self.confidence, device=self.device, verbose=False
        )[0]
        boxes = results.boxes

        detections = list(zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ))

        by_label: dict[str, float] = {}
        for cls_id, conf, _ in detections:
            label = self.class_names[int(cls_id)]
            by_label[label] = max(by_label.get(label, 0.0), conf)

        pred = Prediction(expert=self.name, meta=scan.meta)
        pred.class_probs = by_label
        pred.confidence = max(by_label.values()) if by_label else None
        pred.meta.extra = dict(pred.meta.extra or {})
        pred.meta.extra["detections"] = detections
        return pred

    def findings_from_prediction(self, scan: Scan, pred: Prediction) -> list[Finding]:
        detections = pred.meta.extra.get("detections", [])
        return findings_from_detections(detections, self.class_names)
