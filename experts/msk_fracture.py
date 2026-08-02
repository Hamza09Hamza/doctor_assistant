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
a software-integration candidate; swapping in ResCBAM later is a possible weights change
only after compatibility and real-data evaluation justify the extra integration cost.

Nothing here is trained locally; this adapter and checkpoint are not yet validated for
the project's target setting. `ultralytics` imports lazily so
importing this module stays cheap offline. The checkpoint isn't on PyPI/HF, so unlike the
other adapters we download it ourselves (once, cached locally) from its GitHub release —
no API, no network at inference.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
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
_WEIGHTS_BYTES = 22_484_659
_WEIGHTS_SHA256 = "301abc1774c28dd7c5adbcf1e8a79ed6771273615238b62dd179b622292b3a81"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_weights(
    path: str,
    expected_sha256: str,
    *,
    expected_bytes: int | None = None,
) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"fracture checkpoint does not exist: {path}")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("weights_sha256 must be a lowercase 64-character SHA-256")
    if expected_bytes is not None and os.path.getsize(path) != expected_bytes:
        raise RuntimeError(
            f"fracture checkpoint byte-size mismatch: {path}"
        )
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"fracture checkpoint SHA-256 mismatch: got {actual}, "
            f"expected {expected_sha256}"
        )


def _default_weights_path() -> str:
    if os.path.isfile(_WEIGHTS_CACHE):
        try:
            _verify_weights(
                _WEIGHTS_CACHE,
                _WEIGHTS_SHA256,
                expected_bytes=_WEIGHTS_BYTES,
            )
        except RuntimeError:
            # Keep the suspect file in place until a complete verified replacement
            # is ready; os.replace below makes recovery atomic.
            pass
        else:
            return _WEIGHTS_CACHE

    cache_dir = os.path.dirname(_WEIGHTS_CACHE)
    os.makedirs(cache_dir, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=".yolov8_fracture_best.",
            suffix=".download",
            delete=False,
        ) as handle:
            temporary = handle.name
        urllib.request.urlretrieve(_WEIGHTS_URL, temporary)
        _verify_weights(
            temporary,
            _WEIGHTS_SHA256,
            expected_bytes=_WEIGHTS_BYTES,
        )
        os.replace(temporary, _WEIGHTS_CACHE)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
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
        weights_sha256: str | None = None,
        allow_unverified_weights: bool = False,
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.BONE
        self.weights_path = weights_path
        self.weights_sha256 = weights_sha256
        self.allow_unverified_weights = bool(allow_unverified_weights)
        self.confidence = float(confidence)
        self.device = device
        self.class_names: list[str] = list(GRAZPEDWRI_LABELS)
        self._model = None
        # Provenance: the checksum of whichever weights this instance will load — the
        # verified default's known SHA-256, a caller-supplied one, or an explicit
        # "unverified" marker rather than a fabricated identity.
        if weights_sha256 is not None:
            self.version = f"msk_fracture_yolov8:{weights_sha256}"
        elif weights_path is None:
            self.version = f"msk_fracture_yolov8:{_WEIGHTS_SHA256}"
        else:
            self.version = "msk_fracture_yolov8:custom_unverified"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if self.weights_path is None:
            resolved_weights = _default_weights_path()
        else:
            resolved_weights = os.path.expanduser(self.weights_path)
            if self.weights_sha256 is None:
                if not self.allow_unverified_weights:
                    raise ValueError(
                        "custom fracture weights require weights_sha256; set "
                        "allow_unverified_weights=True only for an explicit "
                        "non-evidence experiment"
                    )
            else:
                _verify_weights(resolved_weights, self.weights_sha256)
        from ultralytics import YOLO

        self._model = YOLO(resolved_weights)

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
