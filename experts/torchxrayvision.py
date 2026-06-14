"""TorchXRayVision — a strong, already-trained chest-X-ray classifier, wrapped as an expert.

Why this exists: training our own DenseNet on a Colab budget only reached ~0.74 AUC with
badly-calibrated logits (every probability squashed below ~0.15), so real findings never
crossed threshold. TorchXRayVision (Cohen et al.) ships DenseNet-121 weights trained on the
*union* of NIH ChestX-ray14 + CheXpert + MIMIC-CXR + PadChest, with calibrated multi-label
outputs that actually fire on true pathology. We wrap it as an `ExpertModel` — same contract
as every other expert — so the router/orchestrator use it unchanged:

  * `predict(scan)` runs the pretrained net and fills `Prediction.class_probs` with the
    subset of TorchXRayVision's pathologies that match our ChestX-ray14 vocabulary. The
    pipeline's threshold step turns those scores into `Finding`s — no custom hook needed.

Nothing here is trained; it is a deploy-and-go expert. Like the other adapters, the heavy
deps (`torchxrayvision`, `torch`) import lazily, so importing this module stays cheap
offline. Weights download once from the TorchXRayVision release and are cached locally — no
API, no network at inference (the project rule: local weights only).
"""

from __future__ import annotations

from collections.abc import Sequence

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan

from .chest_xray import CHESTXRAY14_LABELS


class TorchXRayVisionExpert:
    """Pretrained TorchXRayVision DenseNet as a routable (XRAY, CHEST) classifier.

    `weights="densenet121-res224-all"` is the model trained on every public dataset at once
    — the most robust default. Register it under (XRAY, CHEST) — alongside the trained
    classifier if you want both, the router returns both and their findings pool. The model
    loads lazily on the first `predict` (GPU used when available, but it's small enough for
    CPU). Outputs are mapped to `labels` (default: the ChestX-ray14 14) so guidelines and the
    verifier key off the same vocabulary as the rest of the system.
    """

    def __init__(
        self,
        *,
        name: str = "chest_xrv",
        weights: str = "densenet121-res224-all",
        labels: Sequence[str] = CHESTXRAY14_LABELS,
        resolution: int = 224,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.CHEST
        self.weights = weights
        self.resolution = int(resolution)
        self.device = device
        # Advertised vocabulary (the verifier's "named-but-not-present" check keys off this).
        self.class_names: list[str] = list(labels)
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchxrayvision as xrv

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = xrv.models.DenseNet(weights=self.weights).eval().to(device)
        self.device = device

    def _preprocess(self, data):
        """(C,H,W) or (H,W) tensor -> (1,1,res,res) in TorchXRayVision's [-1024,1024] range.

        Robust to whatever upstream preprocessing produced `data`: we re-normalize by the
        image's own min/max, so a tensor already scaled to [0,1] (our pipeline) and a raw
        [0,255] image both land in the range the net was trained on.
        """
        import torch
        import torch.nn.functional as F

        x = data.as_tensor() if hasattr(data, "as_tensor") else data
        x = x.detach().float()
        if x.ndim == 2:                       # (H, W) -> (1, H, W)
            x = x.unsqueeze(0)
        if x.shape[0] > 1:                    # (C, H, W) -> single channel
            x = x.mean(dim=0, keepdim=True)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo + 1e-8)       # -> [0, 1]
        x = x * 2048.0 - 1024.0               # -> [-1024, 1024] (xrv convention)
        x = x.unsqueeze(0)                    # (1, 1, H, W)
        x = F.interpolate(
            x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        return x.to(self.device)

    def predict(self, scan: Scan) -> Prediction:
        import torch
        import torchxrayvision as xrv

        self._ensure_loaded()
        x = self._preprocess(scan.data)
        with torch.no_grad():
            raw = self._model(x).detach().float().cpu()  # (1, n_path), sigmoid probs
        # xrv applies the sigmoid in its forward; guard a weights variant that doesn't.
        if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
            raw = torch.sigmoid(raw)

        # RAW xrv scores are NOT comparable across pathologies — each has its own operating
        # point (model.op_threshs), so a flat threshold over-calls wildly (a normal study
        # lights up because everything clusters near 0.5). op_norm remaps each score through
        # its operating point so 0.5 == the calibrated decision boundary; then one pipeline
        # threshold is meaningful and normals stay quiet while true findings still cross.
        op = getattr(self._model, "op_threshs", None)
        if op is not None:
            op = op.detach().float().cpu()
            scores = xrv.models.op_norm(raw, op)[0]
            # Pathologies xrv never calibrated come back as a neutral 0.5; don't let that
            # trip the threshold — treat "uncalibrated" as "not reported".
            scores = torch.where(torch.isnan(op), torch.zeros_like(scores), scores)
        else:
            scores = raw[0]

        by_path = {p: float(v) for p, v in zip(self._model.pathologies, scores) if p}

        pred = Prediction(expert=self.name, meta=scan.meta)
        pred.class_probs = {lbl: by_path[lbl] for lbl in self.class_names if lbl in by_path}
        # Highest calibrated score doubles as a coarse study-level confidence.
        pred.confidence = max(pred.class_probs.values()) if pred.class_probs else None
        return pred
