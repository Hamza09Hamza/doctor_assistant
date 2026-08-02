"""TorchXRayVision chest-X-ray engineering control, wrapped as an expert.

Why this exists: training our own DenseNet on a Colab budget only reached ~0.74 AUC with
badly-calibrated logits (every probability squashed below ~0.15), so real findings never
crossed threshold. TorchXRayVision (Cohen et al.) ships DenseNet-121 weights trained on
several different combinations of NIH ChestX-ray14 + CheXpert + MIMIC-CXR + PadChest, with
multi-label ranking scores and published operating-point normalization. Because its
``all`` weights include NIH training data, it is not an independent NIH comparator and
its normalized scores are not accepted probabilities for this project. We wrap it as an
`ExpertModel` so the router/orchestrator can use it as an explicit engineering control:

  * `predict(scan)` runs the pretrained net(s) and fills `Prediction.class_probs` with the
    subset of TorchXRayVision's pathologies that match our ChestX-ray14 vocabulary. The
    pipeline's threshold step turns those scores into `Finding`s — no custom hook needed.

Nothing here is trained locally. Like the other adapters, the heavy
deps (`torchxrayvision`, `torch`) import lazily, so importing this module stays cheap
offline. Weights download once from the TorchXRayVision release and are cached locally — no
API, no network at inference (the project rule: local weights only).

Ensembling: `weights` takes *either* a single string (one model) or a sequence of them
(loads each, averages their op-norm-calibrated scores per pathology) — the machinery exists
because it is a valid experiment. But don't reach for it by default: measured on 300
images from an unverified third-party mirror partition named ``test``
(`scripts/eval_chest_xrv.py`), plain "all" alone scored
0.7582 macro AUC, and every ensemble compared with "all" scored *worse* than "all" alone —
"all+nih" 0.7558 (a wash), "all+nih+chex" 0.7327, "nih" alone 0.7325, "chex" alone 0.5960.
Two reasons this "free win" doesn't materialize here: "all" is trained on the *union*
including nih and chex, so adding those checkpoints back in isn't an independent second
opinion, just diluted noise; and "chex" alone is missing several NIH pathologies from its
own label vocabulary entirely (Fibrosis/Infiltration/Mass/Nodule/Pleural_Thickening came
back at exactly 0.5 AUC — a flat, uncalibrated score), so folding it in actively hurts.
That observation is exploratory: source filenames were unavailable, so the rows could
not be reconciled against NIH's official manifests. Re-run under a provenance-checked
protocol before trusting an ensemble config; it is not a held-out performance claim or a
general "ensembling doesn't work" claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan

from .chest_xray import CHESTXRAY14_LABELS


class TorchXRayVisionExpert:
    """Pretrained TorchXRayVision DenseNet(s) as a routable (XRAY, CHEST) classifier.

    Register it under (XRAY, CHEST) — alongside the trained classifier if you want both, the
    router returns both and their findings pool. Model(s) load lazily on the first `predict`
    (GPU used when available, but small enough for CPU). Outputs are mapped to `labels`
    (default: the ChestX-ray14 14) so guidelines and the verifier key off the same vocabulary
    as the rest of the system.
    """

    def __init__(
        self,
        *,
        name: str = "chest_xrv",
        weights: str | Sequence[str] = "densenet121-res224-all",
        labels: Sequence[str] = CHESTXRAY14_LABELS,
        resolution: int = 224,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.modality = Modality.XRAY
        self.body_part = BodyPart.CHEST
        self.weights: tuple[str, ...] = (weights,) if isinstance(weights, str) else tuple(weights)
        self.resolution = int(resolution)
        self.device = device
        # Provenance: the exact released weight set(s) this instance loads.
        self.version = f"torchxrayvision:{'+'.join(self.weights)}"
        # Advertised vocabulary (the verifier's "named-but-not-present" check keys off this).
        self.class_names: list[str] = list(labels)
        self._models: list = []

    def _ensure_loaded(self) -> None:
        if self._models:
            return
        import torch
        import torchxrayvision as xrv

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models = [
            xrv.models.DenseNet(weights=w).eval().to(device) for w in self.weights
        ]
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

        # Sum calibrated per-pathology scores across the ensemble, dividing by how many
        # models actually calibrated that pathology (not len(self._models)) — a weight set
        # that never saw a label shouldn't dilute the average toward zero for it.
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for model in self._models:
            with torch.no_grad():
                raw = model(x).detach().float().cpu()  # (1, n_path), sigmoid probs
            # xrv applies the sigmoid in its forward; guard a weights variant that doesn't.
            if float(raw.min()) < 0.0 or float(raw.max()) > 1.0:
                raw = torch.sigmoid(raw)

            # RAW xrv scores are NOT comparable across pathologies — each has its own
            # operating point (model.op_threshs), so a flat threshold over-calls wildly (a
            # normal study lights up because everything clusters near 0.5). op_norm remaps
            # each score through its operating point so 0.5 == the calibrated decision
            # boundary; then one pipeline threshold is meaningful and normals stay quiet
            # while true findings still cross.
            op = getattr(model, "op_threshs", None)
            if op is not None:
                op = op.detach().float().cpu()
                scores = xrv.models.op_norm(raw, op)[0]
                uncalibrated = torch.isnan(op)
            else:
                scores = raw[0]
                uncalibrated = torch.zeros_like(scores, dtype=torch.bool)

            for i, path in enumerate(model.pathologies):
                # Pathologies this weight set never calibrated come back as a neutral 0.5 —
                # exclude them from this model's contribution rather than average them in
                # as "not reported"; another weight set in the ensemble may still cover it.
                if not path or bool(uncalibrated[i]):
                    continue
                sums[path] = sums.get(path, 0.0) + float(scores[i])
                counts[path] = counts.get(path, 0) + 1

        by_path = {p: sums[p] / counts[p] for p in sums}

        pred = Prediction(expert=self.name, meta=scan.meta)
        pred.class_probs = {lbl: by_path[lbl] for lbl in self.class_names if lbl in by_path}
        # Highest calibrated score doubles as a coarse study-level confidence.
        pred.confidence = max(pred.class_probs.values()) if pred.class_probs else None
        return pred
