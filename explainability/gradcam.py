"""Grad-CAM — *where* the expert looked, for findings that have no mask.

Chest X-ray classification gives a finding's *kind* and *score* but no pixels. To put
a finding in a lung zone we ask: which region of the shared feature map drove this
class's logit? Grad-CAM answers that by weighting the encoder's activations by the
gradient of the target logit, giving a coarse saliency map. That map feeds a
`Localizer` (see `reporting.findings.GridZoneLocalizer`) to label laterality/zone, and
it is also what the `Prediction.heatmap` field surfaces for the radiologist to eyeball.

It taps the *shared backbone* feature map — the exact tensor every head reads — so the
explanation is faithful to what the model actually used, and it works for any expert
built on `BaseExpert` (2D or 3D) without per-model wiring.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from core.enums import TaskType


class GradCAM:
    """Grad-CAM over an expert's shared backbone, for a chosen classification head.

    Usage:
        cam = GradCAM(expert)                 # picks the sole classification head
        maps = cam(x, targets=[3, 7])         # {class_idx: heatmap} for those classes
        named = cam.for_labels(x)             # {label: heatmap} for the top labels
    """

    def __init__(self, expert, head_name: str | None = None) -> None:
        self.expert = expert
        self.head_name = head_name or self._infer_classification_head(expert)
        self.spatial_dims = getattr(expert.backbone, "spatial_dims", 2)

    @staticmethod
    def _infer_classification_head(expert) -> str:
        for name, head in expert.heads.items():
            if getattr(head, "task", None) is TaskType.CLASSIFICATION:
                return name
        raise ValueError("expert has no classification head to explain")

    def __call__(
        self, x: torch.Tensor, targets: Sequence[int] | None = None
    ) -> dict[int, torch.Tensor]:
        """Return {class_index: heatmap} for `x` (single image, batch dim allowed = 1).

        Each heatmap is upsampled to the input's spatial size and normalized to [0, 1].
        With `targets=None` it explains the single argmax class.
        """
        self.expert.eval()
        if x.dim() == self.spatial_dims + 1:  # (C, ...) -> add batch
            x = x.unsqueeze(0)
        x = x.to(self.expert.device)

        # Forward only the path we need, keeping the feature map in the graph.
        feat = self.expert.backbone(x)
        feat.retain_grad()
        logits = self.expert.heads[self.head_name](feat)  # (1, num_classes)

        if targets is None:
            targets = [int(logits[0].argmax())]

        spatial = tuple(x.shape[2:])
        cams: dict[int, torch.Tensor] = {}
        for cls in targets:
            self.expert.zero_grad(set_to_none=True)
            if feat.grad is not None:
                feat.grad = None
            logits[0, cls].backward(retain_graph=True)
            cams[int(cls)] = self._weight_and_pool(feat.detach(), feat.grad.detach(), spatial)
        return cams

    def for_labels(
        self, x: torch.Tensor, labels: Sequence[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Same as `__call__` but keyed by class name (uses `expert.class_names`)."""
        names = self.expert.class_names
        if not names:
            raise ValueError("expert.class_names is empty; cannot map labels")
        wanted = labels if labels is not None else names
        idx = [names.index(l) for l in wanted]
        cams = self(x, targets=idx)
        return {names[i]: cams[i] for i in idx}

    def _weight_and_pool(
        self, feat: torch.Tensor, grad: torch.Tensor, out_size: tuple[int, ...]
    ) -> torch.Tensor:
        """Grad-weighted, ReLU'd, upsampled, [0,1]-normalized saliency map."""
        reduce_dims = tuple(range(2, feat.dim()))  # spatial axes
        weights = grad.mean(dim=reduce_dims, keepdim=True)  # importance per channel
        cam = (weights * feat).sum(dim=1, keepdim=True)     # (1, 1, *spatial')
        cam = F.relu(cam)
        mode = "bilinear" if self.spatial_dims == 2 else "trilinear"
        cam = F.interpolate(cam, size=out_size, mode=mode, align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu()
