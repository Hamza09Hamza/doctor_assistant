"""Multi-task loss: one number that trains every head at once.

The loss sums a weighted per-head term over whatever heads are present and have a
target available, so the same loss object works for a classification-only expert
or a classification+segmentation+confidence one. This shared-gradient training is
what makes the heads help each other (see models/heads.py).

The confidence head is special: it has no ground-truth label. We train it to
predict whether the classifier was *correct* on this example — a self-supervised
"how much should you trust me" signal.
"""

from __future__ import annotations

import torch
from torch import nn

from core.enums import TaskType
from core.types import HeadOutput


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = weights or {}
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self._seg_loss = None  # built lazily so MONAI import isn't required for cls-only

    def _seg(self):
        if self._seg_loss is None:
            from monai.losses import DiceCELoss

            self._seg_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        return self._seg_loss

    def forward(
        self, outputs: dict[str, HeadOutput], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cls_logits = _classification_logits(outputs)
        total: torch.Tensor | None = None
        components: dict[str, float] = {}

        for name, out in outputs.items():
            loss = self._term(out, targets, cls_logits)
            if loss is None:
                continue
            weighted = self.weights.get(name, 1.0) * loss
            total = weighted if total is None else total + weighted
            components[name] = float(loss.detach())

        if total is None:
            raise ValueError("No head had a usable target; nothing to optimize.")
        components["total"] = float(total.detach())
        return total, components

    def _term(self, out: HeadOutput, targets, cls_logits):
        if out.task is TaskType.CLASSIFICATION and "label" in targets:
            return self.ce(out.tensor, targets["label"])
        if out.task is TaskType.SEGMENTATION and "mask" in targets:
            return self._seg()(out.tensor, targets["mask"])
        if out.task is TaskType.CONFIDENCE and cls_logits is not None and "label" in targets:
            correct = (cls_logits.argmax(dim=1) == targets["label"]).float().detach()
            return self.bce(out.tensor, correct)
        return None


def _classification_logits(outputs: dict[str, HeadOutput]) -> torch.Tensor | None:
    for out in outputs.values():
        if out.task is TaskType.CLASSIFICATION:
            return out.tensor
    return None
