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


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification (Ben-Baruch et al., 2021).

    Chest-X-ray-style multi-label data is heavily imbalanced *per class* (Hernia is
    ~0.2% positive, Infiltration ~18%), and every label is dominated by easy negatives.
    Plain BCE spends most of its gradient on those easy negatives, drowning out the
    rare-positive signal — the mechanism behind "14 labels tanks accuracy vs. a few".
    ASL applies a focal-style down-weighting to negatives only (`gamma_neg > gamma_pos`)
    and a probability-shifting `clip` on negatives, so confidently-correct negatives
    contribute almost nothing while positives keep a normal gradient.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "sum",
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        anti_targets = 1.0 - targets

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            # Shift negative-class probability up before the log so a negative sitting
            # just inside the decision boundary isn't penalized at all — only genuinely
            # wrong negatives contribute loss.
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss = loss + anti_targets * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # Match the official ASL implementation: the focal modulation is a
            # detached weight. Gradients flow through the positive/negative
            # log-likelihood term, not through the dynamically computed weight.
            with torch.no_grad():
                xs_pos_w = xs_pos * targets
                xs_neg_w = xs_neg * anti_targets
                asymmetric_w = torch.pow(
                    (1.0 - xs_pos_w - xs_neg_w).clamp(min=0.0),
                    self.gamma_pos * targets + self.gamma_neg * anti_targets,
                )
            loss = loss * asymmetric_w

        # The reference ASL uses a sum over batch and labels. Keep mean available
        # only as an explicit opt-in for callers that also retune the learning rate.
        return -loss.sum() if self.reduction == "sum" else -loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        multilabel: bool = False,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 1.0,
        asl_clip: float = 0.05,
        asl_reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.weights = weights or {}
        # multilabel=True: chest X-ray style — labels co-occur, each treated independently.
        # multilabel=False: brain-tumour style — mutually exclusive, softmax + CE.
        self.multilabel = multilabel
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()  # still used for the confidence head (see _term)
        # Multilabel classification uses ASL instead of plain BCE (see AsymmetricLoss
        # docstring); the confidence head's binary "was I correct" target is not the
        # imbalanced-label problem ASL targets, so it keeps self.bce regardless of this.
        self.multilabel_loss = (
            AsymmetricLoss(
                gamma_neg=asl_gamma_neg,
                gamma_pos=asl_gamma_pos,
                clip=asl_clip,
                reduction=asl_reduction,
            )
            if multilabel
            else None
        )
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
            if self.multilabel:
                # targets["label"] is a float multi-hot vector (batch, num_classes)
                return self.multilabel_loss(out.tensor, targets["label"])
            return self.ce(out.tensor, targets["label"])
        if out.task is TaskType.SEGMENTATION and "mask" in targets:
            return self._seg()(out.tensor, targets["mask"])
        if out.task is TaskType.CONFIDENCE and cls_logits is not None and "label" in targets:
            if self.multilabel:
                # "correct" = all labels matched at threshold 0.5
                pred = (cls_logits.sigmoid() > 0.5).float()
                correct = (pred == targets["label"].float()).all(dim=1).float().detach()
            else:
                correct = (cls_logits.argmax(dim=1) == targets["label"]).float().detach()
            return self.bce(out.tensor, correct)
        return None


def _classification_logits(outputs: dict[str, HeadOutput]) -> torch.Tensor | None:
    for out in outputs.values():
        if out.task is TaskType.CLASSIFICATION:
            return out.tensor
    return None
