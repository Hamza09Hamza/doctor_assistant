"""`BaseExpert` — one shared backbone wired to several task heads.

This is where the architecture's core idea lives: a single forward pass through
the encoder feeds every head, so the model is trained multi-task and the heads
regularize one another. `BaseExpert` satisfies the `ExpertModel` protocol, so the
router and orchestrator treat any expert uniformly via `.predict(scan)`.
"""

from __future__ import annotations

import torch
from torch import nn

from core.enums import BodyPart, Modality, TaskType
from core.types import HeadOutput, Prediction, Scan
from .backbones import Backbone
from .heads import Head, SegmentationHead


class BaseExpert(nn.Module):
    name: str
    modality: Modality
    body_part: BodyPart

    def __init__(
        self,
        name: str,
        modality: Modality,
        body_part: BodyPart,
        backbone: Backbone,
        heads: dict[str, Head],
        class_names: list[str] | None = None,
        preprocess=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.modality = modality
        self.body_part = body_part
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.class_names = class_names or []
        self.preprocess = preprocess

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> dict[str, HeadOutput]:
        """Encode once, run every head. Segmentation heads get the input size."""
        feat = self.backbone(x)
        out_size = tuple(x.shape[2:])
        outputs: dict[str, HeadOutput] = {}
        for head_name, head in self.heads.items():
            if isinstance(head, SegmentationHead):
                tensor = head(feat, out_size=out_size)
            else:
                tensor = head(feat)
            outputs[head_name] = HeadOutput(task=head.task, name=head_name, tensor=tensor)
        return outputs

    @torch.no_grad()
    def predict(self, scan: Scan) -> Prediction:
        """Run the full forward pass on one scan and decode it into a `Prediction`."""
        self.eval()
        data = self.preprocess(scan.data) if self.preprocess is not None else scan.data
        x = data.unsqueeze(0).to(self.device)  # add batch dim
        outputs = self.forward(x)

        prediction = Prediction(expert=self.name, meta=scan.meta)
        for out in outputs.values():
            t = out.tensor[0]  # drop batch dim
            if out.task is TaskType.CLASSIFICATION:
                probs = torch.softmax(t, dim=0).cpu()
                names = self._class_names(probs.numel())
                prediction.class_probs = {n: float(p) for n, p in zip(names, probs)}
            elif out.task is TaskType.SEGMENTATION:
                prediction.segmentation = self._decode_mask(t).cpu()
            elif out.task is TaskType.CONFIDENCE:
                prediction.confidence = float(torch.sigmoid(t))
        return prediction

    def _class_names(self, n: int) -> list[str]:
        if len(self.class_names) == n:
            return self.class_names
        return [f"class_{i}" for i in range(n)]

    @staticmethod
    def _decode_mask(logits: torch.Tensor) -> torch.Tensor:
        """(C, *spatial) logits -> integer label mask; binary if single channel."""
        if logits.shape[0] == 1:
            return (torch.sigmoid(logits[0]) > 0.5).long()
        return logits.argmax(dim=0).long()
