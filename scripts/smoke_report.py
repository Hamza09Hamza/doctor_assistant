"""Smoke test for the findings -> report path. Runs without timm/monai.

Builds a tiny real expert (so Grad-CAM actually backprops), exercises both finding
extractors, and renders a report through the deterministic path. Run from repo root:

    PYTHONPATH=. python3 scripts/smoke_report.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata
from explainability import GradCAM
from models.backbones import Backbone
from models.experts import BaseExpert
from models.heads import ClassificationHead, ConfidenceHead
from reporting import (
    GridZoneLocalizer,
    Reporter,
    findings_from_classification,
    findings_from_mask,
)


class TinyBackbone(Backbone):
    """A 3-layer conv encoder standing in for DenseNet so the test needs no timm."""

    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, out_channels, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.out_channels = out_channels
        self.spatial_dims = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_tiny_chest_expert() -> BaseExpert:
    labels = ["Cardiomegaly", "Effusion", "Pneumothorax", "Nodule"]
    bb = TinyBackbone()
    return BaseExpert(
        name="chest_xray_tiny",
        modality=Modality.XRAY,
        body_part=BodyPart.CHEST,
        backbone=bb,
        heads={
            "cls": ClassificationHead(bb.out_channels, len(labels), multilabel=True),
            "confidence": ConfidenceHead(bb.out_channels),
        },
        class_names=labels,
        preprocess=None,
    )


def demo_chest() -> None:
    print("\n=== CHEST X-RAY: classification + Grad-CAM -> findings -> report ===")
    torch.manual_seed(0)
    expert = build_tiny_chest_expert()
    scan = Scan(
        data=torch.rand(3, 64, 64),
        meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST),
    )

    pred = expert.predict(scan)
    probs_sum = sum(pred.class_probs.values())
    print("class_probs:", {k: round(v, 3) for k, v in pred.class_probs.items()})
    print(f"sum of probs = {probs_sum:.3f}  (multi-label: need NOT be 1.0)")
    print("confidence:", round(pred.confidence, 3) if pred.confidence is not None else None)

    # Grad-CAM gives a per-label heatmap; the localizer turns it into a lung zone.
    cams = GradCAM(expert).for_labels(scan.data)
    findings = findings_from_classification(
        pred.class_probs,
        thresholds=0.45,
        confidence=pred.confidence,
        heatmaps=cams,
        localizer=GridZoneLocalizer(),
    )
    print(f"\n{len(findings)} finding(s) above threshold:")
    for f in findings:
        print(" -", f.to_facts())

    report = Reporter(auto_llm=False).report(findings, scan.meta)  # deterministic path
    print(f"\n--- REPORT (generator: {report.generator}) ---")
    print(report.to_text())


def demo_brain_mask() -> None:
    print("\n=== BRAIN MRI: mask geometry -> measured findings ===")
    # Synthetic 3D mask (D,H,W): a blob on the left half.
    mask = np.zeros((40, 64, 64), dtype=np.int64)
    mask[15:25, 20:40, 8:28] = 1  # ~10 x 20 x 20 voxels
    spacing = (2.0, 1.0, 1.0)  # mm per voxel (z, y, x)

    findings = findings_from_mask(
        mask, label="glioma", spacing=spacing, confidence=0.82, probability=0.89
    )
    for f in findings:
        print(" -", f.to_facts())

    meta = ScanMetadata(modality=Modality.MRI, body_part=BodyPart.BRAIN)
    report = Reporter(auto_llm=False).report(findings, meta)
    print(f"\n--- REPORT (generator: {report.generator}) ---")
    print(report.to_text())


if __name__ == "__main__":
    demo_chest()
    demo_brain_mask()
    print("\nOK: findings -> report path works end to end.")
