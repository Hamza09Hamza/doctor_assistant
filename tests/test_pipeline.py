from __future__ import annotations

import unittest

import torch
from torch import nn

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan, ScanMetadata
from models.backbones import Backbone
from models.experts import BaseExpert
from models.heads import ClassificationHead
from pipeline import Pipeline, PipelineExecutionError
from reporting import GridZoneLocalizer, Reporter
from routing import ExpertRegistry, ModalityRouter


class FailingExpert:
    name = "broken"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Effusion"]

    def predict(self, scan):
        raise RuntimeError("weights unavailable")


class MutatingExpert:
    name = "mutator"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Effusion"]

    def predict(self, scan):
        scan.meta.extra["private"] = self.name
        return Prediction(
            expert=self.name,
            class_probs={"Effusion": 0.9},
            meta=scan.meta,
        )


class ObservingExpert:
    name = "observer"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Mass"]

    def predict(self, scan):
        if scan.meta.extra:
            raise AssertionError(f"metadata leaked: {scan.meta.extra}")
        return Prediction(
            expert=self.name,
            class_probs={"Mass": 0.1},
            meta=scan.meta,
        )


class TinyBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        nn.init.constant_(self.net.weight, 1.0)
        self.out_channels = 2
        self.spatial_dims = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _scan() -> Scan:
    data = torch.zeros(1, 16, 16)
    data[:, :8, :8] = 1.0
    return Scan(
        data=data,
        meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST),
    )


class PipelineTests(unittest.TestCase):
    def test_all_experts_failing_never_produces_a_normal_report(self) -> None:
        registry = ExpertRegistry()
        registry.register(FailingExpert())
        pipe = Pipeline(ModalityRouter(registry), reporter=Reporter(llm=None))

        with self.assertRaisesRegex(PipelineExecutionError, "All routed experts failed"):
            pipe.analyze_scan(_scan())

    def test_partial_failure_and_metadata_isolation(self) -> None:
        registry = ExpertRegistry()
        registry.register(MutatingExpert())
        registry.register(ObservingExpert())
        registry.register(FailingExpert())
        pipe = Pipeline(
            ModalityRouter(registry),
            reporter=Reporter(llm=None),
            thresholds=0.5,
        )

        result = pipe.analyze_scan(_scan())

        self.assertEqual(result.experts, ["mutator", "observer"])
        self.assertIn("broken", result.expert_failures)
        self.assertEqual(result.scan.meta.extra, {})
        self.assertTrue(result.verification.ok, result.verification.summary())

    def test_pipeline_computes_gradcam_when_localization_is_requested(self) -> None:
        backbone = TinyBackbone()
        head = ClassificationHead(2, 1, multilabel=True, dropout=0.0)
        nn.init.constant_(head.fc.weight, 1.0)
        nn.init.constant_(head.fc.bias, 0.0)
        expert = BaseExpert(
            name="tiny",
            modality=Modality.XRAY,
            body_part=BodyPart.CHEST,
            backbone=backbone,
            heads={"cls": head},
            class_names=["Effusion"],
        )
        registry = ExpertRegistry()
        registry.register(expert)
        pipe = Pipeline(
            ModalityRouter(registry),
            reporter=Reporter(llm=None),
            thresholds=0.5,
            localizer=GridZoneLocalizer(),
        )

        result = pipe.analyze_scan(_scan())

        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].laterality, "right")
        self.assertEqual(result.findings[0].location, "upper zone")

    def test_unverified_generated_draft_is_replaced_by_verified_template(self) -> None:
        class HallucinatingLLM:
            model_id = "test/hallucinator"

            def complete(self, system: str, user: str) -> str:
                return (
                    '{"technique":"XRAY chest","findings":"Pneumothorax.",'
                    '"impression":"Pneumothorax.","recommendation":"None."}'
                )

        registry = ExpertRegistry()
        registry.register(MutatingExpert())  # produces Effusion, not Pneumothorax
        pipe = Pipeline(
            ModalityRouter(registry),
            reporter=Reporter(llm=HallucinatingLLM()),
            thresholds=0.5,
        )

        result = pipe.analyze_scan(_scan())

        self.assertIsNotNone(result.rejected_report)
        self.assertFalse(result.rejected_verification.ok)
        self.assertEqual(result.report.generator, "template (verification-fallback)")
        self.assertTrue(result.verification.ok, result.verification.summary())
        self.assertIn("Effusion", result.report.findings)
        self.assertNotIn("Pneumothorax", result.report.findings)


if __name__ == "__main__":
    unittest.main()
