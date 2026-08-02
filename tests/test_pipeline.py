from __future__ import annotations

import unittest

import torch
from torch import nn

from core.enums import BodyPart, Modality
from core.types import Prediction, Scan, ScanMetadata
from models.backbones import Backbone
from models.experts import BaseExpert
from models.heads import ClassificationHead
from pipeline import AnalysisStatus, Pipeline, PipelineExecutionError
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


class VersionedExpert:
    name = "versioned"
    modality = Modality.XRAY
    body_part = BodyPart.CHEST
    class_names = ["Effusion"]
    version = "versioned-expert:v1"
    preprocessing_version = "prep:v1"

    def predict(self, scan):
        return Prediction(expert=self.name, class_probs={"Effusion": 0.9}, meta=scan.meta)


class SegmentationExpert:
    modality = Modality.XRAY
    body_part = BodyPart.CHEST

    def __init__(self, name: str, label: str, score: float | None) -> None:
        self.name = name
        self.label = label
        self.score = score
        self.class_names = [label]

    def predict(self, scan):
        class_probs = {} if self.score is None else {self.label: self.score}
        return Prediction(
            expert=self.name,
            class_probs=class_probs,
            segmentation=torch.ones(16, 16),
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

        self.assertEqual(result.status, AnalysisStatus.PARTIAL)
        self.assertTrue(result.analysis_id)
        by_expert = {e.expert: e for e in result.expert_executions}
        self.assertEqual({"mutator", "observer", "broken"}, set(by_expert))
        self.assertEqual(by_expert["mutator"].status, "completed")
        self.assertEqual(by_expert["observer"].status, "completed")
        self.assertEqual(by_expert["broken"].status, "failed")
        self.assertIsNotNone(by_expert["broken"].error)
        self.assertIsNone(by_expert["broken"].expert_version)
        execution_ids = {e.execution_id for e in result.expert_executions}
        self.assertEqual(len(execution_ids), 3)  # all unique

    def test_provenance_is_stamped_on_predictions_and_findings(self) -> None:
        registry = ExpertRegistry()
        registry.register(VersionedExpert())
        pipe = Pipeline(ModalityRouter(registry), reporter=Reporter(llm=None), thresholds=0.5)

        result = pipe.analyze_scan(_scan())

        self.assertEqual(result.status, AnalysisStatus.COMPLETE)
        self.assertEqual(len(result.expert_executions), 1)
        execution = result.expert_executions[0]
        self.assertEqual(execution.status, "completed")
        self.assertEqual(execution.expert_version, "versioned-expert:v1")
        self.assertEqual(execution.preprocessing_version, "prep:v1")

        [prediction] = result.predictions
        self.assertEqual(prediction.expert_version, "versioned-expert:v1")
        self.assertEqual(prediction.preprocessing_version, "prep:v1")
        self.assertEqual(prediction.execution_id, execution.execution_id)

        self.assertTrue(result.findings)
        for finding in result.findings:
            self.assertEqual(finding.execution_id, execution.execution_id)
            self.assertEqual(finding.canonical_label, "effusion")

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

    def test_segmentation_preserves_zero_score_and_leaves_missing_score_unknown(self) -> None:
        registry = ExpertRegistry()
        registry.register(SegmentationExpert("zero", "Lesion", 0.0))
        registry.register(SegmentationExpert("unscored", "chest", None))
        pipe = Pipeline(ModalityRouter(registry), reporter=Reporter(llm=None))

        result = pipe.analyze_scan(_scan())

        self.assertEqual([finding.probability for finding in result.findings], [0.0, None])
        self.assertEqual(result.report.findings.lower().count("score"), 1)
        self.assertIn("score 0.00", result.report.findings)
        self.assertNotIn("score 1.00", result.report.findings)
        self.assertTrue(result.verification.ok, result.verification.summary())


if __name__ == "__main__":
    unittest.main()
