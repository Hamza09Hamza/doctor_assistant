from __future__ import annotations

import unittest

import numpy as np

from core.enums import BodyPart, Modality
from core.types import ScanMetadata
from reporting.findings import Finding, findings_from_mask
from reporting.guidelines import GuidelineEngine
from reporting.reporter import Reporter, StructuredReport
from reporting.verifier import Verifier


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST)

    def test_template_report_is_complete_and_grounded(self) -> None:
        findings = [
            Finding(
                label="Pleural_Thickening",
                probability=0.814,
                size_mm=12.4,
                volume_ml=3.46,
            )
        ]
        report = Reporter(llm=None).report(findings, self.meta)
        verdict = Verifier(known_labels=["Pleural_Thickening"]).verify(report)

        self.assertTrue(verdict.ok, verdict.summary())
        self.assertIn("pleural thickening", report.findings.lower())

    def test_unsupported_pathology_is_a_hard_failure(self) -> None:
        report = StructuredReport(
            findings="Pneumothorax is present.",
            source_findings=[Finding(label="Effusion", probability=0.8)],
        )
        verdict = Verifier(known_labels=["Effusion", "Pneumothorax"]).verify(report)

        self.assertFalse(verdict.ok)
        self.assertTrue(any("Pneumothorax" in flag for flag in verdict.flags))

    def test_missing_present_finding_is_a_hard_failure(self) -> None:
        report = StructuredReport(
            findings="No focal abnormality.",
            source_findings=[Finding(label="Effusion", probability=0.8)],
        )
        verdict = Verifier(known_labels=["Effusion"]).verify(report)

        self.assertFalse(verdict.ok)
        self.assertTrue(any("missing from the report" in flag for flag in verdict.flags))

    def test_equal_number_with_wrong_unit_is_not_grounded(self) -> None:
        finding = Finding(label="Mass", probability=0.9, size_mm=12.0)
        report = StructuredReport(
            findings="Mass with a volume of 12 mL.",
            source_findings=[finding],
        )
        verdict = Verifier(known_labels=["Mass"]).verify(report)

        self.assertFalse(verdict.ok)
        self.assertTrue(any("12 mL" in flag for flag in verdict.flags))

    def test_present_finding_cannot_be_negated(self) -> None:
        report = StructuredReport(
            findings="There is no evidence of effusion.",
            source_findings=[Finding(label="Effusion", probability=0.8)],
        )
        verdict = Verifier(known_labels=["Effusion"]).verify(report)

        self.assertFalse(verdict.ok)
        self.assertTrue(any("negates present finding" in flag for flag in verdict.flags))

    def test_count_claim_is_unit_grounded(self) -> None:
        report = Reporter(llm=None).report(
            [Finding(label="Nodule", probability=0.8, count=3)],
            self.meta,
        )
        verdict = Verifier(known_labels=["Nodule"]).verify(report)
        self.assertTrue(verdict.ok, verdict.summary())

    def test_recommendations_are_deduplicated_by_label(self) -> None:
        findings = [
            Finding(label="Effusion", probability=0.9, source="reader-a"),
            Finding(label="Effusion", probability=0.7, source="reader-b"),
        ]
        recommendations = GuidelineEngine().recommend(findings)
        self.assertEqual([r.label for r in recommendations], ["Effusion"])

    def test_no_finding_report_never_calls_language_model(self) -> None:
        class FailingIfCalled:
            def complete(self, system: str, user: str) -> str:
                raise AssertionError("LLM must not be called for no-positive studies")

        report = Reporter(llm=FailingIfCalled()).report([], self.meta)
        self.assertEqual(report.generator, "template")
        self.assertIn("No significant abnormality", report.findings)

    def test_unscored_mask_never_fabricates_a_probability(self) -> None:
        mask = np.ones((4, 4), dtype=np.uint8)

        findings = findings_from_mask(mask, "Liver", min_voxels=10)
        report = Reporter(llm=None).report(findings, self.meta)
        verdict = Verifier(known_labels=["Liver"]).verify(report)

        self.assertEqual(len(findings), 1)
        self.assertIsNone(findings[0].probability)
        self.assertNotIn("probability", findings[0].to_facts())
        self.assertNotIn("score", report.findings.lower())
        self.assertTrue(verdict.ok, verdict.summary())

    def test_verifier_rejects_score_when_source_has_no_probability(self) -> None:
        report = StructuredReport(
            findings="Liver — score 1.00.",
            source_findings=[Finding(label="Liver", probability=None)],
        )

        verdict = Verifier(known_labels=["Liver"]).verify(report)

        self.assertFalse(verdict.ok)
        self.assertTrue(
            any("ungrounded number '1.00'" in flag for flag in verdict.flags),
            verdict.summary(),
        )


if __name__ == "__main__":
    unittest.main()
