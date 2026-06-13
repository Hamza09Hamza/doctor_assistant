"""End-to-end wiring smoke test for the full pipeline — no GPU, no training, no key.

Builds a randomly-initialised chest expert, registers it, routes a synthetic scan
through the orchestrator, and prints the report + guideline + verification. This proves
every seam connects (router → expert → findings → reporter → verifier → guidelines)
without needing weights or the dataset. Predictions are meaningless (random weights);
only the *plumbing* is under test.

Run:  python scripts/smoke_system.py
"""

from __future__ import annotations

import torch

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata
from experts.chest_xray import build_chest_xray_expert
from pipeline import Pipeline
from reporting import GridZoneLocalizer, Verifier
from reporting.findings import Finding
from routing import ExpertRegistry, ModalityRouter


def main() -> None:
    torch.manual_seed(0)

    # 1. Build an expert (no pretrained download, no real weights needed for wiring).
    expert = build_chest_xray_expert(pretrained=False, image_size=224)
    expert.eval()

    # 2. Register + router.
    registry = ExpertRegistry()
    registry.register(expert)
    router = ModalityRouter(registry)

    # 3. Orchestrator. Low threshold so the random model surfaces *some* findings to
    #    exercise the report/guideline/verifier path.
    pipe = Pipeline(router, thresholds=0.3, localizer=GridZoneLocalizer())

    # 4. Synthetic chest "scan".
    scan = Scan(
        data=torch.rand(3, 224, 224),
        meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST),
    )

    result = pipe.analyze_scan(scan)

    print("=" * 70)
    print(f"Experts run     : {result.experts}")
    print(f"Findings        : {[f.label for f in result.findings if f.present]}")
    print(f"Triage urgency  : {result.triage_urgency.name}")
    print(f"Report generator: {result.report.generator}")
    print("=" * 70)
    print(result.to_text())
    print("=" * 70)

    # 5. Verifier unit-check: a deliberately hallucinated report must fail.
    bad = result.report
    bad.findings = "There is a 999 mm mass in the left upper zone with volume 42.0 mL."
    verdict = Verifier(known_labels=list(expert.class_names)).verify(bad)
    print("Adversarial check (expect FAIL):")
    print(verdict.summary())
    assert not verdict.ok, "verifier should have flagged the ungrounded 999 mm / 42 mL"

    # 6. Verifier unit-check: a faithful template report over a known finding must pass.
    from reporting.reporter import _template_report
    good_findings = [Finding(label="Effusion", probability=0.81, present=True, confidence=0.7)]
    good = _template_report(good_findings, scan.meta)
    ok_verdict = Verifier(known_labels=list(expert.class_names)).verify(good)
    print("\nFaithful check (expect PASS):")
    print(ok_verdict.summary())
    assert ok_verdict.ok, "faithful report should pass verification"

    print("\nALL WIRING OK ✓")


if __name__ == "__main__":
    main()
