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
from experts.ct_totalsegmentator import findings_from_label_counts
from experts.maira2 import parse_maira2_grounding
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

    # 7. Pretrained-adapter wiring (no weights downloaded; pure parsing/geometry cores).
    print("\n" + "=" * 70)
    print("Pretrained-adapter checks:")
    _check_register_niche(expert)
    _check_findings_hook()
    _check_totalsegmentator_core()
    _check_maira2_parser()

    print("\nALL WIRING OK ✓")


def _check_register_niche(chest_expert) -> None:
    """One model instance registered under several niches is reachable from each."""
    reg = ExpertRegistry()
    reg.register(chest_expert)
    reg.register_niche(Modality.CT, BodyPart.CHEST, chest_expert)
    reg.register_niche(Modality.CT, BodyPart.ABDOMEN, chest_expert)
    assert reg.match(Modality.CT, BodyPart.CHEST), "CT/chest niche not reachable"
    assert reg.match(Modality.CT, BodyPart.ABDOMEN), "CT/abdomen niche not reachable"
    print("  register_niche: same model reachable under 3 niches OK")


def _check_findings_hook() -> None:
    """Pipeline must prefer an expert's own `findings_from_prediction` when present."""
    from core.types import Prediction

    class FakeExpert:
        name, modality, body_part = "fake", Modality.XRAY, BodyPart.CHEST
        class_names: list[str] = []

        def predict(self, scan):
            return Prediction(expert=self.name, meta=scan.meta)

        def findings_from_prediction(self, scan, pred):
            return [Finding(label="Mass", probability=0.99, present=True, size_mm=12.0)]

    reg = ExpertRegistry()
    reg.register(FakeExpert())
    result = Pipeline(ModalityRouter(reg), thresholds=0.5).analyze_scan(
        Scan(data=torch.rand(3, 64, 64),
             meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST))
    )
    labels = [f.label for f in result.findings]
    assert labels == ["Mass"], f"findings hook not used: {labels}"
    print("  findings_from_prediction hook honored OK")


def _check_totalsegmentator_core() -> None:
    """Voxel counts + spacing -> measured organ-volume findings."""
    counts = {0: 1_000_000, 1: 200_000, 5: 50_000, 9: 50}  # label 0 is background
    class_map = {1: "liver", 5: "spleen", 9: "gallbladder"}
    spacing = (1.5, 1.5, 1.5)  # mm -> 3.375 mm^3/voxel -> /1000 mL
    findings = findings_from_label_counts(
        counts, class_map, spacing, min_volume_ml=1.0
    )
    by_label = {f.label: f for f in findings}
    assert "liver" in by_label and "spleen" in by_label, by_label
    assert "structure_0" not in by_label, "background must be dropped"
    assert "gallbladder" not in by_label, "sub-threshold structure must be dropped"
    liver_ml = by_label["liver"].volume_ml
    assert abs(liver_ml - 200_000 * 3.375 / 1000.0) < 1e-3, liver_ml
    assert findings[0].label == "liver", "should sort largest-volume first"
    print(f"  TotalSegmentator volumetry OK: liver {liver_ml:.0f} mL, spleen "
          f"{by_label['spleen'].volume_ml:.0f} mL")


def _check_maira2_parser() -> None:
    """MAIRA-2 grounded sequence -> canonical, located, present/absent findings."""
    grounded = [
        # patient-left effusion -> appears on the image's right (cx > 0.5), lower zone
        ("Small left pleural effusion.", [(0.55, 0.6, 0.95, 0.95)]),
        ("There is no pneumothorax.", None),
        ("Moderate cardiomegaly is present.", None),
        "Streaky opacity at the right base.",  # no canonical keyword -> generic label
    ]
    narrative, findings = parse_maira2_grounding(grounded)
    by_label = {f.label: f for f in findings}
    eff = by_label["Effusion"]
    assert eff.present and eff.laterality == "left", (eff.present, eff.laterality)
    assert eff.location == "lower zone", eff.location
    ptx = next(f for f in findings if "pneumothorax" in f.extra["text"].lower())
    assert not ptx.present, "negated sentence must be marked not-present"
    assert "Cardiomegaly" in by_label, by_label
    assert "Finding" in by_label, "unmatched sentence should keep a generic label"
    assert "effusion" in narrative.lower()
    print("  MAIRA-2 grounding parser OK: Effusion(left/lower), Pneumothorax(absent), "
          "Cardiomegaly, +generic")


if __name__ == "__main__":
    main()
