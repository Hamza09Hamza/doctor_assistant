"""The orchestrator — one call that runs the whole grounded-reporting pipeline.

    ingest → route → expert(s) → structured findings → report → verify → guidelines

This is the seam that turns a pile of components into a *system*. Everything upstream
(loaders, experts) and downstream (reporter, verifier, guidelines) is swappable; the
orchestrator only speaks the fixed contracts (`Scan`, `Prediction`, `Finding`,
`StructuredReport`). The design discipline holds end to end: vision models decide *what*
is true and produce measured findings, the reporter only verbalizes them, the verifier
checks the prose against those findings, and the guideline agent attaches the
conventional next step. No stage invents facts another stage didn't supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.interfaces import ExpertModel, Router
from core.types import Prediction, Scan
from ingest.loaders import load_scan
from reporting.findings import (
    Finding,
    Localizer,
    findings_from_classification,
    findings_from_mask,
)
from reporting.guidelines import GuidelineEngine, Recommendation, Urgency
from reporting.reporter import Reporter, StructuredReport
from reporting.verifier import Verifier, VerificationResult


@dataclass
class AnalysisResult:
    """Everything the pipeline produced for one scan, in audit order."""

    scan: Scan
    experts: list[str] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report: StructuredReport | None = None
    verification: VerificationResult | None = None
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def triage_urgency(self) -> Urgency:
        return max((r.urgency for r in self.recommendations), default=Urgency.ROUTINE)

    def to_text(self) -> str:
        """Human-readable dump: report + verification verdict + recommendations."""
        lines: list[str] = []
        if self.report is not None:
            lines.append(self.report.to_text())
        if self.recommendations:
            lines.append("\nRECOMMENDATIONS:")
            for r in self.recommendations:
                lines.append(f"  [{r.urgency.name}] {r.label}: {r.text}")
        if self.verification is not None:
            lines.append("\nVERIFICATION:")
            lines.append(self.verification.summary())
        return "\n".join(lines)


class Pipeline:
    """Wire the stages together and run them for a scan.

    Pass your own `Reporter` / `Verifier` / `GuidelineEngine` to customize behaviour;
    sensible defaults are created otherwise. `thresholds` and `localizer` control how a
    `Prediction` becomes `Finding`s (per-label decision thresholds and the optional
    Grad-CAM zone localizer for chest studies).
    """

    def __init__(
        self,
        router: Router,
        *,
        reporter: Reporter | None = None,
        verifier: Verifier | None = None,
        guidelines: GuidelineEngine | None = None,
        thresholds: float | dict[str, float] = 0.5,
        localizer: Localizer | None = None,
    ) -> None:
        self.router = router
        self.reporter = reporter if reporter is not None else Reporter()
        self.verifier = verifier  # None -> built per-expert so known_labels are set
        self.guidelines = guidelines if guidelines is not None else GuidelineEngine()
        self.thresholds = thresholds
        self.localizer = localizer

    # -- entry points --------------------------------------------------------
    def analyze(self, path: str, **load_kwargs) -> AnalysisResult:
        """Load a scan from disk and run the full pipeline. `load_kwargs` are passed to
        `load_scan` (e.g. modality=, body_part= when not detectable from the file)."""
        scan = load_scan(path, **load_kwargs)
        return self.analyze_scan(scan)

    def analyze_scan(self, scan: Scan) -> AnalysisResult:
        """Run the pipeline on an already-loaded `Scan`."""
        result = AnalysisResult(scan=scan)
        experts = self.router.route(scan)

        all_findings: list[Finding] = []
        for expert in experts:
            pred = expert.predict(scan)
            result.experts.append(expert.name)
            result.predictions.append(pred)
            all_findings.extend(self._findings_for(expert, pred, scan))

        # Salience order: present first, then by probability.
        all_findings.sort(key=lambda f: (f.present, f.probability), reverse=True)
        result.findings = all_findings

        result.report = self.reporter.report(all_findings, scan.meta)
        result.recommendations = self.guidelines.recommend(all_findings)
        result.verification = self._verify(result.report, experts)
        return result

    # -- internals -----------------------------------------------------------
    def _findings_for(
        self, expert: ExpertModel, pred: Prediction, scan: Scan
    ) -> list[Finding]:
        """Pick the extraction path by what the expert produced.

        An expert may own its findings extraction by exposing
        `findings_from_prediction(scan, prediction) -> list[Finding]` — used when the
        model's output is richer than the generic decoders (TotalSegmentator's many
        organ masks, MAIRA-2's grounded sentences). Otherwise: a segmentation mask gives
        measured geometry (brain MRI), and class scores give thresholded findings with
        optional Grad-CAM localization (chest X-ray).
        """
        provider = getattr(expert, "findings_from_prediction", None)
        if callable(provider):
            return list(provider(scan, pred))

        if pred.segmentation is not None:
            label = pred.top_label or getattr(expert, "body_part", "lesion")
            label = label.value if hasattr(label, "value") else str(label)
            return findings_from_mask(
                pred.segmentation,
                label=label,
                spacing=scan.meta.spacing,
                confidence=pred.confidence,
                probability=pred.top_score or 1.0,
            )

        heatmaps = {pred.top_label: pred.heatmap} if (
            pred.heatmap is not None and pred.top_label is not None
        ) else None
        return findings_from_classification(
            pred.class_probs,
            thresholds=self.thresholds,
            confidence=pred.confidence,
            heatmaps=heatmaps,
            localizer=self.localizer,
        )

    def _verify(
        self, report: StructuredReport, experts: list[ExpertModel]
    ) -> VerificationResult:
        if self.verifier is not None:
            return self.verifier.verify(report)
        # Build one whose label vocabulary is the union of the experts' classes, so the
        # "named a finding the model didn't flag" check is active.
        known: list[str] = []
        for e in experts:
            known.extend(getattr(e, "class_names", []) or [])
        return Verifier(known_labels=known).verify(report)
