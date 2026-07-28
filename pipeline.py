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

import copy
import warnings
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
    expert_failures: dict[str, str] = field(default_factory=dict)
    predictions: list[Prediction] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    report: StructuredReport | None = None
    verification: VerificationResult | None = None
    rejected_report: StructuredReport | None = None
    rejected_verification: VerificationResult | None = None
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


class PipelineExecutionError(RuntimeError):
    """Raised when experts were routed but none completed successfully."""


class ReportVerificationError(RuntimeError):
    """Raised when neither the drafted report nor deterministic fallback verifies."""


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
        # Deterministic reporting is the safe default. Local LLM wording remains an
        # explicit opt-in by passing Reporter(llm=...), and is still verification-gated.
        self.reporter = reporter if reporter is not None else Reporter(llm=None)
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
        successful_experts: list[ExpertModel] = []

        all_findings: list[Finding] = []
        for expert in experts:
            # One reader failing must not sink the others. A pretrained adapter can be
            # unavailable for reasons unrelated to the rest of the panel — a gated/
            # un-authenticated HF repo (MAIRA-2), an OOM, a network drop. Warn with the
            # expert name + error and carry on; the remaining experts still produce a
            # report. A failed expert is NOT recorded in result.experts/predictions, so
            # the summary reflects only what actually ran.
            try:
                # Several pretrained adapters attach model-specific data to
                # ScanMetadata.extra. Give every expert a private metadata object so one
                # reader cannot leak state into another reader or mutate result.scan.
                expert_scan = Scan(data=scan.data, meta=copy.deepcopy(scan.meta))
                pred = expert.predict(expert_scan)
                findings = self._findings_for(expert, pred, expert_scan)
            except Exception as exc:  # noqa: BLE001 — deliberately broad; isolate the expert
                result.expert_failures[expert.name] = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"Pipeline: expert {expert.name!r} failed and was skipped "
                    f"({type(exc).__name__}: {exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            successful_experts.append(expert)
            result.experts.append(expert.name)
            result.predictions.append(pred)
            all_findings.extend(findings)

        # Failure isolation is useful only while at least one independent reader
        # completed. Treating "every model crashed" as a normal study would be a
        # dangerous false-negative report, so stop explicitly instead.
        if experts and not successful_experts:
            details = "; ".join(
                f"{name}: {error}" for name, error in result.expert_failures.items()
            )
            raise PipelineExecutionError(
                "All routed experts failed; no report was generated."
                + (f" Failures: {details}" if details else "")
            )

        # Salience order: present first, then by probability.
        all_findings.sort(key=lambda f: (f.present, f.probability), reverse=True)
        result.findings = all_findings

        result.report = self.reporter.report(all_findings, scan.meta)
        result.recommendations = self.guidelines.recommend(all_findings)
        result.verification = self._verify(result.report, successful_experts)
        if not result.verification.ok:
            # Never hand an ungrounded generated draft to the caller as the active
            # report. Preserve it for audit, replace it with the deterministic
            # fact-only template, and verify that fallback too.
            result.rejected_report = result.report
            result.rejected_verification = result.verification
            warnings.warn(
                "Pipeline: drafted report failed verification; using the "
                "deterministic reporting fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            fallback = Reporter(llm=None).report(all_findings, scan.meta)
            fallback.generator = "template (verification-fallback)"
            fallback_verification = self._verify(fallback, successful_experts)
            if not fallback_verification.ok:
                raise ReportVerificationError(
                    "Draft and deterministic fallback both failed report verification. "
                    f"Fallback verdict: {fallback_verification.summary()}"
                )
            result.report = fallback
            result.verification = fallback_verification
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
        generic measured lesion geometry, and class scores give thresholded findings with
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

        heatmaps = self._classification_heatmaps(expert, pred, scan)
        return findings_from_classification(
            pred.class_probs,
            thresholds=self.thresholds,
            confidence=pred.confidence,
            heatmaps=heatmaps,
            localizer=self.localizer,
        )

    def _classification_heatmaps(
        self, expert: ExpertModel, pred: Prediction, scan: Scan
    ) -> dict[str, object] | None:
        """Return available heatmaps, computing Grad-CAM for present labels when possible.

        Pretrained adapters may provide a single heatmap directly. Trainable
        ``BaseExpert``-style models expose a backbone and classification heads, so when
        localization is requested we can compute a map for every above-threshold label.
        Explainability failure is non-fatal: the scored finding remains useful without a
        location and the failure is surfaced as a warning.
        """
        if pred.heatmap is not None and pred.top_label is not None:
            return {pred.top_label: pred.heatmap}
        if self.localizer is None:
            return None
        if not hasattr(expert, "backbone") or not hasattr(expert, "heads"):
            return None

        wanted = [
            label
            for label, probability in pred.class_probs.items()
            if probability >= self._threshold_for(label)
        ]
        if not wanted:
            return None

        try:
            from explainability import GradCAM

            data = (
                expert.preprocess(scan.data)
                if getattr(expert, "preprocess", None) is not None
                else scan.data
            )
            return GradCAM(expert).for_labels(data, labels=wanted)
        except Exception as exc:  # noqa: BLE001 — localization must not discard prediction
            warnings.warn(
                f"Pipeline: localization for expert {expert.name!r} failed "
                f"({type(exc).__name__}: {exc}); findings will be unlocalized.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _threshold_for(self, label: str) -> float:
        if isinstance(self.thresholds, dict):
            return float(
                self.thresholds.get(label, self.thresholds.get("__default__", 0.5))
            )
        return float(self.thresholds)

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
