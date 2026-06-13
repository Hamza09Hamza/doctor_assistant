"""The guideline agent — map findings to standardized next-step recommendations.

This is the "Guideline Agent" box in the agentic-radiology diagrams, but built as a
curated, auditable lookup rather than a RAG over a document store — so it needs no
corpus and gives deterministic, defensible output. Each present finding maps to a
recommendation and an urgency tier; a few are modulated by measured size (e.g. a
pulmonary nodule follows Fleischner-style size bands when a diameter is available).

The output is advisory and explicitly non-prescriptive — it surfaces the conventional
work-up a radiologist would consider, tagged by urgency so triage can sort on it. It
never overrides the report; the orchestrator attaches it alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from enum import IntEnum

from .findings import Finding


class Urgency(IntEnum):
    """Triage tiers. Higher sorts first so the most time-critical surfaces at the top."""

    ROUTINE = 0
    PROMPT = 1      # should be actioned this visit / same day
    URGENT = 2      # potentially time-critical; flag for immediate human review


@dataclass
class Recommendation:
    label: str
    text: str
    urgency: Urgency

    def to_facts(self) -> dict:
        return {"label": self.label, "urgency": self.urgency.name.lower(), "text": self.text}


# Per-label recommendation + base urgency for the ChestX-ray14 vocabulary. Phrasing is
# deliberately neutral ("consider", "correlate") — decision support, not orders.
_CHEST_GUIDELINES: dict[str, tuple[str, Urgency]] = {
    "pneumothorax": (
        "Assess size and clinical status; a large or symptomatic pneumothorax may "
        "warrant urgent decompression. Immediate clinical correlation advised.",
        Urgency.URGENT,
    ),
    "mass": (
        "Further characterization with contrast-enhanced CT; consider tissue sampling. "
        "Compare with any prior imaging.",
        Urgency.PROMPT,
    ),
    "consolidation": (
        "Correlate clinically for infection; consider follow-up radiograph after "
        "treatment to confirm resolution.",
        Urgency.PROMPT,
    ),
    "pneumonia": (
        "Correlate with clinical and laboratory findings; follow-up imaging after "
        "treatment to document resolution.",
        Urgency.PROMPT,
    ),
    "edema": (
        "Correlate with cardiac and renal status; assess volume status.",
        Urgency.PROMPT,
    ),
    "effusion": (
        "Consider decubitus views or ultrasound to quantify; thoracentesis if "
        "clinically indicated.",
        Urgency.ROUTINE,
    ),
    "cardiomegaly": (
        "Correlate with echocardiography and clinical assessment of cardiac function.",
        Urgency.ROUTINE,
    ),
    "atelectasis": (
        "Often nonspecific; correlate clinically. Consider follow-up if persistent.",
        Urgency.ROUTINE,
    ),
    "infiltration": (
        "Nonspecific airspace opacity; correlate clinically and with prior imaging.",
        Urgency.ROUTINE,
    ),
    "emphysema": (
        "Correlate with pulmonary function tests; HRCT for further characterization.",
        Urgency.ROUTINE,
    ),
    "fibrosis": (
        "HRCT and pulmonary function tests for further characterization.",
        Urgency.ROUTINE,
    ),
    "pleural_thickening": (
        "Correlate with history of prior infection or asbestos exposure; CT if "
        "progressive or nodular.",
        Urgency.ROUTINE,
    ),
    "hernia": (
        "Surgical correlation if symptomatic.",
        Urgency.ROUTINE,
    ),
    # Nodule handled specially (size-banded) in `_nodule_recommendation`.
}


def _nodule_recommendation(f: Finding) -> Recommendation:
    """Fleischner-style follow-up for a pulmonary nodule, banded by diameter when known.

    Sizes are simplified from the Fleischner Society guidance for a single solid nodule;
    when no diameter is measured (chest X-ray has no mask), fall back to the general
    recommendation. This is decision support, not a substitute for the full criteria
    (which also weigh patient risk and morphology).
    """
    size = f.size_mm
    if size is None:
        text = ("Pulmonary nodule. Follow-up imaging advised per Fleischner Society "
                "criteria; compare with any prior studies.")
        return Recommendation("Nodule", text, Urgency.PROMPT)
    if size < 6:
        text = (f"Small nodule (~{size:.0f} mm). For low-risk patients routine follow-up "
                "may not be required; correlate with risk factors.")
        urg = Urgency.ROUTINE
    elif size < 8:
        text = (f"Nodule (~{size:.0f} mm). CT follow-up at 6–12 months advised "
                "(Fleischner).")
        urg = Urgency.PROMPT
    else:
        text = (f"Nodule (~{size:.0f} mm). CT at 3 months, PET/CT, or tissue sampling "
                "as appropriate (Fleischner).")
        urg = Urgency.PROMPT
    return Recommendation("Nodule", text, urg)


class GuidelineEngine:
    """Turn present findings into a sorted list of `Recommendation`s.

    `guidelines` defaults to the ChestX-ray14 map but can be swapped per expert (a brain
    or bone pack would pass its own). Unknown labels get a generic, low-urgency note so
    nothing is silently dropped.
    """

    def __init__(self, guidelines: dict[str, tuple[str, Urgency]] | None = None) -> None:
        self.guidelines = guidelines if guidelines is not None else _CHEST_GUIDELINES

    def recommend(self, findings: Sequence[Finding]) -> list[Recommendation]:
        present = [f for f in findings if f.present]
        recs: list[Recommendation] = []
        for f in present:
            key = f.label.strip().lower()
            if key == "nodule":
                recs.append(_nodule_recommendation(f))
                continue
            entry = self.guidelines.get(key)
            if entry is None:
                recs.append(Recommendation(
                    f.label,
                    "Correlate clinically; consider radiologist review.",
                    Urgency.ROUTINE,
                ))
            else:
                text, urgency = entry
                recs.append(Recommendation(f.label, text, urgency))
        # Most urgent first; stable within a tier preserves the findings' salience order.
        recs.sort(key=lambda r: r.urgency, reverse=True)
        return recs

    def top_urgency(self, findings: Sequence[Finding]) -> Urgency:
        """The single highest urgency across all findings — the triage signal."""
        recs = self.recommend(findings)
        return max((r.urgency for r in recs), default=Urgency.ROUTINE)
