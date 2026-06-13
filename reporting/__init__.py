"""Reporting layer: structured findings -> auditable radiology-style draft.

Flow: a `Prediction` (+ optional Grad-CAM heatmaps) is turned into a list of
`Finding`s here, then the `Reporter` verbalizes only those facts. The model decides
*what* is true; the reporter only decides *how to say it*.
"""

from .findings import (
    Finding,
    GridZoneLocalizer,
    Localizer,
    findings_from_classification,
    findings_from_mask,
)
from .guidelines import GuidelineEngine, Recommendation, Urgency
from .reporter import (
    DISCLAIMER,
    LLMClient,
    LocalLLMClient,
    Reporter,
    StructuredReport,
)
from .verifier import Verifier, VerificationResult

__all__ = [
    "Finding",
    "Localizer",
    "GridZoneLocalizer",
    "findings_from_classification",
    "findings_from_mask",
    "Reporter",
    "StructuredReport",
    "LLMClient",
    "LocalLLMClient",
    "DISCLAIMER",
    "Verifier",
    "VerificationResult",
    "GuidelineEngine",
    "Recommendation",
    "Urgency",
]
