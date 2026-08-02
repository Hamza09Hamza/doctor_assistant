"""Canonical finding vocabulary — one stable concept code per clinical finding.

Different experts spell the same finding differently: ChestX-ray14 says "Effusion",
KAD says "Pleural_Effusion" in some phrasing, a report template says "pleural
effusion". Without a shared code, `guidelines.py` and `verifier.py` each had their own
ad hoc `.lower()`/`.replace("_", " ")` normalization, which is duplicated logic that can
drift out of sync. This module is the single source of truth both now import.

The canonical codes are exactly `reporting.guidelines`'s existing `_CHEST_GUIDELINES`
keys (already a de facto vocabulary) plus `"nodule"` (already special-cased there) plus
two new codes this session's KAD-512 evaluation work needs: `"nodule_or_mass"` and
`"airspace_opacity"`.

Those two are deliberately their own codes, not aliased onto a narrower existing one.
KAD's phase-1 endpoints query a genuinely combined/broader concept
(`scripts/export_kad_query_pack.py::PHASE1_QUERY_SPECS` prompts "lung nodule or mass"
and "airspace opacity") — aliasing `Nodule_or_mass` to `"nodule"` would let a KAD-flagged
*mass* silently trigger `guidelines.py::_nodule_recommendation`'s Fleischner
size-banding, which does not apply to masses. A combined-scope model concept must get
its own code, never be folded into a more specific one it doesn't fully match.
"""

from __future__ import annotations

# The 13 pre-existing `_CHEST_GUIDELINES` keys, "nodule" (already implicit there), and
# the 2 new KAD phase-1 concepts. Kept as a literal here (not imported from
# `guidelines.py`) to avoid a circular import; `tests/test_vocabulary.py` asserts this
# stays a superset of `_CHEST_GUIDELINES`'s keys so the two can't silently drift apart.
CANONICAL_LABELS: frozenset[str] = frozenset(
    {
        "pneumothorax",
        "mass",
        "consolidation",
        "pneumonia",
        "edema",
        "effusion",
        "cardiomegaly",
        "atelectasis",
        "infiltration",
        "emphysema",
        "fibrosis",
        "pleural_thickening",
        "hernia",
        "nodule",
        "nodule_or_mass",
        "airspace_opacity",
    }
)

# Raw variant string (any casing/underscore/spacing) -> canonical code. Seeded from the
# real label strings already in this repo: the 14 ChestX-ray14 labels
# (experts/torchxrayvision.py), KAD's 3 phase-1 endpoint names
# (scripts/export_kad_query_pack.py::PHASE1_QUERY_SPECS), and the guideline keys
# themselves (idempotent self-mapping).
ALIASES: dict[str, str] = {
    # ChestX-ray14
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "effusion": "effusion",
    "pleural effusion": "effusion",
    "infiltration": "infiltration",
    "mass": "mass",
    "nodule": "nodule",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "consolidation": "consolidation",
    "edema": "edema",
    "emphysema": "emphysema",
    "fibrosis": "fibrosis",
    "pleural thickening": "pleural_thickening",
    "pleural_thickening": "pleural_thickening",
    "hernia": "hernia",
    # No Finding / normal are filtered out upstream in findings_from_classification
    # (the `normal_labels` parameter) and never reach canonicalize() in practice, so
    # they are intentionally not mapped here.
    # KAD phase-1 endpoints (own codes — see module docstring for why)
    "nodule_or_mass": "nodule_or_mass",
    "nodule or mass": "nodule_or_mass",
    "airspace_opacity": "airspace_opacity",
    "airspace opacity": "airspace_opacity",
}


def canonicalize(label: str) -> str:
    """Map a raw expert label to its canonical concept code.

    Looks up `ALIASES` case/underscore-insensitively. On a miss, falls back to a
    normalized slug instead of raising, so an unrecognized label still gets *a* stable
    code rather than crashing the pipeline — the same "don't invent, but don't crash"
    posture `GuidelineEngine`'s generic fallback recommendation already uses for
    unknown labels.
    """
    key = label.strip().lower().replace("_", " ")
    canonical = ALIASES.get(key)
    if canonical is not None:
        return canonical
    return key.replace(" ", "_")
