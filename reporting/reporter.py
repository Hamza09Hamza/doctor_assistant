"""The reporter — turns structured `Finding`s into a radiology-style draft.

Design principle (the whole point): **the language model is a typist, not a doctor.**
It receives a JSON of facts that the vision models already produced and measured in
`findings.py`, and it is instructed to verbalize *only* those facts — no new findings,
no invented measurements, no diagnosis. That keeps the prose auditable: every sentence
maps back to a number, and the `StructuredReport` carries the source findings alongside
the text so a reviewer can check the two against each other.

The LLM path is the default (the system was specced LLM-driven). A deterministic
template path is kept as a safety net for offline/no-key/failed-call situations, so the
pipeline degrades to plain prose rather than crashing — it never silently drops to a
worse mode without that being visible in `StructuredReport.generator`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.types import ScanMetadata

from .findings import Finding

DISCLAIMER = (
    "AI-generated draft for clinical decision support. Not a diagnosis. "
    "All findings require verification by a qualified radiologist."
)

_SYSTEM_PROMPT = """You are a drafting assistant that converts STRUCTURED RADIOLOGY \
FINDINGS — already produced and validated by computer-vision models — into a concise \
radiology-style draft for a radiologist to review.

ABSOLUTE RULES:
1. Use ONLY the facts in the provided JSON. Never add findings, measurements, \
diagnoses, anatomy, or comparisons that are not present in the JSON.
2. If a detail (size, location, laterality) is absent, omit it. Do not infer it.
3. Never change a numeric value, label, or probability.
4. Reflect uncertainty faithfully: low confidence or borderline probability -> hedge \
("possible", "cannot exclude"); high confidence -> state plainly.
5. This is a draft that ASSISTS a radiologist; it is not a diagnosis.

Respond with ONLY a JSON object, no prose around it, with these string keys:
  "technique"      one short sentence naming the study (modality / body part).
  "findings"       the observations, one finding per sentence, faithful to the JSON.
  "impression"     a brief synthesis of the salient positive findings.
  "recommendation" a neutral, non-prescriptive next step (e.g. "clinical correlation \
advised"); if nothing actionable, say so.
"""


@dataclass
class StructuredReport:
    """The report object handed downstream. Prose + the facts it was built from."""

    technique: str = ""
    findings: str = ""
    impression: str = ""
    recommendation: str = ""
    disclaimer: str = DISCLAIMER
    source_findings: list[Finding] = field(default_factory=list)  # audit trail
    generator: str = ""  # "anthropic:claude-...", "template", "template (llm-fallback)"
    meta: ScanMetadata | None = None

    def to_text(self) -> str:
        blocks = [
            ("TECHNIQUE", self.technique),
            ("FINDINGS", self.findings),
            ("IMPRESSION", self.impression),
            ("RECOMMENDATION", self.recommendation),
        ]
        body = "\n\n".join(f"{title}:\n{text}" for title, text in blocks if text)
        return f"{body}\n\n---\n{self.disclaimer}"


@runtime_checkable
class LLMClient(Protocol):
    """Minimal completion interface so any provider can back the reporter."""

    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    """Claude-backed completion. Lazy-imports `anthropic` and reads the API key from
    the environment by default. A report-drafting *typist* task, so a fast model
    (Sonnet) is the sensible default; override `model` for higher-stakes phrasing."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,  # deterministic phrasing; facts are fixed
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None  # created on first use

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            import anthropic  # lazy: keep import cost off the critical path

            self._client = anthropic.Anthropic(api_key=self._api_key)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class Reporter:
    """Build a `StructuredReport` from structured findings.

    Pass an `LLMClient` to use the language-model path (default behaviour: it
    auto-creates an `AnthropicClient` if a key is present). With no client and no key,
    or if the model call/parse fails, it falls back to the deterministic template and
    records that in `StructuredReport.generator`.
    """

    def __init__(self, llm: LLMClient | None = None, *, auto_llm: bool = True) -> None:
        if llm is None and auto_llm and os.environ.get("ANTHROPIC_API_KEY"):
            llm = AnthropicClient()
        self.llm = llm

    def report(
        self, findings: Sequence[Finding], meta: ScanMetadata | None = None
    ) -> StructuredReport:
        facts = self._build_facts(findings, meta)
        if self.llm is not None:
            try:
                return self._llm_report(facts, findings, meta)
            except Exception:  # noqa: BLE001 - never let drafting crash the pipeline
                rep = _template_report(findings, meta)
                rep.generator = "template (llm-fallback)"
                return rep
        return _template_report(findings, meta)

    def _build_facts(
        self, findings: Sequence[Finding], meta: ScanMetadata | None
    ) -> dict[str, Any]:
        present = [f for f in findings if f.present]
        return {
            "study": {
                "modality": meta.modality.value if meta else "unknown",
                "body_part": meta.body_part.value if meta else "unknown",
            },
            "normal": len(present) == 0,
            "findings": [f.to_facts() for f in present],
        }

    def _llm_report(
        self,
        facts: dict[str, Any],
        findings: Sequence[Finding],
        meta: ScanMetadata | None,
    ) -> StructuredReport:
        assert self.llm is not None
        raw = self.llm.complete(_SYSTEM_PROMPT, json.dumps(facts, ensure_ascii=False))
        data = _parse_json(raw)
        model_name = getattr(self.llm, "model", type(self.llm).__name__)
        return StructuredReport(
            technique=str(data.get("technique", "")).strip(),
            findings=str(data.get("findings", "")).strip(),
            impression=str(data.get("impression", "")).strip(),
            recommendation=str(data.get("recommendation", "")).strip(),
            source_findings=list(findings),
            generator=f"anthropic:{model_name}" if isinstance(self.llm, AnthropicClient)
            else f"llm:{model_name}",
            meta=meta,
        )


def _parse_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON extraction: handles a clean object or one wrapped in stray text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _phrase_finding(f: Finding) -> str:
    """Deterministic, fact-faithful sentence for one finding (template path)."""
    parts: list[str] = []
    if f.count > 1:
        parts.append(f"{f.count} foci of")
    if f.laterality:
        parts.append(f.laterality)
    parts.append(f.label.lower() if f.label[:1].isupper() and " " in f.label else f.label)
    if f.location:
        parts.append(f"in the {f.location}")
    detail: list[str] = []
    if f.size_mm is not None:
        detail.append(f"largest diameter {f.size_mm:.0f} mm")
    if f.volume_ml is not None:
        detail.append(f"volume {f.volume_ml:.1f} mL")
    sentence = " ".join(parts)
    if detail:
        sentence += f" ({', '.join(detail)})"
    sentence += f" — score {f.probability:.2f}"
    if f.confidence is not None:
        sentence += f", confidence {f.confidence:.2f}"
    return sentence[0].upper() + sentence[1:] + "."


def _template_report(
    findings: Sequence[Finding], meta: ScanMetadata | None
) -> StructuredReport:
    """Deterministic fallback: structured findings -> plain prose, no LLM."""
    present = [f for f in findings if f.present]
    modality = meta.modality.value if meta else "imaging"
    body_part = meta.body_part.value if meta else "study"
    technique = f"{modality.upper()} of the {body_part}."

    if not present:
        return StructuredReport(
            technique=technique,
            findings="No significant abnormality detected by the model.",
            impression="No acute finding flagged.",
            recommendation="Clinical correlation advised.",
            source_findings=list(findings),
            generator="template",
            meta=meta,
        )

    findings_text = " ".join(_phrase_finding(f) for f in present)
    top = present[0]
    impression = f"Findings most consistent with {top.label.lower()}."
    return StructuredReport(
        technique=technique,
        findings=findings_text,
        impression=impression,
        recommendation="Clinical correlation advised; radiologist review required.",
        source_findings=list(findings),
        generator="template",
        meta=meta,
    )
