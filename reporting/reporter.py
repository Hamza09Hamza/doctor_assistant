"""The reporter — turns structured `Finding`s into a radiology-style draft.

Design principle (the whole point): **the language model is a typist, not a doctor.**
It receives a JSON of facts that the vision models already produced and measured in
`findings.py`, and it is instructed to verbalize *only* those facts — no new findings,
no invented measurements, no diagnosis. That keeps the prose auditable: every sentence
maps back to a number, and the `StructuredReport` carries the source findings alongside
the text so a reviewer can check the two against each other.

All LLM inference runs **locally** — weights are downloaded once from HuggingFace and
cached on disk; no API key, no external call at inference time. `LocalLLMClient` is the
default. A deterministic template path is kept as a safety net for CPU-only / no-GPU
situations, so the pipeline degrades gracefully rather than crashing — the fallback is
always visible in `StructuredReport.generator`.
"""

from __future__ import annotations

import json
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
    generator: str = ""  # "local:Phi-3-mini-...", "template", "template (llm-fallback)"
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
    """Minimal completion interface — implement this to swap in any local model."""

    def complete(self, system: str, user: str) -> str: ...


# Default local model: Phi-3 Mini (3.8B, ~2 GB in 4-bit).
# Fits alongside the vision models on an L4 (24 GB VRAM).
# Swap to a larger model by passing model_id to LocalLLMClient — e.g.:
#   "mistralai/Mistral-7B-Instruct-v0.3"   (~14 GB fp16, ~4 GB 4-bit)
#   "meta-llama/Meta-Llama-3.1-8B-Instruct" (~16 GB fp16, ~4 GB 4-bit)
_DEFAULT_LOCAL_MODEL = "microsoft/Phi-3-mini-4k-instruct"


class LocalLLMClient:
    """Locally-running LLM via HuggingFace transformers — no API key, no network call
    at inference time. Weights download once and are cached in ~/.cache/huggingface.

    The model is loaded lazily on the first `complete()` call so importing this module
    stays cheap. `load_in_4bit=True` (default) keeps VRAM usage low enough to coexist
    with the vision models on an L4 GPU — Phi-3 Mini at 4-bit uses ~2 GB VRAM.

    Prompt format: a `[INST] system \\n user [/INST]` template that works for
    Phi-3 / Mistral / LLaMA instruct models. The model is instructed to output only
    a JSON object (the same contract the Anthropic path used).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_LOCAL_MODEL,
        max_new_tokens: int = 512,
        load_in_4bit: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.device = device
        self._pipe = None   # lazy-loaded

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline, BitsAndBytesConfig

        quant = None
        if self.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        device_map = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"quantization_config": quant} if quant else {},
            device_map=device_map,
            trust_remote_code=True,
        )
        self._pipe.tokenizer.padding_side = "left"

    def complete(self, system: str, user: str) -> str:
        self._ensure_loaded()
        # Chat-template format supported by Phi-3, Mistral, and LLaMA instruct variants.
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        out = self._pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,       # greedy — deterministic phrasing
            return_full_text=False,
        )
        # transformers returns list[dict]; extract the assistant turn
        generated = out[0]["generated_text"]
        if isinstance(generated, list):
            # chat format: last message is assistant turn
            return str(generated[-1].get("content", ""))
        return str(generated)


class Reporter:
    """Build a `StructuredReport` from structured findings.

    Default: auto-creates a `LocalLLMClient` (Phi-3 Mini, 4-bit) when a GPU is
    available. Pass `auto_llm=False` or an explicit `llm=None` to force the
    deterministic template. If the model call or JSON parse fails for any reason,
    it falls back to the template and records that in `StructuredReport.generator`.
    """

    def __init__(self, llm: LLMClient | None = None, *, auto_llm: bool = True) -> None:
        if llm is None and auto_llm:
            try:
                import torch
                if torch.cuda.is_available():
                    llm = LocalLLMClient()
            except Exception:  # noqa: BLE001
                pass  # no GPU or transformers not installed → template fallback
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
            generator=f"local:{model_name}",
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
