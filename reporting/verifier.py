"""The verifier — does the prose say only what the findings support?

The reporter is instructed to be a typist, but "instructed" is not "guaranteed". The
verifier closes that loop: it re-reads the generated report against the very
`source_findings` the report was built from and flags anything that doesn't trace back
to a number. This is the evidence-checking stage of the agentic radiology systems
(EviAgent's evidence collector, the multi-agent "verification" pass) — except it can run
with zero model calls, because all the ground truth is already attached to the report.

Two layers, cheapest first:

  - Deterministic grounding (always on): every numeric value in the prose must match a
    measured value in the findings (size, volume, probability, confidence, count); every
    *named pathology* drawn from the known label set must correspond to a present finding.
    Numbers that match nothing, or findings asserted that the model never flagged, are
    hard flags — these are the hallucinations that matter clinically.

  - Optional LLM entailment (if an `LLMClient` is given): a second pass asks a model
    whether each report sentence is entailed by the findings JSON. Useful for catching
    *qualitative* drift ("large", "worsening") that the numeric check can't see. It only
    ever adds warnings; it never overrides the deterministic verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections.abc import Sequence

from .findings import Finding
from .reporter import LLMClient, StructuredReport

# Numbers that carry a unit we measure, e.g. "12 mm", "3.4 mL", "score 0.87", "42%".
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(mm|ml|%)?", re.IGNORECASE)
# Tolerance for matching a prose number to a measured value (covers rounding in to_facts).
_ABS_TOL = 0.05


@dataclass
class VerificationResult:
    """The verdict plus the evidence for it, so a human can see *why* it failed."""

    ok: bool
    grounded_fraction: float                       # share of prose numbers that matched
    flags: list[str] = field(default_factory=list)    # hard problems (hallucinations)
    warnings: list[str] = field(default_factory=list)  # soft concerns (qualitative drift)
    checked_numbers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[{status}] grounded {self.grounded_fraction*100:.0f}% of numeric claims"]
        for f in self.flags:
            lines.append(f"  ✗ {f}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class Verifier:
    """Check a `StructuredReport` against its own `source_findings`.

    `known_labels` is the expert's full label vocabulary (e.g. the 14 ChestX-ray
    classes). With it, the verifier can flag a report that names a pathology the model
    did *not* flag — the most dangerous failure mode. Without it, that check is skipped
    and only numeric grounding runs.
    """

    def __init__(
        self,
        *,
        known_labels: Sequence[str] | None = None,
        llm: LLMClient | None = None,
        abs_tol: float = _ABS_TOL,
    ) -> None:
        self.known_labels = list(known_labels) if known_labels else []
        self.llm = llm
        self.abs_tol = abs_tol

    # -- public API ----------------------------------------------------------
    def verify(self, report: StructuredReport) -> VerificationResult:
        prose = "\n".join([report.findings, report.impression, report.recommendation])
        present = [f for f in report.source_findings if f.present]

        flags: list[str] = []
        warnings: list[str] = []

        grounded, total, checked = self._check_numbers(prose, present)
        flags.extend(f"ungrounded number '{tok}' — no matching measurement in findings"
                     for tok in checked if not tok.endswith("[ok]"))
        checked_display = [t.replace("[ok]", "") for t in checked]

        warnings.extend(self._check_unsupported_labels(prose, present))

        if self.llm is not None:
            warnings.extend(self._llm_entailment(report, present))

        grounded_fraction = (grounded / total) if total else 1.0
        ok = len(flags) == 0
        return VerificationResult(
            ok=ok,
            grounded_fraction=grounded_fraction,
            flags=flags,
            warnings=warnings,
            checked_numbers=checked_display,
        )

    # -- deterministic checks ------------------------------------------------
    def _allowed_numbers(self, present: Sequence[Finding]) -> list[float]:
        """Every numeric value the prose is permitted to contain."""
        allowed: list[float] = []
        for f in present:
            allowed.append(round(f.probability, 3))
            allowed.append(round(f.probability * 100, 1))  # probability stated as %
            if f.confidence is not None:
                allowed.append(round(f.confidence, 3))
                allowed.append(round(f.confidence * 100, 1))
            if f.size_mm is not None:
                allowed.append(round(f.size_mm, 1))
            if f.volume_ml is not None:
                allowed.append(round(f.volume_ml, 2))
            if f.count != 1:
                allowed.append(float(f.count))
        return allowed

    def _check_numbers(
        self, prose: str, present: Sequence[Finding]
    ) -> tuple[int, int, list[str]]:
        """Match each number in the prose to an allowed measured value.

        Returns (grounded_count, total_count, tokens) where each token is tagged
        '[ok]' if it matched. Bare small integers (1-12) without a unit are treated as
        prose ("one focus", section numbers) and skipped — only decimals and
        unit-bearing numbers are held to grounding.
        """
        allowed = self._allowed_numbers(present)
        grounded = total = 0
        tokens: list[str] = []
        for match in _NUMBER_RE.finditer(prose):
            value = float(match.group(1))
            unit = (match.group(2) or "").lower()
            # Skip bare small integers with no unit — almost always ordinary prose.
            if not unit and value == int(value) and value <= 12 and "." not in match.group(1):
                continue
            total += 1
            token = match.group(0).strip()
            if any(abs(value - a) <= self.abs_tol for a in allowed):
                grounded += 1
                tokens.append(token + "[ok]")
            else:
                tokens.append(token)
        return grounded, total, tokens

    def _check_unsupported_labels(
        self, prose: str, present: Sequence[Finding]
    ) -> list[str]:
        """Flag any *known* pathology named in the prose that isn't a present finding."""
        if not self.known_labels:
            return []
        prose_l = prose.lower()
        present_terms = {self._normalize(f.label) for f in present}
        warnings: list[str] = []
        for label in self.known_labels:
            term = self._normalize(label)
            if term and term in prose_l and term not in present_terms:
                warnings.append(
                    f"report mentions '{label}' but it is not a present finding"
                )
        return warnings

    @staticmethod
    def _normalize(label: str) -> str:
        return label.replace("_", " ").strip().lower()

    # -- optional LLM entailment --------------------------------------------
    def _llm_entailment(
        self, report: StructuredReport, present: Sequence[Finding]
    ) -> list[str]:
        """Ask a model whether the report is entailed by the findings. Warnings only."""
        import json

        facts = json.dumps([f.to_facts() for f in present], ensure_ascii=False)
        system = (
            "You are a strict fact-checker for radiology drafts. You are given FINDINGS "
            "(JSON, the ground truth) and a DRAFT report. List ONLY claims in the draft "
            "that are NOT supported by the findings — invented measurements, pathologies, "
            "locations, or severity not in the JSON. If everything is supported, reply "
            "exactly 'OK'. Be terse: one short line per unsupported claim."
        )
        user = f"FINDINGS:\n{facts}\n\nDRAFT:\n{report.to_text()}"
        try:
            verdict = self.llm.complete(system, user).strip()
        except Exception:  # noqa: BLE001 - verification must never crash the pipeline
            return ["LLM entailment check skipped (call failed)"]
        if verdict.upper().startswith("OK") or not verdict:
            return []
        return [f"LLM: {line.strip(' -•')}" for line in verdict.splitlines() if line.strip()]
