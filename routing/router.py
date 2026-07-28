"""The registry + router that turn a `Scan` into the expert(s) that should see it.

`ExpertRegistry` is a small table keyed by `(modality, body_part)`. `ModalityRouter`
reads a scan's metadata and returns every expert whose niche matches, satisfying the
`core.interfaces.Router` protocol. Matching is exact by default, with optional graceful
fallbacks (body-part-only, then modality-only) so a partially-labelled scan still finds
a plausible reader instead of silently dropping out of the pipeline.
"""

from __future__ import annotations

from core.enums import BodyPart, Modality
from core.interfaces import ExpertModel
from core.types import Scan


class RoutingError(RuntimeError):
    """Raised when no registered expert can handle a scan and no fallback applies."""


class ExpertRegistry:
    """A lookup of experts by the (modality, body_part) niche each advertises.

    Several experts may share a niche (e.g. two chest-X-ray models to ensemble); the
    registry keeps them all and the router returns the lot for the orchestrator to run.
    """

    def __init__(self) -> None:
        self._by_niche: dict[tuple[Modality, BodyPart], list[ExpertModel]] = {}

    def register(self, expert: ExpertModel) -> ExpertModel:
        """Add an expert under its own advertised (modality, body_part). Returns it."""
        return self.register_niche(expert.modality, expert.body_part, expert)

    def register_niche(
        self, modality: Modality, body_part: BodyPart, expert: ExpertModel
    ) -> ExpertModel:
        """Add an expert under an *explicit* niche, regardless of what it advertises.

        Pretrained models often span several niches — TotalSegmentator reads chest *and*
        abdominal CT from one set of weights. Registering the same instance under each
        niche lets the router reach it without cloning the model. Returns it for chaining.
        """
        group = self._by_niche.setdefault((modality, body_part), [])
        if not any(existing is expert for existing in group):
            group.append(expert)
        return expert

    def experts(self) -> list[ExpertModel]:
        """Every registered expert once, even if it serves several niches."""
        return _unique_experts(
            e for group in self._by_niche.values() for e in group
        )

    def match(
        self, modality: Modality, body_part: BodyPart
    ) -> list[ExpertModel]:
        """Experts that exactly match the given niche (may be empty)."""
        return _unique_experts(self._by_niche.get((modality, body_part), ()))

    def match_body_part(self, body_part: BodyPart) -> list[ExpertModel]:
        """Experts for this body part regardless of modality (fallback path)."""
        return _unique_experts(
            e for (mod, bp), group in self._by_niche.items()
            if bp == body_part for e in group
        )

    def match_modality(self, modality: Modality) -> list[ExpertModel]:
        """Experts for this modality regardless of body part (fallback path)."""
        return _unique_experts(
            e for (mod, bp), group in self._by_niche.items()
            if mod == modality for e in group
        )

    def niches_for_modality(self, modality: Modality) -> set[BodyPart]:
        return {bp for (mod, bp) in self._by_niche if mod == modality}

    def niches_for_body_part(self, body_part: BodyPart) -> set[Modality]:
        return {mod for (mod, bp) in self._by_niche if bp == body_part}


def _unique_experts(experts) -> list[ExpertModel]:
    """Deduplicate by object identity while preserving registration order."""
    seen: set[int] = set()
    unique: list[ExpertModel] = []
    for expert in experts:
        key = id(expert)
        if key not in seen:
            seen.add(key)
            unique.append(expert)
    return unique


class ModalityRouter:
    """Route a scan to experts by its `(modality, body_part)` metadata.

    `strict=True` (default) only returns exact-niche matches and raises if there are
    none. With `strict=False`, fallback is allowed only for metadata fields that are
    actually UNKNOWN. A known-but-incompatible modality/body-part pair never crosses
    niches, and an ambiguous partial match raises instead of sending a scan to unrelated
    experts.
    """

    def __init__(self, registry: ExpertRegistry, *, strict: bool = True) -> None:
        self.registry = registry
        self.strict = strict

    def route(self, scan: Scan) -> list[ExpertModel]:
        modality = scan.meta.modality
        body_part = scan.meta.body_part

        exact = self.registry.match(modality, body_part)
        if exact:
            return exact
        if self.strict:
            raise RoutingError(
                f"No expert registered for ({modality.value}, {body_part.value}). "
                f"Registered niches: {sorted((m.value, b.value) for (m, b) in self.registry._by_niche)}"
            )

        # A fallback is safe only when the missing metadata is explicitly UNKNOWN.
        # Known XRAY+BRAIN must never be routed to an MRI brain model merely because
        # the body part matches.
        if modality is not Modality.UNKNOWN and body_part is not BodyPart.UNKNOWN:
            raise RoutingError(
                f"No exact expert for known niche ({modality.value}, {body_part.value}); "
                "fallback is only permitted for UNKNOWN metadata."
            )

        if modality is not Modality.UNKNOWN:
            fallback = self.registry.match_modality(modality)
            parts = self.registry.niches_for_modality(modality)
            if fallback and (len(parts) == 1 or len(fallback) == 1):
                return fallback
            if fallback:
                raise RoutingError(
                    f"Ambiguous {modality.value} fallback across body parts "
                    f"{sorted(p.value for p in parts)}."
                )

        if body_part is not BodyPart.UNKNOWN:
            fallback = self.registry.match_body_part(body_part)
            modalities = self.registry.niches_for_body_part(body_part)
            if fallback and (len(modalities) == 1 or len(fallback) == 1):
                return fallback
            if fallback:
                raise RoutingError(
                    f"Ambiguous {body_part.value} fallback across modalities "
                    f"{sorted(m.value for m in modalities)}."
                )

        if modality is Modality.UNKNOWN and body_part is BodyPart.UNKNOWN:
            niches = list(self.registry._by_niche)
            fallback = self.registry.experts()
            if fallback and (len(niches) == 1 or len(fallback) == 1):
                return fallback
            if fallback:
                raise RoutingError("Both modality and body part are unknown; routing is ambiguous.")
        raise RoutingError(
            f"No expert (even by fallback) for ({modality.value}, {body_part.value})."
        )
