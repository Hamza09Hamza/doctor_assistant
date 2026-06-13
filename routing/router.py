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
        """Add an expert under its (modality, body_part) key. Returns it for chaining."""
        key = (expert.modality, expert.body_part)
        self._by_niche.setdefault(key, []).append(expert)
        return expert

    def experts(self) -> list[ExpertModel]:
        """Every registered expert, flattened."""
        return [e for group in self._by_niche.values() for e in group]

    def match(
        self, modality: Modality, body_part: BodyPart
    ) -> list[ExpertModel]:
        """Experts that exactly match the given niche (may be empty)."""
        return list(self._by_niche.get((modality, body_part), ()))

    def match_body_part(self, body_part: BodyPart) -> list[ExpertModel]:
        """Experts for this body part regardless of modality (fallback path)."""
        return [
            e for (mod, bp), group in self._by_niche.items()
            if bp == body_part for e in group
        ]

    def match_modality(self, modality: Modality) -> list[ExpertModel]:
        """Experts for this modality regardless of body part (fallback path)."""
        return [
            e for (mod, bp), group in self._by_niche.items()
            if mod == modality for e in group
        ]


class ModalityRouter:
    """Route a scan to experts by its `(modality, body_part)` metadata.

    `strict=True` (default) only returns exact-niche matches and raises if there are
    none. With `strict=False` it tries progressively looser fallbacks — same body part,
    then same modality — before giving up, which is useful when an upstream detector
    couldn't pin down both fields.
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

        # Loosen: same body part, then same modality. Stops at the first non-empty set.
        for fallback in (
            self.registry.match_body_part(body_part) if body_part is not BodyPart.UNKNOWN else [],
            self.registry.match_modality(modality) if modality is not Modality.UNKNOWN else [],
        ):
            if fallback:
                return fallback
        raise RoutingError(
            f"No expert (even by fallback) for ({modality.value}, {body_part.value})."
        )
