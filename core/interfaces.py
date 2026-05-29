"""Structural contracts (typing.Protocol) for the swappable pipeline stages.

Using Protocols rather than base classes means a class satisfies an interface
just by having the right shape — no inheritance required. That keeps experts and
loaders loosely coupled to the framework.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .enums import BodyPart, Modality
from .types import Prediction, Scan


@runtime_checkable
class Loader(Protocol):
    """Turns a file path into a model-ready `Scan`."""

    def can_load(self, path: str) -> bool:
        """True if this loader recognizes the file (by extension / magic bytes)."""
        ...

    def load(self, path: str) -> Scan:
        ...


@runtime_checkable
class ExpertModel(Protocol):
    """A specialized diagnostic model for one (modality, body_part) niche.

    Implementations advertise what they handle so the router/registry can match
    a scan to the right expert(s) without hard-coded wiring.
    """

    name: str
    modality: Modality
    body_part: BodyPart

    def predict(self, scan: Scan) -> Prediction:
        ...


@runtime_checkable
class Router(Protocol):
    """Decides which experts should see a scan.

    A scan may match several experts (e.g. a chest CT routed to both a lung and a
    cardiac expert); the orchestrator runs each and hands results to the ensemble.
    """

    def route(self, scan: Scan) -> list[ExpertModel]:
        ...
