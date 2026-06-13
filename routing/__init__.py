"""Routing layer: match a scan to the expert(s) that should read it.

A scan arrives stamped with a `(modality, body_part)`; each registered expert
advertises the niche it handles. The router is the lookup that connects them, so
the orchestrator never hard-wires "chest X-ray -> chest expert". Add an expert to
the registry and it becomes routable — no other code changes.
"""

from .router import ExpertRegistry, ModalityRouter, RoutingError

__all__ = ["ExpertRegistry", "ModalityRouter", "RoutingError"]
