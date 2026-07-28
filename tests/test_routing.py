from __future__ import annotations

import unittest

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata
from routing import ExpertRegistry, ModalityRouter, RoutingError


class FakeExpert:
    def __init__(self, name: str, modality: Modality, body_part: BodyPart) -> None:
        self.name = name
        self.modality = modality
        self.body_part = body_part

    def predict(self, scan):  # pragma: no cover - routing tests do not infer
        raise NotImplementedError


def _scan(modality: Modality, body_part: BodyPart) -> Scan:
    return Scan(data=None, meta=ScanMetadata(modality=modality, body_part=body_part))


class RoutingTests(unittest.TestCase):
    def test_registration_and_flattening_are_identity_deduplicated(self) -> None:
        expert = FakeExpert("ct", Modality.CT, BodyPart.ABDOMEN)
        registry = ExpertRegistry()
        registry.register_niche(Modality.CT, BodyPart.CHEST, expert)
        registry.register_niche(Modality.CT, BodyPart.CHEST, expert)
        registry.register_niche(Modality.CT, BodyPart.ABDOMEN, expert)

        self.assertEqual(registry.experts(), [expert])
        self.assertEqual(registry.match(Modality.CT, BodyPart.CHEST), [expert])

    def test_unknown_body_part_routes_when_modality_is_unambiguous(self) -> None:
        expert = FakeExpert("chest", Modality.XRAY, BodyPart.CHEST)
        registry = ExpertRegistry()
        registry.register(expert)

        routed = ModalityRouter(registry, strict=False).route(
            _scan(Modality.XRAY, BodyPart.UNKNOWN)
        )
        self.assertEqual(routed, [expert])

    def test_unknown_body_part_rejects_ambiguous_modality(self) -> None:
        registry = ExpertRegistry()
        registry.register(FakeExpert("chest", Modality.XRAY, BodyPart.CHEST))
        registry.register(FakeExpert("wrist", Modality.XRAY, BodyPart.BONE))

        with self.assertRaisesRegex(RoutingError, "Ambiguous xray fallback"):
            ModalityRouter(registry, strict=False).route(
                _scan(Modality.XRAY, BodyPart.UNKNOWN)
            )

    def test_known_incompatible_niche_never_crosses_modalities(self) -> None:
        registry = ExpertRegistry()
        registry.register(FakeExpert("brain_mri", Modality.MRI, BodyPart.BRAIN))

        with self.assertRaisesRegex(RoutingError, "fallback is only permitted"):
            ModalityRouter(registry, strict=False).route(
                _scan(Modality.XRAY, BodyPart.BRAIN)
            )


if __name__ == "__main__":
    unittest.main()
