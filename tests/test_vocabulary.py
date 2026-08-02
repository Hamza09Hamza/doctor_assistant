from __future__ import annotations

import unittest

from reporting.guidelines import _CHEST_GUIDELINES
from reporting.vocabulary import ALIASES, CANONICAL_LABELS, canonicalize


class VocabularyTests(unittest.TestCase):
    def test_known_variants_map_to_the_expected_canonical_code(self) -> None:
        self.assertEqual(canonicalize("Effusion"), "effusion")
        self.assertEqual(canonicalize("Pleural Effusion"), "effusion")
        self.assertEqual(canonicalize("Pleural_Thickening"), "pleural_thickening")
        self.assertEqual(canonicalize("cardiomegaly"), "cardiomegaly")

    def test_combined_kad_endpoints_never_alias_to_a_narrower_concept(self) -> None:
        # The specific bug this module exists to prevent: a KAD-flagged mass silently
        # triggering the Fleischner nodule-size-banding guideline logic.
        self.assertEqual(canonicalize("Nodule_or_mass"), "nodule_or_mass")
        self.assertNotEqual(canonicalize("Nodule_or_mass"), canonicalize("Nodule"))
        self.assertNotEqual(canonicalize("Nodule_or_mass"), canonicalize("Mass"))
        self.assertEqual(canonicalize("Airspace_opacity"), "airspace_opacity")
        self.assertNotEqual(canonicalize("Airspace_opacity"), canonicalize("Consolidation"))
        self.assertNotEqual(canonicalize("Airspace_opacity"), canonicalize("Infiltration"))

    def test_unrecognized_label_gets_a_stable_slug_not_a_crash(self) -> None:
        self.assertEqual(canonicalize("Some New Finding"), "some_new_finding")
        # Idempotent: canonicalizing an already-canonical code is a no-op.
        self.assertEqual(canonicalize(canonicalize("Some New Finding")), "some_new_finding")

    def test_every_alias_target_is_a_known_canonical_label(self) -> None:
        for raw, target in ALIASES.items():
            self.assertIn(target, CANONICAL_LABELS, f"alias {raw!r} -> unknown code {target!r}")

    def test_canonical_labels_is_a_superset_of_guideline_keys(self) -> None:
        # Guards against the two vocabularies drifting apart (vocabulary.py duplicates
        # this list rather than importing it, to avoid a circular import).
        self.assertTrue(set(_CHEST_GUIDELINES).issubset(CANONICAL_LABELS))
        self.assertIn("nodule", CANONICAL_LABELS)  # implicit in GuidelineEngine.recommend


if __name__ == "__main__":
    unittest.main()
