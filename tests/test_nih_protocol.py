from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.nih_protocol import (
    filename_from_mirror_row,
    make_patient_disjoint_nih_splits,
    read_nih_filename_manifest,
    reconcile_nih_official_manifests,
)


class NIHManifestReconciliationTests(unittest.TestCase):
    def test_reconciliation_reports_missing_and_unlisted_images(self) -> None:
        reconciliation = reconcile_nih_official_manifests(
            [
                "00000001_000.png",
                "00000002_000.png",
                "00000003_000.png",
                "99999999_000.png",
            ],
            [
                "00000001_000.png",
                "00000002_000.png",
                "00000004_000.png",
            ],
            ["00000003_000.png", "00000005_000.png"],
        )

        self.assertEqual(
            reconciliation.available_train_val,
            {"00000001_000.png", "00000002_000.png"},
        )
        self.assertEqual(reconciliation.available_test, {"00000003_000.png"})
        self.assertEqual(reconciliation.unlisted_available, {"99999999_000.png"})
        self.assertEqual(reconciliation.missing_train_val, {"00000004_000.png"})
        self.assertEqual(reconciliation.missing_test, {"00000005_000.png"})
        self.assertFalse(reconciliation.complete)

    def test_complete_reconciliation_fails_closed_on_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            reconcile_nih_official_manifests(
                ["00000001_000.png", "00000003_000.png"],
                ["00000001_000.png", "00000002_000.png"],
                ["00000003_000.png"],
                require_complete=True,
            )

    def test_manifest_image_or_patient_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap by 1 image"):
            reconcile_nih_official_manifests(
                ["00000001_000.png"],
                ["00000001_000.png"],
                ["00000001_000.png"],
            )

        with self.assertRaisesRegex(ValueError, "overlap by 1 patient"):
            reconcile_nih_official_manifests(
                ["00000001_000.png", "00000001_001.png"],
                ["00000001_000.png"],
                ["00000001_001.png"],
            )

    def test_duplicate_source_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate filename"):
            reconcile_nih_official_manifests(
                ["00000001_000.png", "00000001_000.png"],
                ["00000001_000.png"],
                ["00000002_000.png"],
            )

    def test_malformed_nih_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid NIH image filename"):
            reconcile_nih_official_manifests(
                ["patient_1.png"],
                ["patient_1.png"],
                ["00000002_000.png"],
            )

    def test_manifest_reader_rejects_duplicate_and_malformed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.txt"
            path.write_text(
                "00000001_000.png\n00000001_000.png\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate NIH image filename"):
                read_nih_filename_manifest(path)

            path.write_text("not-an-nih-image.png\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid NIH image filename"):
                read_nih_filename_manifest(path)


class NIHOfficialSplitTests(unittest.TestCase):
    def test_validation_is_patient_disjoint_and_test_membership_is_immutable(self) -> None:
        train_val = [
            "00000001_000.png",
            "00000001_001.png",
            "00000002_000.png",
            "00000002_001.png",
            "00000003_000.png",
            "00000004_000.png",
        ]
        official_test = ["00000005_000.png", "00000006_000.png"]
        available = train_val + official_test

        first = make_patient_disjoint_nih_splits(
            available,
            train_val,
            official_test,
            validation_fraction=0.34,
            seed=7,
            require_complete=True,
        )
        second = make_patient_disjoint_nih_splits(
            available,
            train_val,
            official_test,
            validation_fraction=0.5,
            seed=99,
            require_complete=True,
        )

        train_patients = {name.split("_", 1)[0] for name in first.train}
        validation_patients = {
            name.split("_", 1)[0] for name in first.validation
        }
        self.assertFalse(train_patients & validation_patients)
        self.assertEqual(first.train | first.validation, set(train_val))
        self.assertEqual(first.test, set(official_test))
        self.assertEqual(second.test, set(official_test))
        self.assertFalse((first.train | first.validation) & first.test)

    def test_unlisted_rows_are_not_silently_assigned(self) -> None:
        splits = make_patient_disjoint_nih_splits(
            [
                "00000001_000.png",
                "00000002_000.png",
                "00000003_000.png",
                "00000004_000.png",
            ],
            ["00000001_000.png", "00000002_000.png"],
            ["00000003_000.png"],
            validation_fraction=0.5,
        )
        self.assertEqual(splits.unassigned_available, {"00000004_000.png"})
        self.assertNotIn(
            "00000004_000.png", splits.train | splits.validation | splits.test
        )

    def test_missing_file_does_not_change_official_test_membership(self) -> None:
        splits = make_patient_disjoint_nih_splits(
            [
                "00000001_000.png",
                "00000002_000.png",
                "00000003_000.png",
            ],
            ["00000001_000.png", "00000002_000.png"],
            ["00000003_000.png", "00000004_000.png"],
            validation_fraction=0.5,
        )
        self.assertEqual(
            splits.test, {"00000003_000.png", "00000004_000.png"}
        )
        self.assertEqual(splits.available_test, {"00000003_000.png"})
        self.assertEqual(splits.missing_test, {"00000004_000.png"})


class MirrorIdentityTests(unittest.TestCase):
    def test_original_filename_is_preserved_when_mirror_exposes_it(self) -> None:
        self.assertEqual(
            filename_from_mirror_row(
                {"image": {"path": "nested/00012345_006.png"}}
            ),
            "00012345_006.png",
        )
        self.assertEqual(
            filename_from_mirror_row({"Image Index": "00012345_006.png"}),
            "00012345_006.png",
        )

    def test_opaque_mirror_row_does_not_get_a_fake_official_identity(self) -> None:
        self.assertIsNone(
            filename_from_mirror_row(
                {
                    "image": {"path": "/tmp/datasets/cache-123.png"},
                    "Patient ID": 12345,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
