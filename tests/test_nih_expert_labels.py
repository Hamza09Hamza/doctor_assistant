from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data.nih_expert_labels import (
    CORE_TARGETS,
    canonical_nih_filename,
    load_google_nih_expert_csv,
    prepare_nih_expert_manifest,
)


DOCUMENTED_HEADERS = (
    "Image Index",
    "Patient ID",
    "Set Id",
    "Fracture",
    "Pneumothorax",
    "Airspace opacity",
    "Nodule or mass",
)


class NIHExpertLabelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(
        self,
        name: str,
        rows: list[dict[str, object]],
        *,
        headers: tuple[str, ...] = DOCUMENTED_HEADERS,
        gzip_output: bool = False,
    ) -> Path:
        path = self.root / (name + (".gz" if gzip_output else ""))
        opener = gzip.open if gzip_output else open
        with opener(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_list(self, name: str, filenames: list[str]) -> Path:
        path = self.root / name
        path.write_text("".join(f"{item}\n" for item in filenames), encoding="utf-8")
        return path

    def _documented_row(
        self,
        filename: str,
        split: str,
        *,
        pneumothorax: str = "NO",
        nodule_or_mass: str = "NO",
        airspace_opacity: str = "NO",
        fracture: str = "NO",
    ) -> dict[str, object]:
        return {
            "Image Index": filename,
            "Patient ID": int(filename[:8]),
            "Set Id": split,
            "Fracture": fracture,
            "Pneumothorax": pneumothorax,
            "Airspace opacity": airspace_opacity,
            "Nodule or mass": nodule_or_mass,
        }

    def test_documented_columns_and_gzip_are_loaded(self) -> None:
        path = self._write_csv(
            "expert.csv",
            [
                self._documented_row(
                    "00000013_008.png",
                    "val",
                    pneumothorax="YES",
                    nodule_or_mass="NO",
                    airspace_opacity="YES",
                    fracture="YES",
                )
            ],
            gzip_output=True,
        )

        loaded = load_google_nih_expert_csv(
            path, expected_split="validation", include_fracture=True
        )

        self.assertEqual(len(loaded.rows), 1)
        row = loaded.rows[0]
        self.assertEqual(row.filename, "00000013_008.png")
        self.assertEqual(row.patient_id, "00000013")
        self.assertEqual(row.split, "validation")
        self.assertEqual(
            row.labels,
            {
                "Pneumothorax": 1,
                "Nodule_or_mass": 0,
                "Airspace_opacity": 1,
                "Fracture": 1,
            },
        )

    def test_alias_columns_are_auto_detected(self) -> None:
        headers = (
            "filename",
            "patient_id",
            "split",
            "Pneumothorax",
            "Nodule/Mass",
            "Lung Opacity",
        )
        path = self._write_csv(
            "aliases.csv",
            [
                {
                    "filename": r"folder\00000002_001.PNG",
                    "patient_id": "2",
                    "split": "development",
                    "Pneumothorax": "NO",
                    "Nodule/Mass": "YES",
                    "Lung Opacity": "NO",
                }
            ],
            headers=headers,
        )

        loaded = load_google_nih_expert_csv(
            path, expected_split="validation"
        )

        self.assertEqual(loaded.rows[0].filename, "00000002_001.png")
        self.assertEqual(loaded.rows[0].labels["Nodule_or_mass"], 1)
        self.assertEqual(set(loaded.rows[0].labels), set(CORE_TARGETS))

    def test_non_adjudicated_cells_and_empty_rows_are_explicitly_skipped(self) -> None:
        path = self._write_csv(
            "missing.csv",
            [
                self._documented_row(
                    "00000001_000.png",
                    "val",
                    pneumothorax="YES",
                    nodule_or_mass="",
                    airspace_opacity="NOT ADJUDICATED",
                ),
                self._documented_row(
                    "00000002_000.png",
                    "val",
                    pneumothorax="",
                    nodule_or_mass="N/A",
                    airspace_opacity="UNKNOWN",
                ),
            ],
        )

        loaded = load_google_nih_expert_csv(
            path, expected_split="validation"
        )

        self.assertEqual(len(loaded.rows), 1)
        self.assertEqual(loaded.skipped_rows_no_labels, 1)
        self.assertEqual(loaded.rows[0].labels["Pneumothorax"], 1)
        self.assertIsNone(loaded.rows[0].labels["Nodule_or_mass"])
        self.assertIsNone(loaded.rows[0].labels["Airspace_opacity"])
        self.assertEqual(
            loaded.label_counts["Pneumothorax"],
            {"no": 0, "skipped_non_adjudicated": 1, "yes": 1},
        )

    def test_labels_must_be_exact_uppercase_yes_or_no(self) -> None:
        path = self._write_csv(
            "bad-label.csv",
            [
                self._documented_row(
                    "00000001_000.png", "val", pneumothorax="yes"
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "exactly YES or NO"):
            load_google_nih_expert_csv(path, expected_split="validation")

    def test_unknown_non_adjudicated_marker_is_not_silently_negative(self) -> None:
        path = self._write_csv(
            "bad-marker.csv",
            [
                self._documented_row(
                    "00000001_000.png", "val", pneumothorax="MAYBE"
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "got 'MAYBE'"):
            load_google_nih_expert_csv(path, expected_split="validation")

    def test_missing_or_ambiguous_schema_has_clear_error(self) -> None:
        missing_headers = (
            "Image Index",
            "Pneumothorax",
            "Nodule or mass",
        )
        missing = self._write_csv(
            "missing-column.csv",
            [
                {
                    "Image Index": "00000001_000.png",
                    "Pneumothorax": "NO",
                    "Nodule or mass": "NO",
                }
            ],
            headers=missing_headers,
        )
        with self.assertRaisesRegex(ValueError, "Airspace_opacity label column"):
            load_google_nih_expert_csv(missing, expected_split="validation")

        ambiguous_headers = (
            "Image Index",
            "image_index",
            "Pneumothorax",
            "Nodule or mass",
            "Airspace opacity",
        )
        ambiguous = self._write_csv(
            "ambiguous.csv",
            [
                {
                    "Image Index": "00000001_000.png",
                    "image_index": "00000001_000.png",
                    "Pneumothorax": "NO",
                    "Nodule or mass": "NO",
                    "Airspace opacity": "NO",
                }
            ],
            headers=ambiguous_headers,
        )
        with self.assertRaisesRegex(ValueError, "duplicate/ambiguous headers"):
            load_google_nih_expert_csv(ambiguous, expected_split="validation")

    def test_filename_duplicate_patient_and_declared_split_are_gated(self) -> None:
        malformed = self._write_csv(
            "malformed.csv",
            [
                {
                    **self._documented_row("00000001_000.png", "val"),
                    "Image Index": "not-an-nih-name.png",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "invalid NIH image filename"):
            load_google_nih_expert_csv(malformed, expected_split="validation")

        patient_mismatch_row = self._documented_row("00000001_000.png", "val")
        patient_mismatch_row["Patient ID"] = 2
        patient_mismatch = self._write_csv(
            "patient-mismatch.csv", [patient_mismatch_row]
        )
        with self.assertRaisesRegex(ValueError, "does not match filename-derived"):
            load_google_nih_expert_csv(
                patient_mismatch, expected_split="validation"
            )

        split_mismatch = self._write_csv(
            "split-mismatch.csv",
            [self._documented_row("00000001_000.png", "test")],
        )
        with self.assertRaisesRegex(ValueError, "declared 'validation'"):
            load_google_nih_expert_csv(
                split_mismatch, expected_split="validation"
            )

        duplicate = self._write_csv(
            "duplicate.csv",
            [
                self._documented_row("00000001_000.png", "val"),
                self._documented_row("00000001_000.png", "val"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "duplicate expert-label image"):
            load_google_nih_expert_csv(duplicate, expected_split="validation")

    def test_combined_csv_requires_valid_split_for_every_row(self) -> None:
        missing_split = self._write_csv(
            "combined-no-split.csv",
            [self._documented_row("00000001_000.png", "")],
        )
        with self.assertRaisesRegex(ValueError, "blank split"):
            load_google_nih_expert_csv(missing_split, expected_split=None)

        no_split_header = (
            "Image Index",
            "Pneumothorax",
            "Nodule or mass",
            "Airspace opacity",
        )
        no_split = self._write_csv(
            "no-split-column.csv",
            [
                {
                    "Image Index": "00000001_000.png",
                    "Pneumothorax": "NO",
                    "Nodule or mass": "NO",
                    "Airspace opacity": "NO",
                }
            ],
            headers=no_split_header,
        )
        with self.assertRaisesRegex(ValueError, "requires a split column"):
            load_google_nih_expert_csv(no_split, expected_split=None)

    def test_preparation_reconciles_splits_and_emits_verifiable_hashes(self) -> None:
        combined = self._write_csv(
            "combined.csv",
            [
                self._documented_row(
                    "00000002_001.png", "val", pneumothorax="YES"
                ),
                self._documented_row(
                    "00000001_000.png", "validation", nodule_or_mass="YES"
                ),
                self._documented_row(
                    "00000003_000.png", "test", airspace_opacity="YES"
                ),
            ],
        )
        train_val = self._write_list(
            "train_val_list.txt",
            ["00000001_000.png", "00000002_001.png"],
        )
        test = self._write_list("test_list.txt", ["00000003_000.png"])
        output_csv = self.root / "output" / "canonical.csv"
        output_json = self.root / "output" / "canonical.metadata.json"

        metadata = prepare_nih_expert_manifest(
            combined_csv_path=combined,
            official_train_val_path=train_val,
            official_test_path=test,
            output_csv_path=output_csv,
            output_json_path=output_json,
        )

        with output_csv.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["filename"] for row in output_rows],
            [
                "00000001_000.png",
                "00000002_001.png",
                "00000003_000.png",
            ],
        )
        self.assertEqual(output_rows[0]["patient_id"], "00000001")
        self.assertEqual(output_rows[0]["split"], "validation")
        self.assertEqual(output_rows[0]["Nodule_or_mass"], "1")
        self.assertEqual(output_rows[2]["Airspace_opacity"], "1")
        self.assertNotIn("Fracture", output_rows[0])

        disk_metadata = json.loads(output_json.read_text(encoding="utf-8"))
        output_digest = hashlib.sha256(output_csv.read_bytes()).hexdigest()
        self.assertEqual(metadata, disk_metadata)
        self.assertEqual(metadata["output_csv"]["sha256"], output_digest)
        self.assertEqual(
            metadata["sources"]["combined"]["sha256"],
            hashlib.sha256(combined.read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["rows"], {"total": 3, "validation": 2, "test": 1})

    def test_separate_files_and_optional_fracture_are_supported(self) -> None:
        validation = self._write_csv(
            "validation.csv",
            [
                self._documented_row(
                    "00000001_000.png", "val", fracture="YES"
                )
            ],
        )
        test_csv = self._write_csv(
            "test.csv",
            [
                self._documented_row(
                    "00000002_000.png", "test", fracture="NO"
                )
            ],
        )
        train_val = self._write_list("train_val.txt", ["00000001_000.png"])
        test = self._write_list("test.txt", ["00000002_000.png"])
        output_csv = self.root / "labels.csv"

        prepare_nih_expert_manifest(
            validation_csv_path=validation,
            test_csv_path=test_csv,
            official_train_val_path=train_val,
            official_test_path=test,
            output_csv_path=output_csv,
            output_json_path=self.root / "metadata.json",
            include_fracture=True,
        )

        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("Fracture", rows[0])
        self.assertEqual(rows[0]["Fracture"], "1")
        self.assertEqual(rows[1]["Fracture"], "0")

    def test_development_output_does_not_materialize_test_labels(self) -> None:
        combined = self._write_csv(
            "combined.csv",
            [
                self._documented_row(
                    "00000001_000.png", "val", pneumothorax="YES"
                ),
                self._documented_row(
                    "00000002_000.png", "test", pneumothorax="YES"
                ),
            ],
        )
        train_val = self._write_list("train_val.txt", ["00000001_000.png"])
        test = self._write_list("test.txt", ["00000002_000.png"])
        output_csv = self.root / "development.csv"

        metadata = prepare_nih_expert_manifest(
            combined_csv_path=combined,
            official_train_val_path=train_val,
            official_test_path=test,
            output_csv_path=output_csv,
            output_json_path=self.root / "development.metadata.json",
            output_cohort="development",
        )

        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["filename"] for row in rows], ["00000001_000.png"])
        self.assertEqual(metadata["output_cohort"], "development")
        self.assertEqual(metadata["rows"], {"total": 1, "validation": 1, "test": 0})
        self.assertEqual(
            metadata["validated_source_rows"],
            {"total": 2, "validation": 1, "test": 1},
        )

    def test_wrong_official_split_and_official_patient_overlap_fail_closed(self) -> None:
        validation = self._write_csv(
            "validation.csv",
            [self._documented_row("00000001_000.png", "val")],
        )
        test_csv = self._write_csv(
            "test.csv",
            [self._documented_row("00000002_000.png", "test")],
        )
        train_val = self._write_list("train_val.txt", ["00000099_000.png"])
        test = self._write_list("test.txt", ["00000002_000.png"])

        with self.assertRaisesRegex(ValueError, "absent from the official NIH train_val"):
            prepare_nih_expert_manifest(
                validation_csv_path=validation,
                test_csv_path=test_csv,
                official_train_val_path=train_val,
                official_test_path=test,
                output_csv_path=self.root / "wrong.csv",
                output_json_path=self.root / "wrong.json",
            )

        overlapping_train = self._write_list(
            "overlapping-train.txt", ["00000001_000.png"]
        )
        overlapping_test = self._write_list(
            "overlapping-test.txt", ["00000001_001.png", "00000002_000.png"]
        )
        with self.assertRaisesRegex(ValueError, "overlap by 1 patient"):
            prepare_nih_expert_manifest(
                validation_csv_path=validation,
                test_csv_path=test_csv,
                official_train_val_path=overlapping_train,
                official_test_path=overlapping_test,
                output_csv_path=self.root / "overlap.csv",
                output_json_path=self.root / "overlap.json",
            )

    def test_official_manifest_duplicates_and_output_overwrite_are_rejected(self) -> None:
        validation = self._write_csv(
            "validation.csv",
            [self._documented_row("00000001_000.png", "val")],
        )
        test_csv = self._write_csv(
            "test.csv",
            [self._documented_row("00000002_000.png", "test")],
        )
        duplicate_train = self._write_list(
            "duplicate-train.txt",
            ["00000001_000.png", "00000001_000.png"],
        )
        test = self._write_list("test.txt", ["00000002_000.png"])

        with self.assertRaisesRegex(ValueError, "duplicate filename"):
            prepare_nih_expert_manifest(
                validation_csv_path=validation,
                test_csv_path=test_csv,
                official_train_val_path=duplicate_train,
                official_test_path=test,
                output_csv_path=self.root / "duplicate.csv",
                output_json_path=self.root / "duplicate.json",
            )

        train_val = self._write_list("train-val.txt", ["00000001_000.png"])
        with self.assertRaisesRegex(ValueError, "refusing to overwrite input"):
            prepare_nih_expert_manifest(
                validation_csv_path=validation,
                test_csv_path=test_csv,
                official_train_val_path=train_val,
                official_test_path=test,
                output_csv_path=validation,
                output_json_path=self.root / "overwrite.json",
            )

    def test_source_modes_must_be_exclusive_and_complete(self) -> None:
        common = {
            "official_train_val_path": self.root / "train.txt",
            "official_test_path": self.root / "test.txt",
            "output_csv_path": self.root / "output.csv",
            "output_json_path": self.root / "output.json",
        }
        with self.assertRaisesRegex(ValueError, "not both modes"):
            prepare_nih_expert_manifest(
                combined_csv_path=self.root / "combined.csv",
                validation_csv_path=self.root / "validation.csv",
                test_csv_path=self.root / "test-labels.csv",
                **common,
            )
        with self.assertRaisesRegex(ValueError, "both validation_csv_path"):
            prepare_nih_expert_manifest(
                validation_csv_path=self.root / "validation.csv",
                **common,
            )

    def test_strict_canonical_filename(self) -> None:
        self.assertEqual(
            canonical_nih_filename("/images/00000001_000.PNG"),
            "00000001_000.png",
        )
        for invalid in ("0000001_000.png", "00000001.png", "cache-001.png", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    canonical_nih_filename(invalid)


if __name__ == "__main__":
    unittest.main()
