from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import scripts.benchmark_kad as benchmark_kad
from scripts.benchmark_kad import (
    PHASE1_LABELS,
    PHASE1_PROMPTS,
    build_runtime_contract,
    build_parser,
    deferred_development_scorecard,
    evaluate_expert_scorecard,
    inspect_query_pack,
    load_decision_policy,
    load_expert_manifest,
    load_image_provenance,
    load_manifest_metadata,
    load_source_metadata_provenance,
    prompt_set_sha256,
    run_canonical,
    validate_image_provenance,
    validate_test_lock,
)
from scripts.calibrate_kad_phase1 import (
    DevelopmentInputs,
    build_decision_artifact,
)
from scripts.export_kad_query_pack import PHASE1_QUERY_SPECS


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_decision_artifact(
    root: Path,
    *,
    query_pack_sha256: str,
    patients: int = 320,
) -> dict:
    patient_ids = np.asarray(
        [f"{patient_number:08d}" for patient_number in range(1, patients + 1)]
    )
    sample_ids = np.asarray(
        [f"{patient_id}_000.png" for patient_id in patient_ids]
    )
    active_truth = np.asarray(
        [patient_number % 2 for patient_number in range(1, patients + 1)],
        dtype=np.int8,
    )
    labels = active_truth.reshape(-1, 1)
    active_scores = np.where(active_truth == 1, 0.82, 0.18)
    raw_scores = active_scores.reshape(-1, 1)
    image_hashes = np.asarray(
        [
            hashlib.sha256(f"image:{sample_id}".encode()).hexdigest()
            for sample_id in sample_ids
        ]
    )
    predictions_path = root / "development.predictions.npz"
    np.savez_compressed(
        predictions_path,
        raw_scores=raw_scores,
        labels=labels,
        class_names=np.asarray([PHASE1_LABELS[0]]),
        patient_ids=patient_ids,
        sample_ids=sample_ids,
        image_sha256=image_hashes,
    )
    inputs = DevelopmentInputs(
        benchmark_path=root / "development.json",
        predictions_path=predictions_path,
        benchmark_sha256="3" * 64,
        predictions_sha256=_file_sha256(predictions_path),
        query_pack_sha256=query_pack_sha256,
        prompt_set_sha256=prompt_set_sha256(
            (PHASE1_LABELS[0],),
            (PHASE1_QUERY_SPECS[PHASE1_LABELS[0]]["prompt"],),
        ),
        manifest_sha256="2" * 64,
        active_target=PHASE1_LABELS[0],
        benchmark_metrics_scope=(
            "development_analysis_deferred_until_patient_partition"
        ),
        analysis_deferred=True,
        original_nih_pixels=True,
        benchmark_evidence_status=(
            "development_only_do_not_report_as_test_performance"
        ),
        raw_scores=raw_scores,
        labels=labels,
        patient_ids=patient_ids,
        sample_ids=sample_ids,
        image_sha256=image_hashes,
    )
    artifact = build_decision_artifact(
        inputs,
        seed=20250729,
        split_attempts=512,
        bootstrap_samples=1000,
        min_calibration_positives=10,
        min_calibration_negatives=20,
        min_threshold_positives=10,
        min_threshold_negatives=20,
    )
    if not artifact["thresholds_complete"]:
        raise AssertionError(
            "test fixture did not produce complete decision evidence: "
            + repr(artifact["diagnostics"])
        )
    return artifact


def _image_provenance_payload(
    *,
    filename: str,
    output_sha256: str,
    manifest_sha256: str,
    width: int,
    height: int,
    original_nih_pixels: bool,
    cohort: str = "development",
) -> dict:
    split = "validation" if cohort == "development" else "test"
    payload = {
        "artifact_type": "doctor_assistant.nih_image_provenance",
        "schema_version": 1,
        "source": (
            "NIH official image archive"
            if original_nih_pixels
            else "pinned development mirror"
        ),
        "resolution": {"width": width, "height": height},
        "original_nih_pixels": original_nih_pixels,
        "canonical_manifest": {
            "sha256": manifest_sha256,
            "cohort": cohort,
            "rows_selected": 1,
        },
        "images": [
            {
                "filename": filename,
                "expert_split": split,
                "output_sha256": output_sha256,
                "mirror_revision": (
                    "1c9e054e3336a473be6c01d77cdedf96442e2bad"
                    if not original_nih_pixels
                    else None
                ),
            }
        ],
    }
    if not original_nih_pixels:
        payload.update(
            {
                "development_only": True,
                "official_or_final_evidence_allowed": False,
                "mirror": {
                    "dataset": "arudaev/chest-xray-14-320",
                    "revision": (
                        "1c9e054e3336a473be6c01d77cdedf96442e2bad"
                    ),
                },
            }
        )
    return payload


def _original_pixel_provenance_payload(
    *,
    filename: str,
    output_sha256: str,
    manifest_sha256: str,
    width: int,
    height: int,
    cohort: str = "development",
    agreeing_source_count: int = 2,
    minimum_independent_sources: int = 2,
    corroborating_sources: int = 1,
) -> dict:
    """A schema-2 provenance payload backed by multi-source SHA-256 consensus."""

    split = "validation" if cohort == "development" else "test"
    return {
        "artifact_type": "doctor_assistant.nih_image_provenance",
        "schema_version": 2,
        "source": (
            "NIH Clinical Center direct release, cross-verified against "
            "independently-operated mirrors"
        ),
        "resolution": {"width": width, "height": height},
        "original_nih_pixels": True,
        "development_only": False,
        "official_or_final_evidence_allowed": True,
        "canonical_manifest": {
            "sha256": manifest_sha256,
            "cohort": cohort,
            "rows_selected": 1,
        },
        "archive_verification": {
            "method": "multi_source_sha256_consensus_v1",
            "minimum_independent_sources": minimum_independent_sources,
            "primary_source": {
                "name": "NIH Clinical Center official release",
                "identifier": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
            },
            "corroborating_sources": [
                {
                    "name": "academictorrents (NIH Clinical Center attribution)",
                    "identifier": (
                        "infohash:557481faacd824c83fbf57dcf7b6da9383b3235a"
                    ),
                }
                for _ in range(corroborating_sources)
            ],
            "agreement": "all_selected_files_sha256_identical_across_all_sources",
        },
        "images": [
            {
                "filename": filename,
                "expert_split": split,
                "output_sha256": output_sha256,
                "agreeing_source_count": agreeing_source_count,
            }
        ],
    }


class KADBenchmarkManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.train_names = (
            "00000001_000.png",
            "00000002_000.png",
            "00000003_000.png",
        )
        self.test_names = ("00000004_000.png",)
        for index, name in enumerate(self.train_names):
            Image.fromarray(
                np.full((8, 9), 40 + index, dtype=np.uint8), mode="L"
            ).save(self.images / name)
        self.train_list = self.root / "train_val_list.txt"
        self.test_list = self.root / "test_list.txt"
        self.train_list.write_text("\n".join(self.train_names) + "\n", encoding="utf-8")
        self.test_list.write_text("\n".join(self.test_names) + "\n", encoding="utf-8")
        self.manifest = self.root / "expert.csv"
        self._write_manifest(
            [
                [self.train_names[0], "00000001", "validation", "1", "0", ""],
                [self.train_names[1], "00000002", "validation", "0", "1", "1"],
                [self.train_names[2], "00000003", "validation", "1", "0", "0"],
                [self.test_names[0], "00000004", "test", "1", "0", "1"],
            ]
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self, rows: list[list[str]]) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["filename", "patient_id", "split", *PHASE1_LABELS])
            writer.writerows(rows)

    def _load(self):
        return load_expert_manifest(
            self.manifest,
            self.images,
            cohort="development",
            official_train_val_manifest=self.train_list,
            official_test_manifest=self.test_list,
        )

    def test_loads_only_validation_and_preserves_non_adjudicated_label(self) -> None:
        loaded = self._load()

        self.assertEqual(loaded.sample_ids, self.train_names)
        self.assertEqual(loaded.labels.tolist()[0], [1, 0, -1])
        self.assertTrue(loaded.official_membership_verified)
        self.assertRegex(loaded.manifest_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(loaded.image_set_sha256, r"^[0-9a-f]{64}$")

    def test_rejects_nonbinary_nonblank_label(self) -> None:
        self._write_manifest(
            [
                [self.train_names[0], "00000001", "validation", "YES", "0", "1"],
                [self.train_names[1], "00000002", "validation", "0", "1", "0"],
            ]
        )

        with self.assertRaisesRegex(ValueError, "blank, 0, or 1"):
            self._load()

    def test_rejects_declared_development_image_from_official_test(self) -> None:
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8), mode="L").save(
            self.images / self.test_names[0]
        )
        self._write_manifest(
            [
                [self.train_names[0], "00000001", "validation", "1", "0", "1"],
                [self.train_names[1], "00000002", "validation", "0", "1", "0"],
                [self.test_names[0], "00000004", "validation", "1", "0", "1"],
            ]
        )

        with self.assertRaisesRegex(ValueError, "wrong official NIH partition"):
            self._load()

    def test_missing_image_fails_before_inference(self) -> None:
        (self.images / self.train_names[1]).unlink()

        with self.assertRaisesRegex(FileNotFoundError, "manifest image is missing"):
            self._load()

    def test_only_active_target_must_be_scoreable(self) -> None:
        self._write_manifest(
            [
                [self.train_names[0], "00000001", "validation", "", "1", ""],
                [self.train_names[1], "00000002", "validation", "", "0", ""],
            ]
        )

        loaded = load_expert_manifest(
            self.manifest,
            self.images,
            cohort="development",
            active_target="Nodule_or_mass",
            official_train_val_manifest=self.train_list,
            official_test_manifest=self.test_list,
        )
        self.assertEqual(loaded.sample_ids, self.train_names[:2])

        with self.assertRaisesRegex(ValueError, "Pneumothorax.*not scoreable"):
            self._load()


class KADBenchmarkProvenanceChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_entry = self.root / "Data_Entry_2017.csv"
        self.train_list = self.root / "train_val_list.txt"
        self.test_list = self.root / "test_list.txt"
        self.google_labels = (
            self.root / "google2019_nih-chest-xray-labels.csv.gz"
        )
        self.data_entry.write_text("synthetic pinned table\n", encoding="utf-8")
        self.train_list.write_text(
            "00000001_000.png\n00000002_000.png\n", encoding="utf-8"
        )
        self.test_list.write_text("00000003_000.png\n", encoding="utf-8")
        with gzip.open(
            self.google_labels,
            mode="wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "Image Index",
                    "Patient ID",
                    "Fracture",
                    "Pneumothorax",
                    "Airspace opacity",
                    "Nodule or mass",
                    "Set Id",
                ]
            )
            writer.writerows(
                [
                    ["00000001_000.png", "1", "NO", "YES", "YES", "NO", "val"],
                    ["00000002_000.png", "2", "NO", "NO", "NO", "YES", "val"],
                    ["00000003_000.png", "3", "NO", "YES", "NO", "NO", "test"],
                ]
            )

        self.nih_specs = {
            "Data_Entry_2017.csv": {
                "url": "https://example.test/data",
                "sha256": _file_sha256(self.data_entry),
                "rows": 3,
            },
            "train_val_list.txt": {
                "url": "https://example.test/train",
                "sha256": _file_sha256(self.train_list),
                "rows": 2,
            },
            "test_list.txt": {
                "url": "https://example.test/test",
                "sha256": _file_sha256(self.test_list),
                "rows": 1,
            },
        }
        self.google_spec = {
            "url": "https://example.test/google",
            "sha256": _file_sha256(self.google_labels),
            "rows": 3,
            "split_rows": {"val": 2, "test": 1},
            "google_documented_split_rows": {"val": 1, "test": 1},
            "label_counts": {
                "Fracture": {"NO": 3, "YES": 0},
                "Pneumothorax": {"NO": 1, "YES": 2},
                "Airspace opacity": {"NO": 2, "YES": 1},
                "Nodule or mass": {"NO": 2, "YES": 1},
            },
        }
        self.all_specs = {
            **self.nih_specs,
            self.google_labels.name: self.google_spec,
        }
        paths = {
            "Data_Entry_2017.csv": self.data_entry,
            "train_val_list.txt": self.train_list,
            "test_list.txt": self.test_list,
            self.google_labels.name: self.google_labels,
        }
        self.source_path = self.root / "nih_metadata.provenance.json"
        self.source_data = {
            "schema_version": 1,
            "artifact_type": (
                "doctor_assistant.nih_metadata_and_expert_labels"
            ),
            "sources": {
                "nih_metadata": {
                    "dataset": benchmark_kad.YEIGEN_DATASET,
                    "revision": benchmark_kad.YEIGEN_REVISION,
                },
                "google_expert_labels": {
                    "repository": "mlmed/torchxrayvision",
                    "revision": benchmark_kad.TORCHXRAYVISION_COMMIT,
                },
            },
            "files": {
                name: {
                    "path": str(paths[name]),
                    "url": spec["url"],
                    "sha256": spec["sha256"],
                    "bytes": paths[name].stat().st_size,
                }
                for name, spec in self.all_specs.items()
            },
            "validation": {
                "nih_metadata": {
                    "rows": {
                        name: spec["rows"]
                        for name, spec in self.nih_specs.items()
                    },
                    "split_overlap": 0,
                    "split_union_rows": 3,
                    "data_entry_unique_rows": 3,
                },
                "google_expert_labels": {
                    "rows": 3,
                    "unique_images": 3,
                    "split_rows": {"val": 2, "test": 1},
                    "label_counts": self.google_spec["label_counts"],
                    "all_four_findings_adjudicated_yes_no": True,
                },
            },
            "known_source_discrepancy": {
                "google_documented_split_rows": {"val": 1, "test": 1},
                "pinned_mirror_split_rows": {"val": 2, "test": 1},
                "validation_row_delta": 1,
            },
        }
        self.source_path.write_text(
            json.dumps(self.source_data), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _patched_specs(self):
        return patch.multiple(
            benchmark_kad,
            NIH_METADATA_FILES=self.nih_specs,
            GOOGLE_EXPERT_LABEL_SPEC=self.google_spec,
            PINNED_FILES=self.all_specs,
        )

    def _load_source(self, *, cohort: str = "development"):
        with self._patched_specs():
            return load_source_metadata_provenance(
                self.source_path,
                cohort=cohort,
                official_train_val_manifest=self.train_list,
                official_test_manifest=self.test_list,
            )

    def test_source_receipt_rehashes_and_pins_official_split_files(self) -> None:
        checked = self._load_source()

        self.assertTrue(checked["verified"])
        self.assertEqual(
            checked["official_train_val_manifest_sha256"],
            _file_sha256(self.train_list),
        )

        arbitrary = self.root / "arbitrary_train_val_list.txt"
        arbitrary.write_text(self.train_list.read_text(encoding="utf-8"))
        with self._patched_specs(), self.assertRaisesRegex(
            ValueError, "exact file recorded"
        ):
            load_source_metadata_provenance(
                self.source_path,
                cohort="development",
                official_train_val_manifest=arbitrary,
                official_test_manifest=self.test_list,
            )

    def test_schema_one_source_receipt_cannot_unlock_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "TEST DISABLED.*direct-release"):
            self._load_source(cohort="test")

    def test_manifest_metadata_binds_output_and_full_source_audit(self) -> None:
        images = self.root / "images"
        images.mkdir()
        for index, name in enumerate(("00000001_000.png", "00000002_000.png")):
            Image.fromarray(
                np.full((8, 9), index, dtype=np.uint8), mode="L"
            ).save(images / name)
        manifest_path = self.root / "labels.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["filename", "patient_id", "split", *PHASE1_LABELS])
            writer.writerows(
                [
                    ["00000001_000.png", "00000001", "validation", 1, 0, 1],
                    ["00000002_000.png", "00000002", "validation", 0, 1, 0],
                ]
            )
        manifest = load_expert_manifest(
            manifest_path,
            images,
            cohort="development",
            official_train_val_manifest=self.train_list,
            official_test_manifest=self.test_list,
        )
        metadata_path = self.root / "labels.metadata.json"
        metadata = {
            "schema_version": 1,
            "format": "nih_google_four_findings_canonical",
            "output_cohort": "development",
            "targets": list(PHASE1_LABELS),
            "rows": {"total": 2, "validation": 2, "test": 0},
            "validated_source_rows": {
                "total": 3,
                "validation": 2,
                "test": 1,
            },
            "sources": {
                "combined": {
                    "path": str(self.google_labels),
                    "sha256": _file_sha256(self.google_labels),
                    "source_rows": 3,
                    "rows_emitted": 3,
                    "rows_skipped_no_adjudicated_labels": 0,
                }
            },
            "official_manifests": {
                "train_val": {
                    "sha256": _file_sha256(self.train_list),
                    "images": 2,
                },
                "test": {
                    "sha256": _file_sha256(self.test_list),
                    "images": 1,
                },
            },
            "reconciliation": {
                "validation_rows_in_official_train_val": 2,
                "test_rows_in_official_test": 1,
                "cross_split_image_overlap": 0,
                "official_patient_overlap": 0,
            },
            "output_csv": {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "columns": ["filename", "patient_id", "split", *PHASE1_LABELS],
            },
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with self._patched_specs():
            source = load_source_metadata_provenance(
                self.source_path,
                cohort="development",
                official_train_val_manifest=self.train_list,
                official_test_manifest=self.test_list,
            )
            checked = load_manifest_metadata(
                metadata_path,
                manifest=manifest,
                cohort="development",
                source_metadata=source,
            )
        self.assertTrue(checked["verified"])

        metadata["output_csv"]["sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self._patched_specs(), self.assertRaisesRegex(
            ValueError, "output_csv SHA-256"
        ):
            load_manifest_metadata(
                metadata_path,
                manifest=manifest,
                cohort="development",
                source_metadata=source,
            )


class KADBenchmarkPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pack_hash = "1" * 64
        self.prompt_hash = prompt_set_sha256(
            (PHASE1_LABELS[0],),
            (PHASE1_QUERY_SPECS[PHASE1_LABELS[0]]["prompt"],),
        )
        self.active_target = PHASE1_LABELS[0]
        self.decision_path = self.root / "decision.json"
        self.decision_data = _complete_decision_artifact(
            self.root,
            query_pack_sha256=self.pack_hash,
        )
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_schema_three_policy_loads_and_calibrates_active_target(
        self,
    ) -> None:
        policy = load_decision_policy(
            self.decision_path,
            active_target=self.active_target,
            query_pack_sha256=self.pack_hash,
            prompt_hash=self.prompt_hash,
        )
        scores = np.asarray([[0.2]], dtype=float)
        calibrated = policy.calibrate(scores)

        expected_logit = np.log(scores[0, 0] / (1.0 - scores[0, 0]))
        expected = 1.0 / (
            1.0
            + np.exp(-(expected_logit * policy.slope + policy.intercept))
        )
        self.assertAlmostEqual(calibrated[0, 0], expected)

    def test_stale_schema_one_decision_is_rejected(self) -> None:
        self.decision_data["schema_version"] = 1
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "schema_version"):
            load_decision_policy(
                self.decision_path,
                active_target=self.active_target,
                query_pack_sha256=self.pack_hash,
                prompt_hash=self.prompt_hash,
            )

    def test_strict_loader_rederives_threshold_acceptance_and_role_evidence(
        self,
    ) -> None:
        acceptance_path = (
            "threshold_selection",
            "untouched_study_level_acceptance",
        )

        def acceptance(data: dict) -> dict:
            return data[acceptance_path[0]][acceptance_path[1]]

        mutations = {
            "threshold": lambda data: data["thresholds"].__setitem__(
                self.active_target,
                data["thresholds"][self.active_target] - 0.01,
            ),
            "acceptance_count": lambda data: acceptance(data)["counts"].__setitem__(
                "true_positives",
                acceptance(data)["counts"]["true_positives"] - 1,
            ),
            "acceptance_interval": lambda data: acceptance(data)[
                "confidence_intervals"
            ]["sensitivity"].__setitem__("lower", 0.99),
            "acceptance_pass": lambda data: acceptance(data)["passes"].__setitem__(
                "sensitivity_lower_bound",
                False,
            ),
            "study_outcome": lambda data: acceptance(data)["study_outcomes"][
                0
            ].__setitem__(
                "predicted_positive",
                not acceptance(data)["study_outcomes"][0]["predicted_positive"],
            ),
            "study_selection": lambda data: acceptance(data)["study_selection"][
                "positive"
            ][0].__setitem__("sample_id", "99999999_999.png"),
            "membership_hash": lambda data: data["patient_partition"]["partitions"][
                "acceptance"
            ].__setitem__("membership_sha256", "0" * 64),
            "fit_membership_hash": lambda data: data["calibrator"].__setitem__(
                "fit_membership_sha256",
                "0" * 64,
            ),
            "role_coverage": lambda data: data["patient_partition"][
                "partitions"
            ].pop("acceptance"),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                tampered = copy.deepcopy(self.decision_data)
                mutate(tampered)
                self.decision_path.write_text(
                    json.dumps(tampered), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    load_decision_policy(
                        self.decision_path,
                        active_target=self.active_target,
                        query_pack_sha256=self.pack_hash,
                        prompt_hash=self.prompt_hash,
                    )

    def test_query_pack_requires_pinned_checkpoint_and_query_set(self) -> None:
        pack_path = self.root / "pack.pt"
        pack_path.write_bytes(b"placeholder")
        spec = PHASE1_QUERY_SPECS[self.active_target]
        checked = {
            "labels": [self.active_target],
            "prompts": [spec["prompt"]],
            "source": {
                "checkpoint_sha256": (
                    "eb7223657220aa51eef43b2e155fd73c593eb5821fdcc8741782f6581ddfea76"
                ),
                "query_set": spec["query_set"],
            },
        }
        with (
            patch.object(benchmark_kad.torch, "load", return_value={}),
            patch.object(
                benchmark_kad,
                "preflight_kad512_query_pack",
                return_value=checked,
            ),
            patch.object(
                benchmark_kad,
                "kad512_query_pack_semantic_sha256",
                return_value="a" * 64,
            ),
        ):
            info = inspect_query_pack(
                pack_path,
                expected_labels=(self.active_target,),
                expected_prompts=(spec["prompt"],),
                expected_query_set=spec["query_set"],
            )
        self.assertEqual(
            info["source"]["query_set"], spec["query_set"]
        )

        bad = {**checked, "source": {**checked["source"], "checkpoint_sha256": "0" * 64}}
        with (
            patch.object(benchmark_kad.torch, "load", return_value={}),
            patch.object(
                benchmark_kad,
                "preflight_kad512_query_pack",
                return_value=bad,
            ),
            self.assertRaisesRegex(ValueError, "checkpoint SHA-256"),
        ):
            inspect_query_pack(
                pack_path,
                expected_labels=(self.active_target,),
                expected_prompts=(spec["prompt"],),
                expected_query_set=spec["query_set"],
            )

    def test_phase_one_semantic_sha_rejects_tampered_query_embeddings(self) -> None:
        pack_path = self.root / "tampered-pack.pt"
        pack_path.write_bytes(b"syntactically-valid-but-mutated-embeddings")
        spec = PHASE1_QUERY_SPECS[self.active_target]
        checked = {
            "labels": [self.active_target],
            "prompts": [spec["prompt"]],
            "source": {
                "checkpoint_sha256": (
                    "eb7223657220aa51eef43b2e155fd73c593eb5821fdcc8741782f6581ddfea76"
                ),
                "query_set": spec["query_set"],
            },
        }
        with (
            patch.object(benchmark_kad.torch, "load", return_value={}),
            patch.object(
                benchmark_kad,
                "preflight_kad512_query_pack",
                return_value=checked,
            ),
            patch.object(
                benchmark_kad,
                "kad512_query_pack_semantic_sha256",
                return_value="f" * 64,
            ),
            self.assertRaisesRegex(ValueError, "semantic SHA-256"),
        ):
            inspect_query_pack(
                pack_path,
                expected_labels=(self.active_target,),
                expected_prompts=(spec["prompt"],),
                expected_query_set=spec["query_set"],
                expected_semantic_sha256=spec["semantic_sha256"],
            )

    def test_incomplete_calibration_is_rejected(self) -> None:
        self.decision_data["calibration_complete"] = False
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "calibration_complete"):
            load_decision_policy(
                self.decision_path,
                active_target=self.active_target,
                query_pack_sha256=self.pack_hash,
                prompt_hash=self.prompt_hash,
            )

    def test_incomplete_acceptance_is_rejected(self) -> None:
        self.decision_data["acceptance_complete"] = False
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "acceptance_complete"):
            load_decision_policy(
                self.decision_path,
                active_target=self.active_target,
                query_pack_sha256=self.pack_hash,
                prompt_hash=self.prompt_hash,
            )

    def test_policy_rejects_a_different_or_multi_target_decision(self) -> None:
        self.decision_data["thresholds"]["Nodule_or_mass"] = 0.4
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "active target"):
            load_decision_policy(
                self.decision_path,
                active_target=self.active_target,
                query_pack_sha256=self.pack_hash,
                prompt_hash=self.prompt_hash,
            )

        self.decision_data["thresholds"].pop("Nodule_or_mass")
        self.decision_path.write_text(
            json.dumps(self.decision_data), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "active_target"):
            load_decision_policy(
                self.decision_path,
                active_target="Nodule_or_mass",
                query_pack_sha256=self.pack_hash,
                prompt_hash=self.prompt_hash,
            )

    def test_test_lock_binds_every_input_hash(self) -> None:
        decision_hash = hashlib.sha256(self.decision_path.read_bytes()).hexdigest()
        fields = {
            "query_pack_sha256": self.pack_hash,
            "test_manifest_sha256": "3" * 64,
            "prompt_set_sha256": self.prompt_hash,
            "decision_artifact_sha256": decision_hash,
            "image_set_sha256": "4" * 64,
            "image_provenance_sha256": "7" * 64,
            "runtime_contract_sha256": "9" * 64,
            "official_train_val_manifest_sha256": "5" * 64,
            "official_test_manifest_sha256": "6" * 64,
        }
        lock_path = self.root / "test.lock.json"
        lock_path.write_text(
            json.dumps(
                {
                    "artifact_type": "doctor_assistant.kad_phase1_test_lock",
                    "schema_version": 1,
                    "protocol_frozen": True,
                    "cohort": "test",
                    "active_target": self.active_target,
                    **fields,
                }
            ),
            encoding="utf-8",
        )

        checked = validate_test_lock(
            lock_path, active_target=self.active_target, **fields
        )
        self.assertRegex(checked["sha256"], r"^[0-9a-f]{64}$")

        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_test_lock(
                lock_path,
                active_target=self.active_target,
                **{**fields, "image_set_sha256": "8" * 64},
            )
        with self.assertRaisesRegex(ValueError, "active_target"):
            validate_test_lock(
                lock_path,
                active_target="Nodule_or_mass",
                **fields,
            )

    def test_self_declared_original_pixel_provenance_is_rejected(self) -> None:
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _image_provenance_payload(
                    filename="00000001_000.png",
                    output_sha256="1" * 64,
                    manifest_sha256="2" * 64,
                    width=1024,
                    height=1024,
                    original_nih_pixels=True,
                )
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "ORIGINAL NIH PIXEL EVIDENCE DISABLED",
        ):
            load_image_provenance(path)

    def test_verified_multi_source_original_pixel_provenance_is_accepted(
        self,
    ) -> None:
        image_path = self.root / "00000001_000.png"
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8), mode="L").save(image_path)
        image_hash = _file_sha256(image_path)
        manifest_hash = "2" * 64
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _original_pixel_provenance_payload(
                    filename=image_path.name,
                    output_sha256=image_hash,
                    manifest_sha256=manifest_hash,
                    width=9,
                    height=8,
                )
            ),
            encoding="utf-8",
        )

        provenance = load_image_provenance(path)
        self.assertEqual(provenance.schema_version, 2)
        self.assertTrue(provenance.original_nih_pixels)
        self.assertIsNotNone(provenance.archive_verification)

        manifest = SimpleNamespace(
            manifest_sha256=manifest_hash,
            cohort="development",
            sample_ids=(image_path.name,),
            image_paths=(image_path,),
            image_sha256=(image_hash,),
        )
        validate_image_provenance(provenance, manifest)

    def test_original_pixel_provenance_rejects_insufficient_source_agreement(
        self,
    ) -> None:
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _original_pixel_provenance_payload(
                    filename="00000001_000.png",
                    output_sha256="1" * 64,
                    manifest_sha256="2" * 64,
                    width=1024,
                    height=1024,
                    agreeing_source_count=1,
                )
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "agreeing_source_count"):
            load_image_provenance(path)

    def test_original_pixel_provenance_rejects_too_few_corroborating_sources(
        self,
    ) -> None:
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _original_pixel_provenance_payload(
                    filename="00000001_000.png",
                    output_sha256="1" * 64,
                    manifest_sha256="2" * 64,
                    width=1024,
                    height=1024,
                    corroborating_sources=0,
                )
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "corroborating_sources"):
            load_image_provenance(path)

    def test_original_pixel_provenance_rejects_missing_archive_verification(
        self,
    ) -> None:
        payload = _original_pixel_provenance_payload(
            filename="00000001_000.png",
            output_sha256="1" * 64,
            manifest_sha256="2" * 64,
            width=1024,
            height=1024,
        )
        del payload["archive_verification"]
        path = self.root / "pixels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "archive_verification"):
            load_image_provenance(path)

    def test_schema_two_rejects_a_non_original_declaration(self) -> None:
        payload = _original_pixel_provenance_payload(
            filename="00000001_000.png",
            output_sha256="1" * 64,
            manifest_sha256="2" * 64,
            width=1024,
            height=1024,
        )
        payload["original_nih_pixels"] = False
        path = self.root / "pixels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "schema-2 provenance is reserved"):
            load_image_provenance(path)

    def test_unsupported_schema_version_is_rejected(self) -> None:
        payload = _original_pixel_provenance_payload(
            filename="00000001_000.png",
            output_sha256="1" * 64,
            manifest_sha256="2" * 64,
            width=1024,
            height=1024,
        )
        payload["schema_version"] = 3
        path = self.root / "pixels.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "unsupported image provenance schema_version"
        ):
            load_image_provenance(path)

    def test_image_provenance_resolution_is_checked_against_every_file(self) -> None:
        image_path = self.root / "00000001_000.png"
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8), mode="L").save(image_path)
        image_hash = _file_sha256(image_path)
        manifest_hash = "2" * 64
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _image_provenance_payload(
                    filename=image_path.name,
                    output_sha256=image_hash,
                    manifest_sha256=manifest_hash,
                    width=9,
                    height=8,
                    original_nih_pixels=False,
                )
            ),
            encoding="utf-8",
        )

        provenance = load_image_provenance(path)
        manifest = SimpleNamespace(
            manifest_sha256=manifest_hash,
            cohort="development",
            sample_ids=(image_path.name,),
            image_paths=(image_path,),
            image_sha256=(image_hash,),
        )
        validate_image_provenance(provenance, manifest)

        path.write_text(
            json.dumps(
                _image_provenance_payload(
                    filename=image_path.name,
                    output_sha256=image_hash,
                    manifest_sha256=manifest_hash,
                    width=8,
                    height=8,
                    original_nih_pixels=False,
                )
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "does not match declared"):
            validate_image_provenance(load_image_provenance(path), manifest)

    def test_image_provenance_binds_each_selected_output_hash(self) -> None:
        image_path = self.root / "00000001_000.png"
        Image.fromarray(np.zeros((8, 9), dtype=np.uint8), mode="L").save(image_path)
        manifest_hash = "2" * 64
        path = self.root / "pixels.json"
        path.write_text(
            json.dumps(
                _image_provenance_payload(
                    filename=image_path.name,
                    output_sha256="9" * 64,
                    manifest_sha256=manifest_hash,
                    width=9,
                    height=8,
                    original_nih_pixels=False,
                )
            ),
            encoding="utf-8",
        )
        manifest = SimpleNamespace(
            manifest_sha256=manifest_hash,
            cohort="development",
            sample_ids=(image_path.name,),
            image_paths=(image_path,),
            image_sha256=(_file_sha256(image_path),),
        )

        with self.assertRaisesRegex(ValueError, "output SHA-256 mismatch"):
            validate_image_provenance(load_image_provenance(path), manifest)

    def test_test_run_fails_before_model_load_for_resized_pixels(self) -> None:
        path = self.root / "resized.json"
        path.write_text(
            json.dumps(
                _image_provenance_payload(
                    filename="00000001_000.png",
                    output_sha256="1" * 64,
                    manifest_sha256="2" * 64,
                    width=224,
                    height=224,
                    original_nih_pixels=False,
                    cohort="test",
                )
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "original_nih_pixels=true"):
            run_canonical(
                SimpleNamespace(cohort="test", image_provenance=path)
            )


class KADBenchmarkMetricsTests(unittest.TestCase):
    def test_partial_labels_are_excluded_for_isolated_endpoint(self) -> None:
        labels = np.asarray(
            [[1], [0], [-1], [1]],
            dtype=np.int8,
        )
        scores = np.asarray(
            [[0.9], [0.2], [0.4], [0.8]],
            dtype=float,
        )

        result = evaluate_expert_scorecard(
            scores,
            labels,
            patient_ids=["1", "2", "3", "4"],
            thresholds=None,
            active_target=PHASE1_LABELS[0],
            bootstrap_samples=10,
            seed=4,
        )

        self.assertEqual(
            result["per_label"]["Pneumothorax"]["images"],
            3,
        )
        self.assertEqual(
            result["per_label"]["Pneumothorax"]["non_adjudicated_images"],
            1,
        )
        self.assertNotIn(
            "threshold",
            result["per_label"]["Pneumothorax"],
        )
        self.assertEqual(
            result["metrics_scope"],
            "ranking_only_no_operating_thresholds",
        )

    def test_operating_metrics_are_emitted_for_active_target_only(self) -> None:
        labels = np.asarray(
            [[0], [1], [0], [1]],
            dtype=np.int8,
        )
        scores = np.asarray(
            [[0.1], [0.8], [0.2], [0.9]],
            dtype=float,
        )

        result = evaluate_expert_scorecard(
            scores,
            labels,
            patient_ids=["1", "2", "3", "4"],
            thresholds={"Nodule_or_mass": 0.6},
            active_target="Nodule_or_mass",
            bootstrap_samples=0,
            seed=7,
        )

        self.assertIn("threshold", result["per_label"]["Nodule_or_mass"])
        self.assertEqual(set(result["per_label"]), {"Nodule_or_mass"})
        self.assertEqual(result["exploratory_labels"], [])
        self.assertEqual(result["active_target"], "Nodule_or_mass")

    def test_deferred_development_scorecard_exposes_support_not_ranking(self) -> None:
        labels = np.asarray(
            [[1], [0], [1]],
            dtype=np.int8,
        )

        result = deferred_development_scorecard(
            labels,
            patient_ids=["1", "2", "3"],
            active_target="Pneumothorax",
        )

        self.assertTrue(result["analysis_deferred"])
        self.assertEqual(
            result["metrics_scope"],
            "development_analysis_deferred_until_patient_partition",
        )
        self.assertEqual(
            result["per_label"]["Pneumothorax"]["positives"], 2
        )
        self.assertIsNone(result["macro"]["auroc"])
        self.assertTrue(
            all(
                row["auroc"] is None and row["auprc"] is None
                for row in result["per_label"].values()
            )
        )


class KADBenchmarkArtifactWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "benchmark.json"
        self.raw_scores = np.asarray([[0.8, 0.1, 0.2], [0.2, 0.8, 0.7]])
        self.labels = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.int8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, *, overwrite: bool = False) -> Path:
        return benchmark_kad._write_outputs(
            self.output,
            {
                "artifact_type": "doctor_assistant.test_benchmark",
                "schema_version": 1,
            },
            raw_scores=self.raw_scores,
            labels=self.labels,
            class_names=PHASE1_LABELS,
            patient_ids=("00000001", "00000002"),
            sample_ids=("00000001_000.png", "00000002_000.png"),
            overwrite=overwrite,
        )

    def test_json_binds_the_exact_prediction_npz_sha256(self) -> None:
        predictions = self._write()
        payload = json.loads(self.output.read_text(encoding="utf-8"))

        self.assertEqual(payload["predictions"], str(predictions))
        self.assertEqual(payload["predictions_sha256"], _file_sha256(predictions))

    def test_existing_pair_refuses_implicit_overwrite_and_preserves_bytes(
        self,
    ) -> None:
        predictions = self._write()
        json_bytes = self.output.read_bytes()
        prediction_bytes = predictions.read_bytes()
        self.raw_scores[:] = 0.5

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            self._write()

        self.assertEqual(self.output.read_bytes(), json_bytes)
        self.assertEqual(predictions.read_bytes(), prediction_bytes)

        self._write(overwrite=True)
        self.assertNotEqual(predictions.read_bytes(), prediction_bytes)
        rewritten = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(rewritten["predictions_sha256"], _file_sha256(predictions))


class KADBenchmarkCLITests(unittest.TestCase):
    def test_active_target_is_wired_with_safe_default(self) -> None:
        parser = build_parser()

        default = parser.parse_args(["--query-pack", "pack.pt"])
        selected = parser.parse_args(
            [
                "--query-pack",
                "pack.pt",
                "--active-target",
                "Nodule_or_mass",
            ]
        )

        self.assertEqual(default.active_target, "Pneumothorax")
        self.assertEqual(selected.active_target, "Nodule_or_mass")
        self.assertFalse(default.overwrite)

    def test_provenance_and_deferred_analysis_flags_are_wired(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            [
                "--query-pack",
                "pack.pt",
                "--manifest-metadata",
                "labels.metadata.json",
                "--source-metadata-provenance",
                "nih_metadata.provenance.json",
                "--defer-development-ranking",
            ]
        )

        self.assertEqual(parsed.manifest_metadata, Path("labels.metadata.json"))
        self.assertEqual(
            parsed.source_metadata_provenance,
            Path("nih_metadata.provenance.json"),
        )
        self.assertTrue(parsed.defer_development_ranking)

    def test_runtime_contract_is_deterministic_and_binds_inference_settings(self) -> None:
        args = SimpleNamespace(
            device="cpu",
            no_amp=False,
            batch_size=4,
            seed=19,
        )
        with patch.object(benchmark_kad, "_git_commit", return_value="a" * 40):
            first, first_hash = build_runtime_contract(args)
            second, second_hash = build_runtime_contract(args)

        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertRegex(first_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(first["inference"]["device_resolved"], "cpu")
        self.assertFalse(first["inference"]["amp_enabled"])
        self.assertEqual(first["schema_version"], 3)
        self.assertEqual(first["hardware"]["accelerator"], "cpu")
        self.assertIn("processor", first["hardware"])
        self.assertEqual(first["versions"]["cuda_driver"], "not-applicable")
        self.assertEqual(
            set(first["code_sha256"]),
            {"experts/kad.py", "scripts/benchmark_kad.py"},
        )


if __name__ == "__main__":
    unittest.main()
