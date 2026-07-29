from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.benchmark_kad import load_decision_policy
from scripts.calibrate_kad_phase1 import (
    _canonical_json_sha256,
    build_parser,
    build_decision_artifact,
    clustered_ranking_bootstrap,
    load_development_inputs,
    main as calibration_main,
    prompt_set_sha256,
    study_level_acceptance,
    wilson_interval,
    write_decision_artifact,
)
from scripts.export_kad_query_pack import PHASE1_LABELS, PHASE1_QUERY_SPECS


class KADPhase1CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "development.json"
        self.predictions = self.root / "development.predictions.npz"
        self.query_hash = "a" * 64
        self.manifest_hash = "b" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_pair(
        self,
        *,
        patients: int = 120,
        images_per_patient: int = 1,
        constant_scores: bool = False,
        deferred_analysis: bool = False,
        original_nih_pixels: bool = True,
    ) -> dict:
        sample_ids: list[str] = []
        patient_ids: list[str] = []
        image_hashes: list[str] = []
        labels: list[list[int]] = []
        scores: list[list[float]] = []
        for patient_number in range(1, patients + 1):
            patient_id = f"{patient_number:08d}"
            target = patient_number % 2
            for image_number in range(images_per_patient):
                sample_id = f"{patient_id}_{image_number:03d}.png"
                sample_ids.append(sample_id)
                patient_ids.append(patient_id)
                image_hashes.append(
                    hashlib.sha256(f"image:{sample_id}".encode()).hexdigest()
                )
                labels.append([target])
                if constant_scores:
                    active_score = 0.5
                else:
                    active_score = (
                        0.78 + 0.01 * (image_number % 2)
                        if target
                        else 0.22 - 0.01 * (image_number % 2)
                    )
                scores.append([active_score])

        raw_scores = np.asarray(scores, dtype=np.float32)
        truth = np.asarray(labels, dtype=np.int8)
        np.savez_compressed(
            self.predictions,
            raw_scores=raw_scores,
            labels=truth,
            class_names=np.asarray([PHASE1_LABELS[0]]),
            patient_ids=np.asarray(patient_ids),
            sample_ids=np.asarray(sample_ids),
            image_sha256=np.asarray(image_hashes),
        )
        predictions_hash = hashlib.sha256(self.predictions.read_bytes()).hexdigest()
        image_set_hash = _canonical_json_sha256(
            [
                {
                    "sample_id": sample_id,
                    "image_path": sample_id,
                    "sha256": image_hash,
                }
                for sample_id, image_hash in zip(sample_ids, image_hashes)
            ]
        )
        positives = int((truth[:, 0] == 1).sum())
        negatives = int((truth[:, 0] == 0).sum())
        runtime = {
            "schema_version": 3,
            "git_commit": "1" * 40,
            "git_worktree": {
                "clean": True,
                "untracked_files_checked": True,
            },
            "versions": {
                "python": "test",
                "torch": "test",
                "torchvision": "test",
                "numpy": "test",
                "Pillow": "test",
                "torch_cuda_runtime": "not-applicable",
                "cudnn_runtime": "not-applicable",
                "cuda_driver": "not-applicable",
            },
            "hardware": {
                "accelerator": "cpu",
                "machine": "test",
                "processor": "test",
            },
            "inference": {
                "device_requested": "cpu",
                "device_resolved": "cpu",
                "amp_requested": False,
                "amp_enabled": False,
                "batch_size": 1,
                "seed": 42,
            },
            "determinism": {
                "python_random_seeded": True,
                "numpy_seeded": True,
                "torch_cpu_seeded": True,
                "torch_cuda_seeded": False,
                "deterministic_algorithms_enabled": True,
                "deterministic_algorithms_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "reproducibility_scope": (
                    "deterministic_algorithms_on_identical_hardware_software_and_inputs;"
                    "cross_hardware_bitwise_identity_not_claimed"
                ),
            },
        }
        artifact = {
            "artifact_type": "doctor_assistant.kad_phase1_benchmark",
            "schema_version": 1,
            "purpose": "expert_development_evaluation",
            "cohort": "development",
            "active_target": PHASE1_LABELS[0],
            "candidate_under_decision": PHASE1_LABELS[0],
            "exploratory_labels": [],
            "smoke_only": False,
            "official_manifest_reconciled": True,
            "evidence_status": (
                "development_only_do_not_report_as_test_performance"
                if original_nih_pixels
                else "resized_mirror_candidate_selection_only"
            ),
            "model": {
                "sha256": self.query_hash,
                "semantic_sha256": PHASE1_QUERY_SPECS[PHASE1_LABELS[0]][
                    "semantic_sha256"
                ],
                "labels": [PHASE1_LABELS[0]],
                "prompts": [
                    PHASE1_QUERY_SPECS[PHASE1_LABELS[0]]["prompt"]
                ],
                "prompt_set_sha256": prompt_set_sha256(
                    (PHASE1_LABELS[0],),
                    (PHASE1_QUERY_SPECS[PHASE1_LABELS[0]]["prompt"],),
                ),
            },
            "runtime": runtime,
            "runtime_contract_sha256": _canonical_json_sha256(runtime),
            "inputs": {
                "manifest_sha256": self.manifest_hash,
                "image_set_sha256": image_set_hash,
                "images": len(sample_ids),
                "patients": patients,
                "official_train_val_manifest_sha256": "c" * 64,
                "official_test_manifest_sha256": "d" * 64,
                "image_provenance": {
                    "sha256": "e" * 64,
                    "original_nih_pixels": original_nih_pixels,
                },
            },
            "decision": {
                "status": "not_supplied",
                "operating_point_metrics": "disabled",
            },
            "test_lock": None,
            "scorecard": {
                "images": len(sample_ids),
                "patients": patients,
                "metrics_scope": (
                    "development_analysis_deferred_until_patient_partition"
                    if deferred_analysis
                    else "ranking_only_no_operating_thresholds"
                ),
                "analysis_deferred": deferred_analysis,
                "active_target": PHASE1_LABELS[0],
                "per_label": {
                    PHASE1_LABELS[0]: {
                        "images": len(sample_ids),
                        "positives": positives,
                        "negatives": negatives,
                    }
                },
            },
            "predictions": str(self.predictions),
            "predictions_sha256": predictions_hash,
        }
        self.benchmark.write_text(json.dumps(artifact), encoding="utf-8")
        return artifact

    def test_complete_decision_is_four_way_patient_disjoint_and_loadable(self) -> None:
        self._write_pair(
            patients=300,
            images_per_patient=2,
            deferred_analysis=True,
        )
        inputs = load_development_inputs(
            self.benchmark,
            self.predictions,
            active_target=PHASE1_LABELS[0],
            expected_query_pack_sha256=self.query_hash,
            require_deferred_analysis=True,
        )

        artifact = build_decision_artifact(
            inputs,
            min_calibration_positives=10,
            min_calibration_negatives=20,
            min_threshold_positives=10,
            min_threshold_negatives=20,
            seed=20250729,
            bootstrap_samples=1000,
        )

        self.assertTrue(artifact["calibration_complete"])
        self.assertTrue(artifact["acceptance_complete"])
        self.assertTrue(artifact["thresholds_complete"])
        self.assertEqual(
            artifact["requirements"]["minimum_acceptance_positive_patients"],
            22,
        )
        self.assertEqual(
            set(artifact["calibrator"]["parameters"]), {PHASE1_LABELS[0]}
        )
        self.assertEqual(set(artifact["calibrator"]["metrics"]), {PHASE1_LABELS[0]})
        self.assertEqual(set(artifact["thresholds"]), {PHASE1_LABELS[0]})
        self.assertEqual(
            artifact["threshold_selection"]["active_target"], PHASE1_LABELS[0]
        )
        partition = artifact["patient_partition"]
        self.assertEqual(
            set(partition["pairwise_patient_overlap"].values()), {0}
        )
        records = partition["partitions"]
        all_patients: set[str] = set()
        all_samples: set[str] = set()
        for role in (
            "model_selection",
            "calibration",
            "threshold_selection",
            "acceptance",
        ):
            role_patients = set(records[role]["patient_ids"])
            role_samples = set(records[role]["sample_ids"])
            self.assertFalse(all_patients & role_patients)
            self.assertFalse(all_samples & role_samples)
            all_patients |= role_patients
            all_samples |= role_samples
            self.assertRegex(records[role]["membership_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(all_patients), 300)
        self.assertEqual(len(all_samples), 600)
        self.assertIsNotNone(artifact["model_selection_ranking"]["auroc"])
        ranking_intervals = artifact["model_selection_ranking"][
            "confidence_intervals"
        ]
        self.assertEqual(ranking_intervals["requested_replicates"], 1000)
        self.assertEqual(ranking_intervals["successful_replicates"]["auroc"], 1000)
        self.assertEqual(
            ranking_intervals["method"],
            "patient_clustered_percentile_bootstrap",
        )
        self.assertNotIn(
            "model_selection", artifact["calibrator"]["metrics"][PHASE1_LABELS[0]]
        )
        self.assertNotIn("auroc", json.dumps(artifact["calibrator"]["metrics"]))
        self.assertEqual(
            artifact["model_selection_ranking"]["membership_sha256"],
            records["model_selection"]["membership_sha256"],
        )
        self.assertEqual(
            artifact["calibrator"]["fit_membership_sha256"],
            records["calibration"]["membership_sha256"],
        )
        self.assertEqual(
            artifact["threshold_selection"]["fit_membership_sha256"],
            records["threshold_selection"]["membership_sha256"],
        )
        acceptance = artifact["threshold_selection"][
            "untouched_study_level_acceptance"
        ]
        self.assertEqual(acceptance["unit"], "study")
        self.assertEqual(
            acceptance["study_selection"]["method"],
            "minimum_sha256_independent_of_scores_v1",
        )
        self.assertEqual(
            acceptance["study_selection"]["hash_domain"],
            "kad-phase1-acceptance-study-v1",
        )
        self.assertEqual(
            acceptance["evaluation_membership_sha256"],
            records["acceptance"]["membership_sha256"],
        )

        output = write_decision_artifact(self.root / "decision.json", artifact)
        policy = load_decision_policy(
            output,
            active_target=PHASE1_LABELS[0],
            query_pack_sha256=self.query_hash,
            prompt_hash=prompt_set_sha256(
                (PHASE1_LABELS[0],),
                (PHASE1_QUERY_SPECS[PHASE1_LABELS[0]]["prompt"],),
            ),
        )
        self.assertEqual(policy.active_target, PHASE1_LABELS[0])

    def test_partition_is_deterministic(self) -> None:
        self._write_pair(patients=100)
        inputs = load_development_inputs(self.benchmark, self.predictions)

        first = build_decision_artifact(
            inputs,
            min_calibration_positives=5,
            min_calibration_negatives=5,
            min_threshold_positives=5,
            min_threshold_negatives=5,
            seed=7,
            bootstrap_samples=50,
        )
        second = build_decision_artifact(
            inputs,
            min_calibration_positives=5,
            min_calibration_negatives=5,
            min_threshold_positives=5,
            min_threshold_negatives=5,
            seed=7,
            bootstrap_samples=50,
        )

        self.assertEqual(first["patient_partition"], second["patient_partition"])
        self.assertEqual(first["thresholds"], second["thresholds"])
        self.assertEqual(
            first["model_selection_ranking"]["confidence_intervals"],
            second["model_selection_ranking"]["confidence_intervals"],
        )

    def test_patient_clustered_ranking_bootstrap_is_deterministic(self) -> None:
        patients = np.asarray(
            ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"]
        )
        truth = np.asarray([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        scores = np.asarray(
            [0.9, 0.8, 0.3, 0.2, 0.7, 0.4, 0.6, 0.1, 0.55, 0.45, 0.5, 0.35]
        )

        first = clustered_ranking_bootstrap(
            patients, truth, scores, samples=100, seed=44
        )
        second = clustered_ranking_bootstrap(
            patients, truth, scores, samples=100, seed=44
        )

        self.assertEqual(first, second)
        self.assertEqual(first["patients_resampled_per_replicate"], 6)
        self.assertGreater(first["successful_replicates"]["auroc"], 0)
        self.assertLessEqual(first["successful_replicates"]["auroc"], 100)
        self.assertIsNotNone(first["intervals"]["auroc"])
        self.assertIsNotNone(first["intervals"]["auprc"])

    def test_cli_defaults_to_one_thousand_patient_bootstraps(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--benchmark",
                "development.json",
                "--predictions",
                "development.predictions.npz",
                "--output",
                "decision.json",
            ]
        )

        self.assertEqual(parsed.bootstrap_samples, 1000)
        self.assertEqual(parsed.model_selection_fraction, 0.30)
        self.assertEqual(parsed.calibration_fraction, 0.20)
        self.assertEqual(parsed.threshold_fraction, 0.20)
        self.assertEqual(parsed.acceptance_fraction, 0.30)
        self.assertEqual(parsed.seed, 20250729)
        self.assertEqual(parsed.split_attempts, 512)
        self.assertFalse(parsed.overwrite)

    def test_canonical_cli_rejects_role_recycling_configuration(self) -> None:
        base = [
            "--benchmark",
            "missing-development.json",
            "--predictions",
            "missing-development.predictions.npz",
            "--output",
            "missing-decision.json",
            "--require-deferred-analysis",
        ]
        for changed in (
            ["--model-selection-fraction", "0.25"],
            ["--seed", "7"],
            ["--split-attempts", "128"],
        ):
            with self.subTest(changed=changed), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    calibration_main([*base, *changed])

    def test_canonical_cli_rejects_weakened_support_or_gate(self) -> None:
        base = [
            "--benchmark",
            "missing-development.json",
            "--predictions",
            "missing-development.predictions.npz",
            "--output",
            "missing-decision.json",
            "--require-deferred-analysis",
        ]
        for changed in (
            ["--min-calibration-positives", "1"],
            ["--min-threshold-negatives", "1"],
            ["--sensitivity-target", "0.84"],
            ["--specificity-floor", "0.59"],
            ["--bootstrap-samples", "100"],
        ):
            with self.subTest(changed=changed), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    calibration_main([*base, *changed])

    def test_model_selection_scores_never_affect_calibrator_or_threshold(self) -> None:
        self._write_pair(patients=120)
        inputs = load_development_inputs(self.benchmark, self.predictions)
        arguments = {
            "min_calibration_positives": 5,
            "min_calibration_negatives": 5,
            "min_threshold_positives": 5,
            "min_threshold_negatives": 5,
            "seed": 12,
            "bootstrap_samples": 50,
        }
        original = build_decision_artifact(inputs, **arguments)
        model_selection_ids = original["patient_partition"]["partitions"][
            "model_selection"
        ]["sample_ids"]
        changed_scores = inputs.raw_scores.copy()
        changed_rows = np.isin(inputs.sample_ids, model_selection_ids)
        changed_scores[changed_rows, 0] = 1.0 - changed_scores[changed_rows, 0]

        changed = build_decision_artifact(
            replace(inputs, raw_scores=changed_scores), **arguments
        )

        self.assertEqual(
            original["calibrator"]["parameters"],
            changed["calibrator"]["parameters"],
        )
        self.assertEqual(
            original["candidate_thresholds"], changed["candidate_thresholds"]
        )
        self.assertNotEqual(
            original["model_selection_ranking"]["auroc"],
            changed["model_selection_ranking"]["auroc"],
        )

    def test_acceptance_scores_never_affect_calibrator_or_candidate_threshold(
        self,
    ) -> None:
        self._write_pair(
            patients=320,
            images_per_patient=2,
            deferred_analysis=True,
        )
        inputs = load_development_inputs(self.benchmark, self.predictions)
        arguments = {
            "min_calibration_positives": 5,
            "min_calibration_negatives": 5,
            "min_threshold_positives": 5,
            "min_threshold_negatives": 5,
            "seed": 84,
            "bootstrap_samples": 50,
        }
        original = build_decision_artifact(inputs, **arguments)
        acceptance_ids = original["patient_partition"]["partitions"]["acceptance"][
            "sample_ids"
        ]
        changed_scores = inputs.raw_scores.copy()
        changed_rows = np.isin(inputs.sample_ids, acceptance_ids)
        changed_scores[changed_rows, 0] = 1.0 - changed_scores[changed_rows, 0]

        changed = build_decision_artifact(
            replace(inputs, raw_scores=changed_scores), **arguments
        )

        self.assertEqual(
            original["calibrator"]["parameters"],
            changed["calibrator"]["parameters"],
        )
        self.assertEqual(
            original["candidate_thresholds"],
            changed["candidate_thresholds"],
        )
        self.assertEqual(
            original["threshold_selection"]["result"],
            changed["threshold_selection"]["result"],
        )
        self.assertEqual(
            original["threshold_selection"]["untouched_study_level_acceptance"][
                "study_selection"
            ],
            changed["threshold_selection"]["untouched_study_level_acceptance"][
                "study_selection"
            ],
        )
        self.assertNotEqual(
            original["threshold_selection"]["untouched_study_level_acceptance"][
                "point_estimates"
            ],
            changed["threshold_selection"]["untouched_study_level_acceptance"][
                "point_estimates"
            ],
        )

    def test_threshold_scores_do_not_refit_calibrator_and_failed_threshold_never_reads_acceptance(
        self,
    ) -> None:
        self._write_pair(
            patients=320,
            images_per_patient=2,
            deferred_analysis=True,
        )
        inputs = load_development_inputs(self.benchmark, self.predictions)
        arguments = {
            "min_calibration_positives": 5,
            "min_calibration_negatives": 5,
            "min_threshold_positives": 5,
            "min_threshold_negatives": 5,
            "seed": 24,
            "bootstrap_samples": 50,
        }
        original = build_decision_artifact(inputs, **arguments)
        threshold_ids = original["patient_partition"]["partitions"][
            "threshold_selection"
        ]["sample_ids"]
        changed_scores = inputs.raw_scores.copy()
        threshold_rows = np.isin(inputs.sample_ids, threshold_ids)
        changed_scores[threshold_rows, 0] = 1.0 - changed_scores[threshold_rows, 0]

        with patch(
            "scripts.calibrate_kad_phase1.study_level_acceptance",
            side_effect=AssertionError("acceptance must remain untouched"),
        ) as acceptance_mock:
            changed = build_decision_artifact(
                replace(inputs, raw_scores=changed_scores), **arguments
            )

        acceptance_mock.assert_not_called()
        self.assertEqual(
            original["calibrator"]["parameters"],
            changed["calibrator"]["parameters"],
        )
        self.assertFalse(changed["threshold_selection"]["result"]["meets_constraints"])
        self.assertIsNone(
            changed["threshold_selection"]["untouched_study_level_acceptance"]
        )
        self.assertFalse(changed["acceptance_complete"])
        self.assertFalse(changed["thresholds_complete"])

    def test_test_cohort_is_rejected_before_calibration(self) -> None:
        artifact = self._write_pair()
        artifact["purpose"] = "locked_expert_test_evaluation"
        artifact["cohort"] = "test"
        self.benchmark.write_text(json.dumps(artifact), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "development_evaluation"):
            load_development_inputs(self.benchmark, self.predictions)

    def test_deferred_development_analysis_is_accepted_and_can_be_required(self) -> None:
        self._write_pair(patients=60, deferred_analysis=True)

        inputs = load_development_inputs(
            self.benchmark,
            self.predictions,
            require_deferred_analysis=True,
        )

        self.assertTrue(inputs.analysis_deferred)
        self.assertEqual(
            inputs.benchmark_metrics_scope,
            "development_analysis_deferred_until_patient_partition",
        )

    def test_canonical_mode_rejects_prepartition_full_cohort_ranking(self) -> None:
        self._write_pair(patients=60)

        with self.assertRaisesRegex(ValueError, "requires development analysis"):
            load_development_inputs(
                self.benchmark,
                self.predictions,
                require_deferred_analysis=True,
            )

    def test_npz_tampering_is_rejected_by_bound_sha256_before_use(self) -> None:
        self._write_pair(patients=30)
        with np.load(self.predictions, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["image_sha256"][0] = "f" * 64
        np.savez_compressed(self.predictions, **arrays)

        with self.assertRaisesRegex(ValueError, "predictions_sha256"):
            load_development_inputs(self.benchmark, self.predictions)

    def test_rebound_npz_still_requires_exact_image_identity(self) -> None:
        artifact = self._write_pair(patients=30)
        with np.load(self.predictions, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["image_sha256"][0] = "f" * 64
        np.savez_compressed(self.predictions, **arrays)
        artifact["predictions_sha256"] = hashlib.sha256(
            self.predictions.read_bytes()
        ).hexdigest()
        self.benchmark.write_text(json.dumps(artifact), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "image_set_sha256"):
            load_development_inputs(self.benchmark, self.predictions)

    def test_repeated_studies_cannot_substitute_for_acceptance_patient_support(
        self,
    ) -> None:
        positive_patients = [f"{index:08d}" for index in range(1, 22)]
        negative_patients = [f"{index:08d}" for index in range(100, 120)]
        patient_ids = np.asarray(
            [
                patient_id
                for patient_id in positive_patients
                for _ in range(10)
            ]
            + negative_patients
        )
        sample_ids = np.asarray(
            [
                f"{patient_id}_{study_number:03d}.png"
                for patient_id in positive_patients
                for study_number in range(10)
            ]
            + [f"{patient_id}_000.png" for patient_id in negative_patients]
        )
        labels = np.asarray(
            [1] * (len(positive_patients) * 10) + [0] * len(negative_patients),
            dtype=np.int8,
        )
        acceptance = study_level_acceptance(
            patient_ids,
            sample_ids,
            labels,
            np.where(labels == 1, 0.9, 0.1),
            threshold=0.5,
            sensitivity_target=0.85,
            specificity_floor=0.60,
            seed=18,
            evaluation_membership_sha256="a" * 64,
        )

        self.assertEqual(acceptance["counts"]["positive_studies"], 210)
        self.assertEqual(acceptance["counts"]["positive_patients"], 21)
        self.assertEqual(acceptance["counts"]["selected_positive_studies"], 21)
        self.assertFalse(acceptance["passes"]["support"])
        self.assertFalse(acceptance["complete"])

    def test_two_sided_wilson_feasibility_requires_at_least_22_perfect_positives(
        self,
    ) -> None:
        self.assertLess(wilson_interval(21, 21)[0], 0.85)
        self.assertGreaterEqual(wilson_interval(22, 22)[0], 0.85)

        def perfect_acceptance(positive_patients: int) -> dict:
            truth = np.asarray(
                [1] * positive_patients + [0] * 20,
                dtype=np.int8,
            )
            patient_ids = np.asarray(
                [
                    f"{patient_number:08d}"
                    for patient_number in range(1, len(truth) + 1)
                ]
            )
            sample_ids = np.asarray(
                [f"{patient_id}_000.png" for patient_id in patient_ids]
            )
            probabilities = np.where(truth == 1, 0.9, 0.1)
            return study_level_acceptance(
                patient_ids,
                sample_ids,
                truth,
                probabilities,
                threshold=0.5,
                sensitivity_target=0.85,
                specificity_floor=0.60,
                seed=17,
                evaluation_membership_sha256="a" * 64,
            )

        twenty_one = perfect_acceptance(21)
        twenty_two = perfect_acceptance(22)
        self.assertFalse(twenty_one["passes"]["support"])
        self.assertFalse(twenty_one["passes"]["sensitivity_lower_bound"])
        self.assertFalse(twenty_one["complete"])
        self.assertTrue(twenty_two["passes"]["support"])
        self.assertTrue(twenty_two["passes"]["sensitivity_lower_bound"])
        self.assertTrue(twenty_two["complete"])

    def test_decision_writer_refuses_implicit_overwrite_and_preserves_bytes(
        self,
    ) -> None:
        output = self.root / "immutable-decision.json"
        original = {"artifact_type": "test", "value": 1}
        replacement = {"artifact_type": "test", "value": 2}
        write_decision_artifact(output, original)
        original_bytes = output.read_bytes()

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            write_decision_artifact(output, replacement)

        self.assertEqual(output.read_bytes(), original_bytes)
        write_decision_artifact(output, replacement, overwrite=True)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), replacement)

    def test_under_supported_run_writes_non_deployable_diagnostic(self) -> None:
        self._write_pair(patients=30)
        inputs = load_development_inputs(self.benchmark, self.predictions)

        artifact = build_decision_artifact(
            inputs,
            min_calibration_positives=20,
            min_calibration_negatives=20,
            min_threshold_positives=20,
            min_threshold_negatives=20,
            bootstrap_samples=50,
        )

        self.assertFalse(artifact["calibration_complete"])
        self.assertFalse(artifact["thresholds_complete"])
        self.assertEqual(artifact["thresholds"], {})
        self.assertFalse(artifact["diagnostics"]["deployable"])
        self.assertIn(
            "four_way_patient_split_has_insufficient_class_support",
            artifact["diagnostics"]["reasons"],
        )
        write_decision_artifact(self.root / "diagnostic.json", artifact)

    def test_nonpositive_platt_slope_is_not_exported(self) -> None:
        self._write_pair(patients=120, constant_scores=True)
        inputs = load_development_inputs(self.benchmark, self.predictions)

        artifact = build_decision_artifact(
            inputs,
            min_calibration_positives=5,
            min_calibration_negatives=5,
            min_threshold_positives=5,
            min_threshold_negatives=5,
            bootstrap_samples=50,
        )

        self.assertFalse(artifact["calibration_complete"])
        self.assertFalse(artifact["thresholds_complete"])
        self.assertEqual(artifact["thresholds"], {})
        self.assertIn(
            "platt_slope_is_not_positive", artifact["diagnostics"]["reasons"]
        )

    def test_resized_mirror_pixels_can_never_produce_acceptance_evidence(
        self,
    ) -> None:
        self._write_pair(
            patients=320,
            deferred_analysis=True,
            original_nih_pixels=False,
        )
        inputs = load_development_inputs(self.benchmark, self.predictions)

        artifact = build_decision_artifact(
            inputs,
            min_calibration_positives=5,
            min_calibration_negatives=5,
            min_threshold_positives=5,
            min_threshold_negatives=5,
            seed=92,
            bootstrap_samples=50,
        )

        self.assertTrue(artifact["calibration_complete"])
        self.assertFalse(artifact["acceptance_complete"])
        self.assertFalse(artifact["thresholds_complete"])
        self.assertIsNone(
            artifact["threshold_selection"]["untouched_study_level_acceptance"]
        )
        self.assertFalse(
            artifact["development_pixel_evidence"]["acceptance_eligible"]
        )
        self.assertIn(
            "non_original_development_pixels_not_acceptance_eligible",
            artifact["diagnostics"]["reasons"],
        )

    def test_development_benchmark_requires_clean_reproducible_runtime(self) -> None:
        artifact = self._write_pair(patients=60, deferred_analysis=True)
        artifact["runtime"]["git_worktree"]["clean"] = False
        artifact["runtime_contract_sha256"] = _canonical_json_sha256(
            artifact["runtime"]
        )
        self.benchmark.write_text(json.dumps(artifact), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "clean Git worktree"):
            load_development_inputs(self.benchmark, self.predictions)

    def test_acceptance_uses_prespecified_patient_weighted_study_units(
        self,
    ) -> None:
        self._write_pair(
            patients=320,
            images_per_patient=3,
            deferred_analysis=True,
        )
        inputs = load_development_inputs(self.benchmark, self.predictions)

        artifact = build_decision_artifact(
            inputs,
            min_calibration_positives=5,
            min_calibration_negatives=5,
            min_threshold_positives=5,
            min_threshold_negatives=5,
            seed=33,
            bootstrap_samples=50,
        )

        selection = artifact["threshold_selection"]
        acceptance = selection["untouched_study_level_acceptance"]
        self.assertTrue(selection["result"]["meets_constraints"])
        self.assertEqual(acceptance["unit"], "study")
        self.assertEqual(
            acceptance["counts"]["studies"],
            artifact["patient_partition"]["partitions"]["acceptance"]["images"],
        )
        self.assertEqual(
            acceptance["counts"]["selected_positive_studies"],
            acceptance["counts"]["true_positives"]
            + acceptance["counts"]["false_negatives"],
        )
        self.assertEqual(
            acceptance["counts"]["positive_studies"],
            acceptance["all_study_diagnostics"]["counts"]["positive_studies"],
        )
        self.assertEqual(
            acceptance["counts"]["selected_positive_studies"],
            acceptance["counts"]["positive_patients"],
        )
        self.assertEqual(
            acceptance["counts"]["selected_negative_studies"],
            acceptance["counts"]["negative_patients"],
        )
        self.assertEqual(
            acceptance["interval"],
            "two_sided_wilson_score",
        )
        self.assertEqual(
            acceptance["all_study_diagnostics"]["gate_role"],
            "diagnostic_only_not_used_for_acceptance",
        )
        self.assertTrue(acceptance["complete"])
        self.assertTrue(artifact["acceptance_complete"])
        self.assertTrue(artifact["thresholds_complete"])

    def test_acceptance_targets_cannot_be_relaxed_below_protocol(self) -> None:
        self._write_pair(patients=120)
        inputs = load_development_inputs(self.benchmark, self.predictions)

        with self.assertRaisesRegex(ValueError, "cannot be lower than 0.85"):
            build_decision_artifact(inputs, sensitivity_target=0.84)
        with self.assertRaisesRegex(ValueError, "cannot be lower than 0.6"):
            build_decision_artifact(inputs, specificity_floor=0.59)

    def test_expected_artifact_hash_mismatch_is_rejected(self) -> None:
        self._write_pair(patients=30)

        with self.assertRaisesRegex(ValueError, "expected_predictions_sha256"):
            load_development_inputs(
                self.benchmark,
                self.predictions,
                expected_predictions_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
