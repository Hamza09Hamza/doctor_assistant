"""Fit one leakage-resistant KAD phase-1 decision endpoint.

This command consumes the *development* JSON/NPZ pair written by
``scripts/benchmark_kad.py``.  It fails closed on smoke/test artifacts, validates
the cryptographic and row-level bindings between the two files, then assigns
patients once to four disjoint roles:

* ``model_selection``: ranking metrics only; never used for fitting or selection.
* ``calibration``: fit one Platt calibrator on clipped raw-score logits.
* ``threshold_selection``: select one candidate operating threshold after calibration.
* ``acceptance``: evaluate that frozen candidate once; never used to tune it.

Only ``--active-target`` receives a calibrator or threshold.  A valid but
under-supported/infeasible run still writes a diagnostic artifact, with completion
flags false and no deployable threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from statistics import NormalDist
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from data.nih_protocol import is_nih_image_filename, nih_patient_id
from evaluation.thresholds import ThresholdSelection, select_per_label_thresholds
from scripts.export_kad_query_pack import (
    PHASE1_LABELS,
    PHASE1_PROMPTS,
    PHASE1_QUERY_SPECS,
    sha256_file,
)


_BENCHMARK_ARTIFACT_TYPE = "doctor_assistant.kad_phase1_benchmark"
_DECISION_ARTIFACT_TYPE = "doctor_assistant.kad_phase1_decision"
_DECISION_SCHEMA_VERSION = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLIP_EPSILON = 1e-7
_ACCEPTANCE_CONFIDENCE = 0.95
_MIN_SENSITIVITY_TARGET = 0.85
_MIN_SPECIFICITY_FLOOR = 0.60
_MIN_ACCEPTANCE_POSITIVE_PATIENTS = 22
_MIN_ACCEPTANCE_NEGATIVE_PATIENTS = 20
_PROTOCOL_PARTITION_SEED = 20250729
_PROTOCOL_SPLIT_ATTEMPTS = 512
_PROTOCOL_BOOTSTRAP_SAMPLES = 1000
_PROTOCOL_ROLE_FRACTIONS = {
    "model_selection": 0.30,
    "calibration": 0.20,
    "threshold_selection": 0.20,
    "acceptance": 0.30,
}
_PROTOCOL_MINIMUM_SUPPORT = {
    "calibration_positives": 10,
    "calibration_negatives": 20,
    "threshold_positives": 10,
    "threshold_negatives": 20,
}
_NPZ_FIELDS = {
    "raw_scores",
    "labels",
    "class_names",
    "patient_ids",
    "sample_ids",
    "image_sha256",
}


@dataclass(frozen=True)
class DevelopmentInputs:
    benchmark_path: Path
    predictions_path: Path
    benchmark_sha256: str
    predictions_sha256: str
    query_pack_sha256: str
    prompt_set_sha256: str
    manifest_sha256: str
    active_target: str
    benchmark_metrics_scope: str
    analysis_deferred: bool
    original_nih_pixels: bool
    benchmark_evidence_status: str
    raw_scores: np.ndarray
    labels: np.ndarray
    patient_ids: np.ndarray
    sample_ids: np.ndarray
    image_sha256: np.ndarray


@dataclass(frozen=True)
class PatientPartitions:
    model_selection: np.ndarray
    calibration: np.ndarray
    threshold_selection: np.ndarray
    acceptance: np.ndarray
    selected_attempt: int
    support_feasible: bool


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _calibration_runtime_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "scipy": _distribution_version("scipy"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }


def prompt_set_sha256(
    labels: Sequence[str] = PHASE1_LABELS,
    prompts: Sequence[str] = PHASE1_PROMPTS,
) -> str:
    return _canonical_json_sha256(
        {"labels": list(labels), "prompts": list(prompts)}
    )


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_exact_int(value: Any, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{field} must be {expected}, got {value!r}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse benchmark JSON {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("benchmark JSON must contain an object")
    return parsed


def _string_array(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"predictions {field} must be a one-dimensional string array")
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    return np.asarray(array, dtype=str)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            fields = set(archive.files)
            if fields != _NPZ_FIELDS:
                missing = sorted(_NPZ_FIELDS - fields)
                extra = sorted(fields - _NPZ_FIELDS)
                raise ValueError(
                    "predictions NPZ fields do not match the canonical raw "
                    f"development schema; missing={missing}, extra={extra}"
                )
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("predictions NPZ"):
            raise
        raise ValueError(f"could not safely load predictions NPZ {path}: {exc}") from exc


def load_development_inputs(
    benchmark_path: str | Path,
    predictions_path: str | Path,
    *,
    active_target: str = PHASE1_LABELS[0],
    expected_benchmark_sha256: str | None = None,
    expected_predictions_sha256: str | None = None,
    expected_query_pack_sha256: str | None = None,
    require_deferred_analysis: bool = False,
) -> DevelopmentInputs:
    """Validate and bind one canonical development benchmark/NPZ pair."""

    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    benchmark = Path(benchmark_path).expanduser().resolve()
    predictions = Path(predictions_path).expanduser().resolve()
    if not benchmark.is_file():
        raise FileNotFoundError(f"development benchmark not found: {benchmark}")
    if not predictions.is_file():
        raise FileNotFoundError(f"development predictions not found: {predictions}")
    expected_sibling = benchmark.with_suffix(".predictions.npz")
    if predictions != expected_sibling:
        raise ValueError(
            "predictions must be the canonical sibling of the benchmark: "
            f"{expected_sibling}"
        )

    benchmark_hash = sha256_file(benchmark)
    predictions_hash = sha256_file(predictions)
    for actual, expected, field in (
        (benchmark_hash, expected_benchmark_sha256, "expected_benchmark_sha256"),
        (
            predictions_hash,
            expected_predictions_sha256,
            "expected_predictions_sha256",
        ),
    ):
        if expected is not None:
            _require_sha256(expected, field)
            if actual != expected:
                raise ValueError(f"{field} does not match the supplied artifact")

    data = _read_json_object(benchmark)
    if data.get("artifact_type") != _BENCHMARK_ARTIFACT_TYPE:
        raise ValueError(
            f"benchmark artifact_type must be {_BENCHMARK_ARTIFACT_TYPE!r}"
        )
    if data.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema_version; expected 1")
    if data.get("purpose") != "expert_development_evaluation":
        raise ValueError("calibration accepts only purpose='expert_development_evaluation'")
    if data.get("cohort") != "development":
        raise ValueError("TEST DISABLED: calibration accepts only cohort='development'")
    if data.get("smoke_only") is not False:
        raise ValueError("calibration rejects smoke-only benchmark artifacts")
    if data.get("official_manifest_reconciled") is not True:
        raise ValueError(
            "development benchmark must be reconciled to the official NIH manifests"
        )
    if data.get("active_target") != active_target:
        raise ValueError(f"benchmark active_target must be {active_target!r}")
    if data.get("candidate_under_decision") != active_target:
        raise ValueError("benchmark candidate_under_decision does not match active_target")
    if data.get("exploratory_labels") != []:
        raise ValueError(
            "endpoint-isolated benchmark exploratory_labels must be empty"
        )
    if data.get("test_lock") is not None:
        raise ValueError("development benchmark must not contain a test lock")

    decision = _require_object(data.get("decision"), "benchmark decision")
    if decision.get("status") != "not_supplied":
        raise ValueError(
            "calibration requires raw development inference with no decision artifact"
        )
    if decision.get("operating_point_metrics") != "disabled":
        raise ValueError("raw development operating-point metrics must be disabled")

    model = _require_object(data.get("model"), "benchmark model")
    query_spec = PHASE1_QUERY_SPECS[active_target]
    if model.get("labels") != [active_target]:
        raise ValueError("benchmark model must contain only the active endpoint")
    if model.get("prompts") != [query_spec["prompt"]]:
        raise ValueError("benchmark model prompt does not match the frozen endpoint prompt")
    if model.get("semantic_sha256") != query_spec["semantic_sha256"]:
        raise ValueError(
            "benchmark model semantic SHA-256 is not the reviewed endpoint query pack"
        )
    query_hash = _require_sha256(model.get("sha256"), "benchmark model sha256")
    if expected_query_pack_sha256 is not None:
        _require_sha256(
            expected_query_pack_sha256, "expected_query_pack_sha256"
        )
        if query_hash != expected_query_pack_sha256:
            raise ValueError(
                "expected_query_pack_sha256 does not match the benchmark model"
            )
    frozen_prompt_hash = prompt_set_sha256(
        (active_target,),
        (query_spec["prompt"],),
    )
    if model.get("prompt_set_sha256") != frozen_prompt_hash:
        raise ValueError("benchmark prompt_set_sha256 does not match frozen prompts")

    runtime = _require_object(data.get("runtime"), "benchmark runtime")
    _require_exact_int(runtime.get("schema_version"), 3, "benchmark runtime schema")
    git_commit = runtime.get("git_commit")
    if not isinstance(git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("benchmark runtime git_commit is invalid")
    worktree = _require_object(
        runtime.get("git_worktree"),
        "benchmark runtime git_worktree",
    )
    if worktree != {"clean": True, "untracked_files_checked": True}:
        raise ValueError(
            "benchmark evidence must come from a clean Git worktree including "
            "untracked files"
        )
    versions = _require_object(runtime.get("versions"), "benchmark runtime versions")
    for field in (
        "python",
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "torch_cuda_runtime",
        "cudnn_runtime",
        "cuda_driver",
    ):
        if not isinstance(versions.get(field), str) or not versions[field]:
            raise ValueError(f"benchmark runtime versions.{field} must be non-empty text")
    inference = _require_object(
        runtime.get("inference"), "benchmark runtime inference"
    )
    device_resolved = inference.get("device_resolved")
    if not isinstance(device_resolved, str) or not device_resolved:
        raise ValueError("benchmark runtime inference.device_resolved is invalid")
    hardware = _require_object(runtime.get("hardware"), "benchmark runtime hardware")
    accelerator = hardware.get("accelerator")
    if accelerator not in {"cpu", "cuda"}:
        raise ValueError("benchmark runtime hardware.accelerator must be cpu or cuda")
    if (accelerator == "cuda") != device_resolved.startswith("cuda"):
        raise ValueError("benchmark runtime hardware does not match resolved device")
    if accelerator == "cuda":
        required_cuda_hardware = {
            "device_index": int,
            "name": str,
            "compute_capability": list,
            "total_memory_bytes": int,
            "multiprocessor_count": int,
        }
        for field, expected_type in required_cuda_hardware.items():
            value = hardware.get(field)
            if isinstance(value, bool) or not isinstance(value, expected_type):
                raise ValueError(
                    f"benchmark runtime hardware.{field} has invalid type"
                )
        if (
            not hardware["name"]
            or len(hardware["compute_capability"]) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in hardware["compute_capability"]
            )
            or hardware["total_memory_bytes"] <= 0
            or hardware["multiprocessor_count"] <= 0
        ):
            raise ValueError("benchmark runtime CUDA hardware identity is invalid")
    determinism = _require_object(
        runtime.get("determinism"),
        "benchmark runtime determinism",
    )
    required_determinism = {
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    for field, expected in required_determinism.items():
        if determinism.get(field) != expected:
            raise ValueError(
                f"benchmark runtime determinism.{field} must be {expected!r}"
            )
    if not isinstance(determinism.get("torch_cuda_seeded"), bool):
        raise ValueError("benchmark runtime torch_cuda_seeded must be boolean")
    if determinism.get("reproducibility_scope") != (
        "deterministic_algorithms_on_identical_hardware_software_and_inputs;"
        "cross_hardware_bitwise_identity_not_claimed"
    ):
        raise ValueError("benchmark runtime reproducibility scope is invalid")
    runtime_hash = _require_sha256(
        data.get("runtime_contract_sha256"),
        "benchmark runtime_contract_sha256",
    )
    if runtime_hash != _canonical_json_sha256(runtime):
        raise ValueError("benchmark runtime_contract_sha256 is inconsistent")

    inputs = _require_object(data.get("inputs"), "benchmark inputs")
    manifest_hash = _require_sha256(
        inputs.get("manifest_sha256"), "benchmark manifest_sha256"
    )
    image_set_hash = _require_sha256(
        inputs.get("image_set_sha256"), "benchmark image_set_sha256"
    )
    for field in (
        "official_train_val_manifest_sha256",
        "official_test_manifest_sha256",
    ):
        _require_sha256(inputs.get(field), f"benchmark {field}")
    image_provenance = _require_object(
        inputs.get("image_provenance"), "benchmark image_provenance"
    )
    _require_sha256(
        image_provenance.get("sha256"), "benchmark image_provenance sha256"
    )
    original_nih_pixels = image_provenance.get("original_nih_pixels")
    if not isinstance(original_nih_pixels, bool):
        raise ValueError("benchmark image provenance pixel origin must be boolean")
    evidence_status = data.get("evidence_status")
    expected_evidence_status = (
        "development_only_do_not_report_as_test_performance"
        if original_nih_pixels
        else "resized_mirror_candidate_selection_only"
    )
    if evidence_status != expected_evidence_status:
        raise ValueError(
            "benchmark evidence_status is inconsistent with development pixel origin"
        )

    declared_predictions = data.get("predictions")
    if not isinstance(declared_predictions, str) or not declared_predictions:
        raise ValueError("benchmark predictions must name the NPZ artifact")
    if Path(declared_predictions).name != predictions.name:
        raise ValueError("benchmark predictions declaration does not match supplied NPZ")
    declared_predictions_hash = _require_sha256(
        data.get("predictions_sha256"),
        "benchmark predictions_sha256",
    )
    if declared_predictions_hash != predictions_hash:
        raise ValueError(
            "benchmark predictions_sha256 does not match the supplied NPZ artifact"
        )

    arrays = _load_npz(predictions)
    raw_scores = arrays["raw_scores"]
    labels = arrays["labels"]
    if raw_scores.ndim != 2 or raw_scores.shape[1] != 1:
        raise ValueError(
            "endpoint-isolated raw_scores must have shape "
            f"(samples, 1), got {raw_scores.shape}"
        )
    if raw_scores.dtype.kind != "f":
        raise ValueError("raw_scores must use a floating-point dtype")
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if not np.isfinite(raw_scores).all():
        raise ValueError("raw_scores contain NaN or infinity")
    if ((raw_scores < 0.0) | (raw_scores > 1.0)).any():
        raise ValueError("raw_scores must lie in [0, 1]")
    if labels.shape != raw_scores.shape or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("labels must be an integer array matching raw_scores shape")
    labels = np.asarray(labels, dtype=np.int8)
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("labels must contain only -1 (missing), 0, or 1")

    class_names = _string_array(arrays["class_names"], "class_names")
    if class_names.tolist() != [active_target]:
        raise ValueError("predictions class_names must contain only active_target")
    patient_ids = _string_array(arrays["patient_ids"], "patient_ids")
    sample_ids = _string_array(arrays["sample_ids"], "sample_ids")
    image_hashes = _string_array(arrays["image_sha256"], "image_sha256")
    samples = raw_scores.shape[0]
    if any(len(array) != samples for array in (patient_ids, sample_ids, image_hashes)):
        raise ValueError("predictions row metadata lengths do not match raw_scores")
    if samples == 0:
        raise ValueError("development predictions contain no samples")
    if len(set(sample_ids.tolist())) != samples:
        raise ValueError("predictions sample_ids must be unique")
    for index, (patient_id, sample_id, image_hash) in enumerate(
        zip(patient_ids, sample_ids, image_hashes)
    ):
        if not patient_id:
            raise ValueError(f"predictions patient_ids[{index}] is empty")
        if not is_nih_image_filename(sample_id):
            raise ValueError(
                f"predictions sample_ids[{index}] is not a canonical NIH filename"
            )
        if patient_id != nih_patient_id(sample_id):
            raise ValueError(
                f"predictions patient/sample mismatch at row {index}: "
                f"{patient_id!r}, {sample_id!r}"
            )
        _require_sha256(image_hash, f"predictions image_sha256[{index}]")

    reconstructed_image_set_hash = _canonical_json_sha256(
        [
            {
                "sample_id": sample_id,
                "image_path": sample_id,
                "sha256": image_hash,
            }
            for sample_id, image_hash in zip(sample_ids.tolist(), image_hashes.tolist())
        ]
    )
    if reconstructed_image_set_hash != image_set_hash:
        raise ValueError(
            "predictions sample/image hashes do not match benchmark image_set_sha256"
        )

    _require_exact_int(inputs.get("images"), samples, "benchmark inputs.images")
    unique_patients = len(set(patient_ids.tolist()))
    _require_exact_int(
        inputs.get("patients"), unique_patients, "benchmark inputs.patients"
    )
    scorecard = _require_object(data.get("scorecard"), "benchmark scorecard")
    _require_exact_int(scorecard.get("images"), samples, "benchmark scorecard.images")
    _require_exact_int(
        scorecard.get("patients"), unique_patients, "benchmark scorecard.patients"
    )
    metrics_scope = scorecard.get("metrics_scope")
    accepted_scopes = {
        "ranking_only_no_operating_thresholds",
        "development_analysis_deferred_until_patient_partition",
    }
    if metrics_scope not in accepted_scopes:
        raise ValueError(
            "development scorecard must be ranking-only or explicitly deferred "
            "until the patient partition"
        )
    analysis_deferred = (
        metrics_scope == "development_analysis_deferred_until_patient_partition"
    )
    if analysis_deferred and scorecard.get("analysis_deferred") is not True:
        raise ValueError(
            "deferred development scorecard must set analysis_deferred=true"
        )
    if require_deferred_analysis and not analysis_deferred:
        raise ValueError(
            "canonical calibration requires development analysis to be deferred "
            "until the patient partition"
        )
    if scorecard.get("active_target") != active_target:
        raise ValueError("development scorecard active_target mismatch")
    per_label = _require_object(scorecard.get("per_label"), "scorecard per_label")
    row = _require_object(per_label.get(active_target), f"scorecard {active_target}")
    adjudicated = labels[:, 0] >= 0
    positives = int((labels[:, 0] == 1).sum())
    negatives = int((labels[:, 0] == 0).sum())
    _require_exact_int(
        row.get("images"), int(adjudicated.sum()), f"scorecard {active_target}.images"
    )
    _require_exact_int(
        row.get("positives"), positives, f"scorecard {active_target}.positives"
    )
    _require_exact_int(
        row.get("negatives"), negatives, f"scorecard {active_target}.negatives"
    )
    if positives == 0 or negatives == 0:
        raise ValueError(f"active target {active_target!r} is not scoreable")

    return DevelopmentInputs(
        benchmark_path=benchmark,
        predictions_path=predictions,
        benchmark_sha256=benchmark_hash,
        predictions_sha256=predictions_hash,
        query_pack_sha256=query_hash,
        prompt_set_sha256=frozen_prompt_hash,
        manifest_sha256=manifest_hash,
        active_target=active_target,
        benchmark_metrics_scope=str(metrics_scope),
        analysis_deferred=analysis_deferred,
        original_nih_pixels=original_nih_pixels,
        benchmark_evidence_status=expected_evidence_status,
        raw_scores=raw_scores,
        labels=labels,
        patient_ids=patient_ids,
        sample_ids=sample_ids,
        image_sha256=image_hashes,
    )


def _patient_permutation(
    patient_ids: Sequence[str], *, seed: int, attempt: int
) -> list[str]:
    return sorted(
        patient_ids,
        key=lambda patient_id: (
            hashlib.sha256(
                f"kad-phase1-split-v2:{seed}:{attempt}:{patient_id}".encode("utf-8")
            ).digest(),
            patient_id,
        ),
    )


def _support(labels: np.ndarray) -> tuple[int, int]:
    return int((labels == 1).sum()), int((labels == 0).sum())


def _patient_support(
    patient_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, int]:
    """Count independent positive/negative patients using the acceptance truth rule."""

    patients = np.asarray(patient_ids, dtype=str)
    truth = np.asarray(labels, dtype=np.int8)
    if patients.ndim != 1 or truth.ndim != 1 or len(patients) != len(truth):
        raise ValueError("patient support inputs must be equally sized 1-D arrays")
    positive_patients: set[str] = set()
    negative_patients: set[str] = set()
    for patient_id in sorted(set(patients.tolist())):
        patient_truth = truth[patients == patient_id]
        if np.any(patient_truth == 1):
            positive_patients.add(patient_id)
        if np.any(patient_truth == 0):
            negative_patients.add(patient_id)
    return len(positive_patients), len(negative_patients)


def _partition_patient_counts(
    patient_count: int,
    fractions: Sequence[float],
) -> tuple[int, ...]:
    """Allocate every patient across positive-sized roles by largest deficit."""

    if patient_count < len(fractions):
        raise ValueError(
            f"at least {len(fractions)} patients are required for the declared roles"
        )
    targets = [patient_count * float(fraction) for fraction in fractions]
    counts = [max(1, int(math.floor(target))) for target in targets]
    while sum(counts) < patient_count:
        index = max(
            range(len(counts)),
            key=lambda item: (targets[item] - counts[item], -item),
        )
        counts[index] += 1
    while sum(counts) > patient_count:
        eligible = [index for index, count in enumerate(counts) if count > 1]
        if not eligible:
            raise RuntimeError("internal error: could not allocate patient roles")
        index = max(
            eligible,
            key=lambda item: (counts[item] - targets[item], item),
        )
        counts[index] -= 1
    return tuple(counts)


def make_patient_partitions(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    *,
    model_selection_fraction: float = 0.30,
    calibration_fraction: float = 0.20,
    threshold_fraction: float = 0.20,
    acceptance_fraction: float = 0.30,
    seed: int = 20250729,
    attempts: int = 512,
    min_calibration_positives: int = 10,
    min_calibration_negatives: int = 20,
    min_threshold_positives: int = 10,
    min_threshold_negatives: int = 20,
) -> PatientPartitions:
    """Assign each adjudicated patient once, maximizing class-support balance."""

    fractions = (
        model_selection_fraction,
        calibration_fraction,
        threshold_fraction,
        acceptance_fraction,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
        for value in fractions
    ):
        raise ValueError("all four partition fractions must be positive")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "model-selection, calibration, threshold, and acceptance fractions "
            "must sum to 1"
        )
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    support_requirements = (
        min_calibration_positives,
        min_calibration_negatives,
        min_threshold_positives,
        min_threshold_negatives,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in support_requirements
    ):
        raise ValueError("minimum positive/negative supports must be positive integers")

    patients = np.asarray(patient_ids, dtype=str)
    truth = np.asarray(labels, dtype=np.int8)
    if patients.ndim != 1 or truth.ndim != 1 or len(patients) != len(truth):
        raise ValueError("patient_ids and labels must be equally sized 1-D arrays")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("partition labels must be adjudicated binary values")
    unique_patients = sorted(set(patients.tolist()))
    patient_count = len(unique_patients)
    if patient_count < 4:
        model = np.arange(len(truth), dtype=int)
        empty = np.asarray([], dtype=int)
        return PatientPartitions(model, empty, empty, empty, 0, False)

    n_model, n_calibration, n_threshold, _ = _partition_patient_counts(
        patient_count,
        fractions,
    )

    best: tuple[tuple[float, ...], PatientPartitions] | None = None
    for attempt in range(attempts):
        ordered = _patient_permutation(unique_patients, seed=seed, attempt=attempt)
        model_patients = frozenset(ordered[:n_model])
        calibration_patients = frozenset(
            ordered[n_model : n_model + n_calibration]
        )
        threshold_end = n_model + n_calibration + n_threshold
        threshold_patients = frozenset(
            ordered[n_model + n_calibration : threshold_end]
        )
        acceptance_patients = frozenset(ordered[threshold_end:])
        model_indices = np.flatnonzero(np.isin(patients, list(model_patients)))
        calibration_indices = np.flatnonzero(
            np.isin(patients, list(calibration_patients))
        )
        threshold_indices = np.flatnonzero(
            np.isin(patients, list(threshold_patients))
        )
        acceptance_indices = np.flatnonzero(
            np.isin(patients, list(acceptance_patients))
        )
        model_pos, model_neg = _support(truth[model_indices])
        calibration_pos, calibration_neg = _support(truth[calibration_indices])
        threshold_pos, threshold_neg = _support(truth[threshold_indices])
        acceptance_pos, acceptance_neg = _patient_support(
            patients[acceptance_indices],
            truth[acceptance_indices],
        )
        feasible = (
            model_pos >= 1
            and model_neg >= 1
            and calibration_pos >= min_calibration_positives
            and calibration_neg >= min_calibration_negatives
            and threshold_pos >= min_threshold_positives
            and threshold_neg >= min_threshold_negatives
            and acceptance_pos >= _MIN_ACCEPTANCE_POSITIVE_PATIENTS
            and acceptance_neg >= _MIN_ACCEPTANCE_NEGATIVE_PATIENTS
        )
        support_ratio = min(
            model_pos,
            model_neg,
            calibration_pos / min_calibration_positives,
            calibration_neg / min_calibration_negatives,
            threshold_pos / min_threshold_positives,
            threshold_neg / min_threshold_negatives,
            acceptance_pos / _MIN_ACCEPTANCE_POSITIVE_PATIENTS,
            acceptance_neg / _MIN_ACCEPTANCE_NEGATIVE_PATIENTS,
        )
        target_shares = (
            model_selection_fraction,
            calibration_fraction,
            threshold_fraction,
            acceptance_fraction,
        )
        total_pos, total_neg = _support(truth)
        acceptance_image_pos, acceptance_image_neg = _support(
            truth[acceptance_indices]
        )
        observed_shares = (
            (
                model_pos / total_pos,
                calibration_pos / total_pos,
                threshold_pos / total_pos,
                acceptance_image_pos / total_pos,
            ),
            (
                model_neg / total_neg,
                calibration_neg / total_neg,
                threshold_neg / total_neg,
                acceptance_image_neg / total_neg,
            ),
        )
        imbalance = sum(
            abs(observed - target)
            for shares in observed_shares
            for observed, target in zip(shares, target_shares)
        )
        candidate = PatientPartitions(
            model_selection=model_indices,
            calibration=calibration_indices,
            threshold_selection=threshold_indices,
            acceptance=acceptance_indices,
            selected_attempt=attempt,
            support_feasible=feasible,
        )
        rank = (float(feasible), float(support_ratio), -float(imbalance), -float(attempt))
        if best is None or rank > best[0]:
            best = (rank, candidate)
    assert best is not None
    selected = best[1]
    patient_sets = [
        set(patients[indices].tolist())
        for indices in (
            selected.model_selection,
            selected.calibration,
            selected.threshold_selection,
            selected.acceptance,
        )
    ]
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in (
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        )
    ):
        raise RuntimeError("internal error: patient partition overlap")
    return selected


def _clip_logit(scores: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _probability_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
    }


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = _ACCEPTANCE_CONFIDENCE,
) -> tuple[float, float]:
    """Return a deterministic two-sided Wilson score interval."""

    if (
        isinstance(successes, bool)
        or isinstance(total, bool)
        or not isinstance(successes, int)
        or not isinstance(total, int)
        or total <= 0
        or successes < 0
        or successes > total
    ):
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("Wilson confidence must lie strictly between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    estimate = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (estimate + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def study_level_acceptance(
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    sensitivity_target: float,
    specificity_floor: float,
    seed: int,
    evaluation_membership_sha256: str,
) -> dict[str, Any]:
    """Gate a study endpoint using one hash-selected study per patient/class."""

    patients = np.asarray(patient_ids, dtype=str)
    samples = np.asarray(sample_ids, dtype=str)
    labels = np.asarray(truth, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        patients.ndim != 1
        or samples.ndim != 1
        or labels.ndim != 1
        or scores.ndim != 1
        or not (len(patients) == len(samples) == len(labels) == len(scores))
        or len(patients) == 0
    ):
        raise ValueError("study-level acceptance inputs must be non-empty 1-D arrays")
    if len(set(samples.tolist())) != len(samples):
        raise ValueError("study-level acceptance sample IDs must be unique")
    for index, (patient_id, sample_id) in enumerate(zip(patients, samples)):
        if (
            not is_nih_image_filename(sample_id)
            or nih_patient_id(sample_id) != patient_id
        ):
            raise ValueError(
                "study-level acceptance patient/sample mismatch at "
                f"row {index}"
            )
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("study-level acceptance labels must be binary")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("study-level acceptance probabilities must lie in [0, 1]")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("study-level acceptance threshold must lie in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("acceptance study-selection seed must be non-negative")
    _require_sha256(
        evaluation_membership_sha256,
        "acceptance evaluation_membership_sha256",
    )

    predictions = scores >= threshold
    study_outcomes = [
        {
            "patient_id": str(patient_id),
            "sample_id": str(sample_id),
            "truth": int(label),
            "predicted_positive": bool(predicted),
        }
        for patient_id, sample_id, label, predicted in zip(
            patients,
            samples,
            labels,
            predictions,
        )
    ]
    study_outcomes.sort(key=lambda row: (row["patient_id"], row["sample_id"]))

    selection_domain = "kad-phase1-acceptance-study-v1"

    def select_one(patient_id: str, truth_value: int) -> dict[str, str] | None:
        candidates = [
            row
            for row in study_outcomes
            if row["patient_id"] == patient_id and row["truth"] == truth_value
        ]
        if not candidates:
            return None
        selected = min(
            candidates,
            key=lambda row: (
                hashlib.sha256(
                    (
                        f"{selection_domain}:{seed}:{truth_value}:"
                        f"{patient_id}:{row['sample_id']}"
                    ).encode("utf-8")
                ).digest(),
                row["sample_id"],
            ),
        )
        return {
            "patient_id": patient_id,
            "sample_id": str(selected["sample_id"]),
        }

    selected_positive: list[dict[str, str]] = []
    selected_negative: list[dict[str, str]] = []
    for patient_id in sorted(set(patients.tolist())):
        positive = select_one(patient_id, 1)
        negative = select_one(patient_id, 0)
        if positive is not None:
            selected_positive.append(positive)
        if negative is not None:
            selected_negative.append(negative)

    outcomes_by_sample = {
        str(row["sample_id"]): row for row in study_outcomes
    }
    selected_positive_predictions = [
        bool(outcomes_by_sample[row["sample_id"]]["predicted_positive"])
        for row in selected_positive
    ]
    selected_negative_predictions = [
        bool(outcomes_by_sample[row["sample_id"]]["predicted_positive"])
        for row in selected_negative
    ]
    positive_patients = len(selected_positive)
    negative_patients = len(selected_negative)
    true_positives = int(sum(selected_positive_predictions))
    false_negatives = positive_patients - true_positives
    false_positives = int(sum(selected_negative_predictions))
    true_negatives = negative_patients - false_positives
    sensitivity = (
        true_positives / positive_patients if positive_patients else None
    )
    specificity = (
        true_negatives / negative_patients if negative_patients else None
    )
    sensitivity_interval = (
        wilson_interval(true_positives, positive_patients)
        if positive_patients
        else None
    )
    specificity_interval = (
        wilson_interval(true_negatives, negative_patients)
        if negative_patients
        else None
    )

    all_positive = labels == 1
    all_negative = ~all_positive
    all_study_counts = {
        "true_positives": int((all_positive & predictions).sum()),
        "false_negatives": int((all_positive & ~predictions).sum()),
        "true_negatives": int((all_negative & ~predictions).sum()),
        "false_positives": int((all_negative & predictions).sum()),
    }
    all_positive_studies = int(all_positive.sum())
    all_negative_studies = int(all_negative.sum())
    support_passes = (
        positive_patients >= _MIN_ACCEPTANCE_POSITIVE_PATIENTS
        and negative_patients >= _MIN_ACCEPTANCE_NEGATIVE_PATIENTS
    )
    sensitivity_bound_passes = (
        sensitivity_interval is not None
        and sensitivity_interval[0] >= sensitivity_target
    )
    specificity_bound_passes = (
        specificity_interval is not None
        and specificity_interval[0] >= specificity_floor
    )
    return {
        "role": "untouched_acceptance_only_never_tune_or_fit",
        "unit": "study",
        "patient_weighting": (
            "one_hash_selected_positive_and_one_hash_selected_negative_"
            "adjudicated_study_per_patient"
        ),
        "frozen_threshold": float(threshold),
        "truth_definition": "per_adjudicated_radiograph",
        "prediction_definition": "study_probability_meets_frozen_threshold",
        "interval": "two_sided_wilson_score",
        "confidence_level": _ACCEPTANCE_CONFIDENCE,
        "evaluation_membership_sha256": evaluation_membership_sha256,
        "counts": {
            "studies": int(len(labels)),
            "patients": int(len(set(patients.tolist()))),
            "positive_studies": all_positive_studies,
            "negative_studies": all_negative_studies,
            "positive_patients": positive_patients,
            "negative_patients": negative_patients,
            "selected_positive_studies": positive_patients,
            "selected_negative_studies": negative_patients,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
        },
        "study_outcomes": study_outcomes,
        "study_selection": {
            "method": "minimum_sha256_independent_of_scores_v1",
            "hash_domain": selection_domain,
            "seed": seed,
            "positive": selected_positive,
            "negative": selected_negative,
        },
        "point_estimates": {
            "sensitivity": sensitivity,
            "specificity": specificity,
        },
        "all_study_diagnostics": {
            "gate_role": "diagnostic_only_not_used_for_acceptance",
            "counts": {
                "positive_studies": all_positive_studies,
                "negative_studies": all_negative_studies,
                **all_study_counts,
            },
            "point_estimates": {
                "sensitivity": (
                    all_study_counts["true_positives"] / all_positive_studies
                    if all_positive_studies
                    else None
                ),
                "specificity": (
                    all_study_counts["true_negatives"] / all_negative_studies
                    if all_negative_studies
                    else None
                ),
            },
        },
        "confidence_intervals": {
            "sensitivity": (
                {"lower": sensitivity_interval[0], "upper": sensitivity_interval[1]}
                if sensitivity_interval is not None
                else None
            ),
            "specificity": (
                {"lower": specificity_interval[0], "upper": specificity_interval[1]}
                if specificity_interval is not None
                else None
            ),
        },
        "requirements": {
            "minimum_positive_patients": _MIN_ACCEPTANCE_POSITIVE_PATIENTS,
            "minimum_negative_patients": _MIN_ACCEPTANCE_NEGATIVE_PATIENTS,
            "sensitivity_lower_bound": sensitivity_target,
            "specificity_lower_bound": specificity_floor,
        },
        "passes": {
            "support": support_passes,
            "sensitivity_lower_bound": sensitivity_bound_passes,
            "specificity_lower_bound": specificity_bound_passes,
        },
        "complete": (
            support_passes
            and sensitivity_bound_passes
            and specificity_bound_passes
        ),
    }


def clustered_ranking_bootstrap(
    patient_ids: np.ndarray,
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap AUROC/AUPRC by resampling patients with replacement."""

    patients = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(truth, dtype=np.int8)
    probabilities = np.asarray(scores, dtype=np.float64)
    if (
        patients.ndim != 1
        or labels.ndim != 1
        or probabilities.ndim != 1
        or not (len(patients) == len(labels) == len(probabilities))
        or len(patients) == 0
    ):
        raise ValueError("ranking bootstrap inputs must be non-empty 1-D arrays")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    unique_patients = np.unique(patients)
    rows_by_patient = {
        patient_id: np.flatnonzero(patients == patient_id)
        for patient_id in unique_patients
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for _ in range(samples):
        chosen = rng.choice(
            unique_patients, size=len(unique_patients), replace=True
        )
        rows = np.concatenate([rows_by_patient[patient_id] for patient_id in chosen])
        replicate_truth = labels[rows]
        if not ((replicate_truth == 1).any() and (replicate_truth == 0).any()):
            continue
        replicate_scores = probabilities[rows]
        values["auroc"].append(
            float(roc_auc_score(replicate_truth, replicate_scores))
        )
        values["auprc"].append(
            float(average_precision_score(replicate_truth, replicate_scores))
        )

    def interval(metric: str) -> dict[str, float] | None:
        measured = values[metric]
        if not measured:
            return None
        lower, upper = np.percentile(
            np.asarray(measured, dtype=np.float64), [2.5, 97.5]
        )
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "method": "patient_clustered_percentile_bootstrap",
        "confidence_level": 0.95,
        "requested_replicates": samples,
        "patients_resampled_per_replicate": int(len(unique_patients)),
        "seed": seed,
        "successful_replicates": {
            metric: len(metric_values) for metric, metric_values in values.items()
        },
        "intervals": {
            "auroc": interval("auroc"),
            "auprc": interval("auprc"),
        },
        "failed_replicate_policy": (
            "omit_replicates_without_both_active_target_classes"
        ),
    }


def _partition_ranking(
    inputs: DevelopmentInputs,
    patient_ids: np.ndarray,
    indices: np.ndarray,
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    positives, negatives = _support(truth[indices])
    row: dict[str, Any] = {
        "active_target": inputs.active_target,
        "role": "candidate_ranking_only_never_fit_or_threshold",
        "images": int(len(indices)),
        "patients": int(len(set(patient_ids[indices].tolist()))),
        "positives": positives,
        "negatives": negatives,
        "auroc": None,
        "auprc": None,
    }
    if positives and negatives:
        row["auroc"] = float(roc_auc_score(truth[indices], scores[indices]))
        row["auprc"] = float(
            average_precision_score(truth[indices], scores[indices])
        )
        row["confidence_intervals"] = clustered_ranking_bootstrap(
            patient_ids[indices],
            truth[indices],
            scores[indices],
            samples=bootstrap_samples,
            seed=seed,
        )
    else:
        row["confidence_intervals"] = {
            "method": "patient_clustered_percentile_bootstrap",
            "confidence_level": 0.95,
            "requested_replicates": bootstrap_samples,
            "patients_resampled_per_replicate": int(
                len(set(patient_ids[indices].tolist()))
            ),
            "seed": seed,
            "successful_replicates": {"auroc": 0, "auprc": 0},
            "intervals": {"auroc": None, "auprc": None},
            "failed_replicate_policy": (
                "not_run_because_model_selection_partition_is_not_scoreable"
            ),
        }
    return row


def _partition_record(
    inputs: DevelopmentInputs,
    indices: np.ndarray,
) -> dict[str, Any]:
    truth = inputs.labels[indices, 0]
    sample_ids = sorted(inputs.sample_ids[indices].tolist())
    patient_ids = sorted(set(inputs.patient_ids[indices].tolist()))
    positives, negatives = _support(truth)
    members = [
        {
            "sample_id": str(inputs.sample_ids[index]),
            "patient_id": str(inputs.patient_ids[index]),
        }
        for index in indices
    ]
    members.sort(key=lambda row: (row["patient_id"], row["sample_id"]))
    return {
        "membership_sha256": _canonical_json_sha256(members),
        "images": int(len(indices)),
        "patients": int(len(patient_ids)),
        "positives": positives,
        "negatives": negatives,
        "patient_ids": patient_ids,
        "sample_ids": sample_ids,
    }


def build_decision_artifact(
    inputs: DevelopmentInputs,
    *,
    model_selection_fraction: float = 0.30,
    calibration_fraction: float = 0.20,
    threshold_fraction: float = 0.20,
    acceptance_fraction: float = 0.30,
    seed: int = 20250729,
    split_attempts: int = 512,
    bootstrap_samples: int = 1000,
    min_calibration_positives: int = 10,
    min_calibration_negatives: int = 20,
    min_threshold_positives: int = 10,
    min_threshold_negatives: int = 20,
    sensitivity_target: float = 0.85,
    specificity_floor: float = 0.60,
) -> dict[str, Any]:
    """Fit calibration/threshold roles and return a decision or diagnostic artifact."""

    for name, value in (
        ("sensitivity_target", sensitivity_target),
        ("specificity_floor", specificity_floor),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if sensitivity_target < _MIN_SENSITIVITY_TARGET:
        raise ValueError(
            f"sensitivity_target cannot be lower than {_MIN_SENSITIVITY_TARGET}"
        )
    if specificity_floor < _MIN_SPECIFICITY_FLOOR:
        raise ValueError(
            f"specificity_floor cannot be lower than {_MIN_SPECIFICITY_FLOOR}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    adjudicated = inputs.labels[:, 0] >= 0
    active_indices = np.flatnonzero(adjudicated)
    active_truth = np.asarray(
        inputs.labels[active_indices, 0], dtype=np.int8
    )
    active_scores = np.asarray(
        inputs.raw_scores[active_indices, 0], dtype=np.float64
    )
    active_patients = inputs.patient_ids[active_indices]
    partitions = make_patient_partitions(
        active_patients,
        active_truth,
        model_selection_fraction=model_selection_fraction,
        calibration_fraction=calibration_fraction,
        threshold_fraction=threshold_fraction,
        acceptance_fraction=acceptance_fraction,
        seed=seed,
        attempts=split_attempts,
        min_calibration_positives=min_calibration_positives,
        min_calibration_negatives=min_calibration_negatives,
        min_threshold_positives=min_threshold_positives,
        min_threshold_negatives=min_threshold_negatives,
    )
    split_records = {
        "model_selection": _partition_record(
            inputs, active_indices[partitions.model_selection]
        ),
        "calibration": _partition_record(
            inputs, active_indices[partitions.calibration]
        ),
        "threshold_selection": _partition_record(
            inputs, active_indices[partitions.threshold_selection]
        ),
        "acceptance": _partition_record(
            inputs, active_indices[partitions.acceptance]
        ),
    }
    split_patient_sets = [
        set(split_records[name]["patient_ids"])
        for name in (
            "model_selection",
            "calibration",
            "threshold_selection",
            "acceptance",
        )
    ]
    pairwise_overlap = {
        "model_selection_calibration": len(
            split_patient_sets[0] & split_patient_sets[1]
        ),
        "model_selection_threshold_selection": len(
            split_patient_sets[0] & split_patient_sets[2]
        ),
        "model_selection_acceptance": len(
            split_patient_sets[0] & split_patient_sets[3]
        ),
        "calibration_threshold_selection": len(
            split_patient_sets[1] & split_patient_sets[2]
        ),
        "calibration_acceptance": len(
            split_patient_sets[1] & split_patient_sets[3]
        ),
        "threshold_selection_acceptance": len(
            split_patient_sets[2] & split_patient_sets[3]
        ),
    }
    if any(pairwise_overlap.values()):
        raise RuntimeError("internal error: patient leakage across decision partitions")

    reasons: list[str] = []
    if not partitions.support_feasible:
        reasons.append("four_way_patient_split_has_insufficient_class_support")
    if not inputs.original_nih_pixels:
        reasons.append("non_original_development_pixels_not_acceptance_eligible")

    calibrator_parameters: dict[str, dict[str, float]] = {}
    calibrator_metrics: dict[str, dict[str, Any]] = {}
    calibration_complete = False
    slope: float | None = None
    intercept: float | None = None
    optimizer_iterations: int | None = None
    calibration_indices = partitions.calibration
    threshold_indices = partitions.threshold_selection
    acceptance_indices = partitions.acceptance
    calibration_support = _support(active_truth[calibration_indices])
    threshold_support = _support(active_truth[threshold_indices])
    calibration_supported = (
        calibration_support[0] >= min_calibration_positives
        and calibration_support[1] >= min_calibration_negatives
    )
    threshold_supported = (
        threshold_support[0] >= min_threshold_positives
        and threshold_support[1] >= min_threshold_negatives
    )

    if calibration_supported:
        calibration_logits = _clip_logit(
            active_scores[calibration_indices], _CLIP_EPSILON
        )
        model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            fit_intercept=True,
            max_iter=2000,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(
                calibration_logits.reshape(-1, 1),
                active_truth[calibration_indices],
            )
        convergence_warnings = [
            warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
        ]
        slope = float(model.coef_[0, 0])
        intercept = float(model.intercept_[0])
        optimizer_iterations = int(model.n_iter_[0])
        if convergence_warnings:
            reasons.append("platt_optimizer_did_not_converge")
        elif not math.isfinite(slope) or not math.isfinite(intercept):
            reasons.append("platt_parameters_are_not_finite")
        elif slope <= 0.0:
            reasons.append("platt_slope_is_not_positive")
        else:
            calibration_complete = True
            calibrator_parameters = {
                inputs.active_target: {
                    "slope": slope,
                    "intercept": intercept,
                }
            }
            partition_metrics: dict[str, Any] = {}
            for role, indices in (
                ("calibration", calibration_indices),
                ("threshold_selection", threshold_indices),
            ):
                if len(indices) and len(np.unique(active_truth[indices])) == 2:
                    before = active_scores[indices]
                    after = _sigmoid(
                        _clip_logit(before, _CLIP_EPSILON) * slope + intercept
                    )
                    partition_metrics[role] = {
                        "images": int(len(indices)),
                        "before": _probability_metrics(active_truth[indices], before),
                        "after": _probability_metrics(active_truth[indices], after),
                    }
            calibrator_metrics = {inputs.active_target: partition_metrics}
    else:
        reasons.append("calibration_partition_below_minimum_support")

    threshold_selection: ThresholdSelection | None = None
    acceptance: dict[str, Any] | None = None
    candidate_thresholds: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    acceptance_complete = False
    thresholds_complete = False
    if calibration_complete and threshold_supported:
        assert slope is not None and intercept is not None
        threshold_probabilities = _sigmoid(
            _clip_logit(active_scores[threshold_indices], _CLIP_EPSILON) * slope
            + intercept
        )
        threshold_selection = select_per_label_thresholds(
            threshold_probabilities.reshape(-1, 1),
            active_truth[threshold_indices].reshape(-1, 1),
            [inputs.active_target],
            sensitivity_target=sensitivity_target,
            specificity_floor=specificity_floor,
            min_positives=min_threshold_positives,
            min_negatives=min_threshold_negatives,
        )[0]
        candidate_thresholds = {
            inputs.active_target: float(threshold_selection.threshold)
        }
        if not (
            threshold_selection.supported
            and threshold_selection.meets_constraints
        ):
            reasons.append("threshold_operating_constraints_are_infeasible")
        elif not partitions.support_feasible:
            reasons.append("four_way_patient_partition_is_not_acceptance_ready")
        elif not inputs.original_nih_pixels:
            # Resized/re-encoded mirror pixels may support ranking and diagnostics,
            # but never a test-ready acceptance decision for original NIH pixels.
            pass
        else:
            # The acceptance probabilities are deliberately not computed until the
            # calibrator and candidate threshold have passed every upstream gate.
            acceptance_probabilities = _sigmoid(
                _clip_logit(active_scores[acceptance_indices], _CLIP_EPSILON) * slope
                + intercept
            )
            acceptance = study_level_acceptance(
                active_patients[acceptance_indices],
                inputs.sample_ids[
                    active_indices[acceptance_indices]
                ],
                active_truth[acceptance_indices],
                acceptance_probabilities,
                threshold=float(threshold_selection.threshold),
                sensitivity_target=sensitivity_target,
                specificity_floor=specificity_floor,
                seed=seed + 1_000_003,
                evaluation_membership_sha256=split_records["acceptance"][
                    "membership_sha256"
                ],
            )
            acceptance_complete = bool(acceptance["complete"])
            if acceptance_complete:
                thresholds_complete = True
                thresholds = dict(candidate_thresholds)
            passes = acceptance["passes"]
            if not passes["support"]:
                reasons.append("study_level_acceptance_support_is_inadequate")
            if not passes["sensitivity_lower_bound"]:
                reasons.append(
                    "sensitivity_lower_95_percent_confidence_bound_below_target"
                )
            if not passes["specificity_lower_bound"]:
                reasons.append(
                    "specificity_lower_95_percent_confidence_bound_below_floor"
                )
    elif not threshold_supported:
        reasons.append("threshold_partition_below_minimum_support")

    selection_result = (
        threshold_selection.to_dict() if threshold_selection is not None else None
    )
    model_selection_ranking = _partition_ranking(
        inputs,
        active_patients,
        partitions.model_selection,
        active_truth,
        active_scores,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    model_selection_ranking["membership_sha256"] = split_records[
        "model_selection"
    ]["membership_sha256"]
    deployable = calibration_complete and thresholds_complete
    calibration_runtime = _calibration_runtime_contract()
    return {
        "artifact_type": _DECISION_ARTIFACT_TYPE,
        "schema_version": _DECISION_SCHEMA_VERSION,
        "status": (
            "complete_research_decision" if deployable else "diagnostic_not_deployable"
        ),
        "labels": [inputs.active_target],
        "prompts": [PHASE1_QUERY_SPECS[inputs.active_target]["prompt"]],
        "active_target": inputs.active_target,
        "query_pack_sha256": inputs.query_pack_sha256,
        "prompt_set_sha256": inputs.prompt_set_sha256,
        "development_manifest_sha256": inputs.manifest_sha256,
        "development_benchmark_sha256": inputs.benchmark_sha256,
        "development_predictions_sha256": inputs.predictions_sha256,
        "development_predictions_filename": inputs.predictions_path.name,
        "calibration_runtime": calibration_runtime,
        "calibration_runtime_sha256": _canonical_json_sha256(calibration_runtime),
        "source_benchmark_analysis": {
            "metrics_scope": inputs.benchmark_metrics_scope,
            "analysis_deferred": inputs.analysis_deferred,
        },
        "development_pixel_evidence": {
            "original_nih_pixels": inputs.original_nih_pixels,
            "benchmark_evidence_status": inputs.benchmark_evidence_status,
            "acceptance_eligible": inputs.original_nih_pixels,
        },
        "calibration_complete": calibration_complete,
        "acceptance_complete": acceptance_complete,
        "thresholds_complete": thresholds_complete,
        "model_selection_ranking": model_selection_ranking,
        "calibrator": {
            "type": "per_label_platt_logit",
            "input": "clipped_raw_score_logit",
            "clip_epsilon": _CLIP_EPSILON,
            "fit_role": "calibration_only",
            "fit_membership_sha256": split_records["calibration"][
                "membership_sha256"
            ],
            "regularization": {
                "implementation": "sklearn.linear_model.LogisticRegression",
                "C": 1_000_000.0,
            },
            "optimizer_iterations": optimizer_iterations,
            "parameters": calibrator_parameters,
            "metrics": calibrator_metrics,
        },
        "candidate_thresholds": candidate_thresholds,
        "thresholds": thresholds,
        "threshold_selection": {
            "active_target": inputs.active_target,
            "fit_role": "threshold_selection_only",
            "fit_membership_sha256": split_records["threshold_selection"][
                "membership_sha256"
            ],
            "method": "most_specific_observed_probability_meeting_constraints",
            "sensitivity_target": sensitivity_target,
            "specificity_floor": specificity_floor,
            "min_positives": min_threshold_positives,
            "min_negatives": min_threshold_negatives,
            "candidate_threshold": (
                float(threshold_selection.threshold)
                if threshold_selection is not None
                else None
            ),
            "result": selection_result,
            "untouched_study_level_acceptance": acceptance,
        },
        "patient_partition": {
            "method": "deterministic_label_support_aware_patient_hash_search_v2",
            "scope": "endpoint_specific_frozen_across_all_candidates",
            "active_target": inputs.active_target,
            "seed": seed,
            "attempts_evaluated": split_attempts,
            "selected_attempt": partitions.selected_attempt,
            "fractions": {
                "model_selection": model_selection_fraction,
                "calibration": calibration_fraction,
                "threshold_selection": threshold_fraction,
                "acceptance": acceptance_fraction,
            },
            "active_target_adjudicated_images": int(len(active_indices)),
            "active_target_non_adjudicated_images": int((~adjudicated).sum()),
            "support_feasible": partitions.support_feasible,
            "pairwise_patient_overlap": pairwise_overlap,
            "partitions": split_records,
        },
        "requirements": {
            "minimum_calibration_positives": min_calibration_positives,
            "minimum_calibration_negatives": min_calibration_negatives,
            "minimum_threshold_positives": min_threshold_positives,
            "minimum_threshold_negatives": min_threshold_negatives,
            "minimum_acceptance_positive_patients": (
                _MIN_ACCEPTANCE_POSITIVE_PATIENTS
            ),
            "minimum_acceptance_negative_patients": (
                _MIN_ACCEPTANCE_NEGATIVE_PATIENTS
            ),
            "sensitivity_target": sensitivity_target,
            "specificity_floor": specificity_floor,
            "positive_platt_slope_required": True,
            "acceptance_confidence_level": _ACCEPTANCE_CONFIDENCE,
            "acceptance_requires_lower_confidence_bounds": True,
            "acceptance_interval_method": (
                "two_sided_wilson_score"
            ),
        },
        "diagnostics": {
            "deployable": deployable,
            "reasons": sorted(set(reasons)),
        },
        "warning": (
            "Research evaluation only; this artifact does not establish clinical "
            "validity. Only model_selection_ranking may be used for candidate "
            "selection; calibration, threshold-selection, and untouched acceptance "
            "patients are reserved for their declared roles."
        ),
    }


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier after replacing a Drive/local artifact."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_decision_artifact(
    path: str | Path,
    artifact: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one decision artifact without silently replacing a run."""

    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing decision artifact: {output}"
        )
    content = (
        json.dumps(dict(artifact), indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing decision artifact: {output}"
            )
        os.replace(temporary_path, output)
        temporary_path = None
        _fsync_directory(output.parent)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit one KAD phase-1 calibrator and operating threshold."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-target", choices=PHASE1_LABELS, default=PHASE1_LABELS[0])
    parser.add_argument("--expected-benchmark-sha256")
    parser.add_argument("--expected-predictions-sha256")
    parser.add_argument("--expected-query-pack-sha256")
    parser.add_argument(
        "--require-deferred-analysis",
        action="store_true",
        help=(
            "reject a development benchmark that exposed full-cohort ranking "
            "metrics before the patient-role split"
        ),
    )
    parser.add_argument("--model-selection-fraction", type=float, default=0.30)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--threshold-fraction", type=float, default=0.20)
    parser.add_argument("--acceptance-fraction", type=float, default=0.30)
    parser.add_argument(
        "--seed",
        type=int,
        default=_PROTOCOL_PARTITION_SEED,
        help=(
            f"frozen endpoint-role seed; canonical evidence requires "
            f"{_PROTOCOL_PARTITION_SEED}"
        ),
    )
    parser.add_argument(
        "--split-attempts",
        type=int,
        default=_PROTOCOL_SPLIT_ATTEMPTS,
        help=(
            f"frozen score-blind label-support search count; canonical evidence "
            f"requires {_PROTOCOL_SPLIT_ATTEMPTS}"
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=_PROTOCOL_BOOTSTRAP_SAMPLES,
        help=(
            "patient-clustered bootstrap replicates for model-selection "
            "AUROC/AUPRC intervals"
        ),
    )
    parser.add_argument("--min-calibration-positives", type=int, default=10)
    parser.add_argument("--min-calibration-negatives", type=int, default=20)
    parser.add_argument("--min-threshold-positives", type=int, default=10)
    parser.add_argument("--min-threshold-negatives", type=int, default=20)
    parser.add_argument("--sensitivity-target", type=float, default=0.85)
    parser.add_argument("--specificity-floor", type=float, default=0.60)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing decision JSON only when explicitly requested",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.require_deferred_analysis:
        parser.error(
            "--require-deferred-analysis is mandatory for canonical evidence"
        )
    canonical_fractions = {
        "model_selection": args.model_selection_fraction,
        "calibration": args.calibration_fraction,
        "threshold_selection": args.threshold_fraction,
        "acceptance": args.acceptance_fraction,
    }
    for role, expected in _PROTOCOL_ROLE_FRACTIONS.items():
        if not math.isclose(
            canonical_fractions[role],
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            parser.error(
                f"{role.replace('_', '-')} fraction is frozen at {expected:.2f}; "
                "changing it would recycle endpoint-role membership"
            )
    if args.seed != _PROTOCOL_PARTITION_SEED:
        parser.error(
            f"--seed is frozen at {_PROTOCOL_PARTITION_SEED}; changing it would "
            "recycle endpoint-role membership"
        )
    if args.split_attempts != _PROTOCOL_SPLIT_ATTEMPTS:
        parser.error(
            f"--split-attempts is frozen at {_PROTOCOL_SPLIT_ATTEMPTS}; changing "
            "it would recycle endpoint-role membership"
        )
    if args.bootstrap_samples != _PROTOCOL_BOOTSTRAP_SAMPLES:
        parser.error(
            f"--bootstrap-samples is frozen at {_PROTOCOL_BOOTSTRAP_SAMPLES}"
        )
    canonical_support = {
        "calibration_positives": args.min_calibration_positives,
        "calibration_negatives": args.min_calibration_negatives,
        "threshold_positives": args.min_threshold_positives,
        "threshold_negatives": args.min_threshold_negatives,
    }
    for field, expected in _PROTOCOL_MINIMUM_SUPPORT.items():
        if canonical_support[field] != expected:
            parser.error(
                f"minimum {field.replace('_', ' ')} is frozen at {expected}"
            )
    if not math.isclose(
        args.sensitivity_target,
        _MIN_SENSITIVITY_TARGET,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        parser.error(
            f"--sensitivity-target is frozen at {_MIN_SENSITIVITY_TARGET}"
        )
    if not math.isclose(
        args.specificity_floor,
        _MIN_SPECIFICITY_FLOOR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        parser.error(
            f"--specificity-floor is frozen at {_MIN_SPECIFICITY_FLOOR}"
        )
    if args.output.expanduser().exists() and not args.overwrite:
        parser.error(
            f"refusing to overwrite existing decision artifact: "
            f"{args.output.expanduser()}"
        )
    try:
        inputs = load_development_inputs(
            args.benchmark,
            args.predictions,
            active_target=args.active_target,
            expected_benchmark_sha256=args.expected_benchmark_sha256,
            expected_predictions_sha256=args.expected_predictions_sha256,
            expected_query_pack_sha256=args.expected_query_pack_sha256,
            require_deferred_analysis=args.require_deferred_analysis,
        )
        if args.output.expanduser().resolve().parent != inputs.predictions_path.parent:
            raise ValueError(
                "decision output must be written beside its bound development "
                "predictions NPZ so the strict policy loader can rederive evidence"
            )
        artifact = build_decision_artifact(
            inputs,
            model_selection_fraction=args.model_selection_fraction,
            calibration_fraction=args.calibration_fraction,
            threshold_fraction=args.threshold_fraction,
            acceptance_fraction=args.acceptance_fraction,
            seed=args.seed,
            split_attempts=args.split_attempts,
            bootstrap_samples=args.bootstrap_samples,
            min_calibration_positives=args.min_calibration_positives,
            min_calibration_negatives=args.min_calibration_negatives,
            min_threshold_positives=args.min_threshold_positives,
            min_threshold_negatives=args.min_threshold_negatives,
            sensitivity_target=args.sensitivity_target,
            specificity_floor=args.specificity_floor,
        )
        output = write_decision_artifact(
            args.output,
            artifact,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    status = "COMPLETE" if artifact["thresholds_complete"] else "NOT DEPLOYABLE"
    print(f"{status}: wrote {output}")
    if not artifact["thresholds_complete"]:
        print("Reasons: " + ", ".join(artifact["diagnostics"]["reasons"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
