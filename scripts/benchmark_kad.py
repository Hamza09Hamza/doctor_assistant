"""Leakage-resistant KAD-512 evaluation for expert-labeled chest radiographs.

The normal mode consumes the canonical CSV emitted by
``scripts/prepare_nih_expert_manifest.py`` and an image root.  It
defaults to development evaluation.  A test cohort is fail-closed: inference starts
only when a complete calibration/threshold artifact and a previously frozen lock
bind the exact query pack, manifest, prompt set, decision artifact, and image bytes.

Canonical CSV columns (exact spelling):

    filename,patient_id,split,Pneumothorax,Nodule_or_mass,Airspace_opacity

``split`` is ``validation`` or ``test``. Labels must be literal 0 or 1 when
adjudicated; a blank label is excluded independently for that target. ``filename``
is joined directly to ``--image-root``.

``--smoke-mirror`` is deliberately separate evidence-wise.  It exercises a standard
14-query KAD pack on at most 200 images from a pinned third-party Hugging Face mirror.
It always uses the mirror's development-named partition, emits ranking metrics only,
and records that the result is smoke-only and not officially manifest-reconciled.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
from statistics import NormalDist
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from data.chest_xray14 import CHESTXRAY14_LABELS
from data.nih_expert_labels import load_google_nih_expert_csv
from data.nih_protocol import (
    is_nih_image_filename,
    nih_patient_id,
    read_nih_filename_manifest,
    reconcile_nih_official_manifests,
)
from evaluation import evaluate_multilabel_classifier, rank_classification_errors
from experts.kad import (
    KAD512Expert,
    KAD512_LABELS,
    KAD512_PROMPTS,
    kad512_query_pack_semantic_sha256,
    preflight_kad512_query_pack,
    preprocess_kad512,
)
from scripts.calibrate_chest_thresholds import load_split_sample
from scripts.eval_chest_xrv import _DATASET_ID, _DATASET_REVISION
from scripts.export_kad_query_pack import (
    PHASE1_LABELS,
    PHASE1_PROMPTS,
    PHASE1_QUERY_SPECS,
    sha256_file,
)
from scripts.fetch_nih_metadata import (
    GOOGLE_EXPERT_LABEL_FILENAME,
    GOOGLE_EXPERT_LABEL_SPEC,
    NIH_METADATA_FILES,
    PINNED_FILES,
    TORCHXRAYVISION_COMMIT,
    YEIGEN_DATASET,
    YEIGEN_REVISION,
)


_REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "filename",
    "patient_id",
    "split",
    *PHASE1_LABELS,
)
_OPTIONAL_MANIFEST_COLUMNS = {"Fracture"}
_DECISION_ARTIFACT_TYPE = "doctor_assistant.kad_phase1_decision"
_DECISION_SCHEMA_VERSION = 3
_TEST_LOCK_ARTIFACT_TYPE = "doctor_assistant.kad_phase1_test_lock"
_SOURCE_METADATA_ARTIFACT_TYPE = (
    "doctor_assistant.nih_metadata_and_expert_labels"
)
_MANIFEST_METADATA_FORMAT = "nih_google_four_findings_canonical"
_KAD512_CHECKPOINT_SHA256 = (
    "eb7223657220aa51eef43b2e155fd73c593eb5821fdcc8741782f6581ddfea76"
)
_NIH14_SMOKE_QUERY_SET = "doctor_assistant.nih14_smoke.v1"
_DEVELOPMENT_IMAGE_DATASET = "arudaev/chest-xray-14-320"
_DEVELOPMENT_IMAGE_REVISION = (
    "1c9e054e3336a473be6c01d77cdedf96442e2bad"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CALIBRATOR_CLIP_EPSILON = 1e-7
_MIN_SENSITIVITY_TARGET = 0.85
_MIN_SPECIFICITY_FLOOR = 0.60
_MIN_ACCEPTANCE_POSITIVE_PATIENTS = 22
_MIN_ACCEPTANCE_NEGATIVE_PATIENTS = 20
_PROTOCOL_PARTITION_SEED = 20250729
_PROTOCOL_SPLIT_ATTEMPTS = 512
_PROTOCOL_BOOTSTRAP_SAMPLES = 1000
_PROTOCOL_ROLE_FRACTIONS = (0.30, 0.20, 0.20, 0.30)
_PROTOCOL_MINIMUM_SUPPORT = (10, 20, 10, 20)
_DEFAULT_CACHE = Path(
    os.environ.get(
        "DOCTOR_ASSISTANT_CACHE_DIR",
        Path.home() / ".cache" / "doctor_assistant" / "evaluation",
    )
)


@dataclass(frozen=True)
class ExpertManifest:
    path: Path
    image_root: Path
    cohort: str
    sample_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    image_relpaths: tuple[str, ...]
    labels: np.ndarray
    manifest_sha256: str
    image_sha256: tuple[str, ...]
    image_set_sha256: str
    official_train_val_manifest_sha256: str | None
    official_test_manifest_sha256: str | None
    official_membership_verified: bool


@dataclass(frozen=True)
class DecisionPolicy:
    path: Path
    artifact_sha256: str
    active_target: str
    thresholds: dict[str, float]
    slope: float
    intercept: float
    development_manifest_sha256: str
    development_benchmark_sha256: str
    development_predictions_sha256: str

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != 1:
            raise ValueError(
                "endpoint-isolated raw KAD scores must have shape "
                f"(samples, 1), got {scores.shape}"
            )
        calibrated = scores.copy()
        clipped = np.clip(
            scores[:, 0],
            _CALIBRATOR_CLIP_EPSILON,
            1.0 - _CALIBRATOR_CLIP_EPSILON,
        )
        logits = np.log(clipped / (1.0 - clipped))
        calibrated_logits = logits * self.slope + self.intercept
        calibrated[:, 0] = 1.0 / (
            1.0 + np.exp(-np.clip(calibrated_logits, -80.0, 80.0))
        )
        return calibrated


@dataclass(frozen=True)
class ImageProvenance:
    path: Path
    artifact_sha256: str
    source: str
    width: int
    height: int
    original_nih_pixels: bool
    canonical_manifest_sha256: str
    cohort: str
    rows_selected: int
    output_sha256_by_filename: Mapping[str, str]
    expert_split_by_filename: Mapping[str, str]


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


def _cuda_driver_version() -> str:
    """Return the NVIDIA driver version without binding a run to one GPU UUID."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "not-available"
    versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    return ",".join(versions) if versions else "not-available"


def _hardware_contract(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        processor = platform.processor().strip()
        if not processor:
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as handle:
                    processor = next(
                        (
                            line.split(":", 1)[1].strip()
                            for line in handle
                            if line.lower().startswith("model name")
                        ),
                        "unknown",
                    )
            except OSError:
                processor = "unknown"
        return {
            "accelerator": "cpu",
            "machine": platform.machine() or "unknown",
            "processor": processor or "unknown",
        }

    index = device.index
    if index is None:
        index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    return {
        "accelerator": "cuda",
        "device_index": int(index),
        "name": str(properties.name),
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not resolve repository git commit: {exc}") from exc
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _git_worktree_clean() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not inspect repository worktree: {exc}") from exc
    return not completed.stdout.strip()


def _configure_deterministic_inference(seed: int) -> dict[str, Any]:
    """Apply and report deterministic inference controls before model execution."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("inference seed must be a non-negative integer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    warn_only = (
        bool(torch.is_deterministic_algorithms_warn_only_enabled())
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    return {
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_cpu_seeded": True,
        "torch_cuda_seeded": cuda_available,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": warn_only,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(
            getattr(torch.backends.cudnn, "allow_tf32", False)
        ),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "reproducibility_scope": (
            "deterministic_algorithms_on_identical_hardware_software_and_inputs;"
            "cross_hardware_bitwise_identity_not_claimed"
        ),
    }


def build_runtime_contract(
    args: argparse.Namespace,
    *,
    require_clean_worktree: bool = False,
) -> tuple[dict[str, Any], str]:
    """Return deterministic code/environment/inference provenance for this run."""

    worktree_clean = _git_worktree_clean()
    if require_clean_worktree and not worktree_clean:
        raise RuntimeError(
            "evidence run requires a clean Git worktree including no untracked files; "
            "commit/stash local changes and use an immutable run directory"
        )
    determinism = _configure_deterministic_inference(int(args.seed))
    runtime_device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    amp_requested = not args.no_amp
    cuda_runtime = getattr(torch.version, "cuda", None)
    cudnn_runtime = torch.backends.cudnn.version()
    runtime = {
        "schema_version": 3,
        "git_commit": _git_commit(),
        "git_worktree": {
            "clean": worktree_clean,
            "untracked_files_checked": True,
        },
        "code_sha256": {
            "experts/kad.py": sha256_file(_REPO_ROOT / "experts" / "kad.py"),
            "scripts/benchmark_kad.py": sha256_file(Path(__file__).resolve()),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": _distribution_version("torchvision"),
            "numpy": str(np.__version__),
            "Pillow": _distribution_version("Pillow"),
            "torch_cuda_runtime": (
                str(cuda_runtime) if cuda_runtime is not None else "not-applicable"
            ),
            "cudnn_runtime": (
                str(cudnn_runtime) if cudnn_runtime is not None else "not-applicable"
            ),
            "cuda_driver": (
                _cuda_driver_version()
                if runtime_device.type == "cuda"
                else "not-applicable"
            ),
        },
        "hardware": _hardware_contract(runtime_device),
        "inference": {
            "device_requested": args.device,
            "device_resolved": str(runtime_device),
            "amp_requested": amp_requested,
            "amp_enabled": bool(amp_requested and runtime_device.type == "cuda"),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
        },
        "determinism": determinism,
    }
    return runtime, _canonical_json_sha256(runtime)


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


def _safe_string_array(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a one-dimensional string array")
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    return np.asarray(array, dtype=str)


def _load_bound_decision_predictions(
    data: Mapping[str, Any],
    *,
    decision_path: Path,
    active_target: str,
    expected_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Open and validate the exact development NPZ bound into a decision.

    Acceptance outcomes are rederived from these raw scores.  This prevents a
    self-consistent edit of JSON booleans/counts/intervals from being accepted.
    """

    filename = data.get("development_predictions_filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not filename.endswith(".predictions.npz")
    ):
        raise ValueError(
            "decision development_predictions_filename must be a local "
            "*.predictions.npz basename"
        )
    path = decision_path.parent / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"decision-bound development predictions not found: {path}"
        )
    if sha256_file(path) != expected_sha256:
        raise ValueError(
            "decision-bound development predictions SHA-256 does not match"
        )
    expected_fields = {
        "raw_scores",
        "labels",
        "class_names",
        "patient_ids",
        "sample_ids",
        "image_sha256",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_fields:
                raise ValueError(
                    "decision-bound development predictions fields are not canonical"
                )
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("decision-bound"):
            raise
        raise ValueError(
            f"could not safely load decision-bound development predictions: {exc}"
        ) from exc

    raw_scores = arrays["raw_scores"]
    labels = arrays["labels"]
    if (
        raw_scores.ndim != 2
        or raw_scores.shape[1] != 1
        or raw_scores.dtype.kind != "f"
        or labels.shape != raw_scores.shape
        or labels.dtype.kind not in {"i", "u"}
    ):
        raise ValueError(
            "decision-bound raw_scores/labels must be endpoint-isolated (samples, 1)"
        )
    scores = np.asarray(raw_scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int8)
    if (
        not np.isfinite(scores).all()
        or ((scores < 0.0) | (scores > 1.0)).any()
        or not np.isin(truth, (-1, 0, 1)).all()
    ):
        raise ValueError("decision-bound scores or labels are invalid")
    class_names = _safe_string_array(
        arrays["class_names"],
        "decision-bound class_names",
    )
    if class_names.tolist() != [active_target]:
        raise ValueError(
            "decision-bound class_names must contain only the active target"
        )
    patient_ids = _safe_string_array(
        arrays["patient_ids"],
        "decision-bound patient_ids",
    )
    sample_ids = _safe_string_array(
        arrays["sample_ids"],
        "decision-bound sample_ids",
    )
    image_hashes = _safe_string_array(
        arrays["image_sha256"],
        "decision-bound image_sha256",
    )
    samples = scores.shape[0]
    if (
        samples == 0
        or any(len(values) != samples for values in (patient_ids, sample_ids, image_hashes))
        or len(set(sample_ids.tolist())) != samples
    ):
        raise ValueError("decision-bound prediction row metadata is invalid")

    rows: dict[str, dict[str, Any]] = {}
    for index, (patient_id, sample_id, image_hash) in enumerate(
        zip(patient_ids.tolist(), sample_ids.tolist(), image_hashes.tolist())
    ):
        if (
            not is_nih_image_filename(sample_id)
            or nih_patient_id(sample_id) != patient_id
        ):
            raise ValueError(
                "decision-bound patient/sample identity is invalid at "
                f"row {index}"
            )
        _require_sha256(image_hash, f"decision-bound image_sha256[{index}]")
        rows[sample_id] = {
            "patient_id": patient_id,
            "truth": int(truth[index, 0]),
            "raw_score": float(scores[index, 0]),
        }
    return rows


def _require_exact_int(value: Any, expected: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{field} must be {expected}, got {value!r}")


def _read_json_object(path: str | Path, description: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse {description} {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return resolved, data


def _declared_file_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def load_source_metadata_provenance(
    path: str | Path,
    *,
    cohort: str,
    official_train_val_manifest: str | Path,
    official_test_manifest: str | Path,
) -> dict[str, Any]:
    """Verify the immutable NIH/Google source bundle and its fetch receipt.

    The JSON receipt alone is not trusted: every declared file is opened and
    re-hashed, the two split files must be the exact pinned bytes supplied to the
    benchmark, and the fetcher's semantic validation counts must match the locked
    source contract.
    """

    if cohort not in {"development", "test"}:
        raise ValueError("cohort must be 'development' or 'test'")
    resolved, data = _read_json_object(path, "source metadata provenance")
    if data.get("artifact_type") != _SOURCE_METADATA_ARTIFACT_TYPE:
        raise ValueError(
            "source metadata provenance artifact_type must be "
            f"{_SOURCE_METADATA_ARTIFACT_TYPE!r}"
        )
    if data.get("schema_version") != 1:
        raise ValueError(
            "unsupported source metadata provenance schema_version; expected 1"
        )

    sources = _require_object(data.get("sources"), "source metadata sources")
    nih_source = _require_object(
        sources.get("nih_metadata"), "source metadata nih_metadata"
    )
    if (
        nih_source.get("dataset") != YEIGEN_DATASET
        or nih_source.get("revision") != YEIGEN_REVISION
    ):
        raise ValueError("NIH metadata source dataset/revision is not the pinned source")
    google_source = _require_object(
        sources.get("google_expert_labels"),
        "source metadata google_expert_labels",
    )
    if (
        google_source.get("repository") != "mlmed/torchxrayvision"
        or google_source.get("revision") != TORCHXRAYVISION_COMMIT
    ):
        raise ValueError(
            "Google expert-label source repository/revision is not the pinned source"
        )

    files = _require_object(data.get("files"), "source metadata files")
    if set(files) != set(PINNED_FILES):
        raise ValueError(
            "source metadata files must contain exactly the pinned NIH/Google bundle"
        )
    checked_files: dict[str, dict[str, Any]] = {}
    for name, spec in PINNED_FILES.items():
        record = _require_object(files.get(name), f"source metadata file {name}")
        expected_hash = _require_sha256(spec.get("sha256"), f"pinned {name} sha256")
        if record.get("sha256") != expected_hash:
            raise ValueError(f"source metadata {name} SHA-256 is not pinned")
        if record.get("url") != spec.get("url"):
            raise ValueError(f"source metadata {name} URL is not pinned")
        declared_path = _declared_file_path(
            record.get("path"), f"source metadata {name}.path"
        )
        actual_hash = sha256_file(declared_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"source metadata {name} file SHA-256 mismatch: "
                f"got {actual_hash}, expected {expected_hash}"
            )
        _require_exact_int(
            record.get("bytes"),
            declared_path.stat().st_size,
            f"source metadata {name}.bytes",
        )
        checked_files[name] = {
            "path": declared_path,
            "sha256": actual_hash,
            "bytes": declared_path.stat().st_size,
        }

    supplied_official = {
        "train_val_list.txt": Path(official_train_val_manifest).expanduser().resolve(),
        "test_list.txt": Path(official_test_manifest).expanduser().resolve(),
    }
    for name, supplied_path in supplied_official.items():
        if not supplied_path.is_file():
            raise FileNotFoundError(f"official NIH manifest not found: {supplied_path}")
        declared_path = checked_files[name]["path"]
        if supplied_path != declared_path:
            raise ValueError(
                f"--official-{name.removesuffix('_list.txt').replace('_', '-')} "
                "must be the exact file recorded by --source-metadata-provenance"
            )
        expected_hash = str(NIH_METADATA_FILES[name]["sha256"])
        if sha256_file(supplied_path) != expected_hash:
            raise ValueError(f"{name} is not the pinned official NIH split file")
        names = read_nih_filename_manifest(supplied_path)
        _require_exact_int(
            len(names),
            int(NIH_METADATA_FILES[name]["rows"]),
            f"{name} row count",
        )

    validation = _require_object(
        data.get("validation"), "source metadata validation"
    )
    nih_validation = _require_object(
        validation.get("nih_metadata"), "source metadata NIH validation"
    )
    expected_nih_rows = {
        name: int(spec["rows"]) for name, spec in NIH_METADATA_FILES.items()
    }
    if nih_validation.get("rows") != expected_nih_rows:
        raise ValueError("source metadata NIH validation row counts do not match")
    expected_union = (
        expected_nih_rows["train_val_list.txt"]
        + expected_nih_rows["test_list.txt"]
    )
    for field, expected in (
        ("split_overlap", 0),
        ("split_union_rows", expected_union),
        ("data_entry_unique_rows", expected_nih_rows["Data_Entry_2017.csv"]),
    ):
        _require_exact_int(
            nih_validation.get(field), expected, f"source metadata NIH {field}"
        )

    google_validation = _require_object(
        validation.get("google_expert_labels"),
        "source metadata Google validation",
    )
    _require_exact_int(
        google_validation.get("rows"),
        int(GOOGLE_EXPERT_LABEL_SPEC["rows"]),
        "source metadata Google rows",
    )
    _require_exact_int(
        google_validation.get("unique_images"),
        int(GOOGLE_EXPERT_LABEL_SPEC["rows"]),
        "source metadata Google unique_images",
    )
    if google_validation.get("split_rows") != dict(
        GOOGLE_EXPERT_LABEL_SPEC["split_rows"]
    ):
        raise ValueError("source metadata Google split counts do not match")
    if google_validation.get("label_counts") != dict(
        GOOGLE_EXPERT_LABEL_SPEC["label_counts"]
    ):
        raise ValueError("source metadata Google label counts do not match")
    if google_validation.get("all_four_findings_adjudicated_yes_no") is not True:
        raise ValueError("source metadata Google labels are not fully adjudicated")

    discrepancy = _require_object(
        data.get("known_source_discrepancy"),
        "source metadata known_source_discrepancy",
    )
    if discrepancy.get("pinned_mirror_split_rows") != dict(
        GOOGLE_EXPERT_LABEL_SPEC["split_rows"]
    ):
        raise ValueError("source metadata discrepancy does not identify pinned rows")
    if discrepancy.get("google_documented_split_rows") != dict(
        GOOGLE_EXPERT_LABEL_SPEC["google_documented_split_rows"]
    ):
        raise ValueError("source metadata discrepancy does not identify documented rows")
    expected_delta = (
        int(GOOGLE_EXPERT_LABEL_SPEC["split_rows"]["val"])
        - int(GOOGLE_EXPERT_LABEL_SPEC["google_documented_split_rows"]["val"])
    )
    _require_exact_int(
        discrepancy.get("validation_row_delta"),
        expected_delta,
        "source metadata validation_row_delta",
    )
    if cohort == "test":
        raise ValueError(
            "TEST DISABLED: schema-1 source provenance records an unresolved "
            "Google direct-release discrepancy; locked test evaluation is forbidden"
        )

    return {
        "path": resolved,
        "sha256": sha256_file(resolved),
        "google_expert_labels_path": checked_files[
            GOOGLE_EXPERT_LABEL_FILENAME
        ]["path"],
        "google_expert_labels_sha256": checked_files[
            GOOGLE_EXPERT_LABEL_FILENAME
        ]["sha256"],
        "official_train_val_manifest_sha256": checked_files[
            "train_val_list.txt"
        ]["sha256"],
        "official_test_manifest_sha256": checked_files["test_list.txt"]["sha256"],
        "verified": True,
        "locked_test_eligible": False,
    }


def load_manifest_metadata(
    path: str | Path,
    *,
    manifest: ExpertManifest,
    cohort: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the prepared canonical CSV to its full, pinned source audit."""

    resolved, data = _read_json_object(path, "canonical manifest metadata")
    if data.get("schema_version") != 1:
        raise ValueError(
            "unsupported canonical manifest metadata schema_version; expected 1"
        )
    if data.get("format") != _MANIFEST_METADATA_FORMAT:
        raise ValueError(
            f"canonical manifest metadata format must be {_MANIFEST_METADATA_FORMAT!r}"
        )
    expected_output_cohort = "development" if cohort == "development" else "test"
    if data.get("output_cohort") != expected_output_cohort:
        raise ValueError(
            "canonical manifest metadata output_cohort must match the benchmark "
            f"cohort ({expected_output_cohort!r})"
        )
    expected_targets = list(PHASE1_LABELS)
    declared_targets = data.get("targets")
    if declared_targets not in (
        expected_targets,
        [*expected_targets, "Fracture"],
    ):
        raise ValueError("canonical manifest metadata targets do not match phase 1")

    output = _require_object(
        data.get("output_csv"), "canonical manifest metadata output_csv"
    )
    if output.get("sha256") != manifest.manifest_sha256:
        raise ValueError(
            "canonical manifest metadata output_csv SHA-256 does not match --manifest"
        )
    output_path = _declared_file_path(
        output.get("path"), "canonical manifest metadata output_csv.path"
    )
    if output_path != manifest.path:
        raise ValueError(
            "canonical manifest metadata output_csv.path does not match --manifest"
        )
    expected_columns = ["filename", "patient_id", "split", *declared_targets]
    if output.get("columns") != expected_columns:
        raise ValueError("canonical manifest metadata output columns do not match")

    rows = _require_object(data.get("rows"), "canonical manifest metadata rows")
    expected_rows = {
        "total": len(manifest.sample_ids),
        "validation": len(manifest.sample_ids) if cohort == "development" else 0,
        "test": len(manifest.sample_ids) if cohort == "test" else 0,
    }
    if rows != expected_rows:
        raise ValueError(
            "canonical manifest metadata emitted row counts do not match --manifest"
        )
    expected_source_rows = {
        "total": int(GOOGLE_EXPERT_LABEL_SPEC["rows"]),
        "validation": int(GOOGLE_EXPERT_LABEL_SPEC["split_rows"]["val"]),
        "test": int(GOOGLE_EXPERT_LABEL_SPEC["split_rows"]["test"]),
    }
    if data.get("validated_source_rows") != expected_source_rows:
        raise ValueError(
            "canonical manifest metadata full source row counts do not match "
            "the pinned Google table"
        )

    sources = _require_object(data.get("sources"), "canonical manifest sources")
    if set(sources) != {"combined"}:
        raise ValueError(
            "canonical manifest must come from the pinned combined Google table"
        )
    combined = _require_object(
        sources.get("combined"), "canonical manifest combined source"
    )
    expected_google_hash = str(GOOGLE_EXPERT_LABEL_SPEC["sha256"])
    if (
        combined.get("sha256") != expected_google_hash
        or source_metadata.get("google_expert_labels_sha256")
        != expected_google_hash
    ):
        raise ValueError(
            "canonical manifest Google source SHA-256 is not the pinned table"
        )
    combined_path = _declared_file_path(
        combined.get("path"), "canonical manifest combined source path"
    )
    if combined_path != source_metadata.get("google_expert_labels_path"):
        raise ValueError(
            "canonical manifest combined source is not the file recorded by "
            "--source-metadata-provenance"
        )
    for field, expected in (
        ("source_rows", int(GOOGLE_EXPERT_LABEL_SPEC["rows"])),
        ("rows_emitted", int(GOOGLE_EXPERT_LABEL_SPEC["rows"])),
        ("rows_skipped_no_adjudicated_labels", 0),
    ):
        _require_exact_int(
            combined.get(field), expected, f"canonical manifest combined {field}"
        )

    source_load = load_google_nih_expert_csv(
        combined_path,
        expected_split=None,
        include_fracture=False,
    )
    expected_split = "validation" if cohort == "development" else "test"
    expected_source_rows_for_cohort = tuple(
        sorted(
            (row for row in source_load.rows if row.split == expected_split),
            key=lambda row: row.filename,
        )
    )
    expected_filenames = tuple(
        row.filename for row in expected_source_rows_for_cohort
    )
    if manifest.sample_ids != expected_filenames:
        raise ValueError(
            "canonical manifest filenames/order do not exactly match the pinned "
            f"Google {expected_split} rows"
        )
    for index, source_row in enumerate(expected_source_rows_for_cohort):
        expected_patient = source_row.patient_id
        expected_labels = [
            (
                -1
                if source_row.labels.get(label) is None
                else int(source_row.labels[label])
            )
            for label in PHASE1_LABELS
        ]
        if (
            manifest.patient_ids[index] != expected_patient
            or manifest.labels[index].tolist() != expected_labels
        ):
            raise ValueError(
                "canonical manifest row/labels do not match the pinned Google "
                f"source at {source_row.filename}"
            )

    official = _require_object(
        data.get("official_manifests"),
        "canonical manifest official_manifests",
    )
    for key, filename, source_hash_key in (
        (
            "train_val",
            "train_val_list.txt",
            "official_train_val_manifest_sha256",
        ),
        ("test", "test_list.txt", "official_test_manifest_sha256"),
    ):
        record = _require_object(
            official.get(key), f"canonical manifest official {key}"
        )
        expected_hash = str(NIH_METADATA_FILES[filename]["sha256"])
        if (
            record.get("sha256") != expected_hash
            or source_metadata.get(source_hash_key) != expected_hash
        ):
            raise ValueError(
                f"canonical manifest official {key} SHA-256 is not pinned"
            )
        _require_exact_int(
            record.get("images"),
            int(NIH_METADATA_FILES[filename]["rows"]),
            f"canonical manifest official {key} images",
        )

    reconciliation = _require_object(
        data.get("reconciliation"), "canonical manifest reconciliation"
    )
    expected_reconciliation = {
        "validation_rows_in_official_train_val": int(
            GOOGLE_EXPERT_LABEL_SPEC["split_rows"]["val"]
        ),
        "test_rows_in_official_test": int(
            GOOGLE_EXPERT_LABEL_SPEC["split_rows"]["test"]
        ),
        "cross_split_image_overlap": 0,
        "official_patient_overlap": 0,
    }
    if reconciliation != expected_reconciliation:
        raise ValueError("canonical manifest reconciliation audit does not match")

    return {
        "path": resolved,
        "sha256": sha256_file(resolved),
        "output_cohort": expected_output_cohort,
        "manifest_sha256": manifest.manifest_sha256,
        "google_expert_labels_sha256": expected_google_hash,
        "verified": True,
    }


def load_image_provenance(path: str | Path) -> ImageProvenance:
    """Load the explicit pixel-source declaration used by an expert run."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image provenance artifact not found: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse image provenance {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("image provenance must be a JSON object")
    if data.get("artifact_type") != "doctor_assistant.nih_image_provenance":
        raise ValueError(
            "image provenance artifact_type must be "
            "'doctor_assistant.nih_image_provenance'"
        )
    if data.get("schema_version") != 1:
        raise ValueError("unsupported image provenance schema_version; expected 1")
    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("image provenance source must be non-empty text")
    resolution = data.get("resolution")
    if not isinstance(resolution, dict) or set(resolution) != {"width", "height"}:
        raise ValueError(
            "image provenance resolution must contain exactly width and height"
        )
    width = resolution["width"]
    height = resolution["height"]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("image provenance width/height must be positive integers")
    original = data.get("original_nih_pixels")
    if not isinstance(original, bool):
        raise ValueError("image provenance original_nih_pixels must be boolean")
    if original:
        raise ValueError(
            "ORIGINAL NIH PIXEL EVIDENCE DISABLED: schema-1 provenance is a "
            "self-declaration and cannot prove NIH archive origin. Keep "
            "original_nih_pixels=false until a trusted original-pixel ingestion "
            "receipt with archive/source verification is implemented."
        )
    canonical_manifest = _require_object(
        data.get("canonical_manifest"), "image provenance canonical_manifest"
    )
    manifest_hash = _require_sha256(
        canonical_manifest.get("sha256"),
        "image provenance canonical_manifest.sha256",
    )
    cohort = canonical_manifest.get("cohort")
    if cohort not in {"development", "test"}:
        raise ValueError(
            "image provenance canonical_manifest.cohort must be development or test"
        )
    rows_selected = canonical_manifest.get("rows_selected")
    if (
        isinstance(rows_selected, bool)
        or not isinstance(rows_selected, int)
        or rows_selected <= 0
    ):
        raise ValueError(
            "image provenance canonical_manifest.rows_selected must be positive"
        )
    records = data.get("images")
    if not isinstance(records, list) or len(records) != rows_selected:
        raise ValueError(
            "image provenance images must contain exactly rows_selected records"
        )
    output_hashes: dict[str, str] = {}
    expert_splits: dict[str, str] = {}
    for index, value in enumerate(records):
        record = _require_object(value, f"image provenance images[{index}]")
        filename = record.get("filename")
        if not isinstance(filename, str) or not is_nih_image_filename(filename):
            raise ValueError(
                f"image provenance images[{index}].filename is not canonical NIH"
            )
        if filename in output_hashes:
            raise ValueError(f"image provenance contains duplicate image {filename}")
        split = record.get("expert_split")
        if split not in {"validation", "test"}:
            raise ValueError(
                f"image provenance images[{index}].expert_split is invalid"
            )
        output_hashes[filename] = _require_sha256(
            record.get("output_sha256"),
            f"image provenance images[{index}].output_sha256",
        )
        expert_splits[filename] = split

    if not original:
        if (
            data.get("development_only") is not True
            or data.get("official_or_final_evidence_allowed") is not False
        ):
            raise ValueError(
                "non-original image provenance must be development_only=true and "
                "official_or_final_evidence_allowed=false"
            )
        mirror = _require_object(data.get("mirror"), "image provenance mirror")
        if (
            mirror.get("dataset") != _DEVELOPMENT_IMAGE_DATASET
            or mirror.get("revision") != _DEVELOPMENT_IMAGE_REVISION
        ):
            raise ValueError(
                "non-original image provenance is not the pinned development mirror"
            )
        for index, record in enumerate(records):
            if record.get("mirror_revision") != _DEVELOPMENT_IMAGE_REVISION:
                raise ValueError(
                    f"image provenance images[{index}].mirror_revision is not pinned"
                )
    return ImageProvenance(
        path=resolved,
        artifact_sha256=sha256_file(resolved),
        source=source.strip(),
        width=width,
        height=height,
        original_nih_pixels=original,
        canonical_manifest_sha256=manifest_hash,
        cohort=cohort,
        rows_selected=rows_selected,
        output_sha256_by_filename=output_hashes,
        expert_split_by_filename=expert_splits,
    )


def inspect_query_pack(
    path: str | Path,
    *,
    expected_labels: Sequence[str],
    expected_prompts: Sequence[str],
    expected_query_set: str,
    expected_semantic_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate pack architecture and exact query identity before inference."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"KAD query pack not found: {resolved}")
    try:
        raw = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"could not load KAD query pack {resolved}: {exc}") from exc
    checked = preflight_kad512_query_pack(raw)
    labels = tuple(checked["labels"])
    prompts = tuple(checked["prompts"])
    if labels != tuple(expected_labels) or prompts != tuple(expected_prompts):
        raise ValueError(
            "KAD query pack identity mismatch. Expected labels/prompts "
            f"{list(zip(expected_labels, expected_prompts))}, got "
            f"{list(zip(labels, prompts))}."
        )
    source = dict(checked["source"])
    if source.get("checkpoint_sha256") != _KAD512_CHECKPOINT_SHA256:
        raise ValueError(
            "KAD query pack source checkpoint SHA-256 does not match the "
            "pinned official KAD-512 release"
        )
    if source.get("query_set") != expected_query_set:
        raise ValueError(
            f"KAD query pack source query_set must be {expected_query_set!r}"
        )
    semantic_hash = kad512_query_pack_semantic_sha256(checked)
    if expected_semantic_sha256 is not None:
        _require_sha256(
            expected_semantic_sha256,
            "expected_query_pack_semantic_sha256",
        )
        if semantic_hash != expected_semantic_sha256:
            raise ValueError(
                "KAD query pack semantic SHA-256 does not match the reviewed "
                "canonical phase-1 query features"
            )
    del checked, raw
    gc.collect()
    return {
        "path": resolved,
        "sha256": sha256_file(resolved),
        "semantic_sha256": semantic_hash,
        "labels": list(labels),
        "prompts": list(prompts),
        "prompt_set_sha256": prompt_set_sha256(labels, prompts),
        "source": source,
    }


def load_expert_manifest(
    manifest_path: str | Path,
    image_root: str | Path,
    *,
    cohort: str = "development",
    active_target: str = PHASE1_LABELS[0],
    official_train_val_manifest: str | Path | None = None,
    official_test_manifest: str | Path | None = None,
) -> ExpertManifest:
    """Load and exhaustively validate one canonical expert-label cohort."""

    if cohort not in {"development", "test"}:
        raise ValueError("cohort must be 'development' or 'test'")
    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(image_root).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"expert manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"image root is not a directory: {root}")

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"expert manifest is empty: {manifest}")
    header = rows[0]
    if not header or any(not name for name in header):
        raise ValueError("expert manifest has an empty column name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(
            "expert manifest has duplicate column name(s): " + ", ".join(duplicates)
        )
    missing = [name for name in _REQUIRED_MANIFEST_COLUMNS if name not in header]
    if missing:
        raise ValueError(
            "expert manifest is missing canonical column(s): " + ", ".join(missing)
        )
    noncanonical_label_headers = [
        name
        for name in header
        if name not in PHASE1_LABELS
        and _normalize_label_header(name)
        in {_normalize_label_header(label) for label in PHASE1_LABELS}
    ]
    if noncanonical_label_headers:
        raise ValueError(
            "ambiguous/noncanonical label column(s): "
            + ", ".join(noncanonical_label_headers)
            + "; use the exact canonical label names"
        )
    allowed = set(_REQUIRED_MANIFEST_COLUMNS) | _OPTIONAL_MANIFEST_COLUMNS
    unknown = [name for name in header if name not in allowed]
    if unknown:
        raise ValueError(
            "unrecognized canonical manifest column(s): " + ", ".join(unknown)
        )

    positions = {name: header.index(name) for name in header}
    sample_ids: list[str] = []
    patient_ids: list[str] = []
    image_paths: list[Path] = []
    image_relpaths: list[str] = []
    label_rows: list[list[int]] = []
    seen_samples: set[str] = set()
    seen_paths: set[Path] = set()
    requested_split = "validation" if cohort == "development" else "test"

    for line_number, values in enumerate(rows[1:], start=2):
        if not values or all(not value.strip() for value in values):
            continue
        if len(values) != len(header):
            raise ValueError(
                f"manifest line {line_number} has {len(values)} cells; "
                f"expected {len(header)}"
            )
        values = [value.strip() for value in values]
        sample_id = values[positions["filename"]]
        patient_id = values[positions["patient_id"]]
        row_split = values[positions["split"]]
        if not sample_id:
            raise ValueError(f"manifest line {line_number} has an empty filename")
        if not is_nih_image_filename(sample_id):
            raise ValueError(
                f"manifest line {line_number} has a noncanonical NIH filename: "
                f"{sample_id!r}"
            )
        if sample_id in seen_samples:
            raise ValueError(f"duplicate filename in manifest: {sample_id}")
        seen_samples.add(sample_id)
        if not patient_id:
            raise ValueError(f"manifest line {line_number} has an empty patient_id")
        expected_patient = nih_patient_id(sample_id)
        if patient_id != expected_patient:
            raise ValueError(
                f"manifest line {line_number} patient_id {patient_id!r} does not "
                f"match filename patient prefix {expected_patient!r}"
            )
        if row_split not in {"validation", "test"}:
            raise ValueError(
                f"manifest line {line_number} split must be 'validation' or 'test', "
                f"got {row_split!r}"
            )
        parsed_labels: list[int] = []
        for label in PHASE1_LABELS:
            raw = values[positions[label]]
            if raw not in {"", "0", "1"}:
                raise ValueError(
                    f"manifest line {line_number} label {label!r} must be blank, "
                    f"0, or 1, got {raw!r}"
                )
            parsed_labels.append(-1 if raw == "" else int(raw))
        if all(value == -1 for value in parsed_labels):
            raise ValueError(
                f"manifest line {line_number} has no adjudicated phase-1 labels"
            )
        if row_split != requested_split:
            continue

        resolved_image = (root / sample_id).resolve()
        try:
            canonical_relative = resolved_image.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"manifest line {line_number} image escapes --image-root: "
                f"{sample_id!r}"
            ) from exc
        if resolved_image in seen_paths:
            raise ValueError(
                f"manifest resolves more than one row to image: {canonical_relative}"
            )
        if not resolved_image.is_file():
            raise FileNotFoundError(
                f"manifest image is missing at line {line_number}: {resolved_image}"
            )

        seen_paths.add(resolved_image)
        sample_ids.append(sample_id)
        patient_ids.append(patient_id)
        image_paths.append(resolved_image)
        image_relpaths.append(canonical_relative.as_posix())
        label_rows.append(parsed_labels)

    if not sample_ids:
        raise ValueError(f"expert manifest contains no {cohort!r} rows")
    labels_array = np.asarray(label_rows, dtype=np.int8)
    for index, label in enumerate(PHASE1_LABELS):
        adjudicated = labels_array[:, index] >= 0
        positives = int((labels_array[:, index] == 1).sum())
        negatives = int((labels_array[:, index] == 0).sum())
        if label == active_target and (
            not adjudicated.any() or positives == 0 or negatives == 0
        ):
            raise ValueError(
                f"active expert target {label!r} is not scoreable: "
                f"{positives} positives, {negatives} negatives, and "
                f"{int((~adjudicated).sum())} non-adjudicated rows"
            )

    image_hashes = tuple(sha256_file(path) for path in image_paths)
    image_set_hash = _canonical_json_sha256(
        [
            {
                "sample_id": sample_id,
                "image_path": relpath,
                "sha256": digest,
            }
            for sample_id, relpath, digest in zip(
                sample_ids, image_relpaths, image_hashes
            )
        ]
    )
    official_train_val_hash = None
    official_test_hash = None
    official_membership_verified = False
    if (official_train_val_manifest is None) != (official_test_manifest is None):
        raise ValueError(
            "official NIH reconciliation requires both train_val_list.txt and "
            "test_list.txt"
        )
    if official_train_val_manifest is not None:
        train_val_path = Path(official_train_val_manifest).expanduser().resolve()
        test_path = Path(official_test_manifest).expanduser().resolve()
        if not train_val_path.is_file():
            raise FileNotFoundError(
                f"official NIH train_val manifest not found: {train_val_path}"
            )
        if not test_path.is_file():
            raise FileNotFoundError(
                f"official NIH test manifest not found: {test_path}"
            )
        reconciliation = reconcile_nih_official_manifests(
            sample_ids,
            read_nih_filename_manifest(train_val_path),
            read_nih_filename_manifest(test_path),
            require_complete=False,
            require_patient_disjoint=True,
        )
        if reconciliation.unlisted_available:
            examples = ", ".join(sorted(reconciliation.unlisted_available)[:3])
            raise ValueError(
                "expert manifest contains image(s) absent from the official NIH "
                f"manifests, including: {examples}"
            )
        wrong_partition = (
            reconciliation.available_test
            if cohort == "development"
            else reconciliation.available_train_val
        )
        if wrong_partition:
            examples = ", ".join(sorted(wrong_partition)[:3])
            raise ValueError(
                f"{cohort} expert manifest contains {len(wrong_partition)} image(s) "
                f"from the wrong official NIH partition, including: {examples}"
            )
        expected_members = (
            reconciliation.available_train_val
            if cohort == "development"
            else reconciliation.available_test
        )
        if expected_members != frozenset(sample_ids):
            raise RuntimeError(
                "official NIH reconciliation did not account for every expert row"
            )
        official_train_val_hash = sha256_file(train_val_path)
        official_test_hash = sha256_file(test_path)
        official_membership_verified = True

    return ExpertManifest(
        path=manifest,
        image_root=root,
        cohort=cohort,
        sample_ids=tuple(sample_ids),
        patient_ids=tuple(patient_ids),
        image_paths=tuple(image_paths),
        image_relpaths=tuple(image_relpaths),
        labels=labels_array,
        manifest_sha256=sha256_file(manifest),
        image_sha256=image_hashes,
        image_set_sha256=image_set_hash,
        official_train_val_manifest_sha256=official_train_val_hash,
        official_test_manifest_sha256=official_test_hash,
        official_membership_verified=official_membership_verified,
    )


def _normalize_label_header(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _require_int_at_least(value: Any, minimum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_rate(value: Any, field: str) -> float:
    parsed = _finite_float(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return parsed


def _require_close(actual: Any, expected: float, field: str) -> float:
    parsed = _finite_float(actual, field)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} is inconsistent with the decision evidence")
    return parsed


def _wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson interval counts")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_partition_record(
    value: Any,
    *,
    role: str,
) -> dict[str, Any]:
    record = _require_object(value, f"decision patient_partition.{role}")
    sample_ids = record.get("sample_ids")
    patient_ids = record.get("patient_ids")
    if (
        not isinstance(sample_ids, list)
        or not all(isinstance(item, str) for item in sample_ids)
        or sample_ids != sorted(sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError(
            f"decision patient_partition.{role}.sample_ids must be sorted unique strings"
        )
    if (
        not isinstance(patient_ids, list)
        or not all(isinstance(item, str) and item for item in patient_ids)
        or patient_ids != sorted(patient_ids)
        or len(set(patient_ids)) != len(patient_ids)
    ):
        raise ValueError(
            f"decision patient_partition.{role}.patient_ids must be sorted unique strings"
        )
    _require_exact_int(
        record.get("images"),
        len(sample_ids),
        f"decision patient_partition.{role}.images",
    )
    _require_exact_int(
        record.get("patients"),
        len(patient_ids),
        f"decision patient_partition.{role}.patients",
    )
    if not sample_ids or not patient_ids:
        raise ValueError(f"decision patient_partition.{role} must be non-empty")
    positives = _require_int_at_least(
        record.get("positives"),
        0,
        f"decision patient_partition.{role}.positives",
    )
    negatives = _require_int_at_least(
        record.get("negatives"),
        0,
        f"decision patient_partition.{role}.negatives",
    )
    if positives + negatives != len(sample_ids):
        raise ValueError(
            f"decision patient_partition.{role} support does not equal its images"
        )
    declared_patients = set(patient_ids)
    members: list[dict[str, str]] = []
    observed_patients: set[str] = set()
    for index, sample_id in enumerate(sample_ids):
        if not is_nih_image_filename(sample_id):
            raise ValueError(
                f"decision patient_partition.{role}.sample_ids[{index}] is not NIH"
            )
        patient_id = nih_patient_id(sample_id)
        if patient_id not in declared_patients:
            raise ValueError(
                f"decision patient_partition.{role} sample/patient membership mismatch"
            )
        observed_patients.add(patient_id)
        members.append({"sample_id": sample_id, "patient_id": patient_id})
    if observed_patients != declared_patients:
        raise ValueError(
            f"decision patient_partition.{role} contains patient IDs without samples"
        )
    members.sort(key=lambda row: (row["patient_id"], row["sample_id"]))
    expected_membership_hash = _canonical_json_sha256(members)
    membership_hash = _require_sha256(
        record.get("membership_sha256"),
        f"decision patient_partition.{role}.membership_sha256",
    )
    if membership_hash != expected_membership_hash:
        raise ValueError(
            f"decision patient_partition.{role}.membership_sha256 is inconsistent"
        )
    samples_per_patient = {
        patient_id: sum(
            nih_patient_id(sample_id) == patient_id for sample_id in sample_ids
        )
        for patient_id in patient_ids
    }
    return {
        "sample_ids": set(sample_ids),
        "patient_ids": declared_patients,
        "membership_sha256": membership_hash,
        "samples_per_patient": samples_per_patient,
        "images": len(sample_ids),
        "patients": len(patient_ids),
        "positives": positives,
        "negatives": negatives,
    }


def _validate_probability_metrics(
    value: Any,
    *,
    field: str,
) -> None:
    metrics = _require_object(value, field)
    if set(metrics) != {"brier", "log_loss"}:
        raise ValueError(f"{field} must contain exactly brier and log_loss")
    brier = _finite_float(metrics.get("brier"), f"{field}.brier")
    log_loss = _finite_float(metrics.get("log_loss"), f"{field}.log_loss")
    if not 0.0 <= brier <= 1.0 or log_loss < 0.0:
        raise ValueError(f"{field} contains invalid probability metrics")


def _validate_completed_decision_evidence(
    data: Mapping[str, Any],
    *,
    active_target: str,
    development_prediction_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, float]:
    """Derive completion from schema-3 evidence instead of trusting flags."""

    if data.get("status") != "complete_research_decision":
        raise ValueError("decision artifact status is not a completed research decision")
    diagnostics = _require_object(data.get("diagnostics"), "decision diagnostics")
    if diagnostics.get("deployable") is not True or diagnostics.get("reasons") != []:
        raise ValueError("decision diagnostics do not establish a complete policy")
    source_analysis = _require_object(
        data.get("source_benchmark_analysis"),
        "decision source_benchmark_analysis",
    )
    if (
        source_analysis.get("analysis_deferred") is not True
        or source_analysis.get("metrics_scope")
        != "development_analysis_deferred_until_patient_partition"
    ):
        raise ValueError(
            "decision artifact did not defer analysis until patient roles were frozen"
        )
    calibration_runtime = _require_object(
        data.get("calibration_runtime"),
        "decision calibration_runtime",
    )
    _require_exact_int(
        calibration_runtime.get("schema_version"),
        1,
        "decision calibration runtime schema",
    )
    for field in ("python", "numpy", "scipy", "scikit_learn"):
        if (
            not isinstance(calibration_runtime.get(field), str)
            or not calibration_runtime[field]
        ):
            raise ValueError(
                f"decision calibration_runtime.{field} must be non-empty text"
            )
    _require_sha256(
        calibration_runtime.get("code_sha256"),
        "decision calibration runtime code_sha256",
    )
    calibration_runtime_hash = _require_sha256(
        data.get("calibration_runtime_sha256"),
        "decision calibration_runtime_sha256",
    )
    if calibration_runtime_hash != _canonical_json_sha256(calibration_runtime):
        raise ValueError("decision calibration runtime hash is inconsistent")
    pixel_evidence = _require_object(
        data.get("development_pixel_evidence"),
        "decision development_pixel_evidence",
    )
    if pixel_evidence != {
        "original_nih_pixels": True,
        "benchmark_evidence_status": (
            "development_only_do_not_report_as_test_performance"
        ),
        "acceptance_eligible": True,
    }:
        raise ValueError(
            "decision acceptance requires original NIH development pixels"
        )

    requirements = _require_object(data.get("requirements"), "decision requirements")
    sensitivity_target = _require_rate(
        requirements.get("sensitivity_target"),
        "decision requirements.sensitivity_target",
    )
    specificity_floor = _require_rate(
        requirements.get("specificity_floor"),
        "decision requirements.specificity_floor",
    )
    _require_close(
        sensitivity_target,
        _MIN_SENSITIVITY_TARGET,
        "decision sensitivity target",
    )
    _require_close(
        specificity_floor,
        _MIN_SPECIFICITY_FLOOR,
        "decision specificity floor",
    )
    min_calibration_positives = _require_int_at_least(
        requirements.get("minimum_calibration_positives"),
        1,
        "decision minimum_calibration_positives",
    )
    min_calibration_negatives = _require_int_at_least(
        requirements.get("minimum_calibration_negatives"),
        1,
        "decision minimum_calibration_negatives",
    )
    min_threshold_positives = _require_int_at_least(
        requirements.get("minimum_threshold_positives"),
        1,
        "decision minimum_threshold_positives",
    )
    min_threshold_negatives = _require_int_at_least(
        requirements.get("minimum_threshold_negatives"),
        1,
        "decision minimum_threshold_negatives",
    )
    if (
        min_calibration_positives,
        min_calibration_negatives,
        min_threshold_positives,
        min_threshold_negatives,
    ) != _PROTOCOL_MINIMUM_SUPPORT:
        raise ValueError(
            "decision calibration/threshold support minima are not the frozen protocol"
        )
    _require_exact_int(
        requirements.get("minimum_acceptance_positive_patients"),
        _MIN_ACCEPTANCE_POSITIVE_PATIENTS,
        "decision minimum_acceptance_positive_patients",
    )
    _require_exact_int(
        requirements.get("minimum_acceptance_negative_patients"),
        _MIN_ACCEPTANCE_NEGATIVE_PATIENTS,
        "decision minimum_acceptance_negative_patients",
    )
    if (
        requirements.get("positive_platt_slope_required") is not True
        or requirements.get("acceptance_requires_lower_confidence_bounds") is not True
        or requirements.get("acceptance_interval_method")
        != "two_sided_wilson_score"
    ):
        raise ValueError("decision artifact weakens required completion gates")
    _require_close(
        requirements.get("acceptance_confidence_level"),
        0.95,
        "decision acceptance_confidence_level",
    )

    partition = _require_object(
        data.get("patient_partition"),
        "decision patient_partition",
    )
    if (
        partition.get("method")
        != "deterministic_label_support_aware_patient_hash_search_v2"
        or partition.get("scope")
        != "endpoint_specific_frozen_across_all_candidates"
        or partition.get("active_target") != active_target
        or partition.get("support_feasible") is not True
    ):
        raise ValueError("decision artifact does not contain a feasible four-role split")
    split_seed = _require_int_at_least(
        partition.get("seed"),
        0,
        "decision patient_partition.seed",
    )
    attempts = _require_int_at_least(
        partition.get("attempts_evaluated"),
        1,
        "decision patient_partition.attempts_evaluated",
    )
    if split_seed != _PROTOCOL_PARTITION_SEED:
        raise ValueError(
            "decision patient partition seed is not the frozen protocol seed"
        )
    if attempts != _PROTOCOL_SPLIT_ATTEMPTS:
        raise ValueError(
            "decision patient partition search count is not the frozen protocol count"
        )
    selected_attempt = _require_int_at_least(
        partition.get("selected_attempt"),
        0,
        "decision patient_partition.selected_attempt",
    )
    if selected_attempt >= attempts:
        raise ValueError("decision selected split attempt exceeds attempted search")
    fractions = _require_object(
        partition.get("fractions"),
        "decision patient_partition.fractions",
    )
    role_names = (
        "model_selection",
        "calibration",
        "threshold_selection",
        "acceptance",
    )
    if set(fractions) != set(role_names):
        raise ValueError("decision patient fractions must contain exactly four roles")
    parsed_fractions = [
        _finite_float(fractions[role], f"decision patient fraction {role}")
        for role in role_names
    ]
    if any(value <= 0.0 for value in parsed_fractions) or not math.isclose(
        sum(parsed_fractions),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("decision patient fractions must be positive and sum to 1")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(parsed_fractions, _PROTOCOL_ROLE_FRACTIONS)
    ):
        raise ValueError("decision patient fractions are not the frozen protocol")
    records = _require_object(
        partition.get("partitions"),
        "decision patient_partition.partitions",
    )
    if set(records) != set(role_names):
        raise ValueError("decision artifact must contain exactly four patient roles")
    checked_roles = {
        role: _validate_partition_record(records[role], role=role)
        for role in role_names
    }
    for left_index, left in enumerate(role_names):
        for right in role_names[left_index + 1 :]:
            if checked_roles[left]["patient_ids"] & checked_roles[right]["patient_ids"]:
                raise ValueError("decision patient roles overlap")
            if checked_roles[left]["sample_ids"] & checked_roles[right]["sample_ids"]:
                raise ValueError("decision sample roles overlap")
    expected_overlap_keys = {
        "model_selection_calibration",
        "model_selection_threshold_selection",
        "model_selection_acceptance",
        "calibration_threshold_selection",
        "calibration_acceptance",
        "threshold_selection_acceptance",
    }
    overlap = _require_object(
        partition.get("pairwise_patient_overlap"),
        "decision patient_partition.pairwise_patient_overlap",
    )
    if set(overlap) != expected_overlap_keys or any(
        isinstance(value, bool) or value != 0 for value in overlap.values()
    ):
        raise ValueError("decision pairwise patient-overlap evidence is invalid")
    adjudicated_images = sum(
        checked_roles[role]["images"] for role in role_names
    )
    _require_exact_int(
        partition.get("active_target_adjudicated_images"),
        adjudicated_images,
        "decision active_target_adjudicated_images",
    )
    _require_int_at_least(
        partition.get("active_target_non_adjudicated_images"),
        0,
        "decision active_target_non_adjudicated_images",
    )

    ranking = _require_object(
        data.get("model_selection_ranking"),
        "decision model_selection_ranking",
    )
    model_record = checked_roles["model_selection"]
    for field in ("images", "patients", "positives", "negatives"):
        _require_exact_int(
            ranking.get(field),
            model_record[field],
            f"decision model_selection_ranking.{field}",
        )
    if (
        ranking.get("active_target") != active_target
        or ranking.get("role") != "candidate_ranking_only_never_fit_or_threshold"
    ):
        raise ValueError("decision model-selection ranking role/target is invalid")
    if (
        ranking.get("membership_sha256")
        != model_record["membership_sha256"]
    ):
        raise ValueError("decision model-selection ranking membership is not bound")
    _require_rate(ranking.get("auroc"), "decision model-selection AUROC")
    _require_rate(ranking.get("auprc"), "decision model-selection AUPRC")
    ranking_confidence = _require_object(
        ranking.get("confidence_intervals"),
        "decision model-selection confidence_intervals",
    )
    if (
        ranking_confidence.get("method")
        != "patient_clustered_percentile_bootstrap"
    ):
        raise ValueError("decision model-selection interval method is invalid")
    _require_close(
        ranking_confidence.get("confidence_level"),
        0.95,
        "decision model-selection confidence level",
    )
    requested_replicates = _require_int_at_least(
        ranking_confidence.get("requested_replicates"),
        1,
        "decision model-selection requested_replicates",
    )
    if requested_replicates != _PROTOCOL_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "decision bootstrap replicate count is not the frozen protocol"
        )
    _require_exact_int(
        ranking_confidence.get("patients_resampled_per_replicate"),
        model_record["patients"],
        "decision model-selection patients_resampled_per_replicate",
    )
    _require_exact_int(
        ranking_confidence.get("seed"),
        split_seed,
        "decision model-selection bootstrap seed",
    )
    successful = _require_object(
        ranking_confidence.get("successful_replicates"),
        "decision model-selection successful_replicates",
    )
    intervals = _require_object(
        ranking_confidence.get("intervals"),
        "decision model-selection intervals",
    )
    if set(successful) != {"auroc", "auprc"} or set(intervals) != {
        "auroc",
        "auprc",
    }:
        raise ValueError("decision model-selection bootstrap fields are incomplete")
    for metric in ("auroc", "auprc"):
        completed = _require_int_at_least(
            successful[metric],
            1,
            f"decision model-selection successful {metric} replicates",
        )
        if completed > requested_replicates:
            raise ValueError("decision bootstrap successes exceed requested replicates")
        interval = _require_object(
            intervals[metric],
            f"decision model-selection {metric} interval",
        )
        lower = _require_rate(
            interval.get("lower"),
            f"decision model-selection {metric} lower",
        )
        upper = _require_rate(
            interval.get("upper"),
            f"decision model-selection {metric} upper",
        )
        if lower > upper:
            raise ValueError("decision model-selection interval is inverted")

    calibration = _require_object(data.get("calibrator"), "decision calibrator")
    if (
        calibration.get("type") != "per_label_platt_logit"
        or calibration.get("input") != "clipped_raw_score_logit"
        or calibration.get("fit_role") != "calibration_only"
    ):
        raise ValueError("decision calibrator schema/fit role is invalid")
    if (
        calibration.get("fit_membership_sha256")
        != checked_roles["calibration"]["membership_sha256"]
    ):
        raise ValueError("decision calibrator fit membership is not bound")
    _require_close(
        calibration.get("clip_epsilon"),
        _CALIBRATOR_CLIP_EPSILON,
        "decision calibrator.clip_epsilon",
    )
    regularization = _require_object(
        calibration.get("regularization"),
        "decision calibrator.regularization",
    )
    if (
        regularization.get("implementation")
        != "sklearn.linear_model.LogisticRegression"
    ):
        raise ValueError("decision calibrator implementation is invalid")
    if _finite_float(regularization.get("C"), "decision calibrator C") <= 0.0:
        raise ValueError("decision calibrator C must be positive")
    _require_int_at_least(
        calibration.get("optimizer_iterations"),
        1,
        "decision calibrator.optimizer_iterations",
    )
    parameters = _require_object(
        calibration.get("parameters"),
        "decision calibrator.parameters",
    )
    if set(parameters) != {active_target}:
        raise ValueError("decision calibrator parameters must contain active target only")
    parameter_row = _require_object(
        parameters[active_target],
        f"decision calibrator parameters {active_target}",
    )
    if set(parameter_row) != {"slope", "intercept"}:
        raise ValueError("decision calibrator parameters are incomplete")
    slope = _finite_float(
        parameter_row.get("slope"),
        f"{active_target} calibrator slope",
    )
    intercept = _finite_float(
        parameter_row.get("intercept"),
        f"{active_target} calibrator intercept",
    )
    if slope <= 0.0:
        raise ValueError(f"{active_target} calibrator slope must be positive")
    metrics = _require_object(
        calibration.get("metrics"),
        "decision calibrator.metrics",
    )
    if set(metrics) != {active_target}:
        raise ValueError("decision calibrator metrics must contain active target only")
    target_metrics = _require_object(
        metrics[active_target],
        f"decision calibrator metrics {active_target}",
    )
    if set(target_metrics) != {"calibration", "threshold_selection"}:
        raise ValueError(
            "decision calibrator metrics must exclude untouched acceptance patients"
        )
    for role in ("calibration", "threshold_selection"):
        role_metrics = _require_object(
            target_metrics[role],
            f"decision calibrator metrics {active_target}.{role}",
        )
        _require_exact_int(
            role_metrics.get("images"),
            checked_roles[role]["images"],
            f"decision calibrator metrics {role}.images",
        )
        _validate_probability_metrics(
            role_metrics.get("before"),
            field=f"decision calibrator metrics {role}.before",
        )
        _validate_probability_metrics(
            role_metrics.get("after"),
            field=f"decision calibrator metrics {role}.after",
        )

    raw_thresholds = _require_object(data.get("thresholds"), "decision thresholds")
    candidates = _require_object(
        data.get("candidate_thresholds"),
        "decision candidate_thresholds",
    )
    if set(raw_thresholds) != {active_target} or set(candidates) != {active_target}:
        raise ValueError("decision thresholds must contain the active target only")
    threshold = _require_rate(
        raw_thresholds[active_target],
        f"{active_target} threshold",
    )
    _require_close(
        candidates[active_target],
        threshold,
        f"{active_target} candidate threshold",
    )
    threshold_selection = _require_object(
        data.get("threshold_selection"),
        "decision threshold_selection",
    )
    if (
        threshold_selection.get("active_target") != active_target
        or threshold_selection.get("fit_role") != "threshold_selection_only"
        or threshold_selection.get("method")
        != "most_specific_observed_probability_meeting_constraints"
    ):
        raise ValueError("decision threshold-selection schema/role is invalid")
    if (
        threshold_selection.get("fit_membership_sha256")
        != checked_roles["threshold_selection"]["membership_sha256"]
    ):
        raise ValueError("decision threshold-selection fit membership is not bound")
    _require_close(
        threshold_selection.get("sensitivity_target"),
        sensitivity_target,
        "decision threshold-selection sensitivity_target",
    )
    _require_close(
        threshold_selection.get("specificity_floor"),
        specificity_floor,
        "decision threshold-selection specificity_floor",
    )
    _require_exact_int(
        threshold_selection.get("min_positives"),
        min_threshold_positives,
        "decision threshold-selection min_positives",
    )
    _require_exact_int(
        threshold_selection.get("min_negatives"),
        min_threshold_negatives,
        "decision threshold-selection min_negatives",
    )
    _require_close(
        threshold_selection.get("candidate_threshold"),
        threshold,
        "decision threshold-selection candidate_threshold",
    )
    selection_result = _require_object(
        threshold_selection.get("result"),
        "decision threshold-selection result",
    )
    if set(selection_result) != {
        "label",
        "threshold",
        "sensitivity",
        "specificity",
        "positives",
        "negatives",
        "supported",
        "meets_constraints",
    }:
        raise ValueError("decision threshold-selection result schema is invalid")
    if (
        selection_result.get("label") != active_target
        or selection_result.get("supported") is not True
        or selection_result.get("meets_constraints") is not True
    ):
        raise ValueError("decision threshold-selection result did not pass")
    _require_close(
        selection_result.get("threshold"),
        threshold,
        "decision threshold-selection result.threshold",
    )
    selection_sensitivity = _require_rate(
        selection_result.get("sensitivity"),
        "decision threshold-selection sensitivity",
    )
    selection_specificity = _require_rate(
        selection_result.get("specificity"),
        "decision threshold-selection specificity",
    )
    selection_positives = _require_int_at_least(
        selection_result.get("positives"),
        min_threshold_positives,
        "decision threshold-selection positives",
    )
    selection_negatives = _require_int_at_least(
        selection_result.get("negatives"),
        min_threshold_negatives,
        "decision threshold-selection negatives",
    )
    if (
        selection_positives != checked_roles["threshold_selection"]["positives"]
        or selection_negatives != checked_roles["threshold_selection"]["negatives"]
        or selection_sensitivity < sensitivity_target
        or selection_specificity < specificity_floor
    ):
        raise ValueError("decision threshold-selection evidence is inconsistent")

    acceptance = _require_object(
        threshold_selection.get("untouched_study_level_acceptance"),
        "decision untouched acceptance",
    )
    if (
        acceptance.get("role")
        != "untouched_acceptance_only_never_tune_or_fit"
        or acceptance.get("unit") != "study"
        or acceptance.get("patient_weighting")
        != (
            "one_hash_selected_positive_and_one_hash_selected_negative_"
            "adjudicated_study_per_patient"
        )
        or acceptance.get("truth_definition") != "per_adjudicated_radiograph"
        or acceptance.get("prediction_definition")
        != "study_probability_meets_frozen_threshold"
        or acceptance.get("interval") != "two_sided_wilson_score"
    ):
        raise ValueError("decision untouched-acceptance schema/role is invalid")
    acceptance_record = checked_roles["acceptance"]
    if (
        acceptance.get("evaluation_membership_sha256")
        != acceptance_record["membership_sha256"]
    ):
        raise ValueError("decision untouched acceptance membership is not bound")
    _require_close(
        acceptance.get("frozen_threshold"),
        threshold,
        "decision untouched acceptance frozen_threshold",
    )
    _require_close(
        acceptance.get("confidence_level"),
        0.95,
        "decision untouched acceptance confidence_level",
    )

    count_fields = {
        "studies",
        "patients",
        "positive_studies",
        "negative_studies",
        "positive_patients",
        "negative_patients",
        "selected_positive_studies",
        "selected_negative_studies",
        "true_positives",
        "false_negatives",
        "true_negatives",
        "false_positives",
    }
    counts = _require_object(
        acceptance.get("counts"),
        "decision untouched acceptance counts",
    )
    if set(counts) != count_fields:
        raise ValueError("decision untouched acceptance counts schema is invalid")
    parsed_counts = {
        field: _require_int_at_least(
            counts.get(field),
            0,
            f"decision untouched acceptance {field}",
        )
        for field in count_fields
    }

    outcomes = acceptance.get("study_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError(
            "decision untouched acceptance study_outcomes must be non-empty"
        )
    expected_outcome_fields = {
        "patient_id",
        "sample_id",
        "truth",
        "predicted_positive",
    }
    checked_outcomes: list[dict[str, Any]] = []
    for index, value in enumerate(outcomes):
        outcome = _require_object(
            value,
            f"decision untouched acceptance study_outcomes[{index}]",
        )
        if set(outcome) != expected_outcome_fields:
            raise ValueError("decision untouched acceptance outcome schema is invalid")
        patient_id = outcome.get("patient_id")
        sample_id = outcome.get("sample_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("decision untouched acceptance patient_id is invalid")
        if (
            not isinstance(sample_id, str)
            or sample_id not in acceptance_record["sample_ids"]
            or not is_nih_image_filename(sample_id)
            or nih_patient_id(sample_id) != patient_id
        ):
            raise ValueError(
                "decision untouched acceptance outcome is not in its partition"
            )
        truth_value = outcome.get("truth")
        if (
            isinstance(truth_value, bool)
            or not isinstance(truth_value, int)
            or truth_value not in (0, 1)
        ):
            raise ValueError("decision untouched acceptance outcome truth is invalid")
        predicted_positive = outcome.get("predicted_positive")
        if not isinstance(predicted_positive, bool):
            raise ValueError(
                "decision untouched acceptance prediction must be boolean"
            )
        bound_row = development_prediction_rows.get(sample_id)
        if bound_row is None:
            raise ValueError(
                "decision untouched acceptance sample is absent from the bound "
                "development predictions"
            )
        if (
            bound_row.get("patient_id") != patient_id
            or bound_row.get("truth") != truth_value
        ):
            raise ValueError(
                "decision untouched acceptance truth/patient does not match the "
                "bound development predictions"
            )
        raw_score = _finite_float(
            bound_row.get("raw_score"),
            "decision-bound acceptance raw score",
        )
        clipped_score = min(
            max(raw_score, _CALIBRATOR_CLIP_EPSILON),
            1.0 - _CALIBRATOR_CLIP_EPSILON,
        )
        raw_logit = math.log(clipped_score / (1.0 - clipped_score))
        calibrated_probability = 1.0 / (
            1.0
            + math.exp(
                -min(max(raw_logit * slope + intercept, -80.0), 80.0)
            )
        )
        if predicted_positive != (calibrated_probability >= threshold):
            raise ValueError(
                "decision untouched acceptance prediction is inconsistent with "
                "the bound raw score, calibrator, and threshold"
            )
        checked_outcomes.append(
            {
                "patient_id": patient_id,
                "sample_id": sample_id,
                "truth": truth_value,
                "predicted_positive": predicted_positive,
            }
        )
    expected_outcome_order = sorted(
        checked_outcomes,
        key=lambda row: (row["patient_id"], row["sample_id"]),
    )
    sample_ids = [row["sample_id"] for row in checked_outcomes]
    if checked_outcomes != expected_outcome_order or (
        len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError(
            "decision untouched acceptance outcomes must be sorted and unique"
        )
    if set(sample_ids) != acceptance_record["sample_ids"]:
        raise ValueError(
            "decision untouched acceptance outcomes do not cover its partition"
        )

    selection = _require_object(
        acceptance.get("study_selection"),
        "decision untouched acceptance study_selection",
    )
    if set(selection) != {"method", "hash_domain", "seed", "positive", "negative"}:
        raise ValueError("decision untouched acceptance study selection is incomplete")
    if (
        selection.get("method") != "minimum_sha256_independent_of_scores_v1"
        or selection.get("hash_domain") != "kad-phase1-acceptance-study-v1"
    ):
        raise ValueError("decision untouched acceptance study selection is invalid")
    selection_seed = _require_int_at_least(
        selection.get("seed"),
        0,
        "decision untouched acceptance study-selection seed",
    )
    _require_exact_int(
        selection_seed,
        split_seed + 1_000_003,
        "decision untouched acceptance derived study-selection seed",
    )

    outcomes_by_patient: dict[str, list[dict[str, Any]]] = {}
    outcomes_by_sample: dict[str, dict[str, Any]] = {}
    for outcome in checked_outcomes:
        outcomes_by_patient.setdefault(outcome["patient_id"], []).append(outcome)
        outcomes_by_sample[outcome["sample_id"]] = outcome

    selection_domain = "kad-phase1-acceptance-study-v1"

    def derive_selected(truth_value: int) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        for patient_id in sorted(outcomes_by_patient):
            candidates = [
                row
                for row in outcomes_by_patient[patient_id]
                if row["truth"] == truth_value
            ]
            if not candidates:
                continue
            chosen = min(
                candidates,
                key=lambda row: (
                    hashlib.sha256(
                        (
                            f"{selection_domain}:{selection_seed}:{truth_value}:"
                            f"{patient_id}:{row['sample_id']}"
                        ).encode("utf-8")
                    ).digest(),
                    row["sample_id"],
                ),
            )
            selected.append(
                {
                    "patient_id": patient_id,
                    "sample_id": chosen["sample_id"],
                }
            )
        return selected

    expected_positive_selection = derive_selected(1)
    expected_negative_selection = derive_selected(0)
    if (
        selection.get("positive") != expected_positive_selection
        or selection.get("negative") != expected_negative_selection
    ):
        raise ValueError(
            "decision untouched acceptance selected studies are inconsistent"
        )
    positive_patients = len(expected_positive_selection)
    negative_patients = len(expected_negative_selection)
    true_positives = int(
        sum(
            outcomes_by_sample[row["sample_id"]]["predicted_positive"]
            for row in expected_positive_selection
        )
    )
    false_negatives = positive_patients - true_positives
    false_positives = int(
        sum(
            outcomes_by_sample[row["sample_id"]]["predicted_positive"]
            for row in expected_negative_selection
        )
    )
    true_negatives = negative_patients - false_positives
    positive_studies = int(sum(row["truth"] == 1 for row in checked_outcomes))
    negative_studies = len(checked_outcomes) - positive_studies
    derived_counts = {
        "studies": len(checked_outcomes),
        "patients": len(outcomes_by_patient),
        "positive_studies": positive_studies,
        "negative_studies": negative_studies,
        "positive_patients": positive_patients,
        "negative_patients": negative_patients,
        "selected_positive_studies": positive_patients,
        "selected_negative_studies": negative_patients,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
    }
    if parsed_counts != derived_counts:
        raise ValueError("decision untouched acceptance counts are inconsistent")
    if (
        parsed_counts["studies"] != acceptance_record["images"]
        or parsed_counts["patients"] != acceptance_record["patients"]
        or positive_studies != acceptance_record["positives"]
        or negative_studies != acceptance_record["negatives"]
    ):
        raise ValueError(
            "decision untouched acceptance outcomes do not match its partition"
        )
    if positive_patients <= 0 or negative_patients <= 0:
        raise ValueError("decision untouched acceptance is not scoreable")
    sensitivity = true_positives / positive_patients
    specificity = true_negatives / negative_patients
    point_estimates = _require_object(
        acceptance.get("point_estimates"),
        "decision untouched acceptance point_estimates",
    )
    if set(point_estimates) != {"sensitivity", "specificity"}:
        raise ValueError("decision untouched acceptance point estimates are incomplete")
    _require_close(
        point_estimates.get("sensitivity"),
        sensitivity,
        "decision untouched acceptance sensitivity",
    )
    _require_close(
        point_estimates.get("specificity"),
        specificity,
        "decision untouched acceptance specificity",
    )

    all_diagnostics = _require_object(
        acceptance.get("all_study_diagnostics"),
        "decision untouched acceptance all_study_diagnostics",
    )
    if (
        set(all_diagnostics) != {"gate_role", "counts", "point_estimates"}
        or all_diagnostics.get("gate_role")
        != "diagnostic_only_not_used_for_acceptance"
    ):
        raise ValueError("decision untouched all-study diagnostics are invalid")
    all_counts = _require_object(
        all_diagnostics.get("counts"),
        "decision untouched acceptance all-study counts",
    )
    expected_all_counts = {
        "positive_studies": positive_studies,
        "negative_studies": negative_studies,
        "true_positives": int(
            sum(
                row["truth"] == 1 and row["predicted_positive"]
                for row in checked_outcomes
            )
        ),
        "false_negatives": int(
            sum(
                row["truth"] == 1 and not row["predicted_positive"]
                for row in checked_outcomes
            )
        ),
        "true_negatives": int(
            sum(
                row["truth"] == 0 and not row["predicted_positive"]
                for row in checked_outcomes
            )
        ),
        "false_positives": int(
            sum(
                row["truth"] == 0 and row["predicted_positive"]
                for row in checked_outcomes
            )
        ),
    }
    if all_counts != expected_all_counts:
        raise ValueError("decision untouched all-study counts are inconsistent")
    all_points = _require_object(
        all_diagnostics.get("point_estimates"),
        "decision untouched acceptance all-study point estimates",
    )
    if set(all_points) != {"sensitivity", "specificity"}:
        raise ValueError("decision untouched all-study point estimates are incomplete")
    _require_close(
        all_points.get("sensitivity"),
        expected_all_counts["true_positives"] / positive_studies,
        "decision untouched all-study sensitivity",
    )
    _require_close(
        all_points.get("specificity"),
        expected_all_counts["true_negatives"] / negative_studies,
        "decision untouched all-study specificity",
    )

    expected_sensitivity_interval = _wilson_interval(
        true_positives,
        positive_patients,
    )
    expected_specificity_interval = _wilson_interval(
        true_negatives,
        negative_patients,
    )
    expected_intervals = {
        "sensitivity": expected_sensitivity_interval,
        "specificity": expected_specificity_interval,
    }
    intervals = _require_object(
        acceptance.get("confidence_intervals"),
        "decision untouched acceptance confidence_intervals",
    )
    if set(intervals) != {"sensitivity", "specificity"}:
        raise ValueError("decision untouched acceptance intervals are incomplete")
    for metric, expected_interval in expected_intervals.items():
        interval = _require_object(
            intervals.get(metric),
            f"decision untouched acceptance {metric} interval",
        )
        if set(interval) != {"lower", "upper"}:
            raise ValueError("decision untouched acceptance interval schema is invalid")
        _require_close(
            interval.get("lower"),
            expected_interval[0],
            f"decision untouched acceptance {metric} lower",
        )
        _require_close(
            interval.get("upper"),
            expected_interval[1],
            f"decision untouched acceptance {metric} upper",
        )

    acceptance_requirements = _require_object(
        acceptance.get("requirements"),
        "decision untouched acceptance requirements",
    )
    if set(acceptance_requirements) != {
        "minimum_positive_patients",
        "minimum_negative_patients",
        "sensitivity_lower_bound",
        "specificity_lower_bound",
    }:
        raise ValueError("decision untouched acceptance requirements are incomplete")
    _require_exact_int(
        acceptance_requirements.get("minimum_positive_patients"),
        _MIN_ACCEPTANCE_POSITIVE_PATIENTS,
        "decision untouched acceptance minimum_positive_patients",
    )
    _require_exact_int(
        acceptance_requirements.get("minimum_negative_patients"),
        _MIN_ACCEPTANCE_NEGATIVE_PATIENTS,
        "decision untouched acceptance minimum_negative_patients",
    )
    _require_close(
        acceptance_requirements.get("sensitivity_lower_bound"),
        sensitivity_target,
        "decision untouched acceptance sensitivity requirement",
    )
    _require_close(
        acceptance_requirements.get("specificity_lower_bound"),
        specificity_floor,
        "decision untouched acceptance specificity requirement",
    )
    support_passes = (
        positive_patients >= _MIN_ACCEPTANCE_POSITIVE_PATIENTS
        and negative_patients >= _MIN_ACCEPTANCE_NEGATIVE_PATIENTS
    )
    derived_passes = {
        "support": support_passes,
        "sensitivity_lower_bound": (
            expected_intervals["sensitivity"][0] >= sensitivity_target
        ),
        "specificity_lower_bound": (
            expected_intervals["specificity"][0] >= specificity_floor
        ),
    }
    passes = _require_object(
        acceptance.get("passes"),
        "decision untouched acceptance passes",
    )
    if passes != derived_passes or not all(derived_passes.values()):
        raise ValueError("decision untouched acceptance lower-bound gates did not pass")
    if acceptance.get("complete") is not True:
        raise ValueError("decision untouched acceptance is not complete")

    calibration_record = checked_roles["calibration"]
    threshold_record = checked_roles["threshold_selection"]
    if (
        calibration_record["positives"] < min_calibration_positives
        or calibration_record["negatives"] < min_calibration_negatives
        or threshold_record["positives"] < min_threshold_positives
        or threshold_record["negatives"] < min_threshold_negatives
    ):
        raise ValueError("decision role support is below its declared minimum")
    return threshold, slope, intercept


def load_decision_policy(
    path: str | Path,
    *,
    active_target: str,
    query_pack_sha256: str,
    prompt_hash: str,
) -> DecisionPolicy:
    """Load a completed Platt/threshold artifact for exactly one endpoint."""

    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    _require_sha256(query_pack_sha256, "query_pack_sha256")
    _require_sha256(prompt_hash, "prompt_hash")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"decision artifact not found: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse decision artifact {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("decision artifact must be a JSON object")
    if data.get("artifact_type") != _DECISION_ARTIFACT_TYPE:
        raise ValueError(
            f"decision artifact_type must be {_DECISION_ARTIFACT_TYPE!r}"
        )
    if data.get("schema_version") != _DECISION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported decision artifact schema_version; "
            f"expected {_DECISION_SCHEMA_VERSION}"
        )
    if data.get("active_target") != active_target:
        raise ValueError(
            f"decision artifact active_target must be {active_target!r}"
        )
    query_spec = PHASE1_QUERY_SPECS[active_target]
    if data.get("labels") != [active_target]:
        raise ValueError("decision artifact must contain only the active endpoint label")
    if data.get("prompts") != [query_spec["prompt"]]:
        raise ValueError("decision artifact prompt does not match the frozen endpoint prompt")
    if data.get("query_pack_sha256") != query_pack_sha256:
        raise ValueError("decision artifact was fitted for a different KAD query pack")
    if data.get("prompt_set_sha256") != prompt_hash:
        raise ValueError("decision artifact prompt-set hash does not match")
    for field in (
        "calibration_complete",
        "acceptance_complete",
        "thresholds_complete",
    ):
        if data.get(field) is not True:
            raise ValueError(f"decision artifact {field} is not true")

    development_hash = _require_sha256(
        data.get("development_manifest_sha256"),
        "decision artifact development_manifest_sha256",
    )
    development_benchmark_hash = _require_sha256(
        data.get("development_benchmark_sha256"),
        "decision artifact development_benchmark_sha256",
    )
    development_predictions_hash = _require_sha256(
        data.get("development_predictions_sha256"),
        "decision artifact development_predictions_sha256",
    )
    development_prediction_rows = _load_bound_decision_predictions(
        data,
        decision_path=resolved,
        active_target=active_target,
        expected_sha256=development_predictions_hash,
    )
    threshold, slope, intercept = _validate_completed_decision_evidence(
        data,
        active_target=active_target,
        development_prediction_rows=development_prediction_rows,
    )

    return DecisionPolicy(
        path=resolved,
        artifact_sha256=sha256_file(resolved),
        active_target=active_target,
        thresholds={active_target: threshold},
        slope=slope,
        intercept=intercept,
        development_manifest_sha256=development_hash,
        development_benchmark_sha256=development_benchmark_hash,
        development_predictions_sha256=development_predictions_hash,
    )


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite number")
    return parsed


def validate_test_lock(
    path: str | Path,
    *,
    active_target: str,
    query_pack_sha256: str,
    test_manifest_sha256: str,
    prompt_set_sha256: str,
    decision_artifact_sha256: str,
    image_set_sha256: str,
    image_provenance_sha256: str,
    runtime_contract_sha256: str,
    official_train_val_manifest_sha256: str,
    official_test_manifest_sha256: str,
) -> dict[str, Any]:
    """Fail unless every frozen test-protocol hash matches current inputs."""

    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"test lock artifact not found: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse test lock artifact {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("test lock artifact must be a JSON object")
    if data.get("artifact_type") != _TEST_LOCK_ARTIFACT_TYPE:
        raise ValueError(
            f"test lock artifact_type must be {_TEST_LOCK_ARTIFACT_TYPE!r}"
        )
    if data.get("schema_version") != 1:
        raise ValueError("unsupported test lock schema_version; expected 1")
    if data.get("protocol_frozen") is not True or data.get("cohort") != "test":
        raise ValueError(
            "test lock must declare protocol_frozen=true and cohort='test'"
        )
    if data.get("active_target") != active_target:
        raise ValueError(
            f"test lock active_target must be {active_target!r}"
        )
    expected = {
        "query_pack_sha256": query_pack_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "prompt_set_sha256": prompt_set_sha256,
        "decision_artifact_sha256": decision_artifact_sha256,
        "image_set_sha256": image_set_sha256,
        "image_provenance_sha256": image_provenance_sha256,
        "runtime_contract_sha256": runtime_contract_sha256,
        "official_train_val_manifest_sha256": (
            official_train_val_manifest_sha256
        ),
        "official_test_manifest_sha256": official_test_manifest_sha256,
    }
    problems: list[str] = []
    for field, current in expected.items():
        locked = _require_sha256(data.get(field), f"test lock {field}")
        if locked != current:
            problems.append(f"{field}: locked {locked}, current {current}")
    if problems:
        raise ValueError(
            "test lock hash mismatch; refusing test inference: " + " | ".join(problems)
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "protocol_id": data.get("protocol_id"),
    }


def _load_radiograph(path: Path):
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load expert-manifest images") from exc
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(f"multi-frame image is not accepted: {path}")
            image = opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"could not decode manifest image {path}: {exc}") from exc
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"decoded manifest image is empty: {path}")
    return image


def validate_image_provenance(
    provenance: ImageProvenance,
    manifest: ExpertManifest,
) -> None:
    """Bind every selected file to the fetch receipt, identity, hash, and size."""

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate image provenance") from exc
    if provenance.canonical_manifest_sha256 != manifest.manifest_sha256:
        raise ValueError(
            "image provenance canonical manifest SHA-256 does not match --manifest"
        )
    if provenance.cohort != manifest.cohort:
        raise ValueError("image provenance cohort does not match benchmark cohort")
    if provenance.rows_selected != len(manifest.sample_ids):
        raise ValueError(
            "image provenance rows_selected does not match selected manifest rows"
        )
    selected = set(manifest.sample_ids)
    if set(provenance.output_sha256_by_filename) != selected:
        raise ValueError(
            "image provenance filename records do not exactly match selected images"
        )
    expected_split = "validation" if manifest.cohort == "development" else "test"
    expected = (provenance.width, provenance.height)
    for sample_id, path, actual_hash in zip(
        manifest.sample_ids,
        manifest.image_paths,
        manifest.image_sha256,
    ):
        if provenance.expert_split_by_filename[sample_id] != expected_split:
            raise ValueError(
                f"image provenance expert_split mismatch for {sample_id}"
            )
        if provenance.output_sha256_by_filename[sample_id] != actual_hash:
            raise ValueError(
                f"image provenance output SHA-256 mismatch for {sample_id}"
            )
        try:
            with Image.open(path) as image:
                actual = tuple(image.size)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"could not decode manifest image {path}: {exc}") from exc
        if actual != expected:
            raise ValueError(
                f"image {path} resolution {actual} does not match declared "
                f"provenance resolution {expected}"
            )


def infer_kad_paths(
    query_pack: Path,
    image_paths: Sequence[Path],
    *,
    batch_size: int,
    device: str | None,
    amp: bool,
) -> np.ndarray:
    expert = KAD512Expert(
        query_pack_path=query_pack,
        device=device,
        amp=amp,
    )
    rows: list[np.ndarray] = []
    total = len(image_paths)
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = torch.stack(
            [preprocess_kad512(_load_radiograph(path)) for path in batch_paths]
        )
        scores = expert.predict_proba(batch).numpy()
        if scores.shape != (len(batch_paths), len(expert.class_names)):
            raise RuntimeError(
                f"KAD returned unexpected batch shape {scores.shape} for "
                f"{len(batch_paths)} images"
            )
        rows.append(scores)
        completed = min(start + len(batch_paths), total)
        batch_index = start // batch_size
        if completed == total or batch_index % 25 == 0:
            print(f"KAD inference: {completed:,}/{total:,} images")
    probabilities = np.concatenate(rows, axis=0)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("KAD inference returned NaN or infinite scores")
    return probabilities.astype(np.float32, copy=False)


def infer_kad_tensors(
    query_pack: Path,
    images: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: str | None,
    amp: bool,
) -> np.ndarray:
    expert = KAD512Expert(
        query_pack_path=query_pack,
        device=device,
        amp=amp,
    )
    rows: list[np.ndarray] = []
    total = len(images)
    for start in range(0, len(images), batch_size):
        source = images[start : start + batch_size]
        batch = torch.stack([preprocess_kad512(image) for image in source])
        rows.append(expert.predict_proba(batch).numpy())
        completed = min(start + len(source), total)
        batch_index = start // batch_size
        if completed == total or batch_index % 25 == 0:
            print(f"KAD inference: {completed:,}/{total:,} images")
    probabilities = np.concatenate(rows, axis=0)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("KAD inference returned NaN or infinite scores")
    return probabilities.astype(np.float32, copy=False)


def ranking_scorecard(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    patient_ids: Sequence[str | int],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Use shared metrics, then remove every threshold/calibration claim."""

    full = evaluate_multilabel_classifier(
        probabilities,
        labels,
        class_names,
        thresholds=0.5,
        patient_ids=patient_ids,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    per_label = {
        label: {
            key: row[key]
            for key in ("images", "positives", "negatives", "prevalence", "auroc", "auprc")
        }
        for label, row in full["per_label"].items()
    }
    result: dict[str, Any] = {
        "images": full["images"],
        "patients": full["patients"],
        "scoreable_labels": full["scoreable_labels"],
        "metrics_scope": "ranking_only_no_operating_thresholds",
        "macro": {
            "auroc": full["macro"]["auroc"],
            "auprc": full["macro"]["auprc"],
        },
        "per_label": per_label,
    }
    confidence = full.get("confidence_intervals")
    if confidence:
        result["confidence_intervals"] = {
            "method": confidence["method"],
            "samples": confidence["samples"],
            "macro": {
                "auroc": confidence["macro"]["auroc"],
                "auprc": confidence["macro"]["auprc"],
            },
            "per_label_auroc": confidence["per_label_auroc"],
            "per_label_auprc": confidence["per_label_auprc"],
        }
    return result


def deferred_development_scorecard(
    labels: np.ndarray,
    *,
    patient_ids: Sequence[str | int],
    active_target: str,
) -> dict[str, Any]:
    """Expose support only; defer all ranking analysis until patient roles exist."""

    truth = np.asarray(labels)
    expected_shape = (len(patient_ids), 1)
    if truth.shape != expected_shape:
        raise ValueError(
            f"expert labels must have shape {expected_shape}, got {truth.shape}"
        )
    if not np.isin(truth, (-1, 0, 1)).all():
        raise ValueError("expert labels must use -1 for missing and 0/1 for adjudicated")
    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    adjudicated = truth[:, 0] >= 0
    positives = int((truth[:, 0] == 1).sum())
    negatives = int((truth[:, 0] == 0).sum())
    images = int(adjudicated.sum())
    per_label = {
        active_target: {
            "images": images,
            "positives": positives,
            "negatives": negatives,
            "prevalence": (float(positives / images) if images else None),
            "auroc": None,
            "auprc": None,
            "non_adjudicated_images": int((~adjudicated).sum()),
            "decision_role": "active_target",
            "analysis_deferred": True,
        }
    }
    return {
        "images": int(len(truth)),
        "patients": int(len(set(str(value) for value in patient_ids))),
        "scoreable_labels": int(bool(positives and negatives)),
        "metrics_scope": "development_analysis_deferred_until_patient_partition",
        "analysis_deferred": True,
        "analysis_deferred_until": (
            "calibration_artifact_model_selection_patient_partition"
        ),
        "active_target": active_target,
        "candidate_under_decision": active_target,
        "exploratory_labels": [],
        "missing_label_policy": "exclude_independently_per_target",
        "macro": {"auroc": None, "auprc": None},
        "per_label": per_label,
    }


def evaluate_expert_scorecard(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    patient_ids: Sequence[str | int],
    thresholds: Mapping[str, float] | None,
    active_target: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the isolated endpoint only where it was adjudicated."""

    scores = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels)
    expected_shape = (len(patient_ids), 1)
    if scores.shape != expected_shape or truth.shape != expected_shape:
        raise ValueError(
            f"expert scores/labels must both have shape {expected_shape}; got "
            f"{scores.shape} and {truth.shape}"
        )
    if not np.isin(truth, (-1, 0, 1)).all():
        raise ValueError("expert labels must use -1 for missing and 0/1 for adjudicated")
    if active_target not in PHASE1_LABELS:
        raise ValueError(f"active_target must be one of {PHASE1_LABELS}")
    if thresholds is not None and set(thresholds) != {active_target}:
        raise ValueError("operating threshold must exist for the active target only")

    rank_fields = (
        "images",
        "positives",
        "negatives",
        "prevalence",
        "auroc",
        "auprc",
    )
    operating_fields = (
        "brier",
        "ece",
        "threshold",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    )
    per_label: dict[str, dict[str, Any]] = {}
    confidence_by_label: dict[str, dict[str, Any]] = {}
    patient_array = np.asarray([str(value) for value in patient_ids], dtype=object)
    label = active_target
    mask = truth[:, 0] >= 0
    decision_endpoint = thresholds is not None
    if not mask.any():
        per_label[label] = {
            "images": 0,
            "positives": 0,
            "negatives": 0,
            "prevalence": None,
            "auroc": None,
            "auprc": None,
            "non_adjudicated_images": int(len(truth)),
            "decision_role": "active_target",
        }
    else:
        label_threshold = float(thresholds[label]) if decision_endpoint else 0.5
        report = evaluate_multilabel_classifier(
            scores[mask, :],
            truth[mask, :],
            [label],
            thresholds={label: label_threshold},
            patient_ids=patient_array[mask],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        source_row = report["per_label"][label]
        keep = rank_fields + operating_fields if decision_endpoint else rank_fields
        per_label[label] = {field: source_row[field] for field in keep}
        per_label[label]["non_adjudicated_images"] = int((~mask).sum())
        per_label[label]["decision_role"] = "active_target"
        confidence = report.get("confidence_intervals")
        if confidence:
            row = {
                "auroc": confidence["per_label_auroc"][label],
                "auprc": confidence["per_label_auprc"][label],
            }
            if decision_endpoint:
                row["sensitivity"] = confidence["macro"]["sensitivity"]
                row["specificity"] = confidence["macro"]["specificity"]
            confidence_by_label[label] = row

    macro = {
        field: _finite_mean([per_label[active_target].get(field)])
        for field in ("auroc", "auprc")
    }
    result: dict[str, Any] = {
        "images": int(len(truth)),
        "patients": int(len(set(str(value) for value in patient_ids))),
        "scoreable_labels": int(per_label[active_target]["auroc"] is not None),
        "metrics_scope": (
            "ranking_only_no_operating_thresholds"
            if thresholds is None
            else "active_target_calibrated_with_frozen_threshold"
        ),
        "active_target": active_target,
        "candidate_under_decision": active_target,
        "exploratory_labels": [],
        "missing_label_policy": "exclude_independently_per_target",
        "macro": macro,
        "per_label": per_label,
    }
    if bootstrap_samples:
        result["confidence_intervals"] = {
            "method": "patient_clustered_percentile_bootstrap_per_target",
            "samples": int(bootstrap_samples),
            "per_label": confidence_by_label,
            "macro": {
                "status": (
                    "not_estimated_because_adjudicated_patient_sets_can_differ_by_target"
                )
            },
        }
    return result


def rank_expert_errors(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    thresholds: Mapping[str, float],
    active_target: str,
    sample_ids: Sequence[str],
    limit: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Rank errors per target without treating non-adjudicated cells as negatives."""

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ids = np.asarray(sample_ids, dtype=object)
    if set(thresholds) != {active_target}:
        raise ValueError("error ranking thresholds must contain only active_target")
    expected_shape = (len(sample_ids), 1)
    if probabilities.shape != expected_shape or labels.shape != expected_shape:
        raise ValueError(
            "endpoint-isolated probabilities and labels must both have shape "
            f"{expected_shape}"
        )
    mask = labels[:, 0] >= 0
    one = rank_classification_errors(
        probabilities[mask, :],
        labels[mask, :],
        [active_target],
        thresholds={active_target: thresholds[active_target]},
        sample_ids=ids[mask].tolist(),
        limit=limit,
    )
    result[active_target] = one[active_target]
    return result


def _finite_mean(values: Iterable[Any]) -> float | None:
    parsed = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(parsed)) if parsed else None


def _write_outputs(
    output: Path,
    artifact: Mapping[str, Any],
    *,
    raw_scores: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    patient_ids: Sequence[str | int],
    sample_ids: Sequence[str],
    calibrated_scores: np.ndarray | None = None,
    image_sha256: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    output = output.expanduser()
    prediction_path = output.with_suffix(".predictions.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (output, prediction_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing benchmark artifact(s): "
            + ", ".join(str(path) for path in existing)
        )
    arrays: dict[str, Any] = {
        "raw_scores": raw_scores,
        "labels": labels,
        "class_names": np.asarray(class_names),
        "patient_ids": np.asarray([str(value) for value in patient_ids]),
        "sample_ids": np.asarray(sample_ids),
    }
    if calibrated_scores is not None:
        arrays["calibrated_scores"] = calibrated_scores
    if image_sha256 is not None:
        arrays["image_sha256"] = np.asarray(image_sha256)

    prediction_temporary: str | None = None
    json_temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=output.parent,
            prefix=f".{prediction_path.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as handle:
            prediction_temporary = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        prediction_hash = sha256_file(prediction_temporary)
        bound_artifact = dict(artifact)
        bound_artifact["predictions"] = str(prediction_path)
        bound_artifact["predictions_sha256"] = prediction_hash
        json_bytes = (
            json.dumps(
                _json_safe(bound_artifact),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json_temporary = handle.name
            handle.write(json_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        existing = [path for path in (output, prediction_path) if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "refusing to overwrite existing benchmark artifact(s): "
                + ", ".join(str(path) for path in existing)
            )
        # Commit the data first and its hash-bound JSON last. A process interruption
        # can therefore leave a detectable hash mismatch, never a JSON that silently
        # blesses stale prediction bytes.
        os.replace(prediction_temporary, prediction_path)
        prediction_temporary = None
        _fsync_directory(output.parent)
        os.replace(json_temporary, output)
        json_temporary = None
        _fsync_directory(output.parent)
    finally:
        for temporary in (prediction_temporary, json_temporary):
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
    return prediction_path


def _fsync_directory(path: Path) -> None:
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


def _refuse_existing_outputs(output: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    expanded = output.expanduser()
    prediction_path = expanded.with_suffix(".predictions.npz")
    existing = [path for path in (expanded, prediction_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing benchmark artifact(s): "
            + ", ".join(str(path) for path in existing)
        )


def run_canonical(args: argparse.Namespace) -> tuple[Path, Path]:
    image_provenance = load_image_provenance(args.image_provenance)
    if args.cohort == "test" and not image_provenance.original_nih_pixels:
        raise ValueError(
            "TEST DISABLED: image provenance must declare "
            "original_nih_pixels=true"
        )
    source_metadata = load_source_metadata_provenance(
        args.source_metadata_provenance,
        cohort=args.cohort,
        official_train_val_manifest=args.official_train_val_manifest,
        official_test_manifest=args.official_test_manifest,
    )
    manifest = load_expert_manifest(
        args.manifest,
        args.image_root,
        cohort=args.cohort,
        active_target=args.active_target,
        official_train_val_manifest=args.official_train_val_manifest,
        official_test_manifest=args.official_test_manifest,
    )
    manifest_metadata = load_manifest_metadata(
        args.manifest_metadata,
        manifest=manifest,
        cohort=args.cohort,
        source_metadata=source_metadata,
    )
    validate_image_provenance(image_provenance, manifest)
    official_manifest_reconciled = bool(
        manifest.official_membership_verified
        and source_metadata.get("verified") is True
        and manifest_metadata.get("verified") is True
        and manifest.official_train_val_manifest_sha256
        == source_metadata.get("official_train_val_manifest_sha256")
        and manifest.official_test_manifest_sha256
        == source_metadata.get("official_test_manifest_sha256")
    )
    if not official_manifest_reconciled:
        raise RuntimeError(
            "canonical provenance chain did not bind the pinned official NIH manifests"
        )
    query_spec = PHASE1_QUERY_SPECS[args.active_target]
    query_info = inspect_query_pack(
        args.query_pack,
        expected_labels=(args.active_target,),
        expected_prompts=(query_spec["prompt"],),
        expected_query_set=query_spec["query_set"],
        expected_semantic_sha256=query_spec["semantic_sha256"],
    )
    runtime, runtime_contract_sha256 = build_runtime_contract(
        args,
        require_clean_worktree=True,
    )
    policy = None
    if args.decision_artifact is not None:
        policy = load_decision_policy(
            args.decision_artifact,
            query_pack_sha256=query_info["sha256"],
            prompt_hash=query_info["prompt_set_sha256"],
            active_target=args.active_target,
        )
        if (
            args.cohort == "development"
            and policy.development_manifest_sha256 != manifest.manifest_sha256
        ):
            raise ValueError(
                "decision artifact development_manifest_sha256 does not match "
                "the current canonical development manifest"
            )

    lock_info = None
    if args.cohort == "test":
        if policy is None or args.test_lock is None:
            raise ValueError(
                "TEST DISABLED: --cohort test requires both --decision-artifact "
                "and --test-lock; development is the default"
            )
        lock_info = validate_test_lock(
            args.test_lock,
            active_target=args.active_target,
            query_pack_sha256=query_info["sha256"],
            test_manifest_sha256=manifest.manifest_sha256,
            prompt_set_sha256=query_info["prompt_set_sha256"],
            decision_artifact_sha256=policy.artifact_sha256,
            image_set_sha256=manifest.image_set_sha256,
            image_provenance_sha256=image_provenance.artifact_sha256,
            runtime_contract_sha256=runtime_contract_sha256,
            official_train_val_manifest_sha256=(
                manifest.official_train_val_manifest_sha256
            ),
            official_test_manifest_sha256=(
                manifest.official_test_manifest_sha256
            ),
        )
    elif args.test_lock is not None:
        raise ValueError("--test-lock is accepted only with --cohort test")

    raw_scores = infer_kad_paths(
        query_info["path"],
        manifest.image_paths,
        batch_size=args.batch_size,
        device=args.device,
        amp=not args.no_amp,
    )
    active_index = PHASE1_LABELS.index(args.active_target)
    target_labels = manifest.labels[:, active_index : active_index + 1]
    if raw_scores.shape != target_labels.shape:
        raise RuntimeError(
            f"endpoint-isolated KAD scores shape {raw_scores.shape} does not match "
            f"active-target labels {target_labels.shape}"
        )

    calibrated_scores = None
    errors = None
    if policy is None:
        if args.defer_development_ranking:
            scorecard = deferred_development_scorecard(
                target_labels,
                patient_ids=manifest.patient_ids,
                active_target=args.active_target,
            )
        else:
            scorecard = evaluate_expert_scorecard(
                raw_scores,
                target_labels,
                patient_ids=manifest.patient_ids,
                thresholds=None,
                active_target=args.active_target,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        decision_info: dict[str, Any] = {
            "status": "not_supplied",
            "operating_point_metrics": "disabled",
        }
    else:
        calibrated_scores = policy.calibrate(raw_scores)
        scorecard = evaluate_expert_scorecard(
            calibrated_scores,
            target_labels,
            thresholds=policy.thresholds,
            active_target=args.active_target,
            patient_ids=manifest.patient_ids,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        errors = rank_expert_errors(
            calibrated_scores,
            target_labels,
            thresholds=policy.thresholds,
            active_target=args.active_target,
            sample_ids=manifest.sample_ids,
            limit=args.error_limit,
        )
        decision_info = {
            "status": "complete_calibration_and_thresholds",
            "path": str(policy.path),
            "sha256": policy.artifact_sha256,
            "development_manifest_sha256": policy.development_manifest_sha256,
            "development_benchmark_sha256": policy.development_benchmark_sha256,
            "development_predictions_sha256": (
                policy.development_predictions_sha256
            ),
            "calibrator": "per_label_platt_logit",
            "thresholds": policy.thresholds,
        }
        if args.cohort == "development":
            decision_info["operating_point_metrics"] = (
                "full_development_resubstitution_diagnostic_only"
            )
            scorecard["metrics_scope"] = (
                "active_target_full_development_resubstitution_diagnostic"
            )
            scorecard["operating_point_interpretation"] = (
                "The calibrator and threshold were derived from partitions of this "
                "same development cohort. Full-cohort operating metrics are "
                "resubstitution diagnostics, not validation or test estimates."
            )
        else:
            decision_info["operating_point_metrics"] = (
                "locked_held_out_test_with_frozen_policy"
            )

    prediction_path = args.output.with_suffix(".predictions.npz")
    if args.cohort == "test":
        evidence_warning = (
            "The lock validates artifact identity, not clinical validity."
        )
    elif image_provenance.original_nih_pixels:
        evidence_warning = (
            "This is development data and must not be presented as held-out test."
        )
    else:
        evidence_warning = (
            "Pixels are declared non-original/resized mirror data; use only for "
            "candidate selection, never final evidence."
        )
    artifact = {
        "schema_version": 1,
        "artifact_type": "doctor_assistant.kad_phase1_benchmark",
        "purpose": (
            "locked_expert_test_evaluation"
            if args.cohort == "test"
            else "expert_development_evaluation"
        ),
        "cohort": args.cohort,
        "active_target": args.active_target,
        "candidate_under_decision": args.active_target,
        "exploratory_labels": [],
        "smoke_only": False,
        "official_manifest_reconciled": official_manifest_reconciled,
        "evidence_status": (
            "locked_test_run_research_only_not_clinically_validated"
            if args.cohort == "test"
            else (
                "development_only_do_not_report_as_test_performance"
                if image_provenance.original_nih_pixels
                else "resized_mirror_candidate_selection_only"
            )
        ),
        "model": query_info,
        "runtime": runtime,
        "runtime_contract_sha256": runtime_contract_sha256,
        "inputs": {
            "manifest": str(manifest.path),
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_metadata": {
                "path": str(manifest_metadata["path"]),
                "sha256": manifest_metadata["sha256"],
                "output_cohort": manifest_metadata["output_cohort"],
                "google_expert_labels_sha256": (
                    manifest_metadata["google_expert_labels_sha256"]
                ),
            },
            "source_metadata_provenance": {
                "path": str(source_metadata["path"]),
                "sha256": source_metadata["sha256"],
                "google_expert_labels_sha256": (
                    source_metadata["google_expert_labels_sha256"]
                ),
                "locked_test_eligible": source_metadata["locked_test_eligible"],
            },
            "image_root": str(manifest.image_root),
            "image_set_sha256": manifest.image_set_sha256,
            "images": len(manifest.sample_ids),
            "patients": len(set(manifest.patient_ids)),
            "official_train_val_manifest_sha256": (
                manifest.official_train_val_manifest_sha256
            ),
            "official_test_manifest_sha256": manifest.official_test_manifest_sha256,
            "image_provenance": {
                "path": str(image_provenance.path),
                "sha256": image_provenance.artifact_sha256,
                "source": image_provenance.source,
                "resolution": {
                    "width": image_provenance.width,
                    "height": image_provenance.height,
                },
                "original_nih_pixels": image_provenance.original_nih_pixels,
                "canonical_manifest_sha256": (
                    image_provenance.canonical_manifest_sha256
                ),
                "cohort": image_provenance.cohort,
                "rows_selected": image_provenance.rows_selected,
                "all_selected_output_sha256_verified": True,
            },
        },
        "decision": decision_info,
        "test_lock": lock_info,
        "scorecard": scorecard,
        "ranked_errors": errors,
        "predictions": str(prediction_path),
        "warning": (
            "Research evaluation only; no clinical deployment claim. "
            + evidence_warning
        ),
    }
    written_predictions = _write_outputs(
        args.output,
        artifact,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        labels=target_labels,
        class_names=(args.active_target,),
        patient_ids=manifest.patient_ids,
        sample_ids=manifest.sample_ids,
        image_sha256=manifest.image_sha256,
        overwrite=args.overwrite,
    )
    return args.output, written_predictions


def _smoke_sample_sha256(sample: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for sample_id, patient_id, image, labels in zip(
        sample["sample_ids"],
        sample["patient_ids"],
        sample["images"],
        sample["labels"],
    ):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(patient_id).encode("utf-8"))
        digest.update(b"\0")
        array = np.ascontiguousarray(torch.as_tensor(image).cpu().numpy())
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()


def run_smoke_mirror(args: argparse.Namespace) -> tuple[Path, Path]:
    query_info = inspect_query_pack(
        args.query_pack,
        expected_labels=KAD512_LABELS,
        expected_prompts=KAD512_PROMPTS,
        expected_query_set=_NIH14_SMOKE_QUERY_SET,
    )
    runtime, runtime_contract_sha256 = build_runtime_contract(args)
    sample = load_split_sample(
        "valid",
        args.n,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
        require_metadata=True,
    )
    raw_scores = infer_kad_tensors(
        query_info["path"],
        sample["images"],
        batch_size=args.batch_size,
        device=args.device,
        amp=not args.no_amp,
    )
    chest_index = {label: index for index, label in enumerate(CHESTXRAY14_LABELS)}
    label_columns = [chest_index[label] for label in KAD512_LABELS]
    labels = np.asarray(sample["labels"], dtype=np.uint8)[:, label_columns]
    if raw_scores.shape != labels.shape:
        raise RuntimeError(
            f"KAD scores shape {raw_scores.shape} does not match mirror labels "
            f"{labels.shape}"
        )
    scorecard = ranking_scorecard(
        raw_scores,
        labels,
        KAD512_LABELS,
        patient_ids=sample["patient_ids"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    prediction_path = args.output.with_suffix(".predictions.npz")
    artifact = {
        "schema_version": 1,
        "artifact_type": "doctor_assistant.kad_mirror_smoke",
        "purpose": "checkpoint_and_inference_smoke_only",
        "cohort": "third_party_mirror_partition_named_valid",
        "smoke_only": True,
        "official_manifest_reconciled": False,
        "evidence_status": "not_evaluation_evidence",
        "model": query_info,
        "runtime": runtime,
        "runtime_contract_sha256": runtime_contract_sha256,
        "dataset": {
            "id": _DATASET_ID,
            "revision": args.dataset_revision,
            "partition_name": "valid",
            "partition_provenance": sample["partition_provenance"],
            "official_manifest_reconciled": False,
            "n": len(labels),
            "seed": args.seed,
            "sample_content_sha256": _smoke_sample_sha256(sample),
        },
        "decision": {
            "status": "forbidden_in_smoke_mode",
            "calibration_claims": "disabled",
            "threshold_claims": "disabled",
        },
        "scorecard": scorecard,
        "predictions": str(prediction_path),
        "warning": (
            "SMOKE ONLY. The pinned third-party mirror partition is not reconciled "
            "with NIH's official manifests. These ranking numbers cannot be used as "
            "validation, test, calibration, threshold, or clinical-performance claims."
        ),
    }
    written_predictions = _write_outputs(
        args.output,
        artifact,
        raw_scores=raw_scores,
        labels=labels,
        class_names=KAD512_LABELS,
        patient_ids=sample["patient_ids"],
        sample_ids=sample["sample_ids"],
        overwrite=args.overwrite,
    )
    return args.output, written_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate KAD-512 on strict expert labels or run a mirror smoke check."
    )
    parser.add_argument("--query-pack", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--manifest-metadata",
        type=Path,
        help=(
            "JSON audit emitted beside the canonical CSV by "
            "prepare_nih_expert_manifest.py"
        ),
    )
    parser.add_argument(
        "--source-metadata-provenance",
        type=Path,
        help=(
            "nih_metadata.provenance.json emitted by fetch_nih_metadata.py"
        ),
    )
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--image-provenance",
        type=Path,
        help=(
            "JSON declaring pixel source, width/height, and original_nih_pixels"
        ),
    )
    parser.add_argument(
        "--official-train-val-manifest",
        "--official-train-val-list",
        dest="official_train_val_manifest",
        type=Path,
        help="official NIH train_val_list.txt used to verify expert image membership",
    )
    parser.add_argument(
        "--official-test-manifest",
        "--official-test-list",
        dest="official_test_manifest",
        type=Path,
        help="official NIH test_list.txt used to verify expert image membership",
    )
    parser.add_argument(
        "--cohort",
        choices=("development", "test"),
        default="development",
    )
    parser.add_argument(
        "--active-target",
        choices=PHASE1_LABELS,
        default=PHASE1_LABELS[0],
        help=(
            "single phase-1 endpoint eligible for calibration and an operating "
            "threshold; all other endpoints remain exploratory"
        ),
    )
    parser.add_argument("--decision-artifact", type=Path)
    parser.add_argument("--test-lock", type=Path)
    parser.add_argument(
        "--defer-development-ranking",
        action="store_true",
        help=(
            "write raw development predictions and support counts while deferring "
            "AUROC/AUPRC until calibration assigns patient roles"
        ),
    )
    parser.add_argument("--smoke-mirror", action="store_true")
    parser.add_argument("--n", type=int, default=64, help="smoke images, maximum 200")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--error-limit", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--dataset-revision", default=_DATASET_REVISION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kad_phase1_benchmark.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing JSON/NPZ pair only when explicitly requested",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    if args.error_limit < 0:
        parser.error("--error-limit cannot be negative")
    try:
        _refuse_existing_outputs(args.output, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.error(str(exc))

    if args.smoke_mirror:
        if args.defer_development_ranking:
            parser.error(
                "--defer-development-ranking is for canonical development only"
            )
        if not 1 <= args.n <= 200:
            parser.error("--smoke-mirror requires --n between 1 and 200")
        forbidden = {
            "--manifest": args.manifest,
            "--manifest-metadata": args.manifest_metadata,
            "--source-metadata-provenance": args.source_metadata_provenance,
            "--image-root": args.image_root,
            "--image-provenance": args.image_provenance,
            "--official-train-val-manifest": args.official_train_val_manifest,
            "--official-test-manifest": args.official_test_manifest,
            "--decision-artifact": args.decision_artifact,
            "--test-lock": args.test_lock,
        }
        supplied = [name for name, value in forbidden.items() if value is not None]
        if supplied:
            parser.error(
                "--smoke-mirror forbids evidence/decision arguments: "
                + ", ".join(supplied)
            )
        if args.cohort != "development":
            parser.error("--smoke-mirror never permits --cohort test")
        output, predictions = run_smoke_mirror(args)
    else:
        if args.manifest is None or args.image_root is None:
            parser.error(
                "canonical evaluation requires --manifest and --image-root "
                "(or pass --smoke-mirror)"
            )
        if args.manifest_metadata is None:
            parser.error(
                "canonical expert evaluation requires --manifest-metadata"
            )
        if args.source_metadata_provenance is None:
            parser.error(
                "canonical expert evaluation requires "
                "--source-metadata-provenance"
            )
        if args.defer_development_ranking and args.cohort != "development":
            parser.error(
                "--defer-development-ranking requires --cohort development"
            )
        if args.defer_development_ranking and args.decision_artifact is not None:
            parser.error(
                "--defer-development-ranking requires raw inference without "
                "--decision-artifact"
            )
        if args.image_provenance is None:
            parser.error(
                "canonical expert evaluation requires --image-provenance"
            )
        if (
            args.official_train_val_manifest is None
            or args.official_test_manifest is None
        ):
            parser.error(
                "canonical expert evaluation requires --official-train-val-manifest "
                "and --official-test-manifest so cohort membership is verified"
            )
        if args.cohort == "test" and (
            args.decision_artifact is None or args.test_lock is None
        ):
            parser.error(
                "TEST DISABLED: --cohort test requires both a completed "
                "--decision-artifact and a frozen --test-lock"
            )
        output, predictions = run_canonical(args)

    print(f"Wrote {output} and {predictions}")
    return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
