"""Select per-label chest reporting thresholds on validation patients, then test once.

This script deliberately keeps threshold selection and evaluation on different public
dataset splits. It refuses to export a deployable threshold dictionary when any label
has inadequate validation counts or cannot meet the declared constraints.

Example:
    python scripts/calibrate_chest_thresholds.py \
        --n-valid 2000 --n-test 2000 \
        --sensitivity-target 0.85 --specificity-floor 0.60

The output JSON records model weights, sample sizes, patient-overlap checks, every
per-label selection, and held-out metrics. It is a research artifact, not clinical
validation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from data.chest_xray14 import CHESTXRAY14_LABELS
from evaluation.thresholds import (
    calibrated_threshold_dict,
    evaluate_frozen_thresholds,
    select_per_label_thresholds,
)
from experts.torchxrayvision import TorchXRayVisionExpert
from scripts.eval_chest_xrv import (
    _DATASET_ID,
    _DATASET_REVISION,
    _to_multihot,
    compute_aucs,
    run_expert,
)

_DEFAULT_CACHE = Path(
    os.environ.get(
        "DOCTOR_ASSISTANT_CACHE_DIR",
        Path.home() / ".cache" / "doctor_assistant" / "evaluation",
    )
)


def load_split_sample(
    split: str,
    n: int,
    seed: int,
    cache_dir: Path,
    dataset_revision: str,
    *,
    require_metadata: bool = False,
) -> dict:
    """Load/cache images, labels, and patient IDs from one public dataset split."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"nih_{split}_n{n}_seed{seed}_rev{dataset_revision[:12]}.pt"
    )
    if cache_path.is_file():
        blob = torch.load(cache_path, weights_only=False)
        required = {"images", "labels", "patient_ids"}
        if require_metadata:
            required |= {"sample_ids", "patient_ages", "patient_genders", "view_positions"}
        if required <= set(blob):
            print(f"Loading cached {split} sample from {cache_path}")
            return blob

    from datasets import load_dataset

    print(f"Streaming {n} {split} images with patient IDs ...")
    dataset = load_dataset(
        _DATASET_ID,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=min(4000, max(200, n * 4)))

    images, labels, patient_ids = [], [], []
    sample_ids, patient_ages, patient_genders, view_positions = [], [], [], []
    started = time.time()
    for index, row in enumerate(dataset.take(n)):
        image = row["image"].convert("L")
        # Keep cached source images as uint8. Full-resolution NIH images are 1024²;
        # float32 caching costs ~4 GB per 1,000 studies and exhausts a Colab runtime
        # before evaluation starts. Both expert preprocessors accept uint8 input and
        # perform their own normalization.
        array = np.asarray(image, dtype=np.uint8)
        images.append(torch.from_numpy(array).unsqueeze(0))
        labels.append(_to_multihot(row["label"]))
        patient_id = int(row["Patient ID"])
        patient_ids.append(patient_id)
        sample_ids.append(f"{split}:{index}:patient:{patient_id}")
        patient_ages.append(int(row["Patient Age"]))
        patient_genders.append(str(row["Patient Gender"]).strip() or "UNKNOWN")
        view_positions.append(str(row["View Position"]).strip() or "UNKNOWN")

    blob = {
        "images": images,
        "labels": np.stack(labels),
        "patient_ids": patient_ids,
        "sample_ids": sample_ids,
        "patient_ages": patient_ages,
        "patient_genders": patient_genders,
        "view_positions": view_positions,
        "split": split,
        "seed": seed,
        "dataset_revision": dataset_revision,
    }
    torch.save(blob, cache_path)
    print(
        f"  cached {len(images)} images / {len(set(patient_ids))} patients "
        f"in {time.time() - started:.1f}s"
    )
    return blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-valid", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensitivity-target", type=float, default=0.85)
    parser.add_argument("--specificity-floor", type=float, default=0.60)
    parser.add_argument("--min-positives", type=int, default=20)
    parser.add_argument("--min-negatives", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument(
        "--dataset-revision",
        default=_DATASET_REVISION,
        help="pinned Hugging Face dataset commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chest_xrv_thresholds.json"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    valid = load_split_sample(
        "valid",
        args.n_valid,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
    )
    test = load_split_sample(
        "test",
        args.n_test,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
    )

    overlap = set(valid["patient_ids"]) & set(test["patient_ids"])
    if overlap:
        raise RuntimeError(
            f"Validation/test patient leakage detected: {len(overlap)} overlapping IDs"
        )

    expert = TorchXRayVisionExpert(device=args.device)
    print("Running validation inference ...")
    valid_probs = run_expert(expert, valid["images"])
    print("Running held-out test inference ...")
    test_probs = run_expert(expert, test["images"])

    selections = select_per_label_thresholds(
        valid_probs,
        valid["labels"],
        CHESTXRAY14_LABELS,
        sensitivity_target=args.sensitivity_target,
        specificity_floor=args.specificity_floor,
        min_positives=args.min_positives,
        min_negatives=args.min_negatives,
    )
    all_thresholds = {item.label: item.threshold for item in selections}
    held_out = evaluate_frozen_thresholds(
        test_probs,
        test["labels"],
        CHESTXRAY14_LABELS,
        all_thresholds,
    )

    thresholds_complete = True
    try:
        pipeline_thresholds = calibrated_threshold_dict(selections)
    except ValueError as exc:
        thresholds_complete = False
        pipeline_thresholds = None
        print(f"NOT EXPORTABLE: {exc}")

    artifact = {
        "schema_version": 1,
        "dataset": {
            "id": _DATASET_ID,
            "revision": args.dataset_revision,
            "seed": args.seed,
        },
        "model": {
            "expert": expert.name,
            "weights": list(expert.weights),
            "resolution": expert.resolution,
            "torch_version": torch.__version__,
            "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        },
        "selection": {
            "split": "valid",
            "images": len(valid["images"]),
            "patients": len(set(valid["patient_ids"])),
            "sensitivity_target": args.sensitivity_target,
            "specificity_floor": args.specificity_floor,
            "min_positives": args.min_positives,
            "min_negatives": args.min_negatives,
            "per_label": [item.to_dict() for item in selections],
        },
        "held_out_test": {
            "split": "test",
            "images": len(test["images"]),
            "patients": len(set(test["patient_ids"])),
            "patient_overlap": 0,
            "auc_per_label": compute_aucs(test_probs, test["labels"]),
            "operating_metrics": held_out,
        },
        "thresholds_complete": thresholds_complete,
        "pipeline_thresholds": pipeline_thresholds,
        "warning": (
            "Research configuration only. Public retrospective data; not clinical validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(artifact), indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {args.output} (thresholds_complete={thresholds_complete})")

    print("\nPer-label validation selection -> held-out test:")
    for item in selections:
        test_row = held_out[item.label]
        status = "OK" if item.supported and item.meets_constraints else "BLOCKED"
        print(
            f"  {item.label:<20} t={item.threshold:.3f}  "
            f"val sens/spec={item.sensitivity:.3f}/{item.specificity:.3f}  "
            f"test sens/spec={test_row['sensitivity']:.3f}/{test_row['specificity']:.3f}  "
            f"{status}"
        )


def _json_safe(value):
    """Convert undefined metrics (NaN/inf) to JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
