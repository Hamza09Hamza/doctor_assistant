"""Reproducible exploratory scorecard for one chest X-ray classifier at a time.

Development runs default to the third-party mirror partition named ``valid``. The
mirror's ``valid``/``test`` names have not been reconciled against NIH's official
filename manifests, so even a frozen ``--split test`` run is exploratory rather than
official test evidence. The JSON report records this provenance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata
from data.chest_xray14 import CHESTXRAY14_LABELS
from evaluation import (
    evaluate_multilabel_classifier,
    evaluate_subgroups,
    load_calibrated_thresholds,
    rank_classification_errors,
)
from experts.chest_xray import build_chest_xray_expert
from experts.torchxrayvision import TorchXRayVisionExpert
from models.experts import strip_orig_mod
from scripts.calibrate_chest_thresholds import load_split_sample
from scripts.eval_chest_xrv import _DATASET_ID, _DATASET_REVISION, run_expert

_DEFAULT_CACHE = Path(
    os.environ.get(
        "DOCTOR_ASSISTANT_CACHE_DIR",
        Path.home() / ".cache" / "doctor_assistant" / "evaluation",
    )
)


def _load_custom_expert(
    checkpoint_path: Path,
    *,
    backbone: str,
    image_size: int,
    device: str | None,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("checkpoint has no 'model' state dictionary")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(
            "checkpoint has no class_names; label order cannot be inferred safely"
        )
    state = strip_orig_mod(checkpoint["model"])
    with_confidence = any(key.startswith("heads.confidence.") for key in state)
    expert = build_chest_xray_expert(
        backbone=backbone,
        labels=class_names,
        pretrained=False,
        image_size=image_size,
        with_confidence=with_confidence,
    )
    expert.load_state_dict(state, strict=True)
    expert.to(target_device).eval()
    return expert, checkpoint


def _run_custom_expert(expert, images: list[torch.Tensor]) -> np.ndarray:
    label_index = {name: index for index, name in enumerate(CHESTXRAY14_LABELS)}
    probabilities = np.zeros(
        (len(images), len(CHESTXRAY14_LABELS)), dtype=np.float32
    )
    for row_index, image in enumerate(images):
        scan = Scan(
            data=image,
            meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST),
        )
        prediction = expert.predict(scan)
        for label, probability in prediction.class_probs.items():
            if label in label_index:
                probabilities[row_index, label_index[label]] = probability
    return probabilities


def _age_groups(ages: list[int]) -> list[str]:
    groups = []
    for age in ages:
        if age < 18:
            groups.append("under_18")
        elif age < 40:
            groups.append("18_to_39")
        elif age < 65:
            groups.append("40_to_64")
        else:
            groups.append("65_plus")
    return groups


def _model_and_probabilities(args, sample):
    if args.model == "xrv":
        expert = TorchXRayVisionExpert(weights=args.xrv_weights, device=args.device)
        probabilities = run_expert(expert, sample["images"])
        model_info = {
            "kind": "torchxrayvision",
            "name": expert.name,
            "weights": list(expert.weights),
            "resolution": expert.resolution,
            "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        }
        return probabilities, model_info

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --model custom")
    expert, checkpoint = _load_custom_expert(
        args.checkpoint,
        backbone=args.backbone,
        image_size=args.image_size,
        device=args.device,
    )
    probabilities = _run_custom_expert(expert, sample["images"])
    model_info = {
        "kind": "custom",
        "name": expert.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "backbone": args.backbone,
        "image_size": args.image_size,
        "class_names": list(expert.class_names),
    }
    return probabilities, model_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("xrv", "custom"), default="xrv")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", default="timm:densenet121")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--xrv-weights", default="densenet121-res224-all")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--dataset-revision", default=_DATASET_REVISION)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-artifact",
        type=Path,
        help="completed validation-selected threshold artifact",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--subgroup-min-images", type=int, default=30)
    parser.add_argument("--error-limit", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/chest_classifier_benchmark.json"),
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.split == "test" and args.threshold_artifact is None:
        print(
            "WARNING: mirror partition 'test' requested without frozen validation "
            "thresholds; this run is ranking analysis only."
        )

    sample = load_split_sample(
        args.split,
        args.n,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
        require_metadata=True,
    )
    probabilities, model_info = _model_and_probabilities(args, sample)
    thresholds = (
        load_calibrated_thresholds(args.threshold_artifact)
        if args.threshold_artifact
        else args.threshold
    )
    labels = np.asarray(sample["labels"])
    scorecard = evaluate_multilabel_classifier(
        probabilities,
        labels,
        CHESTXRAY14_LABELS,
        thresholds=thresholds,
        patient_ids=sample["patient_ids"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    subgroups = evaluate_subgroups(
        probabilities,
        labels,
        CHESTXRAY14_LABELS,
        {
            "gender": sample["patient_genders"],
            "view_position": sample["view_positions"],
            "age_group": _age_groups(sample["patient_ages"]),
        },
        thresholds=thresholds,
        patient_ids=sample["patient_ids"],
        min_images=args.subgroup_min_images,
    )
    errors = rank_classification_errors(
        probabilities,
        labels,
        CHESTXRAY14_LABELS,
        thresholds=thresholds,
        sample_ids=sample["sample_ids"],
        limit=args.error_limit,
    )

    prediction_path = args.output.with_suffix(".predictions.npz")
    source_filename_count = sum(
        name is not None for name in sample["source_filenames"]
    )
    smoke_only = source_filename_count == 0
    if smoke_only:
        print(
            "SMOKE ONLY: this mirror sample exposes no original NIH filenames; "
            "official manifest membership cannot be checked."
        )
    artifact = {
        "schema_version": 2,
        "smoke_only": smoke_only,
        "purpose": (
            "development_validation"
            if args.split == "valid"
            else "exploratory_mirror_test_analysis"
        ),
        "dataset": {
            "id": _DATASET_ID,
            "revision": args.dataset_revision,
            "split": args.split,
            "partition_provenance": sample["partition_provenance"],
            "official_nih_manifest_reconciled": sample[
                "official_manifest_reconciled"
            ],
            "seed": args.seed,
            "images": len(labels),
            "patients": len(set(sample["patient_ids"])),
            "source_filename_ids_available": source_filename_count,
            "source_filename_ids_total": len(sample["source_filenames"]),
            "smoke_only": smoke_only,
        },
        "model": model_info,
        "threshold_source": (
            str(args.threshold_artifact.resolve())
            if args.threshold_artifact
            else f"unvalidated_global:{args.threshold}"
        ),
        "scorecard": scorecard,
        "subgroups": subgroups,
        "ranked_errors": errors,
        "predictions": str(prediction_path),
        "warning": (
            (
                "SMOKE ONLY: the mirror exposed no original NIH filenames, so image "
                "membership cannot be reconciled. "
            )
            if smoke_only
            else ""
        )
        + (
            "Research benchmark only. This Hugging Face mirror partition was not "
            "reconciled against NIH's official image manifests; it is not official "
            "validation/test evidence or evidence of clinical validity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(artifact), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        prediction_path,
        probabilities=probabilities,
        labels=labels,
        class_names=np.asarray(CHESTXRAY14_LABELS),
        patient_ids=np.asarray(sample["patient_ids"]),
        sample_ids=np.asarray(sample["sample_ids"]),
        source_filenames=np.asarray(
            [name or "" for name in sample["source_filenames"]]
        ),
    )

    macro = scorecard["macro"]
    study = scorecard["study_level"]
    print(f"Wrote {args.output} and {prediction_path}")
    print(
        f"macro AUROC={_fmt(macro['auroc'])}  "
        f"AUPRC={_fmt(macro['auprc'])}  "
        f"Brier={_fmt(macro['brier'])}  ECE={_fmt(macro['ece'])}"
    )
    print(
        f"macro sensitivity={_fmt(macro['sensitivity'])}  "
        f"specificity={_fmt(macro['specificity'])}  "
        f"normal-any-FP={_fmt(study['normal_any_false_positive'])}"
    )


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
