"""Run purpose-built MONAI pathology experts and measure them against ground truth.

Two experts, both Apache 2.0, both from the MONAI Model Zoo, both *purpose-built for
finding disease* rather than segmenting anatomy:

  brain_tumor  -- `brats_mri_segmentation`. 3D SegResNet on 4-sequence brain MRI.
                  Outputs three nested tumour subregions (TC/WT/ET). The bundle's own
                  published validation Dice is TC 0.8559 / WT 0.9026 / ET 0.7905
                  (avg 0.8518) -- that is the number this run is compared against.
  lung_nodule  -- `lung_nodule_ct_detection`. 3D RetinaNet trained on LUNA16.
                  Outputs *boxes*, not masks. Reported mAP 0.852 / mAR 0.998 on its
                  own validation fold.

Why these two and not a "universal abnormality detector": no such model exists with a
usable licence and 3D input. BiomedParse is 2D-only, gated and CC-BY-NC-SA; MedSAM2 is
promptable (it needs a human to point first, so it cannot discover anything); MAIRA-2 is
MSRLA research-only. See docs/AI_HANDOFF.md section 10 for the full licence framework.

WHAT THIS SCRIPT DOES *NOT* ESTABLISH
-------------------------------------
A Dice number produced here is **indicative, not held-out**, unless the contamination
check below says otherwise. Both bundles were trained on the same public benchmarks the
test cases come from, so a case may sit in the model's own training split. Every run
records a `contamination_status` field that is one of:

    CLEAN            -- verified the case is not in the model's training split
    UNVERIFIED       -- overlap could not be ruled out; treat metrics as indicative only
    CONTAMINATED     -- the case is known to be in the training split; metrics are void

This mirrors the discipline already applied to KAD-512 (docs/CHEST_CLASSIFIER_RESET.md)
and the reason TorchXRayVision was disqualified as an NIH comparator. Do not report a
number from this script without also reporting its contamination status.

Usage (Colab, GPU):

    python scripts/run_monai_pathology_experts.py --expert brain_tumor \
        --data-dir /content/drive/MyDrive/doctor_assistant/monai_experts/data \
        --output-dir /content/drive/MyDrive/doctor_assistant/monai_experts/results \
        --cases 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BRAIN_TUMOR_BUNDLE = "brats_mri_segmentation"
LUNG_NODULE_BUNDLE = "lung_nodule_ct_detection"

# Published validation Dice from the bundle's own model card -- the bar this run is
# measured against. Not a claim about our data; a reference point.
BRATS_REFERENCE_DICE = {"TC": 0.8559, "WT": 0.9026, "ET": 0.7905, "average": 0.8518}

# TC/WT/ET are the standard BraTS nested subregions, in MONAI's channel order.
BRATS_CHANNELS = ("TC (tumour core)", "WT (whole tumour)", "ET (enhancing tumour)")


def log(message: str) -> None:
    """Timestamped, flushed. Colab hides buffered child output; see AI_HANDOFF_2 s5."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def download_bundle(name: str, bundle_root: Path) -> Path:
    """Fetch a MONAI Model Zoo bundle, reusing an existing copy when present."""
    from monai.bundle import download

    target = bundle_root / name
    if (target / "configs" / "inference.json").is_file():
        log(f"Reusing bundle already present at {target}")
        return target

    bundle_root.mkdir(parents=True, exist_ok=True)
    log(f"Downloading MONAI bundle '{name}' (weights included; first run only) ...")
    download(name=name, bundle_dir=str(bundle_root))
    if not (target / "configs" / "inference.json").is_file():
        raise RuntimeError(
            f"bundle '{name}' downloaded but configs/inference.json is missing at {target}"
        )
    log(f"Bundle ready: {target}")
    return target


def _load_network_from_bundle(bundle_dir: Path, device):
    """Instantiate the bundle's own network and load its checkpoint.

    The network definition comes from the bundle's config rather than being re-declared
    here -- re-declaring it by hand is how checkpoints silently fail to match.
    """
    import torch
    from monai.bundle import ConfigParser

    parser = ConfigParser()
    parser.read_config(str(bundle_dir / "configs" / "inference.json"))
    metadata_path = bundle_dir / "configs" / "metadata.json"
    if metadata_path.is_file():
        parser.read_meta(str(metadata_path))

    network = None
    for key in ("network_def", "network"):
        try:
            network = parser.get_parsed_content(key, instantiate=True)
            break
        except Exception:  # noqa: BLE001 -- config key naming varies between bundles
            continue
    if network is None:
        raise RuntimeError(f"could not instantiate a network from {bundle_dir}/configs")

    weights = bundle_dir / "models" / "model.pt"
    if not weights.is_file():
        raise RuntimeError(f"bundle checkpoint missing: {weights}")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and not hasattr(state, "state_dict"):
        network.load_state_dict(state)
    else:  # a fully pickled module
        network = state
    return network.to(device).eval()


def run_brain_tumor(args) -> dict:
    """Brain-tumour segmentation on public MSD Task01 (BraTS-derived) MRI."""
    import numpy as np
    import torch
    from monai.apps import DecathlonDataset
    from monai.data import DataLoader
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        Spacingd,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    bundle_dir = download_bundle(BRAIN_TUMOR_BUNDLE, args.data_dir / "bundles")
    network = _load_network_from_bundle(bundle_dir, device)
    log("Network instantiated from the bundle's own config and checkpoint loaded.")

    # MSD Task01_BrainTumour is the public, registration-free BraTS-derived release
    # (MONAI downloads it from its own S3 mirror). Labels ship with it, which is what
    # makes a real Dice measurement possible at all.
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    log("Preparing MSD Task01_BrainTumour (~7 GB on first run; cached afterwards) ...")
    dataset = DecathlonDataset(
        root_dir=str(args.data_dir),
        task="Task01_BrainTumour",
        transform=transform,
        section="validation",
        download=True,
        cache_rate=0.0,
        num_workers=2,
    )
    log(f"Validation section ready: {len(dataset)} studies available.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

    per_case: list[dict] = []
    started = time.time()
    for index, batch in enumerate(loader):
        if index >= args.cases:
            break
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        case_started = time.time()
        log(f"Case {index + 1}/{args.cases}: input {tuple(image.shape)} -> inferring ...")

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=image,
                roi_size=(224, 224, 144),
                sw_batch_size=1,
                predictor=network,
                overlap=0.5,
            )
            prediction = (torch.sigmoid(logits) > 0.5).float()

        dice_metric(y_pred=prediction, y=label)
        scores = dice_metric.aggregate().tolist()
        dice_metric.reset()

        voxels = {
            name: int(prediction[0, c].sum().item())
            for c, name in enumerate(BRATS_CHANNELS)
        }
        case = {
            "index": index,
            "dice": {n: round(float(s), 4) for n, s in zip(("TC", "WT", "ET"), scores)},
            "predicted_voxels": voxels,
            "seconds": round(time.time() - case_started, 2),
        }
        per_case.append(case)
        log(
            f"  Dice  TC={case['dice']['TC']:.4f}  WT={case['dice']['WT']:.4f}  "
            f"ET={case['dice']['ET']:.4f}   ({case['seconds']}s)"
        )

        if args.save_masks:
            import nibabel as nib

            out = args.output_dir / f"brain_tumor_case{index:03d}_pred.nii.gz"
            mask = prediction[0].cpu().numpy().astype(np.uint8)
            nib.save(nib.Nifti1Image(np.moveaxis(mask, 0, -1), np.eye(4)), str(out))
            log(f"  saved mask -> {out}")

    if not per_case:
        raise RuntimeError("no cases were processed; check --cases and the dataset")

    mean = {
        key: round(sum(c["dice"][key] for c in per_case) / len(per_case), 4)
        for key in ("TC", "WT", "ET")
    }
    mean["average"] = round(sum(mean.values()) / 3, 4)

    log("")
    log("=== BRAIN TUMOUR RESULT ===")
    log(f"cases evaluated : {len(per_case)}")
    for key in ("TC", "WT", "ET", "average"):
        ours = mean[key]
        ref = BRATS_REFERENCE_DICE[key]
        log(f"  {key:<8} ours={ours:.4f}   bundle-reported={ref:.4f}   delta={ours - ref:+.4f}")

    return {
        "expert": "brain_tumor",
        "bundle": BRAIN_TUMOR_BUNDLE,
        "licence": "Apache-2.0",
        "dataset": "MSD Task01_BrainTumour (BraTS-derived, public)",
        "section": "validation",
        "cases_evaluated": len(per_case),
        "mean_dice": mean,
        "bundle_reported_dice": BRATS_REFERENCE_DICE,
        "per_case": per_case,
        "total_seconds": round(time.time() - started, 2),
        # The bundle was trained on BraTS 2018 with "the authors' own division scheme",
        # which is not published in a form we can intersect against MSD Task01 ids. So
        # overlap between this validation section and the bundle's training split cannot
        # currently be ruled out. Stated plainly rather than quietly assumed away.
        "contamination_status": "UNVERIFIED",
        "contamination_note": (
            "brats_mri_segmentation was trained on BraTS 2018 using the bundle authors' "
            "own 200/42/43 split, which is not published in a form that can be "
            "intersected with MSD Task01 case ids. Treat Dice as indicative of a working "
            "pipeline, NOT as held-out performance."
        ),
    }


def run_lung_nodule(args) -> dict:
    """Download and stage the LUNA16-trained nodule detector.

    Deliberately stops before claiming a detection metric: the pretrained checkpoint was
    trained *and* validated on LUNA16 fold 0, and the LIDC case this project has been
    using (LIDC-IDRI-0686, series prefix 1.3.6.1.4.1.14519.5.2.1.6279.6001.* -- the
    LUNA16 prefix) may well sit inside that fold. Running it and reporting a hit rate
    before resolving that would be exactly the contamination mistake that disqualified
    TorchXRayVision as an NIH comparator.
    """
    bundle_dir = download_bundle(LUNG_NODULE_BUNDLE, args.data_dir / "bundles")

    split_files = sorted((bundle_dir).rglob("*fold*.json")) + sorted(
        (bundle_dir).rglob("*split*.json")
    )
    log("")
    log("=== LUNG NODULE DETECTOR: STAGED, NOT SCORED ===")
    log(f"bundle    : {bundle_dir}")
    log(f"split files shipped with bundle: {[p.name for p in split_files] or 'none found'}")
    log(
        "Not scoring yet: the checkpoint trained AND validated on LUNA16 fold 0, and the "
        "project's LIDC test case carries the LUNA16 series prefix. Fold membership must "
        "be resolved before any detection number means anything."
    )

    return {
        "expert": "lung_nodule",
        "bundle": LUNG_NODULE_BUNDLE,
        "licence": "Apache-2.0",
        "status": "staged_pending_contamination_check",
        "bundle_dir": str(bundle_dir),
        "split_files_found": [p.name for p in split_files],
        "contamination_status": "UNVERIFIED",
        "contamination_note": (
            "Pretrained checkpoint was trained and validated on LUNA16 fold 0. "
            "LIDC-IDRI-0686's series UID uses the LUNA16 prefix, so it may be in that "
            "fold. Resolve fold membership before reporting any detection metric."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expert",
        choices=("brain_tumor", "lung_nodule", "both"),
        default="brain_tumor",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=int,
        default=3,
        help="how many validation studies to evaluate (brain_tumor only)",
    )
    parser.add_argument("--save-masks", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.expert in ("brain_tumor", "both"):
        results.append(run_brain_tumor(args))
    if args.expert in ("lung_nodule", "both"):
        results.append(run_lung_nodule(args))

    manifest = {
        "workflow": "monai_pathology_experts",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
        "notice": (
            "Experimental research evaluation. Not a diagnosis, not clinical validation, "
            "and not a medical device."
        ),
    }
    out = args.output_dir / "monai_experts_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    log("")
    log(f"Manifest written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
