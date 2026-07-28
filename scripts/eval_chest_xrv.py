"""Local eval: which TorchXRayVision weight-set combination actually wins on real data?

Pulls a real, labeled sample of the NIH ChestX-ray14 test split (via the
BahaaEldin0/NIH-Chest-Xray-14 HF dataset — the same real-image source already used
elsewhere in this project; ChestMNIST is too low-resolution to trust for this), caches it
locally, and runs it through however many weight-set configurations you ask for, reporting
per-label + macro AUC for each so they can be compared head to head.

No training happens here — every configuration is deploy-and-go pretrained weights. This
only answers: "does averaging op-norm-calibrated scores across these particular weight
sets move real AUC on real images, and in which direction" — which turned out to matter:
a first pass on 300 real NIH test images showed the naive all+nih+chex ensemble *losing*
to plain "all" (mean macro AUC -0.0255, winning only 4/13 scoreable labels), which
motivated adding multi-config comparison here instead of trusting the ensemble-is-free-win
assumption from the literature review that originally suggested it.

Extra deps beyond requirements.txt (eval-only, not needed for the core pipeline):
    pip install torchxrayvision datasets huggingface_hub

Run:
    python scripts/eval_chest_xrv.py --n 300
    python scripts/eval_chest_xrv.py --n 300 --configs all,nih,chex,all+nih,all+chex,all+nih+chex
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata
from data.chest_xray14 import CHESTXRAY14_LABELS
from experts.torchxrayvision import TorchXRayVisionExpert

_LABEL_INDEX = {name: i for i, name in enumerate(CHESTXRAY14_LABELS)}
_DATASET_ID = "BahaaEldin0/NIH-Chest-Xray-14"
_DATASET_REVISION = "932bcdba9d7d9590704d4f20bc70fc2c3a1bbad7"

# Short config name -> torchxrayvision weight string. Extend here to try more combinations.
_WEIGHT_MAP = {
    "all": "densenet121-res224-all",
    "nih": "densenet121-res224-nih",
    "chex": "densenet121-res224-chex",
    "mimic_ch": "densenet121-res224-mimic_ch",
    "mimic_nb": "densenet121-res224-mimic_nb",
    "pc": "densenet121-res224-pc",
    "rsna": "densenet121-res224-rsna",
}

_CACHE_DIR = Path(
    os.environ.get(
        "DOCTOR_ASSISTANT_CACHE_DIR",
        Path.home() / ".cache" / "doctor_assistant" / "evaluation",
    )
)


def _to_multihot(raw_labels: list[str]) -> np.ndarray:
    vec = np.zeros(len(CHESTXRAY14_LABELS), dtype=np.float32)
    for label in raw_labels:
        if label in _LABEL_INDEX:
            vec[_LABEL_INDEX[label]] = 1.0
    return vec


def load_sample(
    n: int,
    seed: int,
    cache_dir: Path = _CACHE_DIR,
    dataset_revision: str = _DATASET_REVISION,
):
    """Stream `n` real, labeled test images, caching to disk so repeat comparisons
    against different weight-set configs don't re-stream from HF each time."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"nih_test_n{n}_seed{seed}_rev{dataset_revision[:12]}.pt"
    )
    if cache_path.is_file():
        print(f"Loading cached sample from {cache_path}")
        blob = torch.load(cache_path, weights_only=False)
        return blob["images"], blob["labels"]

    from datasets import load_dataset

    print(f"Streaming {n} test images from BahaaEldin0/NIH-Chest-Xray-14 ...")
    ds = load_dataset(
        _DATASET_ID,
        split="test",
        streaming=True,
        revision=dataset_revision,
    )
    ds = ds.shuffle(seed=seed, buffer_size=min(4000, max(200, n * 4)))

    images, label_vecs = [], []
    t0 = time.time()
    for row in ds.take(n):
        img = row["image"].convert("L")  # (1024, 1024) grayscale PNG
        # Preserve uint8 in the cache; the expert owns normalization. This cuts the
        # full-resolution sample cache and RAM footprint by 4× on Colab.
        arr = np.asarray(img, dtype=np.uint8)
        images.append(torch.from_numpy(arr).unsqueeze(0))  # (1, H, W)
        label_vecs.append(_to_multihot(row["label"]))
    print(f"  done in {time.time() - t0:.1f}s")
    labels = np.stack(label_vecs)
    torch.save({"images": images, "labels": labels}, cache_path)
    return images, labels


def run_expert(expert: TorchXRayVisionExpert, images: list[torch.Tensor]) -> np.ndarray:
    probs = np.zeros((len(images), len(CHESTXRAY14_LABELS)), dtype=np.float32)
    for i, img in enumerate(images):
        scan = Scan(data=img, meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.CHEST))
        pred = expert.predict(scan)
        for label, p in pred.class_probs.items():
            probs[i, _LABEL_INDEX[label]] = p
    return probs


def compute_aucs(probs: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import roc_auc_score

    aucs: dict[str, float | None] = {}
    for i, cname in enumerate(CHESTXRAY14_LABELS):
        gt = labels[:, i]
        if gt.sum() > 0 and (1 - gt).sum() > 0:
            aucs[cname] = float(roc_auc_score(gt, probs[:, i]))
        else:
            aucs[cname] = None
    return aucs


def compute_threshold_metrics(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, float]:
    """Operational metrics at the threshold that turns scores into report findings."""
    pred = probs >= threshold
    truth = labels > 0
    sensitivities: list[float] = []
    specificities: list[float] = []
    for i in range(labels.shape[1]):
        positive = truth[:, i]
        negative = ~positive
        if positive.any():
            sensitivities.append(float(pred[positive, i].mean()))
        if negative.any():
            specificities.append(float((~pred[negative, i]).mean()))

    normal = truth.sum(axis=1) == 0
    normal_any_fp = float(pred[normal].any(axis=1).mean()) if normal.any() else float("nan")
    normal_mean_findings = (
        float(pred[normal].sum(axis=1).mean()) if normal.any() else float("nan")
    )
    return {
        "macro_sensitivity": float(np.mean(sensitivities)) if sensitivities else float("nan"),
        "macro_specificity": float(np.mean(specificities)) if specificities else float("nan"),
        "normal_any_false_positive": normal_any_fp,
        "normal_mean_findings": normal_mean_findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of test images to pull")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=_CACHE_DIR,
        help="sample cache directory (or set DOCTOR_ASSISTANT_CACHE_DIR)",
    )
    ap.add_argument(
        "--dataset-revision",
        default=_DATASET_REVISION,
        help="pinned Hugging Face dataset commit",
    )
    ap.add_argument(
        "--configs",
        type=str,
        default="all,nih,chex,all+nih,all+chex,all+nih+chex",
        help="comma-separated list of +-joined short weight names, e.g. 'all,all+nih'",
    )
    ap.add_argument(
        "--thresholds",
        default="0.5,0.6,0.7,0.8",
        help="comma-separated reporting thresholds for sensitivity/specificity checks",
    )
    args = ap.parse_args()

    images, labels = load_sample(
        args.n,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
    )
    print(f"Sample: {len(images)} images, {labels.sum():.0f} total positive label instances")
    print("Per-label positive counts:",
          {c: int(labels[:, i].sum()) for i, c in enumerate(CHESTXRAY14_LABELS)})

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        ap.error("--thresholds must contain values between 0 and 1")
    results: dict[str, tuple[dict[str, float | None], float, np.ndarray]] = {}

    for config in config_names:
        short_names = config.split("+")
        weights = tuple(_WEIGHT_MAP[s] for s in short_names)
        t0 = time.time()
        expert = TorchXRayVisionExpert(name=f"chest_xrv_{config}", weights=weights)
        probs = run_expert(expert, images)
        aucs = compute_aucs(probs, labels)
        valid = [v for v in aucs.values() if v is not None]
        macro = float(np.mean(valid)) if valid else float("nan")
        results[config] = (aucs, macro, probs)
        print(f"  [{config:<16}] macro AUC over {len(valid)} labels: {macro:.4f}  ({time.time() - t0:.1f}s)")
        print("    reporting-threshold behavior:")
        for threshold in thresholds:
            operational = compute_threshold_metrics(probs, labels, threshold)
            print(
                f"      t={threshold:.2f}  "
                f"macro sensitivity={operational['macro_sensitivity']:.3f}  "
                f"macro specificity={operational['macro_specificity']:.3f}  "
                f"normal studies with any FP={operational['normal_any_false_positive']:.3f}  "
                f"mean findings/normal={operational['normal_mean_findings']:.2f}"
            )

    print("\n=== Summary, best to worst ===")
    for config, (_, macro, _) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"  {config:<16} macro AUC = {macro:.4f}")

    baseline = "all"
    if baseline in results:
        base_aucs, base_macro, _ = results[baseline]
        print(f"\n=== Per-label detail vs baseline '{baseline}' (macro {base_macro:.4f}) ===")
        for config, (aucs, macro, _) in results.items():
            if config == baseline:
                continue
            print(f"\n--- {config} (macro {macro:.4f}, delta {macro - base_macro:+.4f}) ---")
            deltas = []
            for cname in CHESTXRAY14_LABELS:
                b, o = base_aucs[cname], aucs[cname]
                if b is not None and o is not None:
                    d = o - b
                    deltas.append(d)
                    print(f"  {cname:<20} {baseline}={b:.4f}  {config}={o:.4f}  delta={d:+.4f}")
            if deltas:
                print(f"  -> wins on {sum(d > 0 for d in deltas)}/{len(deltas)} labels")


if __name__ == "__main__":
    main()
