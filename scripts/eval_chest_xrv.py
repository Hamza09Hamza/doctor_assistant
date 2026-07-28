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

_CACHE_DIR = Path("/private/tmp/claude-501/-Users-boukaderhamza-Documents-AI-Doctor-Assistant/e6153783-edb0-411d-a995-81f604828633/scratchpad")


def _to_multihot(raw_labels: list[str]) -> np.ndarray:
    vec = np.zeros(len(CHESTXRAY14_LABELS), dtype=np.float32)
    for label in raw_labels:
        if label in _LABEL_INDEX:
            vec[_LABEL_INDEX[label]] = 1.0
    return vec


def load_sample(n: int, seed: int):
    """Stream `n` real, labeled test images, caching to disk so repeat comparisons
    against different weight-set configs don't re-stream from HF each time."""
    cache_path = _CACHE_DIR / f"nih_sample_n{n}_seed{seed}.pt"
    if cache_path.is_file():
        print(f"Loading cached sample from {cache_path}")
        blob = torch.load(cache_path, weights_only=False)
        return blob["images"], blob["labels"]

    from datasets import load_dataset

    print(f"Streaming {n} test images from BahaaEldin0/NIH-Chest-Xray-14 ...")
    ds = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split="test", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=min(4000, max(200, n * 4)))

    images, label_vecs = [], []
    t0 = time.time()
    for row in ds.take(n):
        img = row["image"].convert("L")  # (1024, 1024) grayscale PNG
        arr = np.asarray(img, dtype=np.float32) / 255.0
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of test images to pull")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--configs",
        type=str,
        default="all,nih,chex,all+nih,all+chex,all+nih+chex",
        help="comma-separated list of +-joined short weight names, e.g. 'all,all+nih'",
    )
    args = ap.parse_args()

    images, labels = load_sample(args.n, args.seed)
    print(f"Sample: {len(images)} images, {labels.sum():.0f} total positive label instances")
    print("Per-label positive counts:",
          {c: int(labels[:, i].sum()) for i, c in enumerate(CHESTXRAY14_LABELS)})

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    results: dict[str, tuple[dict[str, float | None], float]] = {}

    for config in config_names:
        short_names = config.split("+")
        weights = tuple(_WEIGHT_MAP[s] for s in short_names)
        t0 = time.time()
        expert = TorchXRayVisionExpert(name=f"chest_xrv_{config}", weights=weights)
        probs = run_expert(expert, images)
        aucs = compute_aucs(probs, labels)
        valid = [v for v in aucs.values() if v is not None]
        macro = float(np.mean(valid)) if valid else float("nan")
        results[config] = (aucs, macro)
        print(f"  [{config:<16}] macro AUC over {len(valid)} labels: {macro:.4f}  ({time.time() - t0:.1f}s)")

    print("\n=== Summary, best to worst ===")
    for config, (_, macro) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"  {config:<16} macro AUC = {macro:.4f}")

    baseline = "all"
    if baseline in results:
        base_aucs, base_macro = results[baseline]
        print(f"\n=== Per-label detail vs baseline '{baseline}' (macro {base_macro:.4f}) ===")
        for config, (aucs, macro) in results.items():
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
