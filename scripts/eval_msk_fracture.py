"""Local eval: does the pretrained YOLOv8 wrist-fracture detector work on real images?

Pulls a real, labeled sample of GRAZPEDWRI-DX wrist X-rays (via the
MuhammadJazib/GRAZPEDWRI-DX_SMALL HF mirror) and checks image-level presence detection
per class against ground truth, with "fracture" (the clinically primary target) reported
first. There are no bounding boxes in this mirror's labels (just which of the 9 GRAZPEDWRI
classes appear in each image), so this measures presence/absence detection via AUC — the
same methodology as scripts/eval_chest_xrv.py — rather than object-detection mAP.

CAVEAT — read before trusting this number: this HF mirror's relationship to the official
GRAZPEDWRI-DX train/valid/test split (the one the pretrained checkpoint was actually
trained and evaluated on) is not confirmed. Some of these images may have been seen during
training. Treat this as "does the pretrained detector behave sensibly on real wrist X-rays",
not a certified held-out generalization benchmark — a true held-out number would need the
authors' exact split (linked from their repo, not scraped here).

Extra deps beyond requirements.txt (eval-only):
    pip install ultralytics datasets huggingface_hub

Run:  python scripts/eval_msk_fracture.py --n 300
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
from experts.msk_fracture import GRAZPEDWRI_LABELS, MSKFractureExpert

_LABEL_INDEX = {name: i for i, name in enumerate(GRAZPEDWRI_LABELS)}
_CACHE_DIR = Path(
    os.environ.get(
        "DOCTOR_ASSISTANT_CACHE_DIR",
        Path.home() / ".cache" / "doctor_assistant" / "evaluation",
    )
)


def _to_multihot(label_str: str) -> np.ndarray:
    vec = np.zeros(len(GRAZPEDWRI_LABELS), dtype=np.float32)
    for tok in label_str.split():
        idx = int(tok)
        if 0 <= idx < len(GRAZPEDWRI_LABELS):
            vec[idx] = 1.0
    return vec


def load_sample(n: int, seed: int, cache_dir: Path = _CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"grazpedwri_sample_n{n}_seed{seed}.pt"
    if cache_path.is_file():
        print(f"Loading cached sample from {cache_path}")
        blob = torch.load(cache_path, weights_only=False)
        return blob["images"], blob["labels"]

    from datasets import load_dataset

    print(f"Streaming {n} images from MuhammadJazib/GRAZPEDWRI-DX_SMALL ...")
    ds = load_dataset("MuhammadJazib/GRAZPEDWRI-DX_SMALL", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=min(1200, max(200, n * 4)))

    images, label_vecs = [], []
    t0 = time.time()
    for row in ds.take(n):
        img = row["image"].convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr).unsqueeze(0))  # (1, H, W)
        label_vecs.append(_to_multihot(row["label"]))
    print(f"  done in {time.time() - t0:.1f}s")
    labels = np.stack(label_vecs)
    torch.save({"images": images, "labels": labels}, cache_path)
    return images, labels


def run_expert(expert: MSKFractureExpert, images: list[torch.Tensor]) -> np.ndarray:
    probs = np.zeros((len(images), len(GRAZPEDWRI_LABELS)), dtype=np.float32)
    t0 = time.time()
    for i, img in enumerate(images):
        scan = Scan(
            data=img,
            meta=ScanMetadata(modality=Modality.XRAY, body_part=BodyPart.BONE, source_path=f"sample_{i}.png"),
        )
        pred = expert.predict(scan)
        for label, p in pred.class_probs.items():
            probs[i, _LABEL_INDEX[label]] = p
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(images)}")
    print(f"  {expert.name}: {time.time() - t0:.1f}s total")
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=_CACHE_DIR,
        help="sample cache directory (or set DOCTOR_ASSISTANT_CACHE_DIR)",
    )
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score

    images, labels = load_sample(args.n, args.seed, args.cache_dir)
    print(f"Sample: {len(images)} images")
    print("Per-class positive counts:",
          {c: int(labels[:, i].sum()) for i, c in enumerate(GRAZPEDWRI_LABELS)})

    expert = MSKFractureExpert(confidence=args.confidence)
    probs = run_expert(expert, images)

    print(f"\n=== Presence-detection AUC (confidence threshold={args.confidence}) ===")
    aucs = {}
    for i, cname in enumerate(GRAZPEDWRI_LABELS):
        gt = labels[:, i]
        if gt.sum() > 0 and (1 - gt).sum() > 0:
            aucs[cname] = float(roc_auc_score(gt, probs[:, i]))
        else:
            aucs[cname] = None

    fracture_auc = aucs.get("fracture")
    print(f"\n*** fracture (primary target): n_pos={int(labels[:, _LABEL_INDEX['fracture']].sum())}  "
          f"AUC={'n/a' if fracture_auc is None else f'{fracture_auc:.4f}'} ***\n")

    for cname, auc in sorted(aucs.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0))):
        n_pos = int(labels[:, _LABEL_INDEX[cname]].sum())
        auc_str = f"{auc:.4f}" if auc is not None else "  n/a  (too few positives in sample)"
        print(f"  {cname:<20} n_pos={n_pos:>4}  AUC={auc_str}")

    # A simple presence-detection confusion breakdown for "fracture" at conf>=threshold,
    # since that's the number a clinician would actually read.
    gt = labels[:, _LABEL_INDEX["fracture"]]
    pred_present = (probs[:, _LABEL_INDEX["fracture"]] >= args.confidence).astype(float)
    tp = int(((pred_present == 1) & (gt == 1)).sum())
    fn = int(((pred_present == 0) & (gt == 1)).sum())
    tn = int(((pred_present == 0) & (gt == 0)).sum())
    fp = int(((pred_present == 1) & (gt == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"\nFracture presence @ conf>={args.confidence}: TP={tp} FN={fn} TN={tn} FP={fp}  "
          f"sensitivity={sens:.3f}  specificity={spec:.3f}")


if __name__ == "__main__":
    main()
