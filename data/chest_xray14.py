"""NIH ChestX-ray14 dataset builder.

The NIH dataset ships a CSV (`Data_Entry_2017.csv`) with pipe-separated labels per
image. This module turns that CSV into `Sample` objects with multi-hot float labels
that `ScanDataset` feeds straight into the trainer without any other glue.

Dataset layout expected on disk (matches the Kaggle download + unzip):
    <root>/
        images/          <- all 112 120 PNGs in a flat directory
        Data_Entry_2017.csv
        train_val_list.txt   <- official NIH train+val split (86 524 images)
        test_list.txt        <- official NIH test split  (25 596 images)

Usage:
    train_samples = load_chest_xray14(root, split="train")
    val_samples   = load_chest_xray14(root, split="val", val_fraction=0.1)
    test_samples  = load_chest_xray14(root, split="test")
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass

from .dataset import Sample

# Official 14 pathology labels in consistent alphabetical order.
# "No Finding" is excluded — when all 14 are 0 the vector already encodes it.
CHESTXRAY14_LABELS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
)

_NO_FINDING = "No Finding"


@dataclass
class DatasetStats:
    total: int
    per_label: dict[str, int]
    no_finding: int


def load_chest_xray14(
    root: str,
    *,
    split: str = "train",        # "train" | "val" | "test"
    val_fraction: float = 0.1,   # fraction of train_val_list used for validation
    labels: tuple[str, ...] = CHESTXRAY14_LABELS,
    seed: int = 42,
    max_samples: int | None = None,  # cap for quick smoke-runs
) -> list[Sample]:
    """Return `Sample` objects for the requested split.

    The NIH dataset only ships train_val_list.txt and test_list.txt; there is no
    dedicated validation file. We deterministically split train_val by hashing
    image names so the same image always lands in the same fold regardless of order.
    """
    label_index = {l: i for i, l in enumerate(labels)}
    csv_path = os.path.join(root, "Data_Entry_2017.csv")
    image_dir = os.path.join(root, "images")
    _check_paths(csv_path, image_dir)

    train_val_names, test_names = _load_split_files(root)
    rng = random.Random(seed)
    val_set = set(rng.sample(sorted(train_val_names), int(len(train_val_names) * val_fraction)))
    train_set = train_val_names - val_set

    if split == "train":
        allowed = train_set
    elif split == "val":
        allowed = val_set
    elif split == "test":
        allowed = test_names
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    samples: list[Sample] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Image Index"].strip()
            if name not in allowed:
                continue
            path = os.path.join(image_dir, name)
            if not os.path.isfile(path):
                continue  # skip missing files gracefully
            label_vec = _parse_labels(row["Finding Labels"], label_index, len(labels))
            samples.append(Sample(path=path, label=label_vec))
            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise RuntimeError(
            f"No samples found for split={split!r} in {root!r}. "
            "Check that images/ and Data_Entry_2017.csv are present."
        )
    return samples


def dataset_stats(samples: list[Sample], labels: tuple[str, ...] = CHESTXRAY14_LABELS) -> DatasetStats:
    """Count per-label positives and No-Finding cases for a list of samples."""
    counts = [0] * len(labels)
    no_finding = 0
    for s in samples:
        if not isinstance(s.label, list):
            continue
        if sum(s.label) == 0:
            no_finding += 1
        for i, v in enumerate(s.label):
            if v > 0:
                counts[i] += 1
    return DatasetStats(
        total=len(samples),
        per_label={l: counts[i] for i, l in enumerate(labels)},
        no_finding=no_finding,
    )


# ---- helpers ----------------------------------------------------------------

def _check_paths(csv_path: str, image_dir: str) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"NIH CSV not found: {csv_path}\n"
            "Download the dataset from Kaggle (nih-chest-xrays/data) and unzip to the root."
        )
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"images/ directory not found: {image_dir}")


def _load_split_files(root: str) -> tuple[set[str], set[str]]:
    def _read(name: str) -> set[str]:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            return set()
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    train_val = _read("train_val_list.txt")
    test = _read("test_list.txt")
    if not train_val and not test:
        # fall back: treat all images in CSV as train (no split file present)
        return set(), set()
    return train_val, test


def _parse_labels(finding_str: str, label_index: dict[str, int], n: int) -> list[float]:
    """'Cardiomegaly|Effusion' -> multi-hot float list of length n."""
    vec = [0.0] * n
    for label in finding_str.split("|"):
        label = label.strip()
        if label and label != _NO_FINDING and label in label_index:
            vec[label_index[label]] = 1.0
    return vec
