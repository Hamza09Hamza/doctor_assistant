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
    stratify: bool = True,
) -> list[Sample]:
    """Return `Sample` objects for the requested split.

    The NIH dataset only ships train_val_list.txt and test_list.txt; there is no
    dedicated validation file. `stratify=True` (default) uses iterative
    stratification (Sechidis et al., 2011) to build train/val so each of the 14
    labels keeps its overall positive rate in both folds — a plain random split can
    otherwise starve val of a rare label like Hernia (~0.2% prevalence) by chance,
    making its val-set AUC meaningless. Pass `stratify=False` for the old
    hash-of-names random split.
    """
    label_index = {l: i for i, l in enumerate(labels)}
    csv_path = os.path.join(root, "Data_Entry_2017.csv")
    image_dir = os.path.join(root, "images")
    _check_paths(csv_path, image_dir)

    rows = _read_rows(csv_path)  # Image Index -> raw "Finding Labels" string, read once
    train_val_names, test_names = _load_split_files(root)

    if split in ("train", "val"):
        if stratify:
            label_vecs = {
                name: _parse_labels(finding_str, label_index, len(labels))
                for name, finding_str in rows.items()
                if name in train_val_names
            }
            val_set = _stratified_val_split(train_val_names, label_vecs, len(labels), val_fraction, seed)
        else:
            rng = random.Random(seed)
            val_set = set(
                rng.sample(sorted(train_val_names), int(len(train_val_names) * val_fraction))
            )
        allowed = val_set if split == "val" else (train_val_names - val_set)
    elif split == "test":
        allowed = test_names
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got {split!r}")

    samples: list[Sample] = []
    for name, finding_str in rows.items():
        if name not in allowed:
            continue
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue  # skip missing files gracefully
        label_vec = _parse_labels(finding_str, label_index, len(labels))
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

def _read_rows(csv_path: str) -> dict[str, str]:
    """Image Index -> raw 'Finding Labels' string, read once and reused for both the
    split decision and sample construction (avoids parsing the CSV twice)."""
    rows: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["Image Index"].strip()] = row["Finding Labels"]
    return rows


def _stratified_val_split(
    train_val_names: set[str],
    label_vecs: dict[str, list[float]],
    n_labels: int,
    val_fraction: float,
    seed: int,
) -> set[str]:
    """Iterative stratification (Sechidis, Tsoumakas & Vlahavas, 2011), specialized to a
    2-way train/val split.

    Repeatedly picks the rarest label that still has unassigned positive examples, and
    sends each of those examples to whichever fold is furthest below its target share
    for that label — this is what keeps a rare label's val-set prevalence close to its
    overall prevalence instead of leaving it to chance. "No Finding" samples (an
    all-zero label vector) carry no stratification signal, so they're distributed last,
    by plain proportional random split, purely to hit the requested fold sizes.
    """
    names = sorted(train_val_names)
    rng = random.Random(seed)
    rng.shuffle(names)

    positive_names = [nm for nm in names if sum(label_vecs[nm]) > 0]
    no_finding_names = [nm for nm in names if sum(label_vecs[nm]) == 0]

    remaining_size = {"val": val_fraction * len(names), "train": (1.0 - val_fraction) * len(names)}
    remaining_label_count = {
        fold: [
            frac * sum(label_vecs[nm][i] for nm in positive_names)
            for i in range(n_labels)
        ]
        for fold, frac in (("val", val_fraction), ("train", 1.0 - val_fraction))
    }

    assigned: dict[str, str] = {}
    unassigned = set(positive_names)

    while unassigned:
        counts_left = [0] * n_labels
        for nm in unassigned:
            vec = label_vecs[nm]
            for i in range(n_labels):
                if vec[i] > 0:
                    counts_left[i] += 1
        candidate_labels = [i for i, c in enumerate(counts_left) if c > 0]
        if not candidate_labels:
            break  # remaining unassigned samples carry only labels outside `labels`
        target_label = min(candidate_labels, key=lambda i: counts_left[i])

        examples = [nm for nm in unassigned if label_vecs[nm][target_label] > 0]
        rng.shuffle(examples)
        for nm in examples:
            fold = max(
                ("val", "train"),
                key=lambda f: (remaining_label_count[f][target_label], remaining_size[f], rng.random()),
            )
            assigned[nm] = fold
            unassigned.discard(nm)
            vec = label_vecs[nm]
            for i in range(n_labels):
                if vec[i] > 0:
                    remaining_label_count[fold][i] -= 1
            remaining_size[fold] -= 1

    rng.shuffle(no_finding_names)
    for nm in no_finding_names:
        fold = "val" if remaining_size["val"] >= remaining_size["train"] else "train"
        assigned[nm] = fold
        remaining_size[fold] -= 1

    return {nm for nm, fold in assigned.items() if fold == "val"}


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
