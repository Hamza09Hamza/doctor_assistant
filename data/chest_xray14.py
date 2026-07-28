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
    group_by_patient: bool = True,
) -> list[Sample]:
    """Return `Sample` objects for the requested split.

    The NIH dataset only ships train_val_list.txt and test_list.txt; there is no
    dedicated validation file. `stratify=True` (default) uses iterative
    stratification (Sechidis et al., 2011) to build train/val so each of the 14
    labels keeps its overall positive rate in both folds. `group_by_patient=True`
    (default) assigns every image sharing the NIH filename patient prefix to the same
    fold, preventing patient leakage. Pass `stratify=False` for a grouped random split,
    or `group_by_patient=False` only for a dataset whose filenames do not carry patient
    identity.
    """
    label_index = {l: i for i, l in enumerate(labels)}
    csv_path = os.path.join(root, "Data_Entry_2017.csv")
    image_dir = os.path.join(root, "images")
    _check_paths(csv_path, image_dir)

    rows = _read_rows(csv_path)  # Image Index -> raw "Finding Labels" string, read once
    train_val_names, test_names = _load_split_files(root)
    test_names &= set(rows)
    if not train_val_names:
        # Some mirrors omit NIH's official split text files. Preserve the documented
        # fallback by treating CSV rows not explicitly assigned to test as
        # train+validation data. Never invent a test partition.
        train_val_names = set(rows) - test_names
    else:
        # Ignore stale split entries that have no matching CSV row.
        train_val_names &= set(rows)

    if split in ("train", "val"):
        if stratify:
            label_vecs = {
                name: _parse_labels(finding_str, label_index, len(labels))
                for name, finding_str in rows.items()
                if name in train_val_names
            }
            val_set = _stratified_val_split(
                train_val_names,
                label_vecs,
                len(labels),
                val_fraction,
                seed,
                group_by_patient=group_by_patient,
            )
        else:
            val_set = _random_val_split(
                train_val_names,
                val_fraction,
                seed,
                group_by_patient=group_by_patient,
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
    *,
    group_by_patient: bool = True,
) -> set[str]:
    """Iterative stratification (Sechidis, Tsoumakas & Vlahavas, 2011), specialized to a
    2-way train/val split.

    The assignment unit is a patient group by default, not an image. Repeatedly picks
    the rarest label that still has unassigned positive groups, and sends each group to
    whichever fold is furthest below its target share for that label. Group label counts
    retain the number of positive images, so balancing remains image-prevalence-aware.
    All-zero groups are distributed last to approach the requested fold sizes.
    """
    names = sorted(train_val_names)
    rng = random.Random(seed)
    rng.shuffle(names)

    groups = _group_names(names, group_by_patient)
    group_labels = {
        group: [
            sum(label_vecs[name][i] for name in members)
            for i in range(n_labels)
        ]
        for group, members in groups.items()
    }
    positive_groups = [group for group, vec in group_labels.items() if sum(vec) > 0]
    no_finding_groups = [group for group, vec in group_labels.items() if sum(vec) == 0]

    remaining_size = {"val": val_fraction * len(names), "train": (1.0 - val_fraction) * len(names)}
    remaining_label_count = {
        fold: [
            frac * sum(group_labels[group][i] for group in positive_groups)
            for i in range(n_labels)
        ]
        for fold, frac in (("val", val_fraction), ("train", 1.0 - val_fraction))
    }

    assigned: dict[str, str] = {}
    unassigned = set(positive_groups)

    while unassigned:
        counts_left = [0] * n_labels
        for group in unassigned:
            vec = group_labels[group]
            for i in range(n_labels):
                if vec[i] > 0:
                    counts_left[i] += 1
        candidate_labels = [i for i, c in enumerate(counts_left) if c > 0]
        if not candidate_labels:
            break  # remaining unassigned samples carry only labels outside `labels`
        target_label = min(candidate_labels, key=lambda i: counts_left[i])

        examples = [
            group for group in unassigned
            if group_labels[group][target_label] > 0
        ]
        rng.shuffle(examples)
        for group in examples:
            fold = max(
                ("val", "train"),
                key=lambda f: (remaining_label_count[f][target_label], remaining_size[f], rng.random()),
            )
            assigned[group] = fold
            unassigned.discard(group)
            vec = group_labels[group]
            for i in range(n_labels):
                if vec[i] > 0:
                    remaining_label_count[fold][i] -= vec[i]
            remaining_size[fold] -= len(groups[group])

    rng.shuffle(no_finding_groups)
    for group in no_finding_groups:
        fold = "val" if remaining_size["val"] >= remaining_size["train"] else "train"
        assigned[group] = fold
        remaining_size[fold] -= len(groups[group])

    return {
        name
        for group, fold in assigned.items()
        if fold == "val"
        for name in groups[group]
    }


def _random_val_split(
    names: set[str],
    val_fraction: float,
    seed: int,
    *,
    group_by_patient: bool,
) -> set[str]:
    """Grouped random split used when iterative stratification is disabled."""
    groups = _group_names(sorted(names), group_by_patient)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    target = val_fraction * len(names)
    chosen: list[str] = []
    size = 0
    for group in group_ids:
        if size >= target:
            break
        chosen.append(group)
        size += len(groups[group])
    return {name for group in chosen for name in groups[group]}


def _group_names(
    names: list[str], group_by_patient: bool
) -> dict[str, list[str]]:
    """Group NIH filenames (`00000001_000.png`) by their patient prefix."""
    groups: dict[str, list[str]] = {}
    for name in names:
        group = _patient_id(name) if group_by_patient else name
        groups.setdefault(group, []).append(name)
    return groups


def _patient_id(name: str) -> str:
    return os.path.basename(name).split("_", 1)[0]


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
    return train_val, test


def _parse_labels(finding_str: str, label_index: dict[str, int], n: int) -> list[float]:
    """'Cardiomegaly|Effusion' -> multi-hot float list of length n."""
    vec = [0.0] * n
    for label in finding_str.split("|"):
        label = label.strip()
        if label and label != _NO_FINDING and label in label_index:
            vec[label_index[label]] = 1.0
    return vec
