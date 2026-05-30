"""Datasets that feed the trainer.

`ScanDataset` reuses the ingestion + preprocessing layers so training and
inference see *identical* data handling — the single most common source of
train/serve skew in medical AI. `image_folder_samples` reads the ubiquitous
`root/<class>/<image>` layout (the Kaggle-style split) into labeled samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.utils.data import Dataset

from core.enums import BodyPart, Modality
from ingest.loaders import load_scan

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".nii.gz", ".dcm")


@dataclass
class Sample:
    """One training example: a scan path, its label, and an optional mask.

    `label` is an int for single-label tasks (brain tumour: one class wins) or a
    list of floats for multi-label tasks (chest X-ray: 14 independent binary labels).
    The dataset converts a list to a float32 tensor; an int to a long scalar.
    """

    path: str
    label: "int | list[float] | None" = None
    mask_path: str | None = None
    meta: dict = field(default_factory=dict)


def image_folder_samples(root: str) -> tuple[list[Sample], list[str]]:
    """Scan `root/<class_name>/<file>` into samples + sorted class-name list.

    Class indices follow the sorted class-name order so they stay stable across
    machines (don't rely on filesystem ordering).
    """
    class_names = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    index = {name: i for i, name in enumerate(class_names)}
    samples: list[Sample] = []
    for name in class_names:
        cls_dir = os.path.join(root, name)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(_IMAGE_EXTS):
                samples.append(Sample(path=os.path.join(cls_dir, fname), label=index[name]))
    return samples, class_names


class ScanDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        preprocess: Callable,
        modality: Modality | None = None,
        body_part: BodyPart | None = None,
        mask_preprocess: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.preprocess = preprocess
        self.modality = modality
        self.body_part = body_part
        self.mask_preprocess = mask_preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
        s = self.samples[i]
        scan = load_scan(s.path, modality=self.modality, body_part=self.body_part)
        x = self.preprocess(scan.data)
        # MONAI transforms return MetaTensor; torch.compile's CUDA graph autotuner
        # calls as_strided on it and fails. Strip to a plain contiguous tensor here.
        if hasattr(x, 'as_tensor'):
            x = x.as_tensor().contiguous()
        elif not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x).contiguous()

        targets: dict[str, torch.Tensor] = {}
        if s.label is not None:
            if isinstance(s.label, list):
                # multi-hot float vector for multilabel tasks (e.g. chest X-ray)
                targets["label"] = torch.tensor(s.label, dtype=torch.float32)
            else:
                targets["label"] = torch.tensor(s.label, dtype=torch.long)
        if s.mask_path is not None and self.mask_preprocess is not None:
            mask_scan = load_scan(s.mask_path)
            targets["mask"] = self.mask_preprocess(mask_scan.data).long()
        return x, targets
