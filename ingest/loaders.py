"""Turn files on disk into model-ready `Scan` objects.

Two loaders cover the radiology spectrum:
  - `Image2DLoader`   planar images (X-ray, mammography, fundus): PNG/JPG/TIFF.
  - `VolumeLoader`    volumetric medical formats (CT/MRI): NIfTI, DICOM (file or
                      series directory), delegated to MONAI for correct geometry.

`load_scan` is the dispatcher: it asks each loader whether it recognizes the path
and uses the first match. Add a format by writing a new Loader and registering it.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from core.enums import BodyPart, Modality
from core.types import Scan, ScanMetadata

_IMAGE_2D_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_VOLUME_EXTS = (".nii", ".nii.gz", ".dcm", ".mha", ".mhd", ".nrrd")


def _has_ext(path: str, exts: tuple[str, ...]) -> bool:
    lower = path.lower()
    return any(lower.endswith(e) for e in exts)


class Image2DLoader:
    """Planar images via Pillow, returned channels-first and scaled to [0, 1]."""

    def can_load(self, path: str) -> bool:
        return os.path.isfile(path) and _has_ext(path, _IMAGE_2D_EXTS)

    def load(self, path: str) -> Scan:
        from PIL import Image

        img = Image.open(path)
        arr = np.asarray(img)
        if arr.ndim == 2:  # grayscale -> (1, H, W)
            arr = arr[None, ...]
        else:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))

        data = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)
        # Scale common integer dtypes to [0, 1]; leave already-float data alone.
        max_val = float(data.max()) if data.numel() else 1.0
        if max_val > 1.0:
            data = data / (255.0 if max_val <= 255.0 else max_val)

        meta = ScanMetadata(
            original_shape=tuple(int(s) for s in arr.shape),
            source_path=path,
        )
        return Scan(data=data, meta=meta)


class VolumeLoader:
    """CT/MRI volumes and DICOM series, delegated to MONAI's readers."""

    def can_load(self, path: str) -> bool:
        if os.path.isdir(path):  # a DICOM series directory
            return True
        return os.path.isfile(path) and _has_ext(path, _VOLUME_EXTS)

    def load(self, path: str) -> Scan:
        from monai.transforms import LoadImage

        reader = LoadImage(image_only=True, ensure_channel_first=True)
        tensor = reader(path)  # MetaTensor, channels-first

        data = torch.as_tensor(tensor.as_tensor() if hasattr(tensor, "as_tensor") else tensor,
                               dtype=torch.float32)

        spacing = None
        affine = getattr(tensor, "affine", None)
        if affine is not None:
            affine = torch.as_tensor(affine)
            spacing = tuple(float(affine[i, i].abs()) for i in range(min(3, affine.shape[0] - 1)))

        meta = ScanMetadata(
            spacing=spacing,
            original_shape=tuple(int(s) for s in data.shape),
            source_path=path,
        )
        return Scan(data=data, meta=meta)


# Registered in priority order; the first loader that recognizes a path wins.
DEFAULT_LOADERS: list = [Image2DLoader(), VolumeLoader()]


def load_scan(
    path: str,
    modality: Modality | None = None,
    body_part: BodyPart | None = None,
    loaders: list | None = None,
) -> Scan:
    """Load `path` into a `Scan`, optionally stamping known modality/body part.

    Modality and body part are caller-supplied here because reliably *detecting*
    them is the router's job (a later layer); at ingestion we only record what we
    already know.
    """
    for loader in loaders or DEFAULT_LOADERS:
        if loader.can_load(path):
            scan = loader.load(path)
            if modality is not None:
                scan.meta.modality = modality
            if body_part is not None:
                scan.meta.body_part = body_part
            return scan
    raise ValueError(f"No registered loader can handle: {path!r}")
