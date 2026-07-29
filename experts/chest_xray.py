"""ChestXray expert pack — the first end-to-end vertical slice.

Why chest X-ray first: the largest public datasets (NIH ChestX-ray14, CheXpert,
MIMIC-CXR) and the fastest path to a trained model. It also stresses the design in the
right places — findings *co-occur* (multi-label, not one-of), and there are no
segmentation masks, so localization rides on Grad-CAM rather than mask geometry. If the
pipeline reports cleanly here, the mask-based packs (brain MRI) are the easier case.

Architecture: one shared 2D backbone (DenseNet-121, the CheXNet standard) feeding a
multi-label classification head. A legacy confidence head remains available explicitly
for checkpoint compatibility, but is not trained by default: exact-match correctness is
a poor auxiliary target for sparse multi-label findings. Segmentation is intentionally
omitted — there is nothing to supervise it with — so 'where' comes from `GradCAM`.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.enums import BodyPart, Modality
from models.backbones import build_backbone
from models.experts import BaseExpert
from models.heads import ClassificationHead, ConfidenceHead
from preprocessing.transforms import PreprocessConfig, build_preprocess

# NIH ChestX-ray14 label set — the common benchmark. Swap for CheXpert's 13 if needed.
#
# ORDER IS LOAD-BEARING: a classification head's logit index i means "class_names[i]",
# so this tuple MUST list the labels in the exact order the head was trained against.
# Training builds its multi-hot targets from `data.chest_xray14.CHESTXRAY14_LABELS`
# (alphabetical), so this list is kept byte-for-byte identical to it. Reordering here
# without retraining silently mislabels every finding (logit for "Consolidation" gets
# read out as "Effusion", etc.). The trained checkpoint also stores its own
# `class_names`; prefer those when loading weights (see BaseExpert / the system test).
CHESTXRAY14_LABELS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
)


def build_chest_xray_expert(
    *,
    backbone: str = "timm:densenet121",
    labels: Sequence[str] = CHESTXRAY14_LABELS,
    pretrained: bool = True,
    image_size: int = 320,
    in_channels: int = 3,
    with_confidence: bool = False,
    train_preprocess: bool = False,
) -> BaseExpert:
    """Assemble a chest X-ray expert ready to train or to load weights into.

    `in_channels=3` so we can use ImageNet-pretrained DenseNet weights (grayscale is
    repeated to 3 by `AdaptChannels`). `train_preprocess=True` enables augmentation —
    use it for the training dataset, keep it False for inference and validation.
    """
    bb = build_backbone(backbone, spatial_dims=2, in_channels=in_channels, pretrained=pretrained)

    heads = {
        "cls": ClassificationHead(
            in_channels=bb.out_channels,
            num_classes=len(labels),
            spatial_dims=2,
            multilabel=True,  # chest findings co-occur -> sigmoid + BCE
        )
    }
    if with_confidence:
        heads["confidence"] = ConfidenceHead(bb.out_channels, spatial_dims=2)

    data_config = bb.data_config or {}
    channel_mean = data_config.get("mean")
    channel_std = data_config.get("std")
    if channel_mean is not None and len(channel_mean) != in_channels:
        raise ValueError(
            f"{backbone!r} publishes {len(channel_mean)} normalization channels, "
            f"but in_channels={in_channels}; use the model's native channel count"
        )
    if channel_std is not None and len(channel_std) != in_channels:
        raise ValueError(
            f"{backbone!r} publishes {len(channel_std)} normalization channels, "
            f"but in_channels={in_channels}; use the model's native channel count"
        )

    cfg = PreprocessConfig(
        spatial_size=(image_size, image_size),
        in_channels=in_channels,
        intensity="scale",  # X-ray: simple min-max to [0,1]
        channel_mean=(
            tuple(float(value) for value in channel_mean)
            if channel_mean is not None
            else None
        ),
        channel_std=(
            tuple(float(value) for value in channel_std)
            if channel_std is not None
            else None
        ),
        interpolation=(
            str(data_config["interpolation"])
            if data_config.get("interpolation") is not None
            else None
        ),
    )
    preprocess = build_preprocess(cfg, train=train_preprocess)

    return BaseExpert(
        name="chest_xray",
        modality=Modality.XRAY,
        body_part=BodyPart.CHEST,
        backbone=bb,
        heads=heads,
        class_names=list(labels),
        preprocess=preprocess,
    )
