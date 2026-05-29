"""ChestXray expert pack — the first end-to-end vertical slice.

Why chest X-ray first: the largest public datasets (NIH ChestX-ray14, CheXpert,
MIMIC-CXR) and the fastest path to a trained model. It also stresses the design in the
right places — findings *co-occur* (multi-label, not one-of), and there are no
segmentation masks, so localization rides on Grad-CAM rather than mask geometry. If the
pipeline reports cleanly here, the mask-based packs (brain MRI) are the easier case.

Architecture: one shared 2D backbone (DenseNet-121, the CheXNet standard) feeding a
multi-label classification head and a confidence head. Segmentation is intentionally
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
CHESTXRAY14_LABELS: tuple[str, ...] = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
)


def build_chest_xray_expert(
    *,
    backbone: str = "timm:densenet121",
    labels: Sequence[str] = CHESTXRAY14_LABELS,
    pretrained: bool = True,
    image_size: int = 320,
    in_channels: int = 3,
    with_confidence: bool = True,
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

    cfg = PreprocessConfig(
        spatial_size=(image_size, image_size),
        in_channels=in_channels,
        intensity="scale",  # X-ray: simple min-max to [0,1]
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
