"""Encoders (backbones) — the *shared* part of every expert.

A backbone maps an image to a feature map (it does NOT classify). That feature
map is what the task heads consume and what Grad-CAM taps, so the contract is
deliberately narrow: `forward(x) -> feature_map` plus `out_channels`.

  - `TimmBackbone`         2D CNNs from `timm` with ImageNet weights.
  - `MonaiDenseNetBackbone` 2D or 3D DenseNet from MONAI (for CT/MRI volumes).
  - `HFViTBackbone`        HuggingFace ViT encoders (RAD-DINO, BiomedCLIP-style) —
                           medical *foundation* models, reshaped to a feature map so
                           the existing pooling heads consume them unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class Backbone(nn.Module):
    """Base contract for encoders. Subclasses set `out_channels`/`spatial_dims`."""

    out_channels: int
    spatial_dims: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class TimmBackbone(Backbone):
    """2D CNN encoder via `timm`, returning the deepest feature map."""

    def __init__(self, name: str, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.net = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )
        self.spatial_dims = 2
        self.out_channels = int(self.net.feature_info.channels()[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[-1]  # deepest feature map


class MonaiDenseNetBackbone(Backbone):
    """DenseNet feature extractor (2D or 3D) from MONAI, minus its classifier."""

    def __init__(
        self,
        variant: str = "densenet121",
        spatial_dims: int = 3,
        in_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        from monai.networks import nets

        factory = {
            "densenet121": nets.DenseNet121,
            "densenet169": nets.DenseNet169,
            "densenet201": nets.DenseNet201,
        }[variant]
        full = factory(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=2,  # classifier is discarded; value is irrelevant
            pretrained=pretrained,
        )
        self.features = full.features  # the convolutional trunk
        self.spatial_dims = spatial_dims
        self.out_channels = self._infer_out_channels(spatial_dims, in_channels)

    def _infer_out_channels(self, spatial_dims: int, in_channels: int) -> int:
        """Run a tiny dummy forward to read the feature-map channel count.

        Robust across MONAI versions, which expose this attribute inconsistently.
        """
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            shape = (2, in_channels) + (32,) * spatial_dims
            feat = self.features(torch.zeros(shape))
        self.features.train(was_training)
        return int(feat.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class HFViTBackbone(Backbone):
    """A HuggingFace ViT encoder (e.g. RAD-DINO) exposed as a 2D feature map.

    Medical foundation encoders such as `microsoft/rad-dino` (a DINOv2 ViT pre-trained
    on chest radiographs) output a sequence of patch tokens, not a conv feature map. We
    drop the CLS token and reshape the patch tokens back to a (B, C, h, w) grid, so the
    existing `ClassificationHead` (which global-pools) and Grad-CAM (which taps a spatial
    map) work without any change.

    Image size must be a multiple of the model's patch size (14 for RAD-DINO). We pass
    `interpolate_pos_encoding=True` so non-default sizes still work by interpolating the
    positional embeddings — but a multiple of the patch size is required for the token
    grid to be square and reshape cleanly (e.g. 518 = 37×14, or 294 = 21×14).
    """

    def __init__(self, model_id: str, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel

        if pretrained:
            self.net = AutoModel.from_pretrained(model_id)
        else:
            from transformers import AutoConfig
            self.net = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
        self.spatial_dims = 2
        self.out_channels = int(self.net.config.hidden_size)
        self.patch_size = int(getattr(self.net.config, "patch_size", 14))
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = self.net(pixel_values=x, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state          # (B, 1 + n_patches, C) — CLS first
        gh, gw = h // self.patch_size, w // self.patch_size
        patches = tokens[:, 1:, :]              # drop CLS
        # Guard against models that emit register tokens: keep the trailing gh*gw.
        patches = patches[:, patches.shape[1] - gh * gw:, :]
        feat = patches.transpose(1, 2).reshape(b, self.out_channels, gh, gw)
        return feat.contiguous()


def build_backbone(
    name: str,
    spatial_dims: int = 2,
    in_channels: int = 3,
    pretrained: bool = True,
) -> Backbone:
    """Construct a backbone from a `prefix:variant` name.

    Examples:
      "timm:resnet50", "timm:densenet121"          ImageNet CNNs (2D only)
      "monai:densenet121"                          2D/3D DenseNet for volumes
      "hf:microsoft/rad-dino"                       medical foundation ViT (2D only)
    """
    if ":" not in name:
        raise ValueError(f"Backbone name must be 'prefix:variant', got {name!r}")
    prefix, variant = name.split(":", 1)

    if prefix == "timm":
        if spatial_dims != 2:
            raise ValueError("timm backbones support 2D only; use 'monai:' for 3D.")
        return TimmBackbone(variant, in_channels=in_channels, pretrained=pretrained)
    if prefix == "monai":
        return MonaiDenseNetBackbone(
            variant=variant,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            pretrained=pretrained,
        )
    if prefix == "hf":
        if spatial_dims != 2:
            raise ValueError("hf ViT backbones support 2D only.")
        return HFViTBackbone(variant, in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unknown backbone prefix {prefix!r} (expected 'timm', 'monai', or 'hf').")
