from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from experts.chest_xray import build_chest_xray_expert
from models.backbones import Backbone, TimmBackbone
from preprocessing.transforms import (
    NormalizeChannels,
    PreprocessConfig,
    build_preprocess,
)
from scripts.benchmark_chest_classifier import _load_custom_expert
from training.losses import AsymmetricLoss


class TinyConfiguredBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.out_channels = 2
        self.spatial_dims = 2
        self.data_config = {
            "mean": (0.1, 0.2, 0.3),
            "std": (0.4, 0.5, 0.6),
            "interpolation": "bicubic",
        }
        self.conv = nn.Conv2d(3, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AsymmetricLossTests(unittest.TestCase):
    def test_matches_official_detached_weight_sum_and_gradient(self) -> None:
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])

        loss = AsymmetricLoss(
            gamma_neg=4.0,
            gamma_pos=1.0,
            clip=0.05,
        )(logits, targets)
        loss.backward()

        positive_probability = 0.5
        shifted_negative_probability = 0.55
        positive_weight = (1.0 - positive_probability) ** 1.0
        negative_weight = (1.0 - shifted_negative_probability) ** 4.0
        expected_loss = -(
            math.log(positive_probability) * positive_weight
            + math.log(shifted_negative_probability) * negative_weight
        )
        # With detached focal weights, only the log-likelihood term contributes
        # to the gradient. These values differ from the non-detached formulation.
        expected_positive_gradient = positive_weight * (positive_probability - 1.0)
        expected_negative_gradient = (
            negative_weight
            * positive_probability
            * (1.0 - positive_probability)
            / shifted_negative_probability
        )

        self.assertAlmostEqual(float(loss.detach()), expected_loss, places=6)
        self.assertAlmostEqual(
            float(logits.grad[0, 0]), expected_positive_gradient, places=6
        )
        self.assertAlmostEqual(
            float(logits.grad[0, 1]), expected_negative_gradient, places=6
        )

    def test_default_reduction_sums_over_batch_and_labels(self) -> None:
        one = AsymmetricLoss()(torch.zeros(1, 2), torch.tensor([[1.0, 0.0]]))
        two = AsymmetricLoss()(
            torch.zeros(2, 2),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        )

        self.assertAlmostEqual(float(two), 2.0 * float(one), places=6)


class ChestPreprocessingTests(unittest.TestCase):
    def test_timm_backbone_resolves_its_weight_data_config(self) -> None:
        backbone = TimmBackbone("densenet121", pretrained=False)

        self.assertEqual(backbone.data_config["mean"], (0.485, 0.456, 0.406))
        self.assertEqual(backbone.data_config["std"], (0.229, 0.224, 0.225))
        self.assertEqual(backbone.data_config["interpolation"], "bicubic")

    def test_chest_expert_uses_backbone_contract_without_confidence_by_default(
        self,
    ) -> None:
        with patch(
            "experts.chest_xray.build_backbone",
            return_value=TinyConfiguredBackbone(),
        ):
            expert = build_chest_xray_expert(pretrained=False, image_size=16)

        self.assertEqual(list(expert.heads), ["cls"])
        normalizers = [
            transform
            for transform in expert.preprocess.transforms
            if isinstance(transform, NormalizeChannels)
        ]
        self.assertEqual(len(normalizers), 1)
        self.assertEqual(normalizers[0].mean, (0.1, 0.2, 0.3))
        self.assertEqual(normalizers[0].std, (0.4, 0.5, 0.6))
        resize = next(
            transform
            for transform in expert.preprocess.transforms
            if type(transform).__name__ == "Resize"
        )
        self.assertEqual(resize.mode, "bicubic")

    def test_training_augmentation_flips_width_axis_only(self) -> None:
        transform = build_preprocess(
            PreprocessConfig(
                spatial_size=(8, 8),
                in_channels=3,
            ),
            train=True,
        )
        flip = next(
            item
            for item in transform.transforms
            if type(item).__name__ == "RandFlip"
        )

        self.assertEqual(flip.flipper.spatial_axis, 1)

    def test_legacy_checkpoint_loader_reenables_confidence_head(self) -> None:
        checkpoint = {
            "model": {
                "heads.cls.fc.weight": torch.zeros(1),
                "heads.confidence.fc.weight": torch.zeros(1),
            },
            "class_names": ["Effusion"],
        }
        expert = MagicMock()
        expert.to.return_value = expert
        expert.eval.return_value = expert

        with tempfile.NamedTemporaryFile() as checkpoint_file, patch(
            "scripts.benchmark_chest_classifier.torch.load",
            return_value=checkpoint,
        ), patch(
            "scripts.benchmark_chest_classifier.build_chest_xray_expert",
            return_value=expert,
        ) as build:
            _load_custom_expert(
                Path(checkpoint_file.name),
                backbone="timm:densenet121",
                image_size=320,
                device="cpu",
            )

        self.assertTrue(build.call_args.kwargs["with_confidence"])


if __name__ == "__main__":
    unittest.main()
