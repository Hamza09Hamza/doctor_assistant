"""Offline contract tests for Torch/MLX-independent volume segmentation helpers."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from experts.medsam2_volume import (
    SAM2MLXVolumeSegmenter,
    _hiera_pos_embed_sidecar_path,
    _use_multimask_official,
    largest_connected_component,
    prompt_connected_component,
)


class FakeMLXPredictor:
    def init_state(self, frame_dir, **_kwargs):
        self.frame_names = sorted(path.name for path in frame_dir.glob("*.png"))
        return {"depth": len(self.frame_names)}

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, box):
        self.prompts = getattr(self, "prompts", [])
        self.prompts.append((state, frame_idx, obj_id, np.asarray(box)))

    def propagate_in_video(self, state, *, start_frame_idx, reverse):
        if reverse:
            indices = range(start_frame_idx - 1, -1, -1)
        else:
            indices = range(start_frame_idx, state["depth"])
        for index in indices:
            logits = np.full((1, 1, 8, 9), -1.0, dtype=np.float32)
            logits[0, 0, 2:6, 3:7] = 1.0
            yield index, [1], logits

    def reset_state(self, state):
        self.resets = getattr(self, "resets", [])
        self.resets.append(state)


class SAM2MLXAdapterTests(unittest.TestCase):
    def test_converted_checkpoint_sidecar_path_is_unambiguous(self) -> None:
        self.assertEqual(
            _hiera_pos_embed_sidecar_path("/models/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors"),
            Path("/models/MedSAM2_CTLesion_hiera_tiny_mlx.pos_embed_raw.npz"),
        )

    def test_multimask_rule_matches_official_point_count_gate(self) -> None:
        self.assertTrue(_use_multimask_official(0))
        self.assertTrue(_use_multimask_official(1))
        # A box is encoded as two points, so official MedSAM2 uses single-mask output.
        self.assertFalse(_use_multimask_official(2))
        self.assertFalse(_use_multimask_official(3))

    def test_box_is_propagated_bidirectionally_in_source_order(self) -> None:
        with mock.patch("importlib.util.find_spec", return_value=object()):
            segmenter = SAM2MLXVolumeSegmenter(
                model="local-test.safetensors",
                keep_prompt_component=False,
            )
        predictor = FakeMLXPredictor()
        segmenter._predictor = predictor

        mask = segmenter.segment_volume(
            np.zeros((5, 8, 9), dtype=np.uint8),
            seed_index=2,
            box_xyxy=(3, 2, 7, 6),
        )

        self.assertEqual(mask.shape, (5, 8, 9))
        self.assertTrue(mask[:, 2:6, 3:7].all())
        self.assertFalse(mask[:, :2].any())
        self.assertEqual(len(predictor.frame_names), 5)
        self.assertEqual(len(predictor.prompts), 2)
        self.assertEqual(predictor.prompts[0][1:3], (2, 1))
        np.testing.assert_array_equal(predictor.prompts[0][3], [3, 2, 7, 6])
        self.assertEqual(len(predictor.resets), 2)
        self.assertIs(predictor.resets[-1], predictor.prompts[0][0])

    def test_largest_component_removes_disconnected_noise(self) -> None:
        mask = np.zeros((3, 8, 8), dtype=bool)
        mask[:, 2:6, 2:6] = True
        mask[0, 0, 0] = True
        cleaned = largest_connected_component(mask)
        self.assertEqual(int(cleaned.sum()), 3 * 4 * 4)
        self.assertFalse(cleaned[0, 0, 0])

    def test_prompt_component_wins_over_larger_remote_hallucination(self) -> None:
        mask = np.zeros((5, 20, 20), dtype=bool)
        mask[1:4, 8:12, 8:12] = True
        mask[:, :7, :7] = True
        cleaned = prompt_connected_component(mask, 2, (7, 7, 13, 13))
        self.assertEqual(int(cleaned.sum()), 3 * 4 * 4)
        self.assertFalse(cleaned[:, :7, :7].any())


if __name__ == "__main__":
    unittest.main()
