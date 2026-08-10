"""Offline checks for the MedSAM interactive-segmentation helpers: RLE encode/decode
round-trip and box-clipping. No checkpoint, torch, or GPU is touched here -- mirrors
tests/test_totalsegmentator_demo.py's approach of unit-testing the seams around the
expensive model call, not the model call itself.
"""

from __future__ import annotations

import unittest

import numpy as np

from experts.medsam_interactive import (
    clip_box_to_image,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)


class RLERoundTripTests(unittest.TestCase):
    def test_round_trip_random_mask(self) -> None:
        rng = np.random.default_rng(0)
        mask = rng.random((17, 23)) > 0.5

        rle = encode_binary_mask_rle(mask)
        decoded = decode_binary_mask_rle(rle)

        np.testing.assert_array_equal(mask, decoded)

    def test_round_trip_all_false(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)

        decoded = decode_binary_mask_rle(encode_binary_mask_rle(mask))

        np.testing.assert_array_equal(mask, decoded)

    def test_round_trip_all_true(self) -> None:
        mask = np.ones((5, 5), dtype=bool)

        decoded = decode_binary_mask_rle(encode_binary_mask_rle(mask))

        np.testing.assert_array_equal(mask, decoded)

    def test_round_trip_starts_on_foreground(self) -> None:
        # First run must be a background run by convention -- exercise the case where
        # the mask itself starts True, which needs the leading zero-run prepended.
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True

        rle = encode_binary_mask_rle(mask)
        self.assertEqual(rle["counts"][0], 0)
        decoded = decode_binary_mask_rle(rle)

        np.testing.assert_array_equal(mask, decoded)

    def test_size_recorded_correctly(self) -> None:
        mask = np.zeros((7, 3), dtype=bool)

        rle = encode_binary_mask_rle(mask)

        self.assertEqual(rle["size"], [7, 3])


class ClipBoxToImageTests(unittest.TestCase):
    def test_box_within_bounds_unchanged(self) -> None:
        box = clip_box_to_image((10, 10, 50, 60), width=100, height=100)
        self.assertEqual(box, (10, 10, 50, 60))

    def test_box_partly_outside_is_clamped(self) -> None:
        box = clip_box_to_image((-20, -20, 50, 60), width=100, height=100)
        self.assertEqual(box, (0, 0, 50, 60))

    def test_box_entirely_outside_raises(self) -> None:
        with self.assertRaises(ValueError):
            clip_box_to_image((200, 200, 250, 260), width=100, height=100)

    def test_degenerate_zero_area_box_raises(self) -> None:
        with self.assertRaises(ValueError):
            clip_box_to_image((10, 10, 10, 10), width=100, height=100)

    def test_reversed_coordinates_are_sorted(self) -> None:
        box = clip_box_to_image((50, 60, 10, 10), width=100, height=100)
        self.assertEqual(box, (10, 10, 50, 60))


if __name__ == "__main__":
    unittest.main()
