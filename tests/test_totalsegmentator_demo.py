"""Offline checks for the visual TotalSegmentator Colab workflow.

No model weights, network, or GPU are used here.  These tests cover the archive
selection, measurement, preview, and checksum seams surrounding the expensive model
call so a Colab run does not discover basic artifact bugs after inference finishes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/doctor-assistant-matplotlib-tests")

from scripts.demo_totalsegmentator import (  # noqa: E402
    ZENODO_FILENAME,
    _ensure_archive,
    _extract_selected_ct,
    _render_preview,
    _select_preview_slice,
    _structure_measurements,
    _verify_archive,
)


class TotalSegmentatorDemoTests(unittest.TestCase):
    def test_extracts_only_the_selected_subject_ct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset/s001/ct.nii.gz", b"first")
                archive.writestr("dataset/s001/segmentations/liver.nii.gz", b"mask")
                archive.writestr("dataset/s002/ct.nii.gz", b"second")

            subject_id, ct_path = _extract_selected_ct(archive_path, root / "cache", 1)

            self.assertEqual(subject_id, "s002")
            self.assertEqual(ct_path.read_bytes(), b"second")
            self.assertFalse((root / "cache" / "subjects" / "s001").exists())

    def test_measurements_use_physical_voxel_volume(self) -> None:
        segmentation = np.array([[[0, 1], [1, 2]], [[1, 2], [2, 2]]], dtype=np.uint8)

        measurements = _structure_measurements(
            segmentation,
            spacing=(1.0, 2.0, 3.0),
            class_map={1: "liver", 2: "spleen"},
        )

        by_name = {item["name"]: item for item in measurements}
        self.assertEqual(by_name["liver"]["voxels"], 3)
        self.assertAlmostEqual(by_name["liver"]["volume_ml"], 0.018)
        self.assertEqual(by_name["spleen"]["voxels"], 4)
        self.assertEqual(measurements[0]["name"], "spleen")

    def test_preview_prefers_slice_with_abdominal_organs(self) -> None:
        segmentation = np.zeros((5, 5, 3), dtype=np.uint8)
        segmentation[:, :, 0] = 2  # non-priority anatomy covers more pixels
        segmentation[1:4, 1:4, 2] = 1  # liver should determine the selected slice

        selected = _select_preview_slice(segmentation, {1: "liver", 2: "vertebrae_L1"})

        self.assertEqual(selected, 2)

    def test_preview_png_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "preview.png"
            ct = np.linspace(-1000, 500, 6 * 6 * 3, dtype=np.float32).reshape(6, 6, 3)
            segmentation = np.zeros((6, 6, 3), dtype=np.uint8)
            segmentation[1:5, 1:5, 1] = 1

            selected = _render_preview(ct, segmentation, {1: "liver"}, destination)

            self.assertEqual(selected, 1)
            self.assertTrue(destination.exists())
            self.assertGreater(destination.stat().st_size, 1_000)

    def test_archive_receipt_is_bound_to_hash_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "sample.zip"
            archive_path.write_bytes(b"verified archive bytes")
            expected = hashlib.md5(archive_path.read_bytes()).hexdigest()  # noqa: S324

            with patch("scripts.demo_totalsegmentator.ZENODO_MD5", expected):
                _verify_archive(archive_path)

            receipt = json.loads(
                archive_path.with_suffix(".zip.verified.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["md5"], expected)
            self.assertEqual(receipt["size_bytes"], archive_path.stat().st_size)

    def test_incomplete_archive_is_resumed_before_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            archive_path = cache / "dataset" / ZENODO_FILENAME
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(b"partial")

            def finish_download(_url: str, destination: Path) -> None:
                destination.write_bytes(b"x" * 20)

            with (
                patch(
                    "scripts.demo_totalsegmentator._resolve_download_url",
                    return_value=("https://example.invalid/archive", 20),
                ),
                patch(
                    "scripts.demo_totalsegmentator._download_with_retries",
                    side_effect=finish_download,
                ) as download,
                patch("scripts.demo_totalsegmentator._verify_archive") as verify,
            ):
                result = _ensure_archive(cache)

            self.assertEqual(result, archive_path)
            self.assertEqual(archive_path.stat().st_size, 20)
            download.assert_called_once()
            verify.assert_called_once_with(archive_path)

    def test_bad_full_archive_is_quarantined_then_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            archive_path = cache / "dataset" / ZENODO_FILENAME
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(b"bad archive")
            expected_size = archive_path.stat().st_size

            def clean_download(_url: str, destination: Path) -> None:
                destination.write_bytes(b"goodarchive")

            with (
                patch(
                    "scripts.demo_totalsegmentator._resolve_download_url",
                    return_value=("https://example.invalid/archive", expected_size),
                ),
                patch(
                    "scripts.demo_totalsegmentator._download_with_retries",
                    side_effect=clean_download,
                ) as download,
                patch(
                    "scripts.demo_totalsegmentator._verify_archive",
                    side_effect=[RuntimeError("bad md5"), None],
                ) as verify,
            ):
                result = _ensure_archive(cache)

            self.assertEqual(result, archive_path)
            self.assertEqual(archive_path.read_bytes(), b"goodarchive")
            self.assertEqual(len(list(archive_path.parent.glob("*.bad-md5-*"))), 1)
            download.assert_called_once()
            self.assertEqual(verify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
