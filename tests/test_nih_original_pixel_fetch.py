from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.benchmark_kad import load_image_provenance, validate_image_provenance
from scripts.fetch_nih_original_images import (
    OriginalPixelIngestionError,
    verify_and_ingest_original_images,
)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "patient_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path: Path, *, size: tuple[int, int] = (9, 8), fill: int = 0) -> None:
    array = np.full((size[1], size[0]), fill, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


class OriginalPixelIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

        self.manifest_path = self.root / "manifest.csv"
        _write_manifest(
            self.manifest_path,
            [
                {
                    "filename": "00000001_000.png",
                    "patient_id": "00000001",
                    "split": "validation",
                },
                {
                    "filename": "00000002_000.png",
                    "patient_id": "00000002",
                    "split": "validation",
                },
            ],
        )

        self.source_a = self.root / "source_a"
        self.source_b = self.root / "source_b"
        self.source_a.mkdir()
        self.source_b.mkdir()
        for source in (self.source_a, self.source_b):
            _write_image(source / "00000001_000.png")
            _write_image(source / "00000002_000.png", fill=40)

        self.output_dir = self.root / "output"

    def test_two_agreeing_sources_are_accepted_and_loadable(self) -> None:
        provenance = verify_and_ingest_original_images(
            self.manifest_path,
            self.output_dir,
            {"NIH direct release": self.source_a, "torrent mirror": self.source_b},
        )
        self.assertEqual(provenance["schema_version"], 2)
        self.assertTrue(provenance["original_nih_pixels"])
        self.assertEqual(len(provenance["images"]), 2)
        for record in provenance["images"]:
            self.assertEqual(record["agreeing_source_count"], 2)

        provenance_path = self.output_dir / "ORIGINAL_NIH_PIXELS.provenance.json"
        self.assertTrue(provenance_path.is_file())

        # The artifact this script writes must be exactly what benchmark_kad's
        # loader accepts as verified original-pixel evidence.
        loaded = load_image_provenance(provenance_path)
        self.assertEqual(loaded.schema_version, 2)
        self.assertTrue(loaded.original_nih_pixels)

        from types import SimpleNamespace

        manifest = SimpleNamespace(
            manifest_sha256=loaded.canonical_manifest_sha256,
            cohort="development",
            sample_ids=("00000001_000.png", "00000002_000.png"),
            image_paths=(
                self.output_dir / "00000001_000.png",
                self.output_dir / "00000002_000.png",
            ),
            image_sha256=(
                loaded.output_sha256_by_filename["00000001_000.png"],
                loaded.output_sha256_by_filename["00000002_000.png"],
            ),
        )
        validate_image_provenance(loaded, manifest)

    def test_disagreeing_sources_hard_fail_naming_the_file(self) -> None:
        _write_image(self.source_b / "00000001_000.png", fill=99)

        with self.assertRaisesRegex(
            OriginalPixelIngestionError,
            r"00000001_000\.png: sources disagree",
        ):
            verify_and_ingest_original_images(
                self.manifest_path,
                self.output_dir,
                {"a": self.source_a, "b": self.source_b},
            )

    def test_too_few_sources_carrying_a_file_hard_fail(self) -> None:
        (self.source_b / "00000002_000.png").unlink()

        with self.assertRaisesRegex(
            OriginalPixelIngestionError,
            r"00000002_000\.png: only 1 of 2 source\(s\)",
        ):
            verify_and_ingest_original_images(
                self.manifest_path,
                self.output_dir,
                {"a": self.source_a, "b": self.source_b},
            )

    def test_partial_coverage_still_meets_the_minimum(self) -> None:
        source_c = self.root / "source_c"
        source_c.mkdir()
        _write_image(source_c / "00000001_000.png")
        # source_c deliberately lacks 00000002_000.png; two other sources still agree.

        provenance = verify_and_ingest_original_images(
            self.manifest_path,
            self.output_dir,
            {"a": self.source_a, "b": self.source_b, "c": source_c},
            min_independent_sources=2,
        )
        counts = {record["filename"]: record["agreeing_source_count"] for record in provenance["images"]}
        self.assertEqual(counts["00000001_000.png"], 3)
        self.assertEqual(counts["00000002_000.png"], 2)

    def test_inconsistent_resolution_across_images_hard_fails(self) -> None:
        _write_image(self.source_a / "00000002_000.png", size=(20, 20), fill=40)
        _write_image(self.source_b / "00000002_000.png", size=(20, 20), fill=40)

        with self.assertRaisesRegex(OriginalPixelIngestionError, "expected"):
            verify_and_ingest_original_images(
                self.manifest_path,
                self.output_dir,
                {"a": self.source_a, "b": self.source_b},
            )

    def test_fewer_than_two_sources_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            verify_and_ingest_original_images(
                self.manifest_path,
                self.output_dir,
                {"a": self.source_a},
            )

    def test_min_independent_sources_cannot_exceed_source_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_independent_sources"):
            verify_and_ingest_original_images(
                self.manifest_path,
                self.output_dir,
                {"a": self.source_a, "b": self.source_b},
                min_independent_sources=3,
            )

    def test_rerun_is_idempotent_and_does_not_recopy_verified_bytes(self) -> None:
        verify_and_ingest_original_images(
            self.manifest_path,
            self.output_dir,
            {"a": self.source_a, "b": self.source_b},
        )
        first_mtime = (self.output_dir / "00000001_000.png").stat().st_mtime_ns

        provenance = verify_and_ingest_original_images(
            self.manifest_path,
            self.output_dir,
            {"a": self.source_a, "b": self.source_b},
        )
        second_mtime = (self.output_dir / "00000001_000.png").stat().st_mtime_ns
        self.assertEqual(first_mtime, second_mtime)
        self.assertEqual(len(provenance["images"]), 2)


if __name__ == "__main__":
    unittest.main()
