from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import fetch_nih_expert_images as images
from scripts import fetch_nih_metadata as metadata


class NIHMetadataFetchTests(unittest.TestCase):
    @staticmethod
    def _write_expert_gzip(path: Path, rows: list[dict[str, str]]) -> None:
        fields = [
            "Image Index",
            "Patient ID",
            "Fracture",
            "Pneumothorax",
            "Airspace opacity",
            "Nodule or mass",
            "Set Id",
        ]
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_google_expert_table_requires_complete_yes_no_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / metadata.GOOGLE_EXPERT_LABEL_FILENAME
            rows = [
                {
                    "Image Index": "00000001_000.png",
                    "Patient ID": "1",
                    "Fracture": "NO",
                    "Pneumothorax": "YES",
                    "Airspace opacity": "NO",
                    "Nodule or mass": "NO",
                    "Set Id": "val",
                },
                {
                    "Image Index": "00000002_000.png",
                    "Patient ID": "2",
                    "Fracture": "YES",
                    "Pneumothorax": "NO",
                    "Airspace opacity": "YES",
                    "Nodule or mass": "NO",
                    "Set Id": "test",
                },
            ]
            self._write_expert_gzip(path, rows)
            validation = metadata.validate_google_expert_labels(
                path,
                expected_rows=2,
                expected_split_rows={"val": 1, "test": 1},
                expected_label_counts={
                    "Fracture": {"NO": 1, "YES": 1},
                    "Pneumothorax": {"NO": 1, "YES": 1},
                    "Airspace opacity": {"NO": 1, "YES": 1},
                    "Nodule or mass": {"NO": 2, "YES": 0},
                },
            )
            self.assertEqual(validation["unique_images"], 2)
            self.assertTrue(
                validation["all_four_findings_adjudicated_yes_no"]
            )

            rows[1]["Nodule or mass"] = ""
            self._write_expert_gzip(path, rows)
            with self.assertRaisesRegex(
                metadata.MetadataFetchError, "adjudicated YES/NO"
            ):
                metadata.validate_google_expert_labels(
                    path,
                    expected_rows=2,
                    expected_split_rows={"val": 1, "test": 1},
                    expected_label_counts={
                        "Fracture": {"NO": 1, "YES": 1},
                        "Pneumothorax": {"NO": 1, "YES": 1},
                        "Airspace opacity": {"NO": 1, "YES": 1},
                        "Nodule or mass": {"NO": 2, "YES": 0},
                    },
                )

    def test_fetch_includes_expert_labels_and_records_known_count_delta(self) -> None:
        payloads = {
            "metadata.csv": b"metadata",
            metadata.GOOGLE_EXPERT_LABEL_FILENAME: b"expert-labels",
        }
        specifications = {
            name: {
                "url": f"https://example.invalid/{name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in payloads.items()
        }

        def downloader(url: str, destination: Path) -> None:
            destination.write_bytes(payloads[destination.name])

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            metadata, "PINNED_FILES", specifications
        ), patch.object(
            metadata,
            "validate_metadata_files",
            return_value={"rows": {"Data_Entry_2017.csv": 1}},
        ), patch.object(
            metadata,
            "validate_google_expert_labels",
            return_value={
                "rows": 4_376,
                "split_rows": {"val": 2_414, "test": 1_962},
            },
        ):
            result = metadata.fetch_metadata(
                temporary, downloader=downloader
            )

        self.assertIn(metadata.GOOGLE_EXPERT_LABEL_FILENAME, result["files"])
        self.assertEqual(
            result["known_source_discrepancy"]["validation_row_delta"], 2
        )
        self.assertEqual(
            result["artifact_type"],
            "doctor_assistant.nih_metadata_and_expert_labels",
        )


class NIHExpertImageFetchTests(unittest.TestCase):
    def test_parquet_fallback_projects_after_loading_stream(self) -> None:
        target = images.ManifestTarget(
            filename="00000001_000.png", expert_split="validation"
        )
        streams = {
            "train": [
                {"filename": "00000001_000.png", "image": object()},
                {"filename": "00000002_000.png", "image": object()},
            ],
            "validation": [
                {"filename": "00000003_000.png", "image": object()}
            ],
        }
        calls: list[tuple[str, dict[str, object], list[str]]] = []

        class FakeStream:
            def __init__(self, split: str, rows: list[dict[str, object]]) -> None:
                self.split = split
                self.rows = rows
                self.selected: list[str] = []

            def select_columns(self, columns: list[str]) -> "FakeStream":
                self.selected = list(columns)
                calls[-1][2].extend(columns)
                return self

            def __iter__(self):
                return iter(self.rows)

        def loader(dataset: str, **kwargs: object) -> FakeStream:
            split = str(kwargs["split"])
            selected: list[str] = []
            calls.append((dataset, dict(kwargs), selected))
            return FakeStream(split, streams[split])

        locations = images.scan_filename_column_locations(
            [target],
            loader=loader,
            expected_split_rows={"train": 2, "validation": 1},
        )

        self.assertEqual(locations[0].mirror_split, "train")
        self.assertTrue(all("columns" not in kwargs for _, kwargs, _ in calls))
        self.assertTrue(all(selected == ["filename"] for _, _, selected in calls))
        self.assertTrue(
            all(kwargs["revision"] == images.MIRROR_REVISION for _, kwargs, _ in calls)
        )

    def test_one_image_fetch_emits_benchmark_compatible_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "filename,patient_id,split\n"
                "00000001_000.png,00000001,validation\n",
                encoding="utf-8",
            )
            image_buffer = io.BytesIO()
            Image.new("L", images.EXPECTED_IMAGE_SIZE, color=127).save(
                image_buffer, format="JPEG"
            )
            jpeg = image_buffer.getvalue()
            asset_url = (
                "https://datasets-server.huggingface.co/assets/"
                f"--/{images.MIRROR_REVISION}/--/{images.MIRROR_CONFIG}/"
                "train/0/image/image.jpg?signature=redacted"
            )
            row_requests = 0
            asset_requests = 0

            def get_json(
                endpoint: str,
                params: dict[str, object] | None = None,
                **_: object,
            ) -> dict[str, object]:
                if "/api/datasets/" in endpoint:
                    return {
                        "sha": images.MIRROR_REVISION,
                        "lastModified": "2026-01-01T00:00:00Z",
                        "tags": ["license:cc0-1.0"],
                    }
                if endpoint.endswith("/filter"):
                    assert params is not None
                    if params["split"] == "train":
                        return {
                            "partial": False,
                            "rows": [
                                {
                                    "row_idx": 0,
                                    "row": {"filename": "00000001_000.png"},
                                }
                            ],
                        }
                    return {"partial": False, "rows": []}
                if endpoint.endswith("/rows"):
                    nonlocal row_requests
                    row_requests += 1
                    return {
                        "rows": [
                            {
                                "row_idx": 0,
                                "row": {
                                    "filename": "00000001_000.png",
                                    "labels": ["No Finding"],
                                    "image": {
                                        "width": 320,
                                        "height": 320,
                                        "src": asset_url,
                                    },
                                },
                            }
                        ]
                    }
                raise AssertionError(endpoint)

            def request_bytes(
                url: str, **_: object
            ) -> tuple[bytes, dict[str, str]]:
                nonlocal asset_requests
                asset_requests += 1
                self.assertEqual(url, asset_url)
                return jpeg, {"Content-Type": "image/jpeg"}

            result = images.fetch_expert_images(
                manifest,
                root / "images",
                workers=1,
                get_json=get_json,
                request_bytes=request_bytes,
            )
            resumed = images.fetch_expert_images(
                manifest,
                root / "images",
                workers=1,
                get_json=get_json,
                request_bytes=request_bytes,
            )
            saved = json.loads(
                (root / "images" / "DEVELOPMENT_ONLY.provenance.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            result["artifact_type"], "doctor_assistant.nih_image_provenance"
        )
        self.assertEqual(saved["resolution"], {"width": 320, "height": 320})
        self.assertFalse(saved["original_nih_pixels"])
        self.assertIn(images.MIRROR_REVISION, saved["source"])
        self.assertEqual(saved["validation"]["images"], 1)
        self.assertEqual(row_requests, 1)
        self.assertEqual(asset_requests, 1)
        self.assertEqual(resumed["validation"]["downloaded_this_run"], 0)
        self.assertEqual(resumed["validation"]["resumed_this_run"], 1)

    def test_locked_test_fetch_requires_explicit_development_acknowledgement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            manifest.write_text(
                "filename,split\n00000001_000.png,test\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "acknowledge-development-only-test-use"
            ):
                images.load_canonical_manifest(manifest, cohort="test")


if __name__ == "__main__":
    unittest.main()
