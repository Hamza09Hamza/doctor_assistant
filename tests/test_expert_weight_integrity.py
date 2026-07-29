from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experts import msk_fracture


class MSKCheckpointIntegrityTests(unittest.TestCase):
    def test_default_checkpoint_is_verified_before_atomic_cache_commit(self) -> None:
        payload = b"pinned fracture checkpoint fixture"
        expected_hash = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "weights" / "best.pt"

            def download(_url: str, destination: str) -> tuple[str, None]:
                Path(destination).write_bytes(payload)
                return destination, None

            with (
                patch.object(msk_fracture, "_WEIGHTS_CACHE", str(cache)),
                patch.object(msk_fracture, "_WEIGHTS_BYTES", len(payload)),
                patch.object(msk_fracture, "_WEIGHTS_SHA256", expected_hash),
                patch.object(
                    msk_fracture.urllib.request,
                    "urlretrieve",
                    side_effect=download,
                ) as downloader,
            ):
                first = msk_fracture._default_weights_path()
                second = msk_fracture._default_weights_path()

            self.assertEqual(first, str(cache))
            self.assertEqual(second, str(cache))
            self.assertEqual(cache.read_bytes(), payload)
            downloader.assert_called_once()

    def test_corrupt_cached_checkpoint_is_replaced_only_after_verification(self) -> None:
        payload = b"reviewed replacement"
        expected_hash = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "best.pt"
            cache.write_bytes(b"corrupt")

            def download(_url: str, destination: str) -> tuple[str, None]:
                Path(destination).write_bytes(payload)
                return destination, None

            with (
                patch.object(msk_fracture, "_WEIGHTS_CACHE", str(cache)),
                patch.object(msk_fracture, "_WEIGHTS_BYTES", len(payload)),
                patch.object(msk_fracture, "_WEIGHTS_SHA256", expected_hash),
                patch.object(
                    msk_fracture.urllib.request,
                    "urlretrieve",
                    side_effect=download,
                ),
            ):
                msk_fracture._default_weights_path()

            self.assertEqual(cache.read_bytes(), payload)

    def test_custom_checkpoint_requires_an_explicit_hash(self) -> None:
        expert = msk_fracture.MSKFractureExpert(weights_path="/tmp/custom.pt")

        with self.assertRaisesRegex(ValueError, "require weights_sha256"):
            expert._ensure_loaded()


if __name__ == "__main__":
    unittest.main()
