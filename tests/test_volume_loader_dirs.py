"""Offline tests for DICOM-series directory flattening around derived artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from ingest.loaders import VolumeLoader


class VolumeLoaderDirectoryTests(unittest.TestCase):
    def test_flat_source_directory_needs_no_scratch_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "instance_0000.dcm").write_bytes(b"DICOM")

            self.assertIsNone(VolumeLoader._flatten_dicom_dir(str(root)))

    def test_subdirectories_are_excluded_and_source_files_are_symlinked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "instance_0000.dcm"
            source.write_bytes(b"DICOM")
            (root / "derived").mkdir()
            (root / "derived" / "result.dcm").write_bytes(b"SEG")

            flattened = VolumeLoader._flatten_dicom_dir(str(root))
            self.assertIsNotNone(flattened)
            try:
                entries = list(Path(flattened).iterdir())
                self.assertEqual([path.name for path in entries], [source.name])
                self.assertTrue(entries[0].is_symlink())
                self.assertEqual(entries[0].read_bytes(), b"DICOM")
            finally:
                shutil.rmtree(flattened, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
