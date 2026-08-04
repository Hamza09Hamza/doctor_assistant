from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import httpx
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, SegmentationStorage, generate_uid

from scripts.publish_dicom_seg_to_orthanc import (
    clinique_amina_url,
    upload_dicom_files,
    verify_study_visible,
)
from scripts.run_totalsegmentator_dicom_seg import (
    _safe_extract_zip,
    _write_bundle,
    discover_ct_series,
    select_ct_series,
    validate_dicom_seg,
)


def _file_dataset(path: Path, sop_class_uid: str, sop_instance_uid: str) -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)


def _write_ct_series(root: Path, *, count: int, study_uid: str, series_uid: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    frame_uid = generate_uid()
    files = []
    for index in range(count):
        path = root / f"slice_{index:03d}.dcm"
        dataset = _file_dataset(path, CTImageStorage, generate_uid())
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "CT"
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.InstanceNumber = index + 1
        dataset.Rows = 2
        dataset.Columns = 2
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelData = b"\0" * 8
        dataset.save_as(path, enforce_file_format=True)
        files.append(path)
    return files


def _write_seg(path: Path, *, study_uid: str, referenced_series_uid: str) -> None:
    dataset = _file_dataset(path, SegmentationStorage, generate_uid())
    dataset.SOPClassUID = SegmentationStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "SEG"
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = generate_uid()
    dataset.NumberOfFrames = 1
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    segment = Dataset()
    segment.SegmentNumber = 1
    segment.SegmentLabel = "liver"
    dataset.SegmentSequence = [segment]
    referenced = Dataset()
    referenced.SeriesInstanceUID = referenced_series_uid
    dataset.ReferencedSeriesSequence = [referenced]
    dataset.PixelData = b"\1\0\0\0"
    dataset.save_as(path, enforce_file_format=True)


class DicomSegWorkflowTests(unittest.TestCase):
    def test_discovers_and_selects_requested_ct_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_uid = generate_uid()
            first_uid = generate_uid()
            second_uid = generate_uid()
            _write_ct_series(root / "first", count=3, study_uid=study_uid, series_uid=first_uid)
            _write_ct_series(root / "second", count=4, study_uid=study_uid, series_uid=second_uid)
            (root / "not-dicom.txt").write_text("ignored")

            series = discover_ct_series(root)
            self.assertEqual(len(series), 2)
            selected = select_ct_series(series, first_uid)
            self.assertEqual(selected.series_instance_uid, first_uid)
            self.assertEqual(selected.instance_count, 3)
            with self.assertRaisesRegex(RuntimeError, "multiple substantial CT series"):
                select_ct_series(series, None)

    def test_validates_seg_references_and_creates_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source_dicom"
            source_dir.mkdir()
            study_uid = generate_uid()
            series_uid = generate_uid()
            _write_ct_series(source_dir, count=3, study_uid=study_uid, series_uid=series_uid)
            source = select_ct_series(discover_ct_series(source_dir), None)
            seg_path = root / "totalsegmentator_seg.dcm"
            _write_seg(seg_path, study_uid=study_uid, referenced_series_uid=series_uid)
            validation = validate_dicom_seg(seg_path, source)
            self.assertEqual(validation.segment_count, 1)
            self.assertEqual(validation.referenced_series_instance_uid, series_uid)

            manifest = root / "run_manifest.json"
            manifest.write_text("{}\n")
            bundle = root / "ohif_viewer_bundle.zip"
            _write_bundle(
                source_dir=source_dir,
                seg_path=seg_path,
                manifest_path=manifest,
                destination=bundle,
            )
            with zipfile.ZipFile(bundle) as zf:
                names = set(zf.namelist())
            self.assertIn("totalsegmentator_seg.dcm", names)
            self.assertIn("run_manifest.json", names)
            self.assertEqual(len([name for name in names if name.startswith("source_dicom/")]), 3)

    def test_rejects_seg_for_a_different_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            series_uid = generate_uid()
            _write_ct_series(
                source_dir,
                count=2,
                study_uid=generate_uid(),
                series_uid=series_uid,
            )
            source = select_ct_series(discover_ct_series(source_dir), None)
            seg_path = root / "bad_seg.dcm"
            _write_seg(
                seg_path,
                study_uid=generate_uid(),
                referenced_series_uid=series_uid,
            )
            with self.assertRaisesRegex(RuntimeError, "StudyInstanceUID"):
                validate_dicom_seg(seg_path, source)

    def test_zip_extraction_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escape.dcm", b"bad")
            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                _safe_extract_zip(archive, root / "output")

    def test_upload_and_qido_verification_use_orthanc_endpoints(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.method == "POST":
                return httpx.Response(200, json={"Status": "Success"})
            return httpx.Response(200, json=[{"0020000D": {"vr": "UI"}}])

        with tempfile.TemporaryDirectory() as tmp:
            dicom_file = Path(tmp) / "object.dcm"
            dicom_file.write_bytes(b"dicom")
            with httpx.Client(
                base_url="http://orthanc.test",
                transport=httpx.MockTransport(handler),
            ) as client:
                uploaded, existing = upload_dicom_files(client, [dicom_file])
                verify_study_visible(client, "1.2.3")
        self.assertEqual((uploaded, existing), (1, 0))
        self.assertEqual(requests, [("POST", "/instances"), ("GET", "/dicom-web/studies")])

    def test_clinique_amina_url_targets_orthanc_data_source(self) -> None:
        self.assertEqual(
            clinique_amina_url("http://localhost:3000/", "1.2.3"),
            "http://localhost:3000/doctor-assistant/orthancProxy?StudyInstanceUIDs=1.2.3",
        )


if __name__ == "__main__":
    unittest.main()
