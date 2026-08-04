from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

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
    _run_totalsegmentator,
    _write_bundle,
    build_parser,
    discover_ct_series,
    select_ct_series,
    validate_dicom_seg,
)
from scripts import run_local_totalsegmentator_ohif_demo as local_demo
from scripts import run_lidc_lung_nodule_colab as lidc_demo


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


def _write_seg(
    path: Path,
    *,
    study_uid: str,
    referenced_series_uid: str,
    segment_label: str = "liver",
    include_frame_identification: bool = False,
) -> None:
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
    segment.SegmentLabel = segment_label
    dataset.SegmentSequence = [segment]
    referenced = Dataset()
    referenced.SeriesInstanceUID = referenced_series_uid
    dataset.ReferencedSeriesSequence = [referenced]
    if include_frame_identification:
        frame_group = Dataset()
        identification = Dataset()
        identification.ReferencedSegmentNumber = 1
        frame_group.SegmentIdentificationSequence = [identification]
        dataset.PerFrameFunctionalGroupsSequence = [frame_group]
    dataset.PixelData = b"\1\0\0\0"
    dataset.save_as(path, enforce_file_format=True)


class DicomSegWorkflowTests(unittest.TestCase):
    def test_lidc_demo_runs_nodule_task_and_builds_comparison_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "output"
            work = root / "work"
            ct_dir = cache / "ct"
            expert_seg = root / "expert.dcm"
            expert_seg.write_bytes(b"expert")
            prediction_bundle = root / "prediction.zip"
            with zipfile.ZipFile(prediction_bundle, "w") as archive:
                archive.writestr("prediction.dcm", b"prediction")

            def fake_inference(command: list[str]) -> int:
                output.mkdir(parents=True, exist_ok=True)
                (output / "latest_dicom_seg_run.json").write_text(
                    '{"viewer_bundle": "' + str(prediction_bundle) + '"}\n'
                )
                return 0

            with (
                mock.patch.object(lidc_demo, "_require_gpu"),
                mock.patch.object(lidc_demo, "ensure_ct", return_value=ct_dir),
                mock.patch.object(lidc_demo, "ensure_expert_seg", return_value=expert_seg),
                mock.patch.object(lidc_demo, "dicom_seg_main", side_effect=fake_inference) as run_seg,
            ):
                result = lidc_demo.main(
                    [
                        "--cache-dir", str(cache),
                        "--output-dir", str(output),
                        "--work-dir", str(work),
                    ]
                )

            self.assertEqual(result, 0)
            command = run_seg.call_args.args[0]
            self.assertIn("lung_nodules", command)
            self.assertIn("--require-segment-label", command)
            latest = (output / "latest_dicom_seg_run.json").read_text()
            self.assertIn("ohif_ai_vs_expert_bundle.zip", latest)

    def test_inference_passes_the_requested_pathology_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("totalsegmentator.python_api.totalsegmentator") as inference:
                _run_totalsegmentator(
                    dicom_dir=root / "dicom",
                    seg_path=root / "seg.dcm",
                    statistics_path=root / "statistics.json",
                    report_path=root / "report.json",
                    task="lung_nodules",
                    fast=False,
                    force_split=False,
                )
            self.assertEqual(inference.call_args.kwargs["task"], "lung_nodules")

    def test_parser_accepts_pathology_task_and_required_label(self) -> None:
        args = build_parser().parse_args(
            [
                "--dicom-input", "input",
                "--output-dir", "output",
                "--work-dir", "work",
                "--task", "lung_nodules",
                "--require-segment-label", "lung_nodules",
            ]
        )
        self.assertEqual(args.task, "lung_nodules")
        self.assertEqual(args.require_segment_label, ["lung_nodules"])

    def test_local_demo_uses_pinned_public_ct_and_split_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "output"
            work = root / "work"
            weights = root / "weights"
            public_files = [cache / "ct-1-001.dcm"]
            with (
                mock.patch.object(
                    local_demo,
                    "_require_local_gpu",
                    return_value=("RTX 3050", 6.0),
                ),
                mock.patch.object(local_demo, "download_public_ct", return_value=public_files),
                mock.patch.object(local_demo, "validate_public_ct", return_value="signature"),
                mock.patch.object(local_demo, "dicom_seg_main", return_value=0) as run_seg,
                mock.patch.dict("os.environ", {}, clear=True),
            ):
                result = local_demo.main(
                    [
                        "--cache-dir",
                        str(cache),
                        "--output-dir",
                        str(output),
                        "--work-dir",
                        str(work),
                        "--weights-dir",
                        str(weights),
                    ]
                )

            self.assertEqual(result, 0)
            command = run_seg.call_args.args[0]
            self.assertIn("--confirm-deidentified", command)
            self.assertIn("--force-split", command)
            self.assertIn(local_demo.EXPECTED_SERIES_UID, command)
            self.assertNotIn("--fast", command)

    def test_local_demo_rejects_insufficient_vram_before_download(self) -> None:
        with (
            mock.patch.object(
                local_demo,
                "_require_local_gpu",
                return_value=("small GPU", 4.0),
            ),
            mock.patch.object(local_demo, "download_public_ct") as download,
        ):
            with self.assertRaisesRegex(RuntimeError, "less than 5.5 GiB"):
                local_demo.main([])
        download.assert_not_called()

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

    def test_pathology_validation_requires_an_encoded_nodule_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            study_uid = generate_uid()
            series_uid = generate_uid()
            _write_ct_series(source_dir, count=2, study_uid=study_uid, series_uid=series_uid)
            source = select_ct_series(discover_ct_series(source_dir), None)

            nodule_seg = root / "nodule_seg.dcm"
            _write_seg(
                nodule_seg,
                study_uid=study_uid,
                referenced_series_uid=series_uid,
                segment_label="lung_nodules",
                include_frame_identification=True,
            )
            validation = validate_dicom_seg(nodule_seg, source, ("lung_nodules",))
            self.assertEqual(validation.segment_labels, ("lung_nodules",))

            anatomy_seg = root / "anatomy_seg.dcm"
            _write_seg(anatomy_seg, study_uid=study_uid, referenced_series_uid=series_uid)
            with self.assertRaisesRegex(RuntimeError, "missing required segment"):
                validate_dicom_seg(anatomy_seg, source, ("lung_nodules",))

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
