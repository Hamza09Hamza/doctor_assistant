#!/usr/bin/env python3
"""Stage the pinned LIDC-IDRI-0117 detector→MedSAM2 OHIF demonstration.

This case's CT SeriesInstanceUID is absent from LUNA16's published 888-series
``candidates.csv`` and produced one true positive, zero false positives, and zero false
negatives at the detector's frozen 0.3 threshold in this project's completed 27-case
run. That UID check reduces known overlap; it is not proof of zero training
contamination. The script performs no inference:
it downloads and identity-checks the 122-slice CT plus four radiologist DICOM SEG
objects, creates the lightweight API database, and optionally uploads the same objects
to local Orthanc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pydicom
from pydicom.uid import SegmentationStorage
from sqlalchemy import select

from api.db import Base, build_engine, build_session_factory
from api.models import Series, Study
from api.orthanc_client import OrthancClient, OrthancConfig
from scripts.publish_dicom_seg_to_orthanc import clinique_amina_url
from scripts.run_totalsegmentator_dicom_seg import _referenced_series_uids


PATIENT_ID = "LIDC-IDRI-0117"
STUDY_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.336137933660116977458622909107"
CT_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.295958572786158575287945391206"
CT_INSTANCE_COUNT = 122
READER_SEG_SERIES_UIDS = (
    "1.2.276.0.7230010.3.1.3.0.7091.1553291225.533878",
    "1.2.276.0.7230010.3.1.3.0.7093.1553291228.332427",
    "1.2.276.0.7230010.3.1.3.0.7095.1553291231.281737",
    "1.2.276.0.7230010.3.1.3.0.7097.1553291234.180512",
)


def _header_identity(dataset) -> tuple[str, str, str, str, str]:
    return (
        str(getattr(dataset, "PatientID", "")),
        str(getattr(dataset, "StudyInstanceUID", "")),
        str(getattr(dataset, "SeriesInstanceUID", "")),
        str(getattr(dataset, "SOPInstanceUID", "")),
        str(getattr(dataset, "Modality", "")),
    )


def _idc_download(client, series_uid: str, destination: Path) -> None:
    print(f"Downloading pinned IDC series {series_uid} ...", flush=True)
    client.download_dicom_series(
        series_uid,
        str(destination),
        quiet=False,
        show_progress_bar=True,
    )


def _dicom_files(root: Path) -> list[Path]:
    found: list[Path] = []
    if not root.exists():
        return found
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        found.append(path)
    return found


def _spatial_position(dataset) -> float:
    import numpy as np

    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    position = np.asarray(dataset.ImagePositionPatient, dtype=float)
    return float(np.dot(position, np.cross(orientation[:3], orientation[3:])))


def _load_ct_files(root: Path) -> list[tuple[object, Path]]:
    found = []
    for path in _dicom_files(root):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        if (
            str(getattr(dataset, "Modality", "")) == "CT"
            and str(getattr(dataset, "SeriesInstanceUID", "")) == CT_SERIES_UID
        ):
            found.append((dataset, path))
    found.sort(key=lambda item: _spatial_position(item[0]))
    return found


def _validate_ct_identity(ct_files: list[tuple[object, Path]]) -> None:
    sop_uids: set[str] = set()
    for dataset, path in ct_files:
        patient_id, study_uid, series_uid, sop_uid, modality = _header_identity(dataset)
        if patient_id != PATIENT_ID or study_uid != STUDY_UID:
            raise RuntimeError(f"CT identity mismatch in {path}")
        if series_uid != CT_SERIES_UID or modality != "CT" or not sop_uid:
            raise RuntimeError(f"CT series/SOP identity mismatch in {path}")
        if sop_uid in sop_uids:
            raise RuntimeError(f"duplicate CT SOPInstanceUID {sop_uid}")
        sop_uids.add(sop_uid)
    if len(sop_uids) != CT_INSTANCE_COUNT:
        raise RuntimeError(
            f"expected {CT_INSTANCE_COUNT} unique CT SOPInstanceUIDs, found {len(sop_uids)}"
        )


def _ensure_data(cache_dir: Path) -> tuple[list[tuple[object, Path]], list[Path]]:
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError(
            "idc-index is required; use .venv-mlx or install idc-index==0.12.5"
        ) from exc

    client = IDCClient()
    ct_root = cache_dir / "ct"
    ct_files = _load_ct_files(ct_root)
    if len(ct_files) != CT_INSTANCE_COUNT:
        _idc_download(client, CT_SERIES_UID, ct_root)
        ct_files = _load_ct_files(ct_root)
    if len(ct_files) != CT_INSTANCE_COUNT:
        raise RuntimeError(
            f"expected {CT_INSTANCE_COUNT} CT instances for {PATIENT_ID}, found {len(ct_files)}"
        )
    _validate_ct_identity(ct_files)

    seg_paths: list[Path] = []
    for reader_index, series_uid in enumerate(READER_SEG_SERIES_UIDS, start=1):
        root = cache_dir / "expert_seg" / f"reader_{reader_index}"
        candidates = _dicom_files(root)
        matching = []
        for path in candidates:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
            if (
                str(getattr(dataset, "Modality", "")) == "SEG"
                and str(getattr(dataset, "SeriesInstanceUID", "")) == series_uid
            ):
                matching.append(path)
        if len(matching) != 1:
            _idc_download(client, series_uid, root)
            matching = []
            for path in _dicom_files(root):
                dataset = pydicom.dcmread(path, stop_before_pixels=True)
                if (
                    str(getattr(dataset, "Modality", "")) == "SEG"
                    and str(getattr(dataset, "SeriesInstanceUID", "")) == series_uid
                ):
                    matching.append(path)
        if len(matching) != 1:
            raise RuntimeError(
                f"expected one radiologist SEG for reader {reader_index}, found {len(matching)}"
            )
        dataset = pydicom.dcmread(matching[0], stop_before_pixels=True)
        patient_id, study_uid, found_series_uid, sop_uid, modality = _header_identity(
            dataset
        )
        if patient_id != PATIENT_ID or study_uid != STUDY_UID:
            raise RuntimeError(f"reader {reader_index} SEG belongs to a different study")
        if (
            found_series_uid != series_uid
            or modality != "SEG"
            or str(getattr(dataset, "SOPClassUID", "")) != str(SegmentationStorage)
            or not sop_uid
        ):
            raise RuntimeError(f"reader {reader_index} SEG identity is invalid")
        if CT_SERIES_UID not in _referenced_series_uids(dataset):
            raise RuntimeError(f"reader {reader_index} SEG does not reference the pinned CT")
        seg_paths.append(matching[0])

    seg_sop_uids = {
        str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
        for path in seg_paths
    }
    if len(seg_sop_uids) != len(READER_SEG_SERIES_UIDS):
        raise RuntimeError("reader SEG objects do not have distinct SOPInstanceUIDs")

    print(f"Verified pinned public CT: {PATIENT_ID}, {len(ct_files)} slices", flush=True)
    print("Verified four radiologist SEG objects reference the exact CT series.", flush=True)
    return ct_files, seg_paths


def _same_file(source: Path, destination: Path) -> bool:
    if not destination.is_file() or source.stat().st_size != destination.stat().st_size:
        return False
    source_digest = hashlib.sha256(source.read_bytes()).digest()
    return source_digest == hashlib.sha256(destination.read_bytes()).digest()


def _stage_file(source: Path, destination: Path) -> None:
    if _same_file(source, destination):
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _stage_ct(source: list[tuple[object, Path]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, (_dataset, source_path) in enumerate(source):
        target = destination / f"instance_{index:04d}.dcm"
        _stage_file(source_path, target)


def _upload_to_orthanc(ct_paths: list[Path], seg_paths: list[Path], args) -> None:
    config = OrthancConfig(
        base_url=args.orthanc_url,
        username=args.orthanc_username,
        password=args.orthanc_password,
    )
    objects = [*ct_paths, *seg_paths]
    with OrthancClient(config) as orthanc:
        for index, path in enumerate(objects, start=1):
            orthanc.upload_instance(path.read_bytes())
            if index == 1 or index % 50 == 0 or index == len(objects):
                print(f"Orthanc accepted {index}/{len(objects)} DICOM objects", flush=True)
        orthanc.query_study(STUDY_UID)


def prepare(args) -> Path:
    cache_dir = args.cache_dir.resolve()
    ct_source, seg_source = _ensure_data(cache_dir)
    first = ct_source[0][0]

    database_path = args.database.resolve()
    storage_root = args.storage_dir.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as db:
        study = db.scalar(select(Study).where(Study.dicom_study_uid == STUDY_UID))
        if study is None:
            study = Study(
                modality="unknown",
                body_part="unknown",
                source_filename=f"IDC {PATIENT_ID}",
                source="idc",
                dicom_study_uid=STUDY_UID,
                patient_reference=hashlib.sha256(PATIENT_ID.encode()).hexdigest()[:16],
            )
            db.add(study)
            db.flush()

        series_dir = (
            storage_root / "studies" / study.id / "series" / CT_SERIES_UID.replace(".", "_")
        )
        _stage_ct(ct_source, series_dir)
        reference_dir = series_dir / "reference"
        reference_dir.mkdir(exist_ok=True)
        staged_segs = []
        for reader_index, source_path in enumerate(seg_source, start=1):
            destination = reference_dir / f"lidc_reader_{reader_index}_nodules.dcm"
            _stage_file(source_path, destination)
            staged_segs.append(destination)

        series = db.scalar(select(Series).where(Series.dicom_series_uid == CT_SERIES_UID))
        if series is None:
            series = Series(
                study_id=study.id,
                dicom_series_uid=CT_SERIES_UID,
                dicom_modality="CT",
                modality="ct",
                body_part="chest",
                instance_count=len(ct_source),
                storage_dir=str(series_dir),
                analysis_eligible=True,
            )
            db.add(series)
        else:
            series.storage_dir = str(series_dir)
            series.instance_count = len(ct_source)
        db.commit()
        db.refresh(study)
        db.refresh(series)

    manifest = {
        "dataset": "LIDC-IDRI",
        "case": PATIENT_ID,
        "source": "NCI Imaging Data Commons / The Cancer Imaging Archive",
        "license": "CC BY 3.0",
        "study_instance_uid": STUDY_UID,
        "series_instance_uid": CT_SERIES_UID,
        "reader_seg_series_instance_uids": list(READER_SEG_SERIES_UIDS),
        "api_study_id": study.id,
        "api_series_id": series.id,
        "database_url": f"sqlite:///{database_path}",
        "storage_dir": str(storage_root),
        "series_dir": str(series_dir),
        "expert_seg_paths": [str(path) for path in staged_segs],
        "rows": int(first.Rows),
        "columns": int(first.Columns),
        "instance_count": len(ct_source),
        "recommended_window_center": -600,
        "recommended_window_width": 1500,
        "detector_score_threshold": 0.3,
        "recorded_case_result_within_27_case_run": {
            "ground_truth_count": 1,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
        },
        "evidence_artifact": "notebooks/build_ohif_bundles_only.ipynb",
        "demo_selected_post_hoc": True,
        "luna16_uid_overlap_status": "not_listed_in_luna16_candidates_csv",
        "contamination_note": (
            "CT SeriesInstanceUID was absent from LUNA16's published 888-series "
            "candidates.csv in the recorded 27-case evaluation. This excludes known "
            "LUNA16-series overlap; it does not prove absence from every upstream or "
            "private training source."
        ),
    }
    manifest_path = cache_dir / "nodule_detector_demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.upload_orthanc:
        _upload_to_orthanc([path for _dataset, path in ct_source], seg_source, args)

    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    if args.upload_orthanc:
        print(f"Clinique Amina: {clinique_amina_url(args.ohif_url, STUDY_UID)}", flush=True)
    else:
        print("Orthanc upload skipped; re-run with --upload-orthanc after Orthanc is running.")
    engine.dispose()
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/validation/lidc_idri_0117_nodule_detector"),
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("api_storage/lidc_nodule_detector_demo"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("api_storage/lidc_nodule_detector_demo.sqlite"),
    )
    parser.add_argument("--upload-orthanc", action="store_true")
    parser.add_argument("--orthanc-url", default="http://localhost:8042")
    parser.add_argument("--orthanc-username", default="doctor_assistant")
    parser.add_argument("--orthanc-password", default="doctor_assistant")
    parser.add_argument("--ohif-url", default="http://localhost:3000")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
