#!/usr/bin/env python3
"""Download and stage a high-quality, expert-annotated CT for the OHIF box demo.

The pinned LIDC-IDRI-0686 series is a 238-slice, 512x512 thin-slice chest CT. A
radiologist DICOM SEG for a clearly visible nodule is downloaded alongside it. The
script creates an idempotent SQLite API database plus a flat series directory matching
the application's normal Orthanc-import layout. Raw scans remain under ignored paths.
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
from sqlalchemy import select

from api.db import Base, build_engine, build_session_factory
from api.models import Series, Study
from api.orthanc_client import OrthancClient, OrthancConfig
from scripts.publish_dicom_seg_to_orthanc import clinique_amina_url
from scripts.run_lidc_lung_nodule_colab import (
    CT_INSTANCE_COUNT,
    CT_SERIES_UID,
    PATIENT_ID,
    STUDY_UID,
    ensure_ct,
    ensure_expert_seg,
)


def _spatial_position(dataset) -> float:
    import numpy as np

    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    position = np.asarray(dataset.ImagePositionPatient, dtype=float)
    return float(np.dot(position, np.cross(orientation[:3], orientation[3:])))


def _source_files(ct_root: Path) -> list[tuple[object, Path]]:
    found = []
    for path in ct_root.rglob("*.dcm"):
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        if str(getattr(dataset, "SeriesInstanceUID", "")) == CT_SERIES_UID:
            found.append((dataset, path))
    found.sort(key=lambda item: _spatial_position(item[0]))
    if len(found) != CT_INSTANCE_COUNT:
        raise RuntimeError(f"expected {CT_INSTANCE_COUNT} CT instances, found {len(found)}")
    return found


def _stage_files(source: list[tuple[object, Path]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, (_dataset, source_path) in enumerate(source):
        target = destination / f"instance_{index:04d}.dcm"
        if target.exists():
            continue
        try:
            os.link(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)


def _expert_prompt(expert_path: Path) -> dict:
    import numpy as np

    dataset = pydicom.dcmread(expert_path)
    frames = dataset.pixel_array.astype(bool)
    # A middle slice gives forward and reverse propagation comparable distances.
    # Choosing the maximum-area frame put this case's seed near one z-boundary.
    frame_index = int(frames.shape[0] // 2)
    frame = frames[frame_index]
    rows, columns = np.where(frame)
    source_uid = str(
        dataset.PerFrameFunctionalGroupsSequence[frame_index]
        .DerivationImageSequence[0]
        .SourceImageSequence[0]
        .ReferencedSOPInstanceUID
    )
    # 3px scored meaningfully better than 6px on the pinned LIDC-0686 case for both the
    # official Torch checkpoint (Dice 0.519 -> 0.636) and the MLX conversion (0.352 ->
    # 0.428); see data/validation/lidc_idri_0686/experiment_torch_pad3_result.json.
    padding = 3
    return {
        "seed_sop_instance_uid": source_uid,
        "recommended_box_xyxy": [
            max(0, int(columns.min()) - padding),
            max(0, int(rows.min()) - padding),
            min(int(dataset.Columns) - 1, int(columns.max()) + padding),
            min(int(dataset.Rows) - 1, int(rows.max()) + padding),
        ],
        "expert_segment_label": str(dataset.SegmentSequence[0].SegmentLabel),
        "expert_segmented_slices": int(frames.shape[0]),
        "expert_voxels": int(frames.sum()),
    }


def _write_preview(
    source: list[tuple[object, Path]],
    expert_path: Path,
    seed_sop_instance_uid: str,
    box_xyxy: list[int],
    output_path: Path,
) -> None:
    """Render the seed slice at lung window with reference mask and prompt box."""
    import numpy as np
    from PIL import Image, ImageDraw

    source_path = next(
        path for dataset, path in source if str(dataset.SOPInstanceUID) == seed_sop_instance_uid
    )
    source_dataset = pydicom.dcmread(source_path)
    pixels = source_dataset.pixel_array.astype(np.float32)
    pixels = pixels * float(getattr(source_dataset, "RescaleSlope", 1.0))
    pixels += float(getattr(source_dataset, "RescaleIntercept", 0.0))
    lower, upper = -600.0 - 1500.0 / 2.0, -600.0 + 1500.0 / 2.0
    grayscale = np.clip((pixels - lower) / (upper - lower), 0.0, 1.0)
    rgb = np.repeat((grayscale * 255.0).astype(np.uint8)[..., None], 3, axis=-1)

    expert = pydicom.dcmread(expert_path)
    for frame_index, frame_group in enumerate(expert.PerFrameFunctionalGroupsSequence):
        source_uid = str(
            frame_group.DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )
        if source_uid == seed_sop_instance_uid:
            mask = expert.pixel_array[frame_index].astype(bool)
            rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
            break

    preview = Image.fromarray(rgb)
    ImageDraw.Draw(preview).rectangle(tuple(box_xyxy), outline=(40, 255, 80), width=2)
    preview.save(output_path)


def _upload_to_orthanc(files: list[Path], expert_path: Path, args) -> None:
    config = OrthancConfig(
        base_url=args.orthanc_url,
        username=args.orthanc_username,
        password=args.orthanc_password,
    )
    objects = [*files, expert_path]
    with OrthancClient(config) as orthanc:
        for index, path in enumerate(objects, start=1):
            orthanc.upload_instance(path.read_bytes())
            if index == 1 or index % 50 == 0 or index == len(objects):
                print(f"Orthanc accepted {index}/{len(objects)} DICOM objects", flush=True)
        orthanc.query_study(STUDY_UID)


def prepare(args) -> Path:
    cache_dir = args.cache_dir.resolve()
    ct_root = ensure_ct(cache_dir)
    expert_path = ensure_expert_seg(cache_dir)
    source = _source_files(ct_root)
    first = source[0][0]

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
            storage_root
            / "studies"
            / study.id
            / "series"
            / CT_SERIES_UID.replace(".", "_")
        )
        _stage_files(source, series_dir)
        reference_dir = series_dir / "reference"
        reference_dir.mkdir(exist_ok=True)
        staged_expert = reference_dir / "lidc_radiologist_nodule_seg.dcm"
        if not staged_expert.exists():
            shutil.copy2(expert_path, staged_expert)

        series = db.scalar(select(Series).where(Series.dicom_series_uid == CT_SERIES_UID))
        if series is None:
            series = Series(
                study_id=study.id,
                dicom_series_uid=CT_SERIES_UID,
                dicom_modality="CT",
                modality="ct",
                body_part="chest",
                instance_count=len(source),
                storage_dir=str(series_dir),
                analysis_eligible=True,
            )
            db.add(series)
        else:
            series.storage_dir = str(series_dir)
            series.instance_count = len(source)
        db.commit()
        db.refresh(study)
        db.refresh(series)

    prompt = _expert_prompt(expert_path)
    seed_dataset = next(ds for ds, _path in source if str(ds.SOPInstanceUID) == prompt["seed_sop_instance_uid"])
    preview_path = cache_dir / f"seed_instance_{int(seed_dataset.InstanceNumber)}_lung_window.png"
    _write_preview(
        source,
        expert_path,
        prompt["seed_sop_instance_uid"],
        prompt["recommended_box_xyxy"],
        preview_path,
    )
    manifest = {
        "dataset": "LIDC-IDRI",
        "case": PATIENT_ID,
        "source": "NCI Imaging Data Commons / The Cancer Imaging Archive",
        "license": "CC BY 3.0",
        "study_instance_uid": STUDY_UID,
        "series_instance_uid": CT_SERIES_UID,
        "api_study_id": study.id,
        "api_series_id": series.id,
        "database_url": f"sqlite:///{database_path}",
        "storage_dir": str(storage_root),
        "series_dir": str(series_dir),
        "expert_seg_path": str(staged_expert),
        "preview_path": str(preview_path),
        "rows": int(first.Rows),
        "columns": int(first.Columns),
        "pixel_spacing_mm": [float(value) for value in first.PixelSpacing],
        "slice_thickness_mm": float(first.SliceThickness),
        "convolution_kernel": str(getattr(first, "ConvolutionKernel", "")),
        "instance_count": len(source),
        "recommended_window_center": -600,
        "recommended_window_width": 1500,
        "seed_instance_number": int(seed_dataset.InstanceNumber),
        **prompt,
    }
    manifest_path = cache_dir / "interactive_demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.upload_orthanc:
        _upload_to_orthanc([path for _dataset, path in source], expert_path, args)

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
        default=Path("data/validation/lidc_idri_0686"),
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path("api_storage/lidc_interactive_demo"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("api_storage/lidc_interactive_demo.sqlite"),
    )
    parser.add_argument("--upload-orthanc", action="store_true")
    parser.add_argument("--orthanc-url", default="http://localhost:8042")
    parser.add_argument("--orthanc-username", default="doctor_assistant")
    parser.add_argument("--orthanc-password", default="doctor_assistant")
    parser.add_argument("--ohif-url", default="http://localhost:3000")
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
