#!/usr/bin/env python3
"""Download and stage a real, DICOM-native brain-tumour MRI case for the BraTS wiring.

Pinned case: UPENN-GBM-00020 (NCI Imaging Data Commons, `upenn_gbm` collection, CC BY
4.0). Four co-registered MRI sequences -- T1, T1c (post-contrast "stealth-post"), T2,
FLAIR -- each a separate DICOM series with its own real `SeriesDescription` tag, as
required by `api/persistence.py::_match_brats_sequences`. Unlike the Medical
Segmentation Decathlon's Task01_BrainTumour (one combined 4-channel NIfTI per case,
not separate DICOM series -- unusable here), this is real per-series DICOM as it would
arrive from a PACS/Orthanc import.

Mirrors `scripts/prepare_lidc_interactive_demo.py`'s staging pattern: an idempotent
SQLite API database plus a flat per-series directory layout matching the application's
normal Orthanc-import layout, with a rendered preview PNG for a visual quality check.
Raw scans remain under an ignored path (`data/validation/`).

One quirk worth documenting up front: IDC's `upenn_gbm` collection is exported from
each patient's CaPTk (Cancer Imaging Phenomics Toolkit) post-processing pipeline, and
each of the four series carries its OWN distinct DICOM `StudyInstanceUID` -- they were
never one PACS study with a shared UID the way LIDC-IDRI-0686 was. That's fine for this
app: `_match_brats_sequences` groups series via the *local* `Series.study_id` foreign
key, never the raw DICOM StudyInstanceUID, so one local `Study` row can legitimately
own all four regardless. `Study.dicom_study_uid` is left `None` here rather than
picking one of the four UIDs and pretending it represents the whole case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

COLLECTION = "upenn_gbm"
PATIENT_ID = "UPENN-GBM-00020"

# {canonical BraTS sequence name: (SeriesInstanceUID, expected instance count)} -- pinned
# so a re-run can verify identity instead of trusting whatever idc-index returns later.
SEQUENCE_SERIES: dict[str, tuple[str, int]] = {
    "t1": ("1.3.6.1.4.1.14519.5.2.1.88442458185890417183040615754756360791", 192),
    "t1c": ("1.3.6.1.4.1.14519.5.2.1.179029676246601157621901214548306088166", 192),
    "t2": ("1.3.6.1.4.1.14519.5.2.1.304664131632340855252559437288625814808", 64),
    "flair": ("1.3.6.1.4.1.14519.5.2.1.173238233038984239243519723361468888961", 60),
}

# The real SeriesDescription tag read off one instance of each downloaded series --
# verified with pydicom against the actual files, not just IDC's index metadata. None
# of these are in `experts/mri_brats.py::_MODALITY_ALIASES` as shipped; see
# docs/BRATS_TEST_CASE.md for what was added and why.
EXPECTED_DESCRIPTIONS: dict[str, str] = {
    "t1": "t1 axial: Processed_CaPTk",
    "t1c": "t1 axial stealth-post : Processed_CaPTk",
    "t2": "Axial T2 tse: Processed_CaPTk",
    "flair": "t2_Flair_axial: Processed_CaPTk",
}


def _idc_download(series_uid: str, destination: Path) -> None:
    from idc_index import IDCClient

    print(f"Downloading IDC series {series_uid} ...", flush=True)
    IDCClient().download_dicom_series(
        series_uid, str(destination), quiet=False, show_progress_bar=False
    )


def ensure_sequence(cache_dir: Path, sequence: str) -> list[Path]:
    series_uid, expected_count = SEQUENCE_SERIES[sequence]
    raw_dir = cache_dir / "raw" / sequence
    existing = list(raw_dir.rglob("*.dcm")) if raw_dir.exists() else []
    if len(existing) != expected_count:
        if existing:
            print(
                f"{sequence}: found {len(existing)}/{expected_count} instances; "
                "re-downloading.",
                flush=True,
            )
        _idc_download(series_uid, raw_dir)
        existing = list(raw_dir.rglob("*.dcm"))
    if len(existing) != expected_count:
        raise RuntimeError(
            f"{sequence}: expected {expected_count} instances, found {len(existing)}"
        )

    first = pydicom.dcmread(str(existing[0]), stop_before_pixels=True)
    if str(first.SeriesInstanceUID) != series_uid:
        raise RuntimeError(f"{sequence}: unexpected SeriesInstanceUID in downloaded files")
    description = str(getattr(first, "SeriesDescription", None) or "")
    if description != EXPECTED_DESCRIPTIONS[sequence]:
        raise RuntimeError(
            f"{sequence}: SeriesDescription changed -- expected "
            f"{EXPECTED_DESCRIPTIONS[sequence]!r}, found {description!r}"
        )
    print(
        f"Verified {sequence}: {len(existing)} instances, SeriesDescription={description!r}",
        flush=True,
    )
    return sorted(existing)


def _stage_files(source: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(source):
        target = destination / f"instance_{index:04d}.dcm"
        if target.exists():
            continue
        try:
            import os

            os.link(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)


def _write_preview(cache_dir: Path, staged: dict[str, list[Path]], output_path: Path) -> None:
    """Render one anatomically-matched axial slice per sequence, side by side.

    The four sequences share a common resampled z-grid (a side effect of CaPTk's
    co-registration) -- picking the physically nearest slice per sequence to the
    middle of the shortest (FLAIR) series lines all four panels up on the same
    anatomy, the same way the LIDC preview lines its box/mask up with a specific
    seed slice.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    def z_position(path: Path) -> float:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return float(ds.ImagePositionPatient[2])

    flair_paths = sorted(staged["flair"], key=z_position)
    target_z = z_position(flair_paths[len(flair_paths) // 2])

    panels = []
    labels = ["t1", "t1c", "t2", "flair"]
    for name in labels:
        paths = staged[name]
        nearest = min(paths, key=lambda p: abs(z_position(p) - target_z))
        ds = pydicom.dcmread(str(nearest))
        arr = ds.pixel_array.astype(np.float32)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        norm = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        img = (norm * 255).astype(np.uint8)
        panels.append(Image.fromarray(img).convert("RGB").resize((256, 256)))

    combo = Image.new("RGB", (256 * 4 + 30, 256 + 30), (30, 30, 30))
    draw = ImageDraw.Draw(combo)
    for i, (im, label) in enumerate(zip(panels, labels)):
        combo.paste(im, (i * 256 + i * 10, 30))
        draw.text((i * 256 + i * 10 + 5, 5), label.upper(), fill=(255, 255, 0))
    combo.save(output_path)


def prepare(args) -> Path:
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    staged_source: dict[str, list[Path]] = {}
    for sequence in SEQUENCE_SERIES:
        staged_source[sequence] = ensure_sequence(cache_dir, sequence)

    database_path = args.database.resolve()
    storage_root = args.storage_dir.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    session_factory = build_session_factory(engine)

    patient_reference = hashlib.sha256(PATIENT_ID.encode()).hexdigest()[:16]
    series_manifest: dict[str, dict] = {}

    with session_factory() as db:
        study = db.scalar(
            select(Study).where(
                Study.source == "idc",
                Study.patient_reference == patient_reference,
                Study.source_filename == f"IDC {PATIENT_ID}",
            )
        )
        if study is None:
            study = Study(
                modality="unknown",
                body_part="unknown",
                source_filename=f"IDC {PATIENT_ID}",
                source="idc",
                # Not one real DICOM StudyInstanceUID -- see module docstring: each of
                # the four series has its own, an IDC/CaPTk packaging artifact.
                dicom_study_uid=None,
                patient_reference=patient_reference,
            )
            db.add(study)
            db.flush()

        for sequence, source_files in staged_source.items():
            series_uid, _count = SEQUENCE_SERIES[sequence]
            series_dir = (
                storage_root / "studies" / study.id / "series" / series_uid.replace(".", "_")
            )
            _stage_files(source_files, series_dir)

            series = db.scalar(select(Series).where(Series.dicom_series_uid == series_uid))
            if series is None:
                series = Series(
                    study_id=study.id,
                    dicom_series_uid=series_uid,
                    dicom_modality="MR",
                    modality="mri",
                    body_part="brain",
                    instance_count=len(source_files),
                    storage_dir=str(series_dir),
                    analysis_eligible=True,
                )
                db.add(series)
            else:
                series.storage_dir = str(series_dir)
                series.instance_count = len(source_files)
            db.commit()
            db.refresh(series)
            series_manifest[sequence] = {
                "api_series_id": series.id,
                "dicom_series_uid": series_uid,
                "series_description": EXPECTED_DESCRIPTIONS[sequence],
                "storage_dir": str(series_dir),
                "instance_count": len(source_files),
            }

        db.refresh(study)

    preview_path = cache_dir / "preview_t1_t1c_t2_flair.png"
    _write_preview(cache_dir, staged_source, preview_path)

    manifest = {
        "dataset": "UPENN-GBM",
        "case": PATIENT_ID,
        "source": "NCI Imaging Data Commons (collection: upenn_gbm)",
        "source_url": "https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=upenn_gbm",
        "license": "CC BY 4.0",
        "api_study_id": study.id,
        "database_url": f"sqlite:///{database_path}",
        "storage_dir": str(storage_root),
        "preview_path": str(preview_path),
        "series": series_manifest,
    }
    manifest_path = cache_dir / "brats_test_case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    engine.dispose()
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/validation/upenn_gbm_00020")
    )
    parser.add_argument(
        "--storage-dir", type=Path, default=Path("api_storage/brats_test_case")
    )
    parser.add_argument(
        "--database", type=Path, default=Path("api_storage/brats_test_case.sqlite")
    )
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())
