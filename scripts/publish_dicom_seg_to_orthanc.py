#!/usr/bin/env python3
"""Publish a CT + DICOM SEG bundle to Orthanc and print its Clinique Amina URL."""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote

import httpx

try:
    from scripts.run_totalsegmentator_dicom_seg import (
        _safe_extract_zip,
        discover_ct_series,
        select_ct_series,
        validate_dicom_seg,
    )
except ModuleNotFoundError:  # Direct execution: ``python scripts/publish_....py``
    from run_totalsegmentator_dicom_seg import (  # type: ignore[no-redef]
        _safe_extract_zip,
        discover_ct_series,
        select_ct_series,
        validate_dicom_seg,
    )


def clinique_amina_url(base_url: str, study_instance_uid: str) -> str:
    base = base_url.rstrip("/")
    uid = quote(study_instance_uid, safe=".")
    return f"{base}/doctor-assistant/orthancProxy?StudyInstanceUIDs={uid}"


def _resolve_bundle(bundle: Path, destination: Path) -> tuple[Path, Path]:
    if not bundle.is_file() or not zipfile.is_zipfile(bundle):
        raise ValueError(f"viewer bundle is not a readable ZIP: {bundle}")
    _safe_extract_zip(bundle, destination)
    source_dir = destination / "source_dicom"
    seg_path = destination / "totalsegmentator_seg.dcm"
    if not source_dir.is_dir() or not seg_path.is_file():
        raise RuntimeError(
            "viewer bundle must contain source_dicom/ and totalsegmentator_seg.dcm"
        )
    return source_dir, seg_path


def upload_dicom_files(client: httpx.Client, files: list[Path]) -> tuple[int, int]:
    uploaded = 0
    already_stored = 0
    for index, path in enumerate(files, start=1):
        response = client.post(
            "/instances",
            content=path.read_bytes(),
            headers={"Content-Type": "application/dicom"},
        )
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("Status", "")).lower()
        if "already" in status:
            already_stored += 1
        else:
            uploaded += 1
        if index == 1 or index == len(files) or index % 50 == 0:
            print(f"  ... {index}/{len(files)} DICOM objects accepted by Orthanc", flush=True)
    return uploaded, already_stored


def verify_study_visible(client: httpx.Client, study_instance_uid: str) -> None:
    response = client.get(
        "/dicom-web/studies",
        params={"StudyInstanceUID": study_instance_uid},
        headers={"Accept": "application/dicom+json"},
    )
    response.raise_for_status()
    if not response.json():
        raise RuntimeError("Orthanc accepted the objects but QIDO-RS cannot see the study")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--orthanc-url",
        default=os.environ.get("ORTHANC_URL", "http://localhost:8042"),
    )
    parser.add_argument(
        "--orthanc-username",
        default=os.environ.get("ORTHANC_USERNAME", "doctor_assistant"),
    )
    parser.add_argument(
        "--orthanc-password",
        default=os.environ.get("ORTHANC_PASSWORD", "doctor_assistant"),
    )
    parser.add_argument("--ohif-url", default="http://localhost:3000")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the bundle and print its URL without contacting Orthanc",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="ohif_bundle_") as tmp:
        source_dir, seg_path = _resolve_bundle(args.bundle, Path(tmp))
        series = discover_ct_series(source_dir)
        selected = select_ct_series(series, None)
        validation = validate_dicom_seg(seg_path, selected)
        dicom_files = list(selected.files) + [seg_path]

        print(
            f"Bundle verified: {selected.instance_count} CT instances, "
            f"{validation.segment_count} non-empty anatomical segments.",
            flush=True,
        )
        viewer_url = clinique_amina_url(args.ohif_url, selected.study_instance_uid)
        if args.dry_run:
            print("Dry run: Orthanc was not contacted.", flush=True)
            print(f"Clinique Amina: {viewer_url}", flush=True)
            return 0

        try:
            with httpx.Client(
                base_url=args.orthanc_url.rstrip("/"),
                auth=(args.orthanc_username, args.orthanc_password),
                timeout=120.0,
            ) as client:
                print(f"Uploading {len(dicom_files)} DICOM objects to Orthanc ...", flush=True)
                uploaded, already_stored = upload_dicom_files(client, dicom_files)
                verify_study_visible(client, selected.study_instance_uid)
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"cannot connect to Orthanc at {args.orthanc_url}; start the Docker services first"
            ) from exc

    print(
        f"SUCCESS: Orthanc stored {uploaded} new objects; {already_stored} were already present.",
        flush=True,
    )
    print(f"Clinique Amina: {viewer_url}", flush=True)
    print("Open that URL, select the SEG series, then use the Segmentation panel.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
