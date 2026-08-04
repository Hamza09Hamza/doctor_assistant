#!/usr/bin/env python3
"""One-command local TotalSegmentator + OHIF demo using a public test CT.

The selected source is the de-identified ACRIN CT series published in the MIT-
licensed ``OHIF/viewer-testdata`` repository.  The repository revision, DICOM
Study/Series UIDs, instance count, and de-identification flag are all pinned.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import sys
from pathlib import Path

import requests

try:
    from scripts.run_totalsegmentator_dicom_seg import main as dicom_seg_main
except ModuleNotFoundError:  # Direct execution from scripts/
    from run_totalsegmentator_dicom_seg import main as dicom_seg_main  # type: ignore[no-redef]


OHIF_TESTDATA_COMMIT = "c16371c6e52894411af23b63a9ca65af5161b00c"
PUBLIC_CT_BASE_URL = (
    "https://raw.githubusercontent.com/OHIF/viewer-testdata/"
    f"{OHIF_TESTDATA_COMMIT}/dcm/acrin"
)
EXPECTED_STUDY_UID = "1.3.6.1.4.1.14519.5.2.1.7009.2403.334240657131972136850343327463"
EXPECTED_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.7009.2403.226151125820845824875394858561"
EXPECTED_INSTANCE_COUNT = 135


def _require_local_gpu() -> tuple[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see the NVIDIA GPU. Run this command in the same normal terminal "
            "where nvidia-smi succeeds."
        )
    properties = torch.cuda.get_device_properties(0)
    return properties.name, properties.total_memory / 1024**3


def _download_one(index: int, destination: Path) -> Path:
    filename = f"ct-1-{index:03d}.dcm"
    final_path = destination / filename
    if final_path.exists() and final_path.stat().st_size > 1024:
        return final_path
    partial_path = final_path.with_suffix(".dcm.partial")
    url = f"{PUBLIC_CT_BASE_URL}/{filename}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with partial_path.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if partial_path.stat().st_size <= 1024:
        raise RuntimeError(f"downloaded DICOM file is unexpectedly small: {filename}")
    partial_path.replace(final_path)
    return final_path


def download_public_ct(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading/reusing {EXPECTED_INSTANCE_COUNT} de-identified public CT slices "
        "from the pinned OHIF test-data revision ...",
        flush=True,
    )
    completed = 0
    paths: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {
            executor.submit(_download_one, index, destination): index
            for index in range(1, EXPECTED_INSTANCE_COUNT + 1)
        }
        for future in concurrent.futures.as_completed(future_map):
            paths.append(future.result())
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == EXPECTED_INSTANCE_COUNT:
                print(f"  ... {completed}/{EXPECTED_INSTANCE_COUNT} slices ready", flush=True)
    return sorted(paths)


def validate_public_ct(paths: list[Path]) -> str:
    import pydicom

    if len(paths) != EXPECTED_INSTANCE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_INSTANCE_COUNT} CT instances, found {len(paths)}"
        )
    study_uids: set[str] = set()
    series_uids: set[str] = set()
    sop_uids: set[str] = set()
    deidentified: set[str] = set()
    digest = hashlib.sha256()
    for path in paths:
        dataset = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            specific_tags=[
                "Modality",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "PatientIdentityRemoved",
            ],
        )
        if str(getattr(dataset, "Modality", "")) != "CT":
            raise RuntimeError(f"public test file is not CT: {path.name}")
        study_uids.add(str(dataset.StudyInstanceUID))
        series_uids.add(str(dataset.SeriesInstanceUID))
        sop_uids.add(str(dataset.SOPInstanceUID))
        deidentified.add(str(getattr(dataset, "PatientIdentityRemoved", "")).upper())
        digest.update(path.name.encode("utf-8"))
        digest.update(str(dataset.SOPInstanceUID).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    if study_uids != {EXPECTED_STUDY_UID} or series_uids != {EXPECTED_SERIES_UID}:
        raise RuntimeError("public CT Study/Series UID does not match the pinned source")
    if len(sop_uids) != EXPECTED_INSTANCE_COUNT:
        raise RuntimeError("public CT contains missing or duplicate SOP Instance UIDs")
    if deidentified != {"YES"}:
        raise RuntimeError("public CT does not consistently declare PatientIdentityRemoved=YES")
    signature = digest.hexdigest()
    print("Public CT validation passed: one de-identified 135-slice CT series.", flush=True)
    print("Source signature:", signature, flush=True)
    return signature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/public_ohif_acrin_ct"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/totalsegmentator_ohif_local_demo"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/totalsegmentator_ohif_local_work"),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("results/totalsegmentator_weights"),
        help="Durable local model-weight cache",
    )
    parser.add_argument("--fast", action="store_true", help="Use 3 mm instead of full 1.5 mm")
    parser.add_argument("--force", action="store_true", help="Intentionally rerun inference")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_name, vram_gib = _require_local_gpu()
    print(f"GPU ready: {gpu_name} ({vram_gib:.1f} GiB VRAM)", flush=True)
    if vram_gib < 5.5 and not args.fast:
        raise RuntimeError("less than 5.5 GiB VRAM detected; rerun with --fast")
    try:
        import highdicom  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "highdicom is missing. Install it once with: "
            f"{sys.executable} -m pip install highdicom==0.27.0"
        ) from exc

    paths = download_public_ct(args.cache_dir)
    validate_public_ct(paths)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", str(args.weights_dir.resolve()))
    print("Model-weight cache:", args.weights_dir.resolve(), flush=True)
    command = [
        "--dicom-input",
        str(args.cache_dir),
        "--output-dir",
        str(args.output_dir),
        "--work-dir",
        str(args.work_dir),
        "--series-instance-uid",
        EXPECTED_SERIES_UID,
        "--confirm-deidentified",
        # Full 1.5 mm inference, split into three spatial chunks to keep peak
        # allocation safely inside the RTX 3050 Laptop GPU's 6 GiB VRAM.
        "--force-split",
    ]
    if args.fast:
        command.append("--fast")
    if args.force:
        command.append("--force")
    return dicom_seg_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
