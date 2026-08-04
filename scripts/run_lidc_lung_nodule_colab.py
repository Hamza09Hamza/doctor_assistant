#!/usr/bin/env python3
"""Run the lung-nodule pathology demo on one pinned, expert-annotated LIDC CT."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

try:
    from scripts.run_totalsegmentator_dicom_seg import (
        _referenced_series_uids,
        discover_ct_series,
        main as dicom_seg_main,
        select_ct_series,
    )
except ModuleNotFoundError:  # Direct execution from scripts/
    from run_totalsegmentator_dicom_seg import (  # type: ignore[no-redef]
        _referenced_series_uids,
        discover_ct_series,
        main as dicom_seg_main,
        select_ct_series,
    )


PATIENT_ID = "LIDC-IDRI-0686"
STUDY_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.530012655070930408996523309860"
CT_SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.6279.6001.195557219224169985110295082004"
CT_INSTANCE_COUNT = 238
# A relatively large single-nodule expert SEG from the 49 annotations for this scan.
EXPERT_SEG_SERIES_UID = "1.2.276.0.7230010.3.1.3.0.47207.1553332188.229147"


def _require_gpu() -> tuple[str, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed; run the notebook setup cell first") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is attached. In Colab select Runtime > L4 GPU.")
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if memory_gib < 12:
        raise RuntimeError(
            f"GPU {name!r} has only {memory_gib:.1f} GiB. Use the Colab L4 runtime for this demo."
        )
    print(f"GPU ready: {name} ({memory_gib:.1f} GiB)", flush=True)
    return name, memory_gib


def _idc_download(series_uid: str, destination: Path) -> None:
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise RuntimeError("idc-index is not installed; run the notebook setup cell first") from exc
    print(f"Downloading pinned IDC series {series_uid} ...", flush=True)
    IDCClient().download_dicom_series(
        series_uid,
        str(destination),
        quiet=False,
        show_progress_bar=True,
    )


def ensure_ct(cache_dir: Path) -> Path:
    ct_dir = cache_dir / "ct"
    existing = discover_ct_series(ct_dir) if ct_dir.exists() else []
    matching = [series for series in existing if series.series_instance_uid == CT_SERIES_UID]
    if not matching or matching[0].instance_count != CT_INSTANCE_COUNT:
        if matching:
            print(
                f"Resuming incomplete CT download: found {matching[0].instance_count}/"
                f"{CT_INSTANCE_COUNT} slices.",
                flush=True,
            )
        _idc_download(CT_SERIES_UID, ct_dir)
        existing = discover_ct_series(ct_dir)
    selected = select_ct_series(existing, CT_SERIES_UID)
    if selected.study_instance_uid != STUDY_UID or selected.instance_count != CT_INSTANCE_COUNT:
        raise RuntimeError(
            "Pinned CT identity check failed: expected the documented LIDC study and 238 slices"
        )
    print(
        f"Verified public benchmark CT: {PATIENT_ID}, {selected.instance_count} slices",
        flush=True,
    )
    return ct_dir


def ensure_expert_seg(cache_dir: Path) -> Path:
    import pydicom

    expert_dir = cache_dir / "expert_seg"
    candidates = list(expert_dir.rglob("*")) if expert_dir.exists() else []
    seg_path = next((path for path in candidates if path.is_file()), None)
    if seg_path is None:
        _idc_download(EXPERT_SEG_SERIES_UID, expert_dir)
        candidates = [path for path in expert_dir.rglob("*") if path.is_file()]
    for path in candidates:
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if (
            str(getattr(dataset, "Modality", "")) == "SEG"
            and str(getattr(dataset, "SeriesInstanceUID", "")) == EXPERT_SEG_SERIES_UID
        ):
            if str(getattr(dataset, "StudyInstanceUID", "")) != STUDY_UID:
                raise RuntimeError("Expert SEG does not belong to the pinned CT study")
            if CT_SERIES_UID not in _referenced_series_uids(dataset):
                raise RuntimeError("Expert SEG does not reference the pinned CT series")
            print("Verified the radiologist SEG references the same CT series.", flush=True)
            return path
    raise RuntimeError("IDC download completed but the pinned expert DICOM SEG was not found")


def _comparison_bundle(prediction_bundle: Path, expert_seg: Path) -> Path:
    destination = prediction_bundle.with_name("ohif_ai_vs_expert_bundle.zip")
    partial = destination.with_suffix(".zip.partial")
    shutil.copy2(prediction_bundle, partial)
    with zipfile.ZipFile(partial, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(expert_seg, "expert_reference/lidc_radiologist_nodule_seg.dcm")
    partial.replace(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_gpu()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ct_dir = ensure_ct(args.cache_dir)
    expert_seg = ensure_expert_seg(args.cache_dir)

    command = [
        "--dicom-input",
        str(ct_dir),
        "--output-dir",
        str(args.output_dir),
        "--work-dir",
        str(args.work_dir),
        "--series-instance-uid",
        CT_SERIES_UID,
        "--task",
        "lung_nodules",
        "--require-segment-label",
        "lung_nodules",
        "--confirm-deidentified",
    ]
    if args.force:
        command.append("--force")
    result = dicom_seg_main(command)
    latest_path = args.output_dir / "latest_dicom_seg_run.json"
    latest = json.loads(latest_path.read_text())
    comparison = _comparison_bundle(Path(latest["viewer_bundle"]), expert_seg)
    latest["expert_seg"] = str(expert_seg)
    latest["comparison_bundle"] = str(comparison)
    latest_path.write_text(json.dumps(latest, indent=2) + "\n")

    print("\nSUCCESS: automatic lung-nodule SEG and expert comparison are ready.", flush=True)
    print(f"OHIF comparison bundle: {comparison}", flush=True)
    print("This is experimental visual evaluation, not a diagnosis.", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
