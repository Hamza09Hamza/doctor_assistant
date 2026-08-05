#!/usr/bin/env python3
"""Create a TotalSegmentator DICOM SEG from one de-identified CT series.

This is the GPU/Colab half of the OHIF workflow.  It accepts a directory or ZIP
containing DICOM, selects one CT series, stages it on fast local storage, runs
TotalSegmentator, validates the resulting DICOM SEG references, and writes a
portable bundle containing both the source series and the SEG object.

No attempt is made to de-identify data here.  ``--confirm-deidentified`` is a
required explicit assertion by the operator; tag values that commonly contain
patient identity are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"


@dataclass(frozen=True)
class DicomSeries:
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str | None
    files: tuple[Path, ...]
    sop_instance_uids: tuple[str, ...]

    @property
    def instance_count(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class SegValidation:
    study_instance_uid: str
    series_instance_uid: str
    referenced_series_instance_uid: str
    segment_count: int
    frame_count: int
    segment_labels: tuple[str, ...]


def _require_pydicom():
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - exercised in minimal environments only
        raise RuntimeError("pydicom is required; install the pinned Colab dependencies first") from exc
    return pydicom


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP without permitting absolute paths or ``..`` traversal."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"unsafe path in DICOM ZIP: {member.filename!r}")
        zf.extractall(destination)


def _materialize_input(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"DICOM input does not exist: {source}")
    if source.is_dir():
        return source
    if not zipfile.is_zipfile(source):
        raise ValueError("DICOM input must be a directory or ZIP file")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting DICOM ZIP to local storage: {destination}", flush=True)
    _safe_extract_zip(source, destination)
    return destination


def discover_ct_series(root: Path, modalities: frozenset[str] = frozenset({"CT"})) -> list[DicomSeries]:
    """Return DICOM series matching `modalities` without reading pixel data or printing
    identifying tags. Defaults to CT only (this module's original, still-only use case);
    scripts/publish_dicom_seg_to_orthanc.py passes {"CT", "MR"} so it can also publish
    the synthetic-MRI brain-tumour bundle built by build_brain_tumor_seg_bundle.py."""
    pydicom = _require_pydicom()
    tags = [
        "Modality",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "FrameOfReferenceUID",
        "InstanceNumber",
    ]
    grouped: dict[tuple[str, str], list[tuple[int, Path, str, str | None]]] = {}

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                specific_tags=tags,
                force=False,
            )
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")).upper() not in modalities:
            continue
        study_uid = str(getattr(dataset, "StudyInstanceUID", ""))
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        sop_uid = str(getattr(dataset, "SOPInstanceUID", ""))
        if not study_uid or not series_uid or not sop_uid:
            continue
        try:
            instance_number = int(getattr(dataset, "InstanceNumber", 0))
        except (TypeError, ValueError):
            instance_number = 0
        frame_uid = str(getattr(dataset, "FrameOfReferenceUID", "")) or None
        grouped.setdefault((study_uid, series_uid), []).append(
            (instance_number, path, sop_uid, frame_uid)
        )

    series: list[DicomSeries] = []
    for (study_uid, series_uid), instances in grouped.items():
        instances.sort(key=lambda item: (item[0], str(item[1])))
        sop_uids = [item[2] for item in instances]
        if len(set(sop_uids)) != len(sop_uids):
            raise RuntimeError(f"CT series {series_uid} contains duplicate SOP Instance UIDs")
        frame_uids = {item[3] for item in instances if item[3]}
        if len(frame_uids) > 1:
            raise RuntimeError(f"CT series {series_uid} contains multiple Frame of Reference UIDs")
        series.append(
            DicomSeries(
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                frame_of_reference_uid=next(iter(frame_uids), None),
                files=tuple(item[1] for item in instances),
                sop_instance_uids=tuple(sop_uids),
            )
        )
    return sorted(series, key=lambda item: item.instance_count, reverse=True)


def select_ct_series(series: list[DicomSeries], requested_uid: str | None) -> DicomSeries:
    if not series:
        raise RuntimeError("no readable CT DICOM series was found in the input")
    if requested_uid:
        matches = [item for item in series if item.series_instance_uid == requested_uid]
        if not matches:
            available = ", ".join(
                f"{item.series_instance_uid} ({item.instance_count} instances)" for item in series
            )
            raise RuntimeError(
                f"requested SeriesInstanceUID {requested_uid!r} was not found; available: {available}"
            )
        return matches[0]
    if len(series) == 1:
        return series[0]

    largest, second = series[0], series[1]
    if second.instance_count <= 10 and largest.instance_count >= 8 * second.instance_count:
        print(
            "Multiple CT series found; selecting the only full stack "
            f"({largest.instance_count} slices).",
            flush=True,
        )
        return largest

    available = "\n".join(
        f"  {item.series_instance_uid}  ({item.instance_count} instances)" for item in series
    )
    raise RuntimeError(
        "multiple substantial CT series were found. Set --series-instance-uid explicitly:\n"
        + available
    )


def _source_signature(series: DicomSeries) -> str:
    payload = "\n".join(
        [series.study_instance_uid, series.series_instance_uid, *series.sop_instance_uids]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stage_series(series: DicomSeries, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    print(f"Staging {series.instance_count} CT slices on local Colab storage ...", flush=True)
    for index, source in enumerate(series.files):
        shutil.copy2(source, destination / f"instance_{index:05d}.dcm")


def _referenced_series_uids(dataset) -> set[str]:
    referenced = set()
    sequence = getattr(dataset, "ReferencedSeriesSequence", None) or []
    for item in sequence:
        uid = str(getattr(item, "SeriesInstanceUID", ""))
        if uid:
            referenced.add(uid)
    return referenced


def validate_dicom_seg(
    seg_path: Path,
    source: DicomSeries,
    required_segment_labels: tuple[str, ...] = (),
) -> SegValidation:
    pydicom = _require_pydicom()
    if not seg_path.is_file() or seg_path.stat().st_size == 0:
        raise RuntimeError(f"DICOM SEG was not created: {seg_path}")
    dataset = pydicom.dcmread(str(seg_path), stop_before_pixels=False)
    if str(getattr(dataset, "SOPClassUID", "")) != SEGMENTATION_STORAGE_UID:
        raise RuntimeError("output is not a DICOM Segmentation Storage object")
    if str(getattr(dataset, "Modality", "")) != "SEG":
        raise RuntimeError("output DICOM object does not declare Modality=SEG")
    study_uid = str(getattr(dataset, "StudyInstanceUID", ""))
    if study_uid != source.study_instance_uid:
        raise RuntimeError("DICOM SEG StudyInstanceUID does not match the source CT study")
    referenced_uids = _referenced_series_uids(dataset)
    if source.series_instance_uid not in referenced_uids:
        raise RuntimeError("DICOM SEG does not reference the selected source CT series")
    segment_sequence = getattr(dataset, "SegmentSequence", []) or []
    segment_count = len(segment_sequence)
    segment_labels = tuple(
        str(getattr(segment, "SegmentLabel", "")).strip() for segment in segment_sequence
    )
    frame_count = int(getattr(dataset, "NumberOfFrames", 0) or 0)
    if segment_count < 1 or frame_count < 1 or not getattr(dataset, "PixelData", b""):
        raise RuntimeError("DICOM SEG contains no usable segments, frames, or pixel data")
    normalized_labels = {label.casefold() for label in segment_labels}
    missing_labels = [
        label for label in required_segment_labels if label.casefold() not in normalized_labels
    ]
    if missing_labels:
        raise RuntimeError(
            "DICOM SEG is missing required segment label(s): " + ", ".join(missing_labels)
        )
    if required_segment_labels:
        number_to_label = {
            int(segment.SegmentNumber): str(segment.SegmentLabel).strip()
            for segment in segment_sequence
        }
        labels_with_frames: set[str] = set()
        for frame_group in getattr(dataset, "PerFrameFunctionalGroupsSequence", []) or []:
            identification = getattr(frame_group, "SegmentIdentificationSequence", []) or []
            if identification:
                number = int(identification[0].ReferencedSegmentNumber)
                if number in number_to_label:
                    labels_with_frames.add(number_to_label[number].casefold())
        empty_labels = [
            label for label in required_segment_labels if label.casefold() not in labels_with_frames
        ]
        if empty_labels:
            raise RuntimeError(
                "DICOM SEG has no encoded frames for required segment label(s): "
                + ", ".join(empty_labels)
            )
    return SegValidation(
        study_instance_uid=study_uid,
        series_instance_uid=str(getattr(dataset, "SeriesInstanceUID", "")),
        referenced_series_instance_uid=source.series_instance_uid,
        segment_count=segment_count,
        frame_count=frame_count,
        segment_labels=segment_labels,
    )


def _run_with_heartbeat(function, *, interval_seconds: float = 30.0):
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not finished.wait(interval_seconds):
            elapsed = (time.monotonic() - started) / 60.0
            print(f"  ... TotalSegmentator is still working ({elapsed:.1f} minutes elapsed)", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return function()
    finally:
        finished.set()
        thread.join(timeout=1)


def _run_totalsegmentator(
    *,
    dicom_dir: Path,
    seg_path: Path,
    statistics_path: Path,
    report_path: Path,
    task: str,
    fast: bool,
    force_split: bool,
) -> None:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise RuntimeError("TotalSegmentator is not installed in this Python environment") from exc
    try:
        import highdicom  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("highdicom is required to create DICOM SEG output") from exc

    seg_path.parent.mkdir(parents=True, exist_ok=True)

    def inference():
        return totalsegmentator(
            input=dicom_dir,
            output=seg_path,
            task=task,
            device="gpu",
            fast=fast,
            output_type="dicom_seg",
            statistics=statistics_path,
            report=report_path,
            nr_thr_saving=1,
            force_split=force_split,
        )

    quality = "fast_3mm" if fast else "full_1p5mm"
    print(f"Running TotalSegmentator: task={task}, quality={quality}, device=gpu", flush=True)
    _run_with_heartbeat(inference)


def _write_bundle(
    *,
    source_dir: Path,
    seg_path: Path,
    manifest_path: Path,
    destination: Path,
) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    tmp_path.unlink(missing_ok=True)
    print("Creating the portable CT + SEG viewer bundle ...", flush=True)
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(source_dir.glob("*.dcm")):
            zf.write(path, f"source_dicom/{path.name}")
        zf.write(seg_path, "totalsegmentator_seg.dcm")
        zf.write(manifest_path, "run_manifest.json")
    tmp_path.replace(destination)


def _copy_result(local_result: Path, durable_result: Path) -> None:
    durable_result.parent.mkdir(parents=True, exist_ok=True)
    staging = durable_result.with_name(durable_result.name + ".copying")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(local_result, staging)
    if durable_result.exists():
        shutil.rmtree(durable_result)
    staging.replace(durable_result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--series-instance-uid")
    parser.add_argument(
        "--task",
        default="total",
        help="TotalSegmentator task name (for example: total or lung_nodules)",
    )
    parser.add_argument(
        "--require-segment-label",
        action="append",
        default=[],
        help="Fail unless this exact label exists in the DICOM SEG (repeatable)",
    )
    parser.add_argument("--fast", action="store_true", help="Use the lower-resolution 3 mm model")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Split inference into three parts to reduce peak GPU memory",
    )
    parser.add_argument("--force", action="store_true", help="Intentionally rerun inference")
    parser.add_argument(
        "--confirm-deidentified",
        action="store_true",
        help="Required assertion that the operator has de-identified the input",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_deidentified:
        raise RuntimeError(
            "refusing to process or bundle DICOM until --confirm-deidentified is supplied; "
            "this script does not remove patient identifiers"
        )
    if args.fast and args.task == "lung_nodules":
        raise RuntimeError("the lung_nodules task has no supported fast model; remove --fast")
    safe_task = "".join(character if character.isalnum() or character in "_-" else "_" for character in args.task)
    if not safe_task:
        raise RuntimeError("--task must contain at least one letter or number")

    print("=== DICOM CT -> TotalSegmentator DICOM SEG started ===", flush=True)
    print("Patient-identifying DICOM tag values will not be printed.", flush=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dicom_input_", dir=args.work_dir) as extraction_tmp:
        input_root = _materialize_input(args.dicom_input, Path(extraction_tmp))
        discovered = discover_ct_series(input_root)
        print(f"Found {len(discovered)} CT series.", flush=True)
        selected = select_ct_series(discovered, args.series_instance_uid)
        print(
            "Selected CT series: "
            f"{selected.series_instance_uid} ({selected.instance_count} instances)",
            flush=True,
        )

        signature = _source_signature(selected)
        quality = "fast_3mm" if args.fast else "full_1p5mm"
        run_key = hashlib.sha256(
            f"{selected.study_instance_uid}|{selected.series_instance_uid}|{args.task}".encode("utf-8")
        ).hexdigest()[:12]
        durable_result = args.output_dir / f"study_{run_key}" / f"task_{safe_task}" / quality
        durable_seg = durable_result / "totalsegmentator_seg.dcm"
        durable_manifest = durable_result / "run_manifest.json"

        durable_bundle = durable_result / "ohif_viewer_bundle.zip"
        if (
            durable_seg.exists()
            and durable_manifest.exists()
            and durable_bundle.exists()
            and not args.force
        ):
            old_manifest = json.loads(durable_manifest.read_text())
            if (
                old_manifest.get("source_signature") == signature
                and old_manifest.get("task", "total") == args.task
            ):
                validation = validate_dicom_seg(
                    durable_seg, selected, tuple(args.require_segment_label)
                )
                print("Reusing the verified DICOM SEG already stored in Drive.", flush=True)
                latest = {
                    "result_dir": str(durable_result),
                    "dicom_seg": str(durable_seg),
                    "viewer_bundle": str(durable_bundle),
                    "manifest": str(durable_manifest),
                    "study_instance_uid": selected.study_instance_uid,
                    "validation": asdict(validation),
                }
                (args.output_dir / "latest_dicom_seg_run.json").write_text(
                    json.dumps(latest, indent=2) + "\n"
                )
                print(f"Viewer bundle: {latest['viewer_bundle']}", flush=True)
                return 0

        local_result = args.work_dir / f"study_{run_key}" / f"task_{safe_task}" / quality
        source_dir = local_result / "source_dicom"
        if local_result.exists():
            shutil.rmtree(local_result)
        local_result.mkdir(parents=True)
        _stage_series(selected, source_dir)

        seg_path = local_result / "totalsegmentator_seg.dcm"
        statistics_path = local_result / "statistics.json"
        report_path = local_result / "totalsegmentator_run_report.json"
        started = time.monotonic()
        _run_totalsegmentator(
            dicom_dir=source_dir,
            seg_path=seg_path,
            statistics_path=statistics_path,
            report_path=report_path,
            task=args.task,
            fast=args.fast,
            force_split=args.force_split,
        )
        runtime_seconds = round(time.monotonic() - started, 2)
        validation = validate_dicom_seg(
            seg_path, selected, tuple(args.require_segment_label)
        )

        manifest = {
            "workflow": "totalsegmentator_dicom_seg",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_signature": signature,
            "study_instance_uid": selected.study_instance_uid,
            "source_series_instance_uid": selected.series_instance_uid,
            "source_instance_count": selected.instance_count,
            "task": args.task,
            "required_segment_labels": args.require_segment_label,
            "quality": quality,
            "force_split": args.force_split,
            "runtime_seconds": runtime_seconds,
            "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"),
            "pydicom_version": importlib.metadata.version("pydicom"),
            "highdicom_version": importlib.metadata.version("highdicom"),
            "validation": asdict(validation),
            "dicom_seg_sha256": _hash_file(seg_path),
            "notice": (
                "Experimental pathology segmentation for visual evaluation; not a diagnosis."
                if args.task != "total"
                else "Anatomical segmentation for visual QC; not a diagnosis."
            ),
        }
        manifest_path = local_result / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        bundle_path = local_result / "ohif_viewer_bundle.zip"
        _write_bundle(
            source_dir=source_dir,
            seg_path=seg_path,
            manifest_path=manifest_path,
            destination=bundle_path,
        )
        # The bundle already contains the selected source series. Avoid storing a
        # second full copy of the CT beside it in Google Drive.
        shutil.rmtree(source_dir)
        _copy_result(local_result, durable_result)

        latest = {
            "result_dir": str(durable_result),
            "dicom_seg": str(durable_result / seg_path.name),
            "viewer_bundle": str(durable_result / bundle_path.name),
            "manifest": str(durable_result / manifest_path.name),
            "study_instance_uid": selected.study_instance_uid,
            "validation": asdict(validation),
        }
        (args.output_dir / "latest_dicom_seg_run.json").write_text(
            json.dumps(latest, indent=2) + "\n"
        )

    print("\nSUCCESS: standards-valid DICOM SEG created and references verified.", flush=True)
    print(f"DICOM SEG:    {latest['dicom_seg']}", flush=True)
    print(f"Viewer bundle: {latest['viewer_bundle']}", flush=True)
    print("Next: publish the bundle to local Orthanc and open the printed OHIF URL.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
