"""Run one real TotalSegmentator CT case and produce inspectable artifacts.

This is the visual proof step before DICOM-SEG/OHIF integration.  It downloads the
official 102-subject TotalSegmentator sample archive, verifies its published MD5,
extracts only the selected CT (not the whole archive), runs the full 1.5 mm model by
default, and writes:

* ``segmentation.nii.gz`` -- the 117-label mask in the source CT geometry;
* ``preview.png`` -- one abdominal CT slice with a colored segmentation overlay;
* ``measurements.json`` -- per-structure voxel counts and volumes;
* ``run_manifest.json`` -- input/output hashes and runtime provenance; and
* TotalSegmentator's own statistics/report JSON files.

The output describes segmented anatomy, not diagnosed pathology.  It deliberately does
not send organ names through the generic finding/recommendation reporter.

Recommended Colab invocation (L4 GPU):

    python scripts/demo_totalsegmentator.py \
      --drive-cache /content/drive/MyDrive/doctor_assistant/totalsegmentator_demo \
      --work-dir /content/totalsegmentator_work

The durable cache holds the archive, model output, and final artifacts.  Large inference
I/O happens under ``--work-dir`` so Google Drive FUSE is not on the model's hot path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ZENODO_RECORD_ID = "10047263"
ZENODO_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
ZENODO_FILENAME = "Totalsegmentator_dataset_small_v201.zip"
# Published by Zenodo for record 10047263, version 2.0.1.
ZENODO_MD5 = "6b5524af4b15e6ba06ef2d700c0c73e0"


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_retries(url: str, destination: Path, *, attempts: int = 5) -> None:
    """Download with resume support and visible progress for the multi-GB archive."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            existing = destination.stat().st_size if destination.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                total = response.headers.get("Content-Range", "").split("/")[-1]
                total_bytes = int(total) if total.isdigit() else None
                mode = "ab" if existing and response.status_code == 206 else "wb"
                written = existing if mode == "ab" else 0
                last_reported_block = written // (200 * 1024 * 1024)
                with destination.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        block = written // (200 * 1024 * 1024)
                        if block != last_reported_block:
                            last_reported_block = block
                            pct = f" ({100 * written / total_bytes:.0f}%)" if total_bytes else ""
                            print(
                                f"  ... {written / 1024 / 1024:.0f} MB downloaded{pct}",
                                flush=True,
                            )
            return
        except Exception as exc:  # noqa: BLE001 -- retried boundary around remote I/O
            print(f"Download attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt == attempts:
                raise
    raise RuntimeError("unreachable")


def _resolve_download_url() -> tuple[str, int | None]:
    """Resolve the one expected Zenodo file and reject changed record metadata."""
    import requests

    response = requests.get(ZENODO_RECORD_URL, timeout=30)
    response.raise_for_status()
    matches = [
        entry
        for entry in response.json().get("files", [])
        if entry.get("key") == ZENODO_FILENAME
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Zenodo record {ZENODO_RECORD_ID} did not contain exactly one {ZENODO_FILENAME!r}"
        )
    entry = matches[0]
    remote_checksum = str(entry.get("checksum", "")).removeprefix("md5:").lower()
    if remote_checksum and remote_checksum != ZENODO_MD5:
        raise RuntimeError(
            f"Zenodo checksum changed: expected {ZENODO_MD5}, metadata reports {remote_checksum}"
        )
    links = entry.get("links", {})
    url = links.get("content") or links.get("self")
    if not url:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD_ID} supplied no download URL")
    size = entry.get("size")
    return str(url), int(size) if size is not None else None


def _verify_archive(path: Path) -> None:
    """Verify once, then reuse a size-bound receipt on later Colab sessions."""
    receipt_path = path.with_suffix(path.suffix + ".verified.json")
    if receipt_path.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = {}
        if (
            receipt.get("md5") == ZENODO_MD5
            and receipt.get("size_bytes") == path.stat().st_size
        ):
            print(f"Archive verification receipt is valid: {path}")
            return

    print("Verifying the published archive MD5 (one-time check) ...", flush=True)
    actual = _hash_file(path, "md5")
    if actual != ZENODO_MD5:
        raise RuntimeError(
            f"Archive checksum mismatch for {path}: expected {ZENODO_MD5}, got {actual}. "
            "The file was left in place for inspection; remove it before retrying."
        )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "md5": actual,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "zenodo_record_id": ZENODO_RECORD_ID,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _ensure_archive(cache_dir: Path) -> Path:
    dataset_dir = cache_dir / "dataset"
    archive_path = dataset_dir / ZENODO_FILENAME
    if not archive_path.exists():
        print("Resolving the official Zenodo download ...", flush=True)
        url, expected_size = _resolve_download_url()
        print(f"Downloading {ZENODO_FILENAME} from Zenodo ...", flush=True)
        _download_with_retries(url, archive_path)
        if expected_size is not None and archive_path.stat().st_size != expected_size:
            raise RuntimeError(
                "Downloaded size mismatch: "
                f"expected {expected_size}, got {archive_path.stat().st_size}"
            )
    else:
        print(f"Reusing downloaded archive: {archive_path}")
    _verify_archive(archive_path)
    return archive_path


def _extract_selected_ct(
    archive_path: Path, cache_dir: Path, subject_index: int
) -> tuple[str, Path]:
    """Extract only one ``ct.nii.gz`` member, rather than all 102 subjects."""
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (member for member in archive.infolist() if member.filename.endswith("/ct.nii.gz")),
            key=lambda member: member.filename,
        )
        if not members:
            raise RuntimeError(f"No subject ct.nii.gz files found in {archive_path}")
        if subject_index < 0 or subject_index >= len(members):
            raise ValueError(
                f"--subject-index {subject_index} out of range; "
                f"archive has {len(members)} subjects"
            )
        member = members[subject_index]
        subject_id = Path(member.filename).parent.name
        if not subject_id:
            raise RuntimeError(
                f"Could not derive subject ID from archive member {member.filename!r}"
            )
        destination = cache_dir / "subjects" / subject_id / "ct.nii.gz"
        if destination.exists() and destination.stat().st_size == member.file_size:
            print(f"Reusing extracted CT: {destination}")
            return subject_id, destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting only {member.filename} ...", flush=True)
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if destination.stat().st_size != member.file_size:
            raise RuntimeError(f"Extracted CT size mismatch for {destination}")
        return subject_id, destination


def _structure_measurements(seg_data, spacing, class_map: dict[int, str]) -> list[dict]:
    import numpy as np

    labels, counts = np.unique(seg_data, return_counts=True)
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    measurements = [
        {
            "label_id": int(label),
            "name": class_map.get(int(label), f"structure_{int(label)}"),
            "voxels": int(count),
            "volume_ml": round(float(count) * voxel_volume_ml, 3),
        }
        for label, count in zip(labels, counts)
        if int(label) != 0
    ]
    measurements.sort(key=lambda item: item["volume_ml"], reverse=True)
    return measurements


def _select_preview_slice(seg_data, class_map: dict[int, str]) -> int:
    """Choose an axial slice rich in abdominal organs, with a safe all-label fallback."""
    import numpy as np

    priority_names = {"liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach"}
    priority_ids = [label for label, name in class_map.items() if name in priority_names]
    if priority_ids:
        priority_mask = np.isin(seg_data, priority_ids)
        scores = priority_mask.sum(axis=(0, 1))
        if scores.max() > 0:
            return int(scores.argmax())
    return int((seg_data > 0).sum(axis=(0, 1)).argmax())


def _render_preview(ct_data, seg_data, class_map: dict[int, str], destination: Path) -> int:
    """Save a three-panel axial visual-QC image and return the chosen slice index."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    slice_index = _select_preview_slice(seg_data, class_map)
    ct_slice = np.rot90(np.asarray(ct_data[:, :, slice_index], dtype=np.float32))
    seg_slice = np.rot90(np.asarray(seg_data[:, :, slice_index], dtype=np.uint8))

    labels, counts = np.unique(seg_slice, return_counts=True)
    present = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0]
    cmap = plt.get_cmap("turbo")
    rgba = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    for label, _count in present:
        color = cmap(label / max(class_map))
        rgba[seg_slice == label] = (*color[:3], 0.52)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
    axes[0].set_title("CT (abdominal window)")
    axes[1].imshow(ct_slice, cmap="gray", vmin=-160, vmax=240)
    axes[1].imshow(rgba)
    axes[1].set_title("TotalSegmentator overlay")
    axes[2].imshow(rgba)
    axes[2].set_facecolor("black")
    axes[2].set_title("Segmentation mask")

    top_labels = sorted(present, key=lambda pair: pair[1], reverse=True)[:12]
    legend = [
        Patch(color=cmap(label / max(class_map)), label=class_map.get(label, str(label)))
        for label, _count in top_labels
    ]
    if legend:
        axes[2].legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.suptitle(
        f"Visual QC only — axial slice {slice_index}; not a diagnosis",
        fontsize=14,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return slice_index


def _runtime_info() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "totalsegmentator": importlib.metadata.version("TotalSegmentator"),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:  # noqa: BLE001 -- provenance should not hide the main result
        info["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _copy_existing_files(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir() if source_dir.exists() else ():
        if source.is_file():
            shutil.copy2(source, destination_dir / source.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-cache",
        type=Path,
        default=Path("/content/drive/MyDrive/doctor_assistant/totalsegmentator_demo"),
        help="durable cache/output directory (normally Google Drive)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/totalsegmentator_work"),
        help="fast local scratch directory used for model I/O",
    )
    parser.add_argument("--subject-index", type=int, default=0, help="subject index, 0-101")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use the lower-resolution 3 mm model; omit for the full 1.5 mm model",
    )
    parser.add_argument("--device", default="gpu", help="TotalSegmentator device, e.g. gpu or cpu")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun inference even when a durable segmentation already exists",
    )
    args = parser.parse_args()

    cache_dir = args.drive_cache.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _ensure_archive(cache_dir)
    subject_id, cached_ct = _extract_selected_ct(archive_path, cache_dir, args.subject_index)

    quality = "fast_3mm" if args.fast else "full_1p5mm"
    durable_result_dir = cache_dir / "results" / subject_id / quality
    local_result_dir = args.work_dir.resolve() / subject_id / quality
    local_result_dir.mkdir(parents=True, exist_ok=True)
    local_ct = local_result_dir / "ct.nii.gz"
    if not local_ct.exists() or local_ct.stat().st_size != cached_ct.stat().st_size:
        print(f"Staging CT on local Colab storage: {local_ct}", flush=True)
        shutil.copy2(cached_ct, local_ct)

    local_seg = local_result_dir / "segmentation.nii.gz"
    fresh_inference = args.force or not (durable_result_dir / "segmentation.nii.gz").exists()
    if fresh_inference:
        from totalsegmentator.python_api import totalsegmentator

        print(
            f"Running TotalSegmentator for {subject_id}: quality={quality}, device={args.device}",
            flush=True,
        )
        started = time.monotonic()
        result = totalsegmentator(
            input=local_ct,
            output=local_seg,  # ml=True requires a file path, not a directory
            ml=True,
            fast=args.fast,
            device=args.device,
            task="total",
            statistics=local_result_dir / "totalsegmentator_statistics.json",
            statistics_extra=True,
            report=local_result_dir / "totalsegmentator_run_report.json",
        )
        _seg_img, _statistics = result
        inference_seconds = time.monotonic() - started
    else:
        print(f"Reusing durable segmentation: {durable_result_dir / 'segmentation.nii.gz'}")
        _copy_existing_files(durable_result_dir, local_result_dir)
        inference_seconds = None

    import nibabel as nib
    import numpy as np
    from totalsegmentator.map_to_binary import class_map

    if not local_seg.exists():
        raise RuntimeError(f"TotalSegmentator did not produce {local_seg}")
    ct_img = nib.load(local_ct)
    seg_img = nib.load(local_seg)
    if ct_img.shape != seg_img.shape:
        raise RuntimeError(
            f"Geometry failure: CT shape {ct_img.shape} != mask shape {seg_img.shape}"
        )
    if not np.allclose(ct_img.affine, seg_img.affine, rtol=1e-4, atol=1e-3):
        raise RuntimeError("Geometry failure: CT and segmentation affines do not match")

    # Keep the CT lazy/memory-mapped: the preview reads only one axial slice.  The
    # segmentation must be materialized once for counts and the non-empty-mask check.
    ct_data = ct_img.dataobj
    seg_data = np.asarray(seg_img.dataobj, dtype=np.uint8)
    structure_map = {int(key): value for key, value in class_map["total"].items()}
    spacing = tuple(float(value) for value in seg_img.header.get_zooms()[:3])
    measurements = _structure_measurements(seg_data, spacing, structure_map)
    if not measurements:
        raise RuntimeError("Segmentation is empty; no anatomical structures were produced")

    measurements_path = local_result_dir / "measurements.json"
    measurements_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subject_id": subject_id,
                "spacing_mm": spacing,
                "structures": measurements,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    preview_path = local_result_dir / "preview.png"
    preview_slice = _render_preview(ct_data, seg_data, structure_map, preview_path)

    manifest_path = local_result_dir / "run_manifest.json"
    if fresh_inference or not manifest_path.exists():
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "visual segmentation proof; not a diagnosis or clinical validation",
            "subject_id": subject_id,
            "dataset": {
                "zenodo_record_id": ZENODO_RECORD_ID,
                "filename": ZENODO_FILENAME,
                "published_md5": ZENODO_MD5,
            },
            "model": {
                "task": "total",
                "quality": quality,
                "fast": args.fast,
                "device_requested": args.device,
            },
            "runtime": _runtime_info(),
            "inference_seconds": inference_seconds,
            "geometry": {
                "shape": list(seg_img.shape),
                "spacing_mm": list(spacing),
                "affine_matches_source": True,
            },
            "preview_slice_index": preview_slice,
            "structure_count": len(measurements),
            "files": {
                "ct_sha256": _hash_file(local_ct),
                "segmentation_sha256": _hash_file(local_seg),
                "measurements_sha256": _hash_file(measurements_path),
                "preview_sha256": _hash_file(preview_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    durable_result_dir.mkdir(parents=True, exist_ok=True)
    for artifact in local_result_dir.iterdir():
        if artifact.is_file() and artifact.name != "ct.nii.gz":
            shutil.copy2(artifact, durable_result_dir / artifact.name)

    (cache_dir / "latest_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subject_id": subject_id,
                "quality": quality,
                "result_dir": str(durable_result_dir),
                "preview": str(durable_result_dir / "preview.png"),
                "segmentation": str(durable_result_dir / "segmentation.nii.gz"),
                "measurements": str(durable_result_dir / "measurements.json"),
                "manifest": str(durable_result_dir / "run_manifest.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSUCCESS: segmented {len(measurements)} anatomical structures for {subject_id}")
    print(f"Preview:       {durable_result_dir / 'preview.png'}")
    print(f"Segmentation:  {durable_result_dir / 'segmentation.nii.gz'}")
    print(f"Measurements:  {durable_result_dir / 'measurements.json'}")
    print(f"Run manifest:  {durable_result_dir / 'run_manifest.json'}")
    print("These are anatomical segmentations for visual QC, not diagnoses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
