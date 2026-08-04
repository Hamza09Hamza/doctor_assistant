"""Standalone TotalSegmentator demo — run this in Colab, not locally.

TotalSegmentator (Apache 2.0, github.com/wasserth/TotalSegmentator) needs a real,
full-resolution CT volume to be worth anything -- toy/downsampled samples don't exercise
the model meaningfully. This script downloads one case from the official small-subset
release of the TotalSegmentator training data (Zenodo record 10047263, "Small subset of
TotalSegmentator Dataset", ~3.2GB, 102 subjects, v2.0.1 -- see
https://zenodo.org/records/10047263), extracts a single subject's CT volume, runs
TotalSegmentator on it, and prints the segmented structures with their measured volumes
(using this repo's own experts/ct_totalsegmentator.py helper, so the output matches
exactly what the real pipeline would produce).

This deliberately stops at "prove segmentation works and show real numbers" -- it does
NOT attempt DICOM SEG conversion or OHIF wiring yet. That's real, separate follow-up work
(the Zenodo dataset is NIfTI, not DICOM, so there's no SOPInstanceUID series to attach a
DICOM SEG to; a NIfTI->DICOM synthesis step is needed first) -- scoped as a distinct next
script rather than crammed in here half-tested.

Usage (Colab, GPU runtime -- an L4 has plenty of VRAM for the full "big" model, no need
for TotalSegmentator's lower-quality "fast" mode):

    !git clone -b classifier-evaluation-colab https://github.com/Hamza09Hamza/doctor_assistant.git
    %cd doctor_assistant
    !pip install -q totalsegmentator
    from google.colab import drive
    drive.mount('/content/drive')
    !python scripts/demo_totalsegmentator.py --drive-cache /content/drive/MyDrive/doctor_assistant/totalsegmentator_demo
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Official small-subset release (v2.0.1, 102 subjects, ~3.2GB) -- confirmed via Zenodo's
# own record page and API (zenodo.org/api/records/10047263) at the time this was written.
ZENODO_RECORD_ID = "10047263"
ZENODO_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def _download_with_retries(url: str, destination: Path, *, attempts: int = 5) -> None:
    """Zenodo has been flaky (observed 504s); resumable, retried download.

    Prints progress every ~200MB -- a multi-GB download with zero intermediate output
    is indistinguishable from a hang when run non-interactively (e.g. piped through a
    notebook cell's subprocess call); this is what actually happened on the first real
    run, 10+ minutes of silence that was really just this loop working.
    """
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
                written = existing
                last_reported_mb = written // (200 * 1024 * 1024)
                with destination.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                        current_mb = written // (200 * 1024 * 1024)
                        if current_mb != last_reported_mb:
                            last_reported_mb = current_mb
                            pct = f" ({100 * written / total_bytes:.0f}%)" if total_bytes else ""
                            print(f"  ... {written / 1024 / 1024:.0f}MB downloaded{pct}")
            return
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, this is a demo script
            print(f"Download attempt {attempt}/{attempts} failed: {exc}")
            if attempt == attempts:
                raise
    raise RuntimeError("unreachable")


def _resolve_zip_url() -> tuple[str, str]:
    """Ask Zenodo's API for the exact filename + download URL (don't hardcode either)."""
    import requests

    response = requests.get(ZENODO_RECORD_URL, timeout=30)
    response.raise_for_status()
    files = response.json().get("files", [])
    if not files:
        raise RuntimeError(f"Zenodo record {ZENODO_RECORD_ID} reports no files")
    entry = files[0]
    return entry["key"], entry["links"]["self"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-cache",
        type=Path,
        default=None,
        help="durable directory (e.g. a Drive-mounted path) to cache the downloaded zip "
        "and extracted case across sessions; defaults to /content if unset",
    )
    parser.add_argument(
        "--subject-index",
        type=int,
        default=0,
        help="which subject subdirectory in the archive to segment (0-based)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use TotalSegmentator's lower-resolution 'fast' mode; omit on an L4 -- there's "
        "enough VRAM for the full-quality 'big' model, which is what you actually want to "
        "evaluate",
    )
    args = parser.parse_args()

    cache_dir = args.drive_cache or Path("/content/totalsegmentator_demo")
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "totalsegmentator_small_subset.zip"
    extract_dir = cache_dir / "extracted"

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        if not zip_path.exists():
            print("Resolving download URL from Zenodo's API ...")
            filename, download_url = _resolve_zip_url()
            print(f"Downloading {filename} from {download_url} ...")
            _download_with_retries(download_url, zip_path)
        else:
            print(f"Reusing already-downloaded zip at {zip_path}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            print(f"Extracting {len(members)} files to {extract_dir} ...")
            for index, member in enumerate(members, start=1):
                archive.extract(member, extract_dir)
                if index % 500 == 0 or index == len(members):
                    print(f"  ... {index}/{len(members)} files extracted")
    else:
        print(f"Reusing already-extracted data at {extract_dir}")

    subject_dirs = sorted(p for p in extract_dir.rglob("*") if p.is_dir() and (p / "ct.nii.gz").exists())
    if not subject_dirs:
        print(f"No subject directories with ct.nii.gz found under {extract_dir}", file=sys.stderr)
        return 1
    if args.subject_index >= len(subject_dirs):
        print(
            f"--subject-index {args.subject_index} out of range (found {len(subject_dirs)} subjects)",
            file=sys.stderr,
        )
        return 1

    subject_dir = subject_dirs[args.subject_index]
    ct_path = subject_dir / "ct.nii.gz"
    print(f"Segmenting subject: {subject_dir.name} ({ct_path})")

    from totalsegmentator.python_api import totalsegmentator

    output_dir = cache_dir / "segmentation_output" / subject_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running TotalSegmentator (fast={args.fast}) -- this downloads model weights on first run ...")
    result = totalsegmentator(
        input=ct_path,
        output=output_dir,
        ml=True,  # single multilabel NIfTI, matches ct_totalsegmentator.py's expected input
        fast=args.fast,
        device="gpu",
    )

    import nibabel as nib
    import numpy as np

    from experts.ct_totalsegmentator import findings_from_label_counts
    from totalsegmentator.map_to_binary import class_map

    seg_img = result if hasattr(result, "get_fdata") else nib.load(output_dir / "segmentation.nii.gz")
    seg_data = np.asarray(seg_img.get_fdata()).astype("int32")
    spacing = seg_img.header.get_zooms()[:3]

    labels, counts = np.unique(seg_data, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(labels, counts) if label != 0}
    structure_names = class_map.get("total", {})

    findings = findings_from_label_counts(
        count_map, structure_names, spacing, min_volume_ml=1.0
    )
    findings.sort(key=lambda f: f.volume_ml or 0, reverse=True)

    print(f"\n=== {len(findings)} segmented structure(s), subject {subject_dir.name} ===")
    for finding in findings:
        print(f"- {finding.label}: {finding.volume_ml:.1f} mL")

    print(f"\nSaved segmentation to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
