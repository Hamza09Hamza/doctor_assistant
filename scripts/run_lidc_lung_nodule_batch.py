#!/usr/bin/env python3
"""Run the lung-nodule detector across every clean LIDC-IDRI case with the same
structure as the project's original verified case (LIDC-IDRI-0672): one CT series,
exactly 4 reader DICOM SEG series, and confirmed absence from LUNA16's 888-scan
training corpus.

Turns the existing n=1 near-miss result into a real small-sample sensitivity number
(and enough points for a rough FROC shape). Reuses every piece of the single-case
pipeline in run_monai_pathology_experts.py and lidc_seg_ground_truth.py unchanged --
this script only adds candidate discovery and orchestration across many cases.

GPU usage note: downloads (I/O-bound, ~1 CT + 4 SEG files per case) are parallelized
with a thread pool; GPU inference is deliberately run one case at a time, AFTER all
data is staged, not interleaved with it. A single Colab GPU has nothing to gain from
running multiple full 3D sliding-window detector passes concurrently -- that's real
risk of an OOM crash for no measurable speedup, since inference is already
compute-bound once it starts. The actual lever for keeping the GPU continuously busy
is eliminating the download-wait between cases, which staging everything first does;
the detector itself is loaded once and reused across every case rather than reloaded
per case.

Usage (Colab, after run_monai_pathology_experts.py has already downloaded the
lung_nodule_ct_detection bundle at least once):

    python scripts/run_lidc_lung_nodule_batch.py \
        --data-dir /content/drive/MyDrive/doctor_assistant/monai_experts/data \
        --scratch-dir /content/monai_scratch \
        --output-dir /content/drive/MyDrive/doctor_assistant/monai_experts/results \
        --max-cases 27 --download-workers 6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_monai_pathology_experts import (  # noqa: E402
    LUNG_NODULE_BUNDLE,
    LUNG_NODULE_MIN_READER_CONSENSUS,
    LUNG_NODULE_SCORE_THRESHOLD,
    _ct_dicom_to_nifti,
    _find_dicom_files,
    _load_lung_nodule_detector,
    _match_detections,
    _run_lung_nodule_detector,
    download_bundle,
    log,
)

try:
    from scripts.lidc_seg_ground_truth import consensus_nodules, extract_reader_nodules
except ModuleNotFoundError:  # Direct execution from scripts/
    from lidc_seg_ground_truth import consensus_nodules, extract_reader_nodules  # type: ignore[no-redef]

# LUNA16's own published files (Zenodo record for the LUNA16 grand-challenge dataset),
# not a redistributed copy -- candidates.csv covers all 888 series LUNA16 actually used
# (positives and negatives), which is what "trained/evaluated on" means for contamination
# purposes; annotations.csv alone only covers series with a positive nodule.
LUNA16_CANDIDATES_URL = "https://zenodo.org/records/3723295/files/candidates.csv"


def _luna16_series_uids(cache_dir: Path) -> set[str]:
    """Every SeriesInstanceUID LUNA16 actually used (888 scans), read from LUNA16's own
    candidates.csv and cached locally so repeat runs don't re-download a ~20MB file."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = cache_dir / "luna16_candidates.csv"
    if not candidates_path.is_file():
        log(f"Downloading LUNA16's own candidates.csv ({LUNA16_CANDIDATES_URL}) ...")
        urllib.request.urlretrieve(LUNA16_CANDIDATES_URL, candidates_path)
    series = set(pd.read_csv(candidates_path)["seriesuid"].unique())
    log(f"LUNA16 training/eval corpus: {len(series)} unique series")
    return series


def filter_clean_candidates(lidc_index, luna_series: set[str]) -> list[dict]:
    """Business logic split out from discover_clean_candidates() so it can be unit
    tested against a small synthetic index, without a live idc-index/network call.

    lidc_index: a DataFrame with at least PatientID/SeriesInstanceUID/Modality columns,
    already filtered to one collection (e.g. the 'lidc_idri' rows of idc-index's index).

    Restricted to patients with exactly one CT series and exactly 4 reader SEG series
    (one comprehensive SEG per reader) -- matching the structure of the project's
    originally-verified case, LIDC-IDRI-0672. LIDC-IDRI patients with a different SEG
    count (per-nodule rather than per-patient SEGs) are excluded from this batch, not
    judged invalid -- generalizing ground-truth extraction to that shape is separate
    scope from this run.
    """
    seg = lidc_index[lidc_index["Modality"] == "SEG"]
    seg_by_patient = seg.groupby("PatientID")["SeriesInstanceUID"].apply(list)
    four_seg_patients = {pid for pid, uids in seg_by_patient.items() if len(uids) == 4}

    ct = lidc_index[lidc_index["Modality"] == "CT"]
    ct_by_patient = ct.groupby("PatientID")["SeriesInstanceUID"].apply(list)

    candidates = []
    for patient_id in sorted(four_seg_patients):
        ct_series = ct_by_patient.get(patient_id, [])
        if len(ct_series) != 1:
            continue
        ct_uid = ct_series[0]
        if ct_uid in luna_series:
            continue
        candidates.append(
            {
                "patient_id": patient_id,
                "ct_series_uid": ct_uid,
                "seg_series_uids": list(seg_by_patient[patient_id]),
            }
        )
    return candidates


def get_idc_client():
    """One shared IDCClient, built once. IDCClient.__init__ parses IDC's full index
    (a large parquet file, twice -- current + prior versions) from scratch on every
    construction; scripts/run_monai_pathology_experts.py's _idc_download() calls
    IDCClient() fresh per download, which is fine for the single-case flow (one call,
    ever) but pathological here -- constructing 6 of them concurrently in the download
    thread pool means 6 threads all redoing that same expensive parse under the GIL
    before any of them even starts the actual network transfer. download_dicom_series()
    only reads self.index (no writes to shared state) for a plain series download, so
    reusing one client instance across threads is safe."""
    from idc_index import IDCClient

    return IDCClient()


def discover_clean_candidates(cache_dir: Path, client=None) -> list[dict]:
    client = client or get_idc_client()
    lidc = client.index[client.index["collection_id"] == "lidc_idri"]
    luna_series = _luna16_series_uids(cache_dir)
    candidates = filter_clean_candidates(lidc, luna_series)
    log(f"Discovered {len(candidates)} clean candidate case(s).")
    return candidates


def _client_download(client, series_uid: str, destination: Path) -> None:
    log(f"Downloading IDC series {series_uid} ...")
    # show_progress_bar=False: tqdm's dynamic cursor-control output is not safe when
    # multiple threads render bars to the same stdout concurrently -- with several
    # download workers active at once this can visibly stall (bars fighting over
    # terminal control), even though the underlying downloads may be proceeding fine.
    # Our own log() lines above/in _stage_case are the real progress signal in batch
    # mode; the bar is purely decorative and not worth the risk here.
    client.download_dicom_series(
        series_uid, str(destination), quiet=False, show_progress_bar=False
    )


def _stage_case(candidate: dict, scratch_dir: Path, client) -> dict:
    """Download one case's CT + 4 reader SEGs. Runs inside a thread pool -- I/O-bound
    (network download via idc-index), safe to parallelize unlike GPU inference. Uses the
    shared client passed in (see get_idc_client()) rather than run_monai_pathology_
    experts._idc_download(), which would construct a new, expensive-to-build client
    per call."""
    patient_id = candidate["patient_id"]
    case_dir = scratch_dir / "lidc_batch" / patient_id
    ct_dir = case_dir / "ct"
    ct_files = _find_dicom_files(ct_dir) if ct_dir.exists() else []
    if not ct_files:
        _client_download(client, candidate["ct_series_uid"], ct_dir)
        ct_files = _find_dicom_files(ct_dir)
    if not ct_files:
        raise RuntimeError(f"{patient_id}: CT download produced no readable DICOM files")
    log(f"  {patient_id}: CT done ({len(ct_files)} files)")

    seg_paths = []
    for i, seg_uid in enumerate(candidate["seg_series_uids"]):
        seg_dir = case_dir / "seg" / f"reader_{i}"
        found = _find_dicom_files(seg_dir) if seg_dir.exists() else []
        if len(found) != 1:
            _client_download(client, seg_uid, seg_dir)
            found = _find_dicom_files(seg_dir)
        log(f"  {patient_id}: reader_{i} SEG done")
        if len(found) != 1:
            raise RuntimeError(
                f"{candidate['patient_id']}: expected exactly one SEG file for "
                f"reader_{i}, found {len(found)}"
            )
        seg_paths.append(found[0])

    return {**candidate, "ct_dir": ct_dir, "seg_paths": seg_paths}


def run_batch(args) -> dict:
    import torch

    idc_client = get_idc_client()
    cache_dir = args.data_dir / "lidc_batch_cache"
    candidates = discover_clean_candidates(cache_dir, client=idc_client)
    if args.max_cases is not None:
        candidates = candidates[: args.max_cases]
    log(f"Running batch on {len(candidates)} case(s).")

    log(f"Staging {len(candidates)} case(s) with {args.download_workers} parallel download workers ...")
    staged = []
    failed_staging = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(_stage_case, c, args.scratch_dir, idc_client): c for c in candidates}
        for future in concurrent.futures.as_completed(futures):
            candidate = futures[future]
            try:
                staged.append(future.result())
                log(f"  staged: {candidate['patient_id']}")
            except Exception as exc:  # noqa: BLE001 -- one bad case must not abort the batch
                log(f"  FAILED staging {candidate['patient_id']}: {exc}")
                failed_staging.append({"patient_id": candidate["patient_id"], "error": str(exc)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    bundle_dir = download_bundle(LUNG_NODULE_BUNDLE, args.data_dir / "bundles")
    log("Loading detector once, reused across every case ...")
    detector = _load_lung_nodule_detector(bundle_dir, device)

    per_case_results = []
    failed_inference = []
    total_tp = total_fp = total_fn = total_gt = 0
    for case in staged:
        patient_id = case["patient_id"]
        try:
            log(f"--- {patient_id} ---")
            nifti_path = args.scratch_dir / "lidc_batch" / patient_id / "ct.nii.gz"
            nifti_path, slice_spacing_mm = _ct_dicom_to_nifti(
                case["ct_dir"], case["ct_series_uid"], nifti_path
            )

            nodules_by_reader = {
                f"reader_{i}": extract_reader_nodules(seg_path, slice_spacing_mm)
                for i, seg_path in enumerate(case["seg_paths"])
            }
            ground_truth = consensus_nodules(
                nodules_by_reader, min_readers=LUNG_NODULE_MIN_READER_CONSENSUS
            )
            log(f"  ground truth: {len(ground_truth)} consensus nodule(s)")

            detections = _run_lung_nodule_detector(nifti_path, detector, device)
            match = _match_detections(detections, ground_truth, LUNG_NODULE_SCORE_THRESHOLD)
            log(
                f"  TP={match['true_positives']} FP={match['false_positives']} "
                f"FN={match['false_negatives']} sensitivity={match['sensitivity']}"
            )

            total_tp += match["true_positives"]
            total_fp += match["false_positives"]
            total_fn += match["false_negatives"]
            total_gt += match["ground_truth_count"]

            per_case_results.append(
                {
                    "patient_id": patient_id,
                    "ct_series_uid": case["ct_series_uid"],
                    "ground_truth": ground_truth,
                    "detections": detections,
                    "match_at_threshold": match,
                }
            )
        except Exception as exc:  # noqa: BLE001 -- one bad case must not abort the batch
            log(f"  FAILED inference on {patient_id}: {exc}")
            failed_inference.append({"patient_id": patient_id, "error": str(exc)})

    overall_sensitivity = total_tp / total_gt if total_gt else None
    n_cases = len(per_case_results)
    log("")
    log("=== BATCH SUMMARY ===")
    log(
        f"Cases evaluated: {n_cases} "
        f"(failed staging: {len(failed_staging)}, failed inference: {len(failed_inference)})"
    )
    log(f"Total ground-truth nodules: {total_gt}")
    log(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    log(f"Overall sensitivity: {overall_sensitivity}")
    log(f"False positives per scan: {total_fp / n_cases if n_cases else None}")

    result = {
        "expert": "lung_nodule_batch",
        "bundle": LUNG_NODULE_BUNDLE,
        "licence": "Apache-2.0",
        "score_threshold": LUNG_NODULE_SCORE_THRESHOLD,
        "cases_evaluated": n_cases,
        "cases_failed_staging": failed_staging,
        "cases_failed_inference": failed_inference,
        "aggregate": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "ground_truth_count": total_gt,
            "sensitivity": overall_sensitivity,
            "false_positives_per_scan": total_fp / n_cases if n_cases else None,
        },
        "per_case": per_case_results,
        "contamination_note": (
            "Every case's CT SeriesInstanceUID was checked against LUNA16's own "
            "published candidates.csv (888 unique series, the complete corpus this "
            "detector was trained/evaluated on) with zero matches, via idc-index's live "
            "collection index -- not a hardcoded list. Candidates were additionally "
            "restricted to patients with exactly 4 reader SEG series (one comprehensive "
            "SEG per reader), matching the structure of the project's originally-verified "
            "case (LIDC-IDRI-0672); LIDC-IDRI patients with a different SEG count "
            "(per-nodule rather than per-patient SEGs) were excluded from this batch, not "
            "judged invalid."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "lidc_lung_nodule_batch_manifest.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log(f"Manifest written: {out_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content/monai_scratch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--download-workers", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
