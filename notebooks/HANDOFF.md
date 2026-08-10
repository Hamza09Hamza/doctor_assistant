# Handoff: MONAI pathology experts — OHIF viewer + lung-nodule batch eval

Repo: `doctor_assistant`, branch `classifier-evaluation-colab`, remote
`git@github.com:Hamza09Hamza/doctor_assistant.git`. This handoff combines the prior
session's summary (everything up through commit `2d9fc01`) with everything done in
this session (`f113ca2` through `d69ba2e`, 20 commits). Paste this whole file into a
fresh conversation to resume with full context.

## 1. The big picture / why any of this exists

`doctor_assistant` is a non-clinical, experimental imaging pipeline (explicitly not a
diagnostic tool). The user wants concrete, portfolio-credible "wins": real pipelines
run on real data with rigorously-verified evidence, not hand-waved results. This
project's established culture (baked in from many prior sessions, not just this one):
no threshold-shopping, no silently discarding inconvenient results, always verify a
diagnostic independently before trusting it, always test with known-geometry synthetic
fixtures before touching real data.

Two MONAI Model Zoo pathology experts are the current focus:
- `brats_mri_segmentation` — 3D SegResNet, brain MRI, outputs TC/WT/ET tumour
  subregions. Published Dice: TC 0.8559 / WT 0.9026 / ET 0.7905.
- `lung_nodule_ct_detection` — 3D RetinaNet trained on LUNA16. Outputs boxes.

Full methodology and prior results are documented in
`docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md` (already committed, still accurate as of
this handoff, though the lung-nodule section will be superseded once the 27-case batch
finishes — see Section 6).

## 2. State as of the last handoff (commit `2d9fc01` and earlier)

- Brain tumour: full 96-case MSD validation, Dice TC=0.7931/WT=0.8890/ET=0.7506/avg
  0.8109 vs published 0.8518. Contamination status UNVERIFIED (bundle's original BraTS
  split isn't public in intersectable form).
- Lung nodule: original test case LIDC-IDRI-0686 confirmed CONTAMINATED (in LUNA16's
  own training split). Replaced with LIDC-IDRI-0672, confirmed CLEAN (checked against
  all 888 LUNA16 scans). Ground truth: 4/4 LIDC readers agree on one 5.0mm nodule.
  Detector's top detection (score 0.995) landed 2.766mm from that nodule's center —
  0.278mm outside LUNA16's strict hit radius (2.489mm). A genuine near-miss, not a
  clean hit, not a clean failure. **This was n=1** — one case, not a benchmark.
- Two real coordinate bugs found and fixed by visual inspection, not code review: a
  DICOM `ImageOrientationPatient` row/col swap (`lidc_seg_ground_truth.py`,
  `plot_lung_nodule_detections.py`), and a self-cancelling bug where the diagnostic
  plot shared the same wrong assumption as the code it was checking.
- Built the NIfTI→DICOM adapter (`scripts/nifti_to_dicom.py`) and two bundle builders
  (`scripts/build_brain_tumor_seg_bundle.py`, `scripts/build_lung_nodule_seg_bundle.py`)
  so both experts' results could be viewed in the project's real OHIF viewer (via the
  existing `scripts/publish_dicom_seg_to_orthanc.py` → Orthanc pipeline), not just read
  as JSON. Pushed as `2d9fc01`. **Not yet run against real Colab data** at that point.

## 3. This session: getting the OHIF bundles actually working

### 3.1 Throwaway bundle-builder notebook
Built `notebooks/build_ohif_bundles_only.ipynb` — standalone notebook that clones the
repo, restages just the small lung-nodule CT, and runs both bundle-builder scripts
against an already-saved manifest, without rerunning the full pathology pipeline.
Made GPU-optional (`31586a4`) since only one single forward pass needs a GPU at all,
and it's fast even on CPU.

### 3.2 Real bugs found by actually looking at the rendered result
The user ran the bundles and reported real problems, in order:

1. **Lung-nodule bundle**: worked correctly on the real CT — ground truth and AI
   detection overlays visible, positioned correctly. This was the first genuinely
   working OHIF result of the whole project.
2. **Brain-tumour bundle**: `publish_dicom_seg_to_orthanc.py` crashed with
   `no readable CT DICOM series was found in the input` — `discover_ct_series()` in
   `scripts/run_totalsegmentator_dicom_seg.py` hardcoded `Modality != "CT"` and
   rejected the brain-tumour bundle's `MR` series outright. Fixed by adding an
   optional `modalities` parameter (default `{"CT"}`, unchanged for the two other
   existing CT-only callers); the publish script now passes `{"CT", "MR"}`. (`a84e6c8`
   in the same commit as the next fix.)
3. **MRI rendered as a blank/black viewport**. Root cause: `nifti_to_dicom.py`
   hardcoded `WindowCenter=2048`/`WindowWidth=4095` in raw **stored-pixel** units
   (0–4095), but per DICOM these tags apply *after* the Modality LUT
   (`RescaleSlope`/`RescaleIntercept`) — i.e. to **real-world values**. Real MSD FLAIR
   data (z-scored) sits around -3..+5, nowhere near the assumed window, so it clipped
   to black. Fixed to default from the real data's own min/max. (`a84e6c8`)
4. **MRI rendered but washed-out/low-contrast**. A few outlier voxels stretched
   literal min/max far past where real tissue actually sits. Fixed to use 1st/99th
   percentile instead of literal min/max as the default window. (`38132a6`)
5. **Background rendered gray, not black**. Real bug, self-inflicted in the immediately
   prior fix: MONAI's `NormalizeIntensityd(nonzero=True)` leaves background at a
   near-constant value (~-0.000365, not exactly 0.0 due to float rounding), and that
   value happened to land *inside* the percentile window instead of below it. Fixed:
   detect the modal (most frequent) value as background, exclude it from the
   percentile calculation, and force it below the window floor with a small margin.
   **First attempt at this fix had the comparison backwards** (`>= p_lo - margin`
   instead of `>= p_lo`, and lowered instead of raised the window edge) — caught by
   writing and running a synthetic verification script before pushing, not by
   guessing. Corrected in the same session. (`496d951`)
6. **No ground-truth overlay for brain tumour** — only the AI prediction existed,
   unlike the lung-nodule bundle which has both. Fixed: `case["label"]` already goes
   through the identical `ConvertToMultiChannelBasedOnBratsClassesd` transform as the
   prediction (same TC/WT/ET channel format), so a second SEG (`algorithm_type=
   "MANUAL"`, no `AlgorithmIdentificationSequence` needed) was added at zero extra
   inference cost. `_build_prediction_seg` renamed to `_build_seg`, parametrized by
   `series_description`/`series_number`/`algorithm_type`. (`496d951`)
7. **User reported the MRI still "not clear enough" / "low quality" after all of the
   above.** Real root cause, found by actually reading the code rather than tuning
   windowing further: the DICOM background image was built from `case["image"]`
   **after** `NormalizeIntensityd` — i.e. the model's own z-scored *input tensor*, not
   real scan intensities. Every windowing fix up to this point was polishing the wrong
   data. Fixed with `CopyItemsd(keys="image", times=1, names="image_raw")` inserted
   right before `NormalizeIntensityd` in the transform pipeline — `image_raw` (real,
   skull-stripped MSD intensities) is now what gets displayed; `image` (normalized)
   still goes to the model. No new dataset needed — MSD's own real data was sitting
   right there unused for display purposes. (`53920a4`)

### 3.3 Architecture question: should OHIF view NIfTI natively instead?
User asked whether the viewer could just accept NIfTI directly instead of the
DICOM-wrapping approach. Answered and closed: **no** — OHIF's SEG rendering, series
browser, and viewport pipeline are all built around DICOM's UID model
(`StudyInstanceUID`/`SeriesInstanceUID`/`FrameOfReferenceUID`); native NIfTI support
would need a custom `DataSource` and a custom segmentation-overlay renderer, real new
engineering, not a config toggle. The NIfTI→DICOM conversion itself was verified exact
by construction (same source list feeds both the image and the SEG) — every bug found
was a *display default* bug, not something inherent to the wrapping approach.

## 4. This session: turning the lung-nodule n=1 result into a real sample

### 4.1 Decision
User pushed for more "wins." Agreed next step: stage more clean LIDC cases (same
proven methodology as case 0672) instead of just 2-3 by hand — queried IDC's own index
live and found the real numbers, not estimates:
- LIDC-IDRI has 1018 total CT series.
- 164 patients have exactly 4 SEG series (same structural shape as case 0672 — one
  comprehensive SEG per reader).
- Of those, **27** have exactly one CT series and that CT's `SeriesInstanceUID`
  confirmed absent from LUNA16's own published `candidates.csv` (888 unique series,
  fetched live from `https://zenodo.org/records/3723295/files/candidates.csv`, cached
  locally).

User's explicit direction: use all 27, and make the download side use available
compute efficiently (not literally "overclock" the GPU — clarified that GPU inference
can't usefully run concurrently on one Colab GPU without real OOM risk; the actual
lever is not leaving the GPU idle waiting on sequential downloads).

### 4.2 `scripts/run_lidc_lung_nodule_batch.py` (new)
Reuses every piece of the existing single-case pipeline
(`run_monai_pathology_experts.py`, `lidc_seg_ground_truth.py`) unchanged. Adds:
- `discover_clean_candidates()` / `filter_clean_candidates()` (business logic split
  out specifically for a synthetic-data unit test, no live network needed for tests) —
  the exact 27-candidate discovery logic above.
- `_stage_case()` — downloads one case's CT + 4 reader SEGs.
- `run_batch()` — orchestrates staging then sequential GPU inference, one shared
  detector instance reused across all cases, aggregates TP/FP/FN/sensitivity, writes
  `lidc_lung_nodule_batch_manifest.json`.

Companion test file `tests/test_run_lidc_lung_nodule_batch.py` (8 tests, all passing).

### 4.3 The debugging cycle (each fix pushed and verified before moving on — this is
### the part worth reading carefully before touching this script again)

1. **`894c4b8`** — initial version, `--download-workers 6` (parallel).
2. **Stalled live** — zero completed downloads for 5+ minutes. Root cause found:
   `_idc_download()` (the shared helper in `run_monai_pathology_experts.py`)
   constructs a brand-new `IDCClient()` per call, and `IDCClient.__init__` parses IDC's
   full index (a large parquet file, twice) from scratch every time — 6 threads doing
   that simultaneously under the GIL before any of them even starts the real network
   transfer. Fixed: `get_idc_client()` builds one shared client, threaded through
   `discover_clean_candidates()` and `_stage_case()`. Verified live: one full case
   staged in 34s with the shared client. (`8d88cde`)
3. **Stalled again, faster this time but same symptom.** Second suspect: `tqdm`
   progress bars are not safe when multiple threads render them to the same stdout
   concurrently — can visibly stall even though downloads are proceeding underneath.
   Fixed: `show_progress_bar=False`. Also added per-file (not just per-case) log lines
   for visibility while waiting. (`a9216b8`)
4. **User, reasonably fed up with two failed guesses**: stopped debugging concurrency
   live and defaulted `--download-workers` to **1** (sequential) — the one approach
   proven reliable every time this session. `> 1` remains available as an opt-in, not
   the default, with the docstring explicitly saying why. (`556bb0a`)
5. **User reported 1400+ lines of pure noise** (`s5cmd` echoing a `cp s3://...` line
   per individual file). Fixed: `quiet=True` on `download_dicom_series()` — a whole
   case now logs ~25 concise lines instead of hundreds, verified live. (`b5403b5`)
6. **User got a stray `KeyboardInterrupt`** mid-run while it was actually working fine
   (staged 11 cases in under a minute). Diagnosed as a Colab UI gotcha: Ctrl+C on a
   focused code cell can be intercepted as "interrupt kernel" rather than "copy text" —
   very plausibly triggered while the user was trying to copy output to paste here.
   Recommended `| tee <logfile>` to Drive instead of ever needing to copy from the
   terminal, and mouse-drag-select instead of Ctrl+C if they do need to copy something.
7. **A real Jupyter kernel crash** ("kernel died and is being automatically
   restarted") happened mid-run, but only *after* 5 real cases had already finished
   with real results (TP/FP/FN/sensitivity). None of that was recoverable — the script
   only ever wrote a result after the *entire* 27-case batch finished. Fixed with a
   proper checkpoint mechanism: `save_partial_result()`/`load_partial_result()`
   (module-level, not closures, specifically so they have their own regression test)
   write/read `output_dir/lidc_lung_nodule_batch_partial/<patient_id>.json` the moment
   each case's result is computed, atomically (temp file + rename). A rerun skips
   already-computed cases (logs `(resumed from checkpoint)`) instead of redoing GPU
   inference. (`46e345d`)
8. **Also fixed during this cycle**: the resume-check for *downloads* only checked
   "folder is non-empty," which would silently treat a network-interrupted partial CT
   download as complete. Fixed by comparing against IDC's own recorded `instanceCount`
   for that series (a real column in the index; verified live against
   `LIDC-IDRI-0079` → `instanceCount=113`, matches exactly what was actually
   downloaded). `filter_clean_candidates()` now carries `ct_instance_count` through per
   candidate. (`fac36f5`)
9. **Real, well-reasoned hypothesis for *why* the kernel crashed** (not just recovered
   from): the detector's sliding-window inferer explicitly runs on **host RAM, not
   GPU** — `device="cpu"` in `_load_lung_nodule_detector()`
   (`run_monai_pathology_experts.py:672-674`), with `roi_size=(512, 512, 192)` windows.
   `nvidia-smi` looking idle during real inference is expected and doesn't mean nothing
   is happening — the real memory pressure is host RAM, invisible there. A Jupyter
   "kernel died" message (as opposed to a caught Python exception, which the existing
   `except Exception:` already handles and logs as `FAILED inference`) is the signature
   of an OS-level OOM kill, which bypasses Python's exception handling entirely.
   Hardened with `gc.collect()` + `torch.cuda.empty_cache()` in a `finally` block after
   every case, success or failure. Not a guaranteed fix (no profiler was attached to
   the actual Colab runtime to confirm), but a real, justified mitigation given the
   evidence. (`d69ba2e`)

## 5. Current live status (as of this handoff)

The batch is running in the user's Colab notebook right now, post-`d69ba2e`. Last
confirmed progress (7 of 27 cases):

| Case | Ground truth | TP | FP | FN | sensitivity |
|---|---|---|---|---|---|
| LIDC-IDRI-0079 | 1 nodule | 1 | 1 | 0 | 1.0 |
| LIDC-IDRI-0104 | 1 nodule | 1 | 7 | 0 | 1.0 |
| LIDC-IDRI-0115 | 1 nodule | 1 | 3 | 0 | 1.0 |
| LIDC-IDRI-0117 | 1 nodule | 1 | 0 | 0 | 1.0 |
| LIDC-IDRI-0146 | 1 nodule | 0 | 5 | 1 | 0.0 (missed — also the case with a SimpleITK "Non uniform sampling" slice-spacing warning; possible correlation, not proven) |
| LIDC-IDRI-0150 | 0 nodules (true negative) | 0 | 1 | 0 | None |
| LIDC-IDRI-0156 | 1 nodule | 1 | 1 | 0 | 1.0 |
| LIDC-IDRI-0296 | (ground truth extracted, inference in progress at last check) |

Running interim sensitivity: 5/6 real nodules found (~83%), with false positives
averaging roughly 2-3 per scan across cases processed so far. Assessed to the user as
a genuinely credible, moderately strong result — not a "wow" number, not broken either.
FP/scan this high is an expected trade-off for high sensitivity at a permissive score
threshold (0.3), and is standard to report *alongside* sensitivity in this field
(FROC-style), not a red flag on its own.

## 6. Pending / next steps

1. **Let the batch finish.** 20 of 27 cases remain as of this handoff. User is running
   with `| tee .../batch_log.txt` so output is saved to Drive, not dependent on
   copy-pasting terminal text (see 4.3.6). If it crashes again, checkpointing means a
   rerun of the exact same command picks up where it left off — no lost work, and no
   need to redo already-finished cases' GPU inference.
2. **Once it completes**: read the real `lidc_lung_nodule_batch_manifest.json` (not
   the running/partial numbers above) and report the actual final aggregate
   sensitivity, false-positives-per-scan, and per-case near-miss detail for the
   FN=1 case (0146) the same way the original single-case near-miss was quantified
   (`_match_detections`'s `nearest_miss_detail` field already does this automatically).
3. **Update `docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md`** once the batch is done — the
   current "Lung nodule" section documents n=1; it should be rewritten with the real
   n=27 (or however many succeeded) result, replacing the single-case framing.
   Contamination methodology section should be updated to describe the batch's live
   IDC-index-based verification (not a hardcoded list) rather than only the original
   single-case check.
4. **If the kernel crashes again after this session's fixes**: that's real new
   information worth investigating further (e.g., whether it's specifically triggered
   by large/irregular-geometry cases like 0146, or a genuine slow memory leak
   regardless of case). Ask which case number it happened on.
5. **Brain-tumour contamination remains UNVERIFIED** (unchanged from before this
   session) — the bundle's original BraTS 2018 split isn't published in a form
   intersectable with MSD Task01 case IDs. Not blocking, just a known caveat already
   documented.
6. **Not done, not currently planned**: a `viewer/ohif/` real-integration workstream
   (findings panel, `api/` backend endpoints, CORS, etc.) exists as a *separate*,
   already-partially-built product track referenced in a saved plan file at
   `~/.claude/plans/shimmering-squishing-comet.md` (UI/UX redesign of the OHIF
   findings panel + WorkList). That plan is unrelated to this session's work except
   that the same Orthanc+OHIF stack (`Clinique Amina` viewer, published to
   `http://localhost:3000/doctor-assistant/orthancProxy?...`) is what all the bundles
   in this session were opened in. Not touched this session; mentioned here only so a
   future session doesn't confuse the two tracks.

## 7. Key files touched this session

- `notebooks/build_ohif_bundles_only.ipynb` (new)
- `notebooks/HANDOFF.md` (this file, new)
- `notebooks/report.txt` (untracked, user's own scratch paste of terminal output —
  not authoritative/live, several stale snapshots were read from it during debugging)
- `scripts/nifti_to_dicom.py` (windowing fixes)
- `scripts/build_brain_tumor_seg_bundle.py` (ground-truth SEG added, raw-intensity
  display fix, `_build_prediction_seg` → `_build_seg` rename)
- `scripts/run_totalsegmentator_dicom_seg.py` (`discover_ct_series` modality param)
- `scripts/publish_dicom_seg_to_orthanc.py` (passes `{"CT", "MR"}`)
- `scripts/run_lidc_lung_nodule_batch.py` (new, the batch runner — see 4.2/4.3)
- `tests/test_nifti_to_dicom.py`, `tests/test_build_brain_tumor_seg_bundle.py`,
  `tests/test_build_lung_nodule_seg_bundle.py`, `tests/test_run_lidc_lung_nodule_batch.py`
  (updated/new, all passing — run via `.venv/bin/python -m unittest discover -s tests`)
- Explicitly NOT touched (another agent's uncommitted local changes, left alone per
  established convention): `.gitignore`, `tests/test_totalsegmentator_dicom_seg.py`

## 8. Environment notes for whoever picks this up

- Local repo has a working `.venv` with `idc-index`, `pydicom`, `highdicom`,
  `SimpleITK`, `pandas` etc. already installed — use `.venv/bin/python`, not bare
  `python3` (system Python is externally-managed/Debian, `pip install` fails without
  `--break-system-packages`).
- The user runs the actual heavy Colab work in a **VS Code Jupyter extension**
  connected to a Colab/remote runtime (not the Colab browser UI directly) — this is
  why kernel crashes show VS Code-specific messaging ("kernel died and is being
  automatically restarted").
- User's GPU in Colab: Tesla T4. Their own local machine has an RTX 3050 6GB, which
  successfully ran earlier full-pipeline work this project (mentioned in the prior
  session, relevant if ever advising on local-vs-Colab tradeoffs again).
- Orthanc + OHIF (`Clinique Amina` viewer) are running locally on the user's machine,
  reachable at `http://localhost:3000` / Orthanc `http://localhost:8042` — publish
  bundles with `.venv/bin/python scripts/publish_dicom_seg_to_orthanc.py --bundle <zip>`.
