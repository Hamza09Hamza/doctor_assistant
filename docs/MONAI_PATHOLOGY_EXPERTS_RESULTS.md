# MONAI pathology experts: results and methodology

Status: updated 2026-08-12. Research evaluation, not a clinical-use claim. The
lung-nodule result now covers all 27 LIDC cases that met the frozen eligibility rule;
their CT SeriesInstanceUIDs were absent from LUNA16's published 888-series
`candidates.csv` corpus.

## Why these two models

`scripts/run_monai_pathology_experts.py` replaced TotalSegmentator's `lung_nodules`
task as the project's CT pathology proof. TotalSegmentator's nodule task has no
published Dice/FROC anywhere — it's an anatomy segmenter's side-task, not a
purpose-built detector — and on this project's original LIDC test case it missed a
clear 10mm expert-annotated nodule while emitting three scattered sub-3mm specks.
Two purpose-built MONAI Model Zoo bundles replaced it, both Apache-2.0:

- `brats_mri_segmentation` — 3D SegResNet, brain MRI, outputs TC/WT/ET tumour
  subregions. Published validation Dice: TC 0.8559 / WT 0.9026 / ET 0.7905
  (avg 0.8518).
- `lung_nodule_ct_detection` — 3D RetinaNet trained on LUNA16. Outputs boxes, not
  masks. Published mAP 0.852 / mAR 0.998 on its own validation fold.

## Brain tumour: TC 0.79 / WT 0.89 / ET 0.75 (avg 0.81), full 96-case split

First attempt scored ET at exactly 0.0000 on every case — the tell that something
was structurally wrong, not just weak. Root cause: **MSD Task01_BrainTumour is
BraTS-derived but not stored in the BraTS convention.** Two independent
reconciliations were needed, both derived at runtime from the dataset's own
`dataset.json` (not hardcoded, so a dataset revision fails loudly instead of
silently):

1. **Label renumbering.** MSD numbers labels `1=edema, 2=non-enhancing core,
   3=enhancing`. The bundle (and MONAI's `ConvertToMultiChannelBasedOnBratsClasses`
   default) expects BraTS-2018's `1=non-enhancing core, 2=edema, 4=enhancing`.
2. **Input channel permutation.** The bundle expects `{0: T1c, 1: T1, 2: T2,
   3: FLAIR}`; MSD stores `{0: FLAIR, 1: T1w, 2: t1gd, 3: T2w}`. Uncorrected, FLAIR
   was fed into the contrast-enhanced slot — the one sequence enhancing tumour is
   defined by.

After the fix, full 96-case validation split:

| | ours | published | delta |
|---|---|---|---|
| TC | 0.7931 | 0.8559 | -0.0628 |
| WT | 0.8890 | 0.9026 | -0.0136 |
| ET | 0.7506 | 0.7905 | -0.0399 |
| avg | 0.8109 | 0.8518 | -0.0409 |

**Contamination status: UNVERIFIED.** The bundle was trained on BraTS 2018 using
the authors' own 200/42/43 split, not published in a form that can be intersected
with MSD Task01 case IDs. This number is evidence the pipeline is correct, not
proof of held-out performance.

## Lung nodule: 91.3% sensitivity across 27 LIDC scans absent from LUNA16's UID corpus

### Contamination: the original test case had to be replaced

The project's original LIDC test case (LIDC-IDRI-0686) is **confirmed** — not
merely suspected — to be in `lung_nodule_ct_detection`'s own training split.
Verified directly: downloaded MONAI's own published LUNA16 10-fold split archive
and found this case's exact SeriesInstanceUID in `dataset_fold0.json`'s
*training* list, with its 9 recorded nodule coordinates matching LUNA16's public
`annotations.csv` exactly (ruling out a UID collision). No score from this
checkpoint on this case can ever mean anything.

Replacement: **LIDC-IDRI-0672**, verified absent from LUNA16's complete 888-scan
corpus (checked its CT SeriesInstanceUID against every row of both
`annotations.csv` and `candidates.csv` — zero matches). LUNA16 excludes it for a
documented data-quality reason (slice thickness/spacing), not a case-quality one.

### Ground truth: derived from the four LIDC radiologists' own DICOM SEGs

LUNA16 has no entry for this case, so there's no reference standard to look up.
`scripts/lidc_seg_ground_truth.py` extracts nodule candidates directly from the
four readers' primary DICOM SEG data and clusters them into a consensus using
LUNA16's own documented rule (≥3 of 4 readers must mark overlapping
segmentations). Result: **4/4 readers independently agreed on one 5.0mm nodule.**

### Original single-case diagnostic

At score ≥ 0.3: TP=0, FP=7, FN=1 (sensitivity 0.0) — but the binary count hides
the real story. The detector's single highest-confidence detection (score 0.995,
box size 5.57mm) is **2.766mm from the true nodule's center**, against a
**2.489mm ground-truth radius** — missing LUNA16's strict hit criterion
(distance ≤ target radius) by **0.278mm**. In practical terms this is the
detector correctly and confidently locating the real nodule, not a genuine
failure to find it; it just falls fractionally outside a deliberately strict
scoring rule. The scoring threshold was not loosened to reclassify this as a hit
— that would be the same kind of after-the-fact threshold-shopping this project
refuses to do for the KAD gates. Instead `_match_detections` now records, for
every missed nodule, the nearest detection's distance and how far outside the
hit radius it landed, so a 0.28mm near-miss and a 200mm total failure are never
indistinguishable again.

**Known-overlap status:** the CT SeriesInstanceUID is absent from LUNA16's published
888-series corpus, with the verification evidence recorded inline in the manifest.
That is a meaningful leakage check, not proof of absence from every upstream/private
training source.

That first case established that the detector, DICOM geometry, and multi-reader
ground-truth extraction were wired correctly. It is retained as a useful miss-analysis,
not the headline result.

### Completed 27-case result

The follow-up batch ran every case discovered by the frozen eligibility rule: exactly
one CT series, exactly four reader DICOM SEG series, and the CT SeriesInstanceUID absent
from LUNA16's complete published `candidates.csv` corpus. No case was selected by model
score. All 27 staged and completed inference successfully at the same fixed score
threshold (`0.3`).

| Measurement | Result |
|---|---:|
| Cases evaluated | 27 |
| Failed staging / inference | 0 / 0 |
| Consensus ground-truth nodules | 23 |
| True positives | 21 |
| False positives | 58 |
| False negatives | 2 |
| Sensitivity | **91.3%** |
| False positives per scan | **2.15** |

These are aggregate detection counts at one frozen operating point, not a calibrated
probability of cancer and not a scan-clearance claim. Four evaluated scans contained no
consensus nodule; the detector still emitted candidates on them, which is why the
viewer must call its output a shortlist and preserve the false-positive warning.

The complete execution log is embedded in
`notebooks/build_ohif_bundles_only.ipynb` (batch summary at notebook lines 416-421).
The run also checkpointed a full JSON manifest to Drive. Recovering that manifest is
still required before computing patient-bootstrap confidence intervals or a proper
multi-threshold FROC curve; the headline above does not pretend those analyses exist.

For the viewer demo, `LIDC-IDRI-0117` was selected after the frozen run as a visually
clean case: one consensus nodule, TP=1, FP=0, FN=0.
`scripts/prepare_lidc_nodule_detector_demo.py` pins its 122-slice CT and four same-study
radiologist SEG objects. This one successful scan is a demonstration case only; the
27-case aggregate is the evidence result.

## A real methodological lesson: two coordinate bugs, both caught by looking, not by code review

Both bugs below passed code review, matched their respective library's real API
(verified against source before writing), and passed unit tests on first write.
Neither was actually correct. Both were only caught by rendering the CT and
looking at where the markers landed.

1. **DICOM `ImageOrientationPatient` row/col swap.** The standard's own wording is
   genuinely confusing: the first direction-cosine triplet is called "the row
   direction," but it actually means the direction you move *traversing along* a
   row — i.e. the direction of increasing **column** index. Both
   `lidc_seg_ground_truth.py` and the diagnostic plotting script had this
   backwards. Confirmed against DICOM PS3.3 C.7.6.2.1.1 directly before fixing.
   The unit tests written alongside the original code did not catch it: they used
   isotropic spacing on an identity orientation, which is swap-invariant. A
   regression test using anisotropic spacing (`tests/test_lidc_seg_ground_truth.py`)
   was added afterward and verified — analytically and empirically — that it
   would have failed against the pre-fix code.

2. **A self-cancelling fix.** After fixing bug 1 in the extraction script, the
   diagnostic plot looked *completely unchanged* — same pixel position, despite
   the ground truth's reported world coordinates genuinely changing. Cause: the
   plotting script's inverse coordinate math shared the *identical* row/col
   convention (and the identical bug) as the extraction script. Fixing the same
   mistake symmetrically in a forward and inverse transform leaves their
   composition unchanged — the diagnostic was checking a buggy extraction against
   an equally-buggy un-projection and calling them "consistent." Fixed by
   rebuilding the plotting script on SimpleITK's `TransformPhysicalPointToContin
   uousIndex`, a mature, independently-implemented library function sharing no
   code with the extraction script — so agreement between the two now means
   something. Its exact index ordering was verified against a synthetic image
   with known geometry before being trusted on real data.

The practical takeaway: a diagnostic that shares assumptions with the code it's
checking can't catch a shared mistake, and passing tests only prove what the
tests happened to exercise. Both lessons apply beyond this module.

## Viewing results in OHIF, not just as JSON

Both results can be turned into a bundle viewable in the project's real OHIF viewer,
through the existing, already-proven Orthanc-upload path
(`scripts/publish_dicom_seg_to_orthanc.py`, run locally — unmodified, no new
backend/frontend code needed at all; OHIF's own stock DICOM SEG viewport renders the
result).

- **Lung nodule** (`scripts/build_lung_nodule_seg_bundle.py`): the real LIDC-IDRI-0672
  CT, plus two independently-toggleable DICOM SEG objects rasterized from the
  manifest's already-computed detections and consensus ground truth — an "AI
  detections" SEG and a "ground truth (4-reader consensus)" SEG. No synthetic data
  needed; this is a real DICOM CT throughout.
- **Brain tumour** (`scripts/build_brain_tumor_seg_bundle.py`): MSD Task01_BrainTumour
  ships as NIfTI only, with no real DICOM series a SEG could reference — the same
  blocker documented for a different NIfTI-only demo elsewhere in this project. Solved
  with a new, reusable adapter, `scripts/nifti_to_dicom.py`, that synthesizes a
  standards-conformant DICOM MRI series from any NIfTI + affine. Every identifying
  field (`PatientID`, `PatientName`, `SeriesDescription`, `Manufacturer`) states plainly
  that the container is synthetic; the pixel data and the model's predicted TC/WT/ET
  mask are both real. Its RAS+→LPS geometry conversion is verified against an
  independent computation with known geometry before ever touching real data
  (`tests/test_nifti_to_dicom.py`), including a real, harmless finding along the way:
  GDCM-based readers (SimpleITK, OHIF/cornerstone) sort a loaded series by projected
  spatial position, not by filename or `InstanceNumber` — a written slice order can
  come back reversed in a viewer without any geometry actually being wrong, since each
  slice's own absolute position stays correct regardless.

Both builder scripts only read this workflow's already-computed results (the manifest,
or one MSD validation case) — no GPU re-inference beyond one single-case forward pass
for the brain-tumour demo. `highdicom` is required only by these two scripts and their
tests, not by `run_monai_pathology_experts.py` itself.

## Files

- `scripts/run_monai_pathology_experts.py` — both experts' full pipeline
- `scripts/lidc_seg_ground_truth.py` — multi-reader DICOM SEG consensus ground truth
- `scripts/plot_lung_nodule_detections.py` — SimpleITK-based visual diagnostic
- `scripts/nifti_to_dicom.py` — reusable NIfTI→synthetic-DICOM adapter
- `scripts/build_lung_nodule_seg_bundle.py` — real-CT AI/ground-truth SEG bundle
- `scripts/build_brain_tumor_seg_bundle.py` — synthetic-DICOM MRI + prediction SEG bundle
- `tests/test_lidc_seg_ground_truth.py` — regression tests, including the
  anisotropic-spacing row/col swap guard
- `tests/test_nifti_to_dicom.py`, `tests/test_build_brain_tumor_seg_bundle.py`,
  `tests/test_build_lung_nodule_seg_bundle.py` — regression tests for the OHIF-bundle
  path, including the sphere-rasterization volume check and the SEG source-series
  reference check
- `notebooks/monai_pathology_experts_colab.ipynb` — the Colab run, including both
  bundle-builder cells

## Next, if continuing this workstream

1. Recover the saved full 27-case manifest and calculate a multi-threshold FROC/CPM,
   patient-level bootstrap confidence intervals, and explicit analysis of the two
   misses. No model rerun is needed if the Drive manifest is intact.
2. Wire the detector shortlist into OHIF, then let a selected detector candidate seed
   MedSAM2 refinement. Arbitrary manual boxes remain prompted-structure segmentation
   and must never be presented as anomaly evidence.
3. Add hard-negative prompt buckets (normal lung, vessel/bifurcation, pleura/scar,
   rib/muscle, outside-lung) and report false-accept rate before adding any automatic
   “supported/unsupported” prompt gate.
4. Brain tumour contamination remains UNVERIFIED; resolving it would need the
   bundle authors' original BraTS 2018 split in a form intersectable with MSD
   Task01 case IDs, which does not currently exist publicly.
