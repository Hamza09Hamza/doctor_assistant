# Development and verification

Doctor Assistant contains model families with incompatible or unusually heavy
dependencies. Use separate environments instead of installing every adapter into one
runtime.

## Chest X-ray, MSK, reporting, and evaluation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-extras.txt
```

Torch should be installed from the wheel index appropriate for the machine when the
default PyPI wheel is unsuitable. See the official PyTorch installation selector for
CUDA-specific commands.

Run the deterministic regression and smoke checks:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_report.py
python scripts/smoke_system.py
```

For classifier development in Google Colab, use
`notebooks/chest_classifier_build_colab.ipynb`. It evaluates the pinned KAD-512
candidate against the adjudicated NIH development cohort, keeps only one endpoint under
decision and one query in the model at a time, and saves hashed JSON/NPZ artifacts to a
new immutable Google Drive
directory for every run. It freezes four patient-disjoint roles: model selection (30%),
calibration (20%), threshold selection (20%), and untouched acceptance (30%). Acceptance
reports study-level sensitivity and specificity from one score-blind, deterministic
SHA-256-selected positive study per positive patient and one negative study per negative
patient. It gates on two-sided 95% Wilson bounds and requires at least 22 positive and
20 negative acceptance patients; all-study metrics are diagnostic only. Its current
320-pixel JPEG mirror is decision-ineligible: calibration and candidate-threshold
outputs are diagnostic, and untouched-acceptance scores are not read at all. The
acceptance stage requires a future trusted original-NIH pixel path. The historical
`classifier_evaluation_colab.ipynb` is exploratory and must not be used as the evidence
workflow.

Role membership is endpoint-specific and frozen across all candidates with seed
`20250729` and exactly 512 score-blind, label-support hash-search attempts. The
canonical CLI rejects changes to either value. Endpoint-specific adaptation must start
from the same frozen base; shared multi-endpoint encoder training requires a separate
global role manifest and is not implemented.

Real-data evaluation downloads public datasets and pretrained weights on first use:

```bash
python scripts/eval_chest_xrv.py --n 300
python scripts/eval_msk_fracture.py --n 300
```

The chest evaluation reports both threshold-independent AUC and operational behavior at
several reporting thresholds: macro sensitivity, macro specificity, the fraction of
dataset-normal studies with any false-positive finding, and the mean number of findings
per normal study. AUC alone is not evidence that `Pipeline(thresholds=...)` is safe.

The older TorchXRayVision threshold script explores named partitions from a third-party
mirror:

```bash
python scripts/calibrate_chest_thresholds.py \
  --n-valid 2000 --n-test 2000 \
  --sensitivity-target 0.85 --specificity-floor 0.60
```

Its outputs are smoke/development diagnostics, not official NIH validation or test
evidence, because that mirror does not expose source filenames for manifest
reconciliation. The KAD workflow instead uses `prepare_nih_expert_manifest.py`,
`benchmark_kad.py`, and `calibrate_kad_phase1.py`; locked test mode remains disabled
until original NIH pixels and a fully frozen decision artifact are supplied. The
benchmark also enforces the active endpoint's canonical query-set ID and semantic
query-pack hash; a multi-query pack or a pack containing different query tensors is
rejected. This isolation is required because KAD decoder self-attention makes scores
depend on every query present. Query-pack format v3 canonicalizes the frozen text
embedding through a BF16 round trip before storing it as float32. Canonical phase-1
export reads the reviewed singleton embeddings from the checksum-pinned
`configs/chest_kad_phase1_query_features.json` asset instead of recomputing them, so
PyTorch and CPU-kernel differences cannot break the semantic identity check. Canonical
benchmarking also requires a clean worktree and records the deterministic inference
controls, GPU/CPU identity, compute capability, CUDA runtime, cuDNN runtime, and NVIDIA
driver in its evidence artifact. The canonical Colab notebook no longer locks to a single
GPU class (it previously required a T4) — it accepts whatever accelerator Colab assigns
(T4, L4, A100, ...) and relies on that per-run recorded GPU/compute-capability/CUDA/cuDNN
identity to keep evidence traceable and segregable by accelerator, rather than preventing
mixed-accelerator evidence by refusing to run on anything but one GPU class. NumPy, SciPy, scikit-learn, pandas,
Pillow, and transformers are pinned in the notebook; the decision artifact separately
hashes its calibration script and records the exact Python/NumPy/SciPy/scikit-learn
versions.

Historical mirror artifacts deliberately set `threshold_export_eligible=false`,
`thresholds_complete=false`, and `pipeline_thresholds=null`.
`load_calibrated_thresholds(...)` rejects them even when every diagnostic candidate
met its point-estimate constraints. No mirror-derived threshold may feed `Pipeline`.

Schema-1 image provenance still rejects a self-authored `original_nih_pixels=true`
declaration unconditionally. A second schema (schema-2) can assert original-pixel
evidence, but only behind a multi-source consensus check: `scripts/fetch_nih_original_images.py`
accepts a file only once its SHA-256 agrees across at least two independently-operated
local source directories (no NIH-published per-file checksum manifest exists to pin
against a single source), and hard-fails, naming the exact file, on any disagreement
or insufficient source coverage. `scripts/benchmark_kad.py`'s `load_image_provenance`
validates this schema-2 `archive_verification` block before accepting
`original_nih_pixels=true`. The script verifies and ingests already-downloaded source
copies; it does not perform the ~42GB download itself. Running it against the full NIH
release from two or more real independent sources, and using its output to unlock a
Gate 4 locked-evaluation run, remains a follow-up step.

Evaluation samples default to
`~/.cache/doctor_assistant/evaluation`. Override this with either
`--cache-dir PATH` or the `DOCTOR_ASSISTANT_CACHE_DIR` environment variable.
The NIH mirror is pinned to a specific Hugging Face dataset commit by default; preserve
that revision when comparing runs.

## CT pathology experts (MONAI)

For the current pathology proof, use
`notebooks/monai_pathology_experts_colab.ipynb` on an L4 (or `--expert brain_tumor` /
`--expert lung_nodule` directly via `scripts/run_monai_pathology_experts.py`). It
replaced the TotalSegmentator `lung_nodules` path below: that task has no published
Dice/FROC anywhere (it's an anatomy segmenter's side-task, not a purpose-built
detector) and missed a clear 10mm expert-annotated nodule on this project's original
LIDC case. Two purpose-built MONAI Model Zoo bundles (`brats_mri_segmentation`,
`lung_nodule_ct_detection`, both Apache-2.0) replaced it. See
`docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md` for full results, methodology, and two real
coordinate-mapping bugs worth reading before touching this pipeline again.

## CT and TotalSegmentator (legacy path, superseded above for pathology detection)

`notebooks/lung_nodule_segmentation_colab.ipynb` on an L4 needs no user-provided
DICOM: it downloads pinned public LIDC series `LIDC-IDRI-0686`, runs the
TotalSegmentator `lung_nodules` task, requires a non-empty nodule segment, and creates
an OHIF bundle with both the prediction and a radiologist DICOM SEG. This is a narrow
lung-nodule benchmark, not a universal abnormality detector, and `LIDC-IDRI-0686` is
separately confirmed to be inside `lung_nodule_ct_detection`'s own LUNA16 training
split (see the MONAI doc above) -- irrelevant for TotalSegmentator, which is not
LUNA16-trained, but worth knowing before reusing this UID elsewhere.

Use a separate environment for TotalSegmentator. Its nnU-Net stack can replace NumPy,
SciPy, and Torch versions required by the chest-data and reporting environment.

```bash
python3 -m venv .venv-ct
source .venv-ct/bin/activate
python -m pip install --upgrade pip
python -m pip install torch monai nibabel pydicom SimpleITK TotalSegmentator highdicom==0.27.0
```

For the first visual proof, run
`python -u scripts/run_local_totalsegmentator_ohif_demo.py`. It automatically uses a
pinned public de-identified OHIF test CT and the full 1.5 mm model with split inference.
This path completed on the RTX 3050 6 GB laptop GPU on 2026-08-04 and created a valid
DICOM SEG with 91 non-empty segments and 3,287 frames. Colab is now an optional fallback
for computers without a suitable local NVIDIA GPU.

The viewer bridge is implemented in `scripts/run_totalsegmentator_dicom_seg.py` and
`scripts/publish_dicom_seg_to_orthanc.py`, with an optional Colab notebook. It uses a
real de-identified DICOM CT,
asks TotalSegmentator 2.17 for direct `output_type="dicom_seg"` output, supports both
anatomy and named specialist tasks, validates the
source references and pixel data, uploads the CT and SEG to Orthanc, and prints the
exact Clinique Amina URL. See `docs/TOTALSEGMENTATOR_OHIF.md` for the short runbook.
The NIfTI-only Zenodo demo has no source DICOM instances to reference and therefore
cannot itself produce a standards-valid DICOM SEG.

For the interactive full-volume path (one OHIF box -> MedSAM2 propagation -> measured
DICOM SEG -> Orthanc), see [MEDSAM2_3D_OHIF.md](MEDSAM2_3D_OHIF.md). The pinned
high-quality CT, radiologist reference, and one-command Mac acceptance run are in
[LIDC_INTERACTIVE_DEMO.md](LIDC_INTERACTIVE_DEMO.md).
To run the Torch backend on a Colab GPU while keeping only OHIF/Orthanc on the Mac, use
[COLAB_INFERENCE_SERVER.md](COLAB_INFERENCE_SERVER.md).

Do not treat a successful synthetic CT wiring run as organ-segmentation validation.
TotalSegmentator must be evaluated on correctly de-identified, appropriately licensed
clinical-format CT data with preserved geometry.

## What the checks establish

- Unit tests establish software contracts such as routing safety, metadata isolation,
  report grounding, official-manifest fail-closed behavior, and checkpoint state.
- Smoke tests establish that the components connect.
- Real-data scripts measure a specific model/dataset/configuration combination.

None of these establishes clinical validity. Record model versions, dataset provenance,
patient-level splits, thresholds, and confidence intervals for every reported result.
