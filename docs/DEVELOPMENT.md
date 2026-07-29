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
depend on every query present. Canonical
benchmarking also requires a clean worktree and records the deterministic inference
controls, GPU/CPU identity, compute capability, CUDA runtime, cuDNN runtime, and NVIDIA
driver in its evidence artifact. The canonical Colab notebook requires a T4 rather than
silently mixing T4, L4, and A100 evidence. NumPy, SciPy, scikit-learn, pandas,
Pillow, and transformers are pinned in the notebook; the decision artifact separately
hashes its calibration script and records the exact Python/NumPy/SciPy/scikit-learn
versions.

Historical mirror artifacts deliberately set `threshold_export_eligible=false`,
`thresholds_complete=false`, and `pipeline_thresholds=null`.
`load_calibrated_thresholds(...)` rejects them even when every diagnostic candidate
met its point-estimate constraints. No mirror-derived threshold may feed `Pipeline`.

The current image-provenance schema also rejects a self-authored
`original_nih_pixels=true` declaration. A trusted original-NIH ingestion receipt that
verifies archive/source identity, exact selected bytes, and the canonical manifest
must be implemented before the original-pixel acceptance and locked-test paths can be
enabled.

Evaluation samples default to
`~/.cache/doctor_assistant/evaluation`. Override this with either
`--cache-dir PATH` or the `DOCTOR_ASSISTANT_CACHE_DIR` environment variable.
The NIH mirror is pinned to a specific Hugging Face dataset commit by default; preserve
that revision when comparing runs.

## CT and TotalSegmentator

Use a separate environment for TotalSegmentator. Its nnU-Net stack can replace NumPy,
SciPy, and Torch versions required by the chest-data and reporting environment.

```bash
python3 -m venv .venv-ct
source .venv-ct/bin/activate
python -m pip install --upgrade pip
python -m pip install torch monai nibabel pydicom SimpleITK TotalSegmentator
```

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
