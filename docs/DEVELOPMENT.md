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
`notebooks/classifier_evaluation_colab.ipynb`. It keeps validation analysis,
threshold selection, and the final held-out test as separate explicit stages and saves
the resulting JSON/NPZ artifacts to Google Drive.

Real-data evaluation downloads public datasets and pretrained weights on first use:

```bash
python scripts/eval_chest_xrv.py --n 300
python scripts/eval_msk_fracture.py --n 300
```

The chest evaluation reports both threshold-independent AUC and operational behavior at
several reporting thresholds: macro sensitivity, macro specificity, the fraction of
dataset-normal studies with any false-positive finding, and the mean number of findings
per normal study. AUC alone is not evidence that `Pipeline(thresholds=...)` is safe.

Per-label threshold selection uses validation patients and evaluates the frozen result
once on separate test patients:

```bash
python scripts/calibrate_chest_thresholds.py \
  --n-valid 2000 --n-test 2000 \
  --sensitivity-target 0.85 --specificity-floor 0.60
```

The script refuses to export `pipeline_thresholds` when any label has inadequate
validation support or cannot meet the declared constraints.

Completed artifacts can be loaded without copying values manually:

```python
from evaluation import load_calibrated_thresholds

thresholds = load_calibrated_thresholds("outputs/chest_xrv_thresholds.json")
pipe = Pipeline(router, thresholds=thresholds)
```

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
  report grounding, split fallback, and checkpoint state.
- Smoke tests establish that the components connect.
- Real-data scripts measure a specific model/dataset/configuration combination.

None of these establishes clinical validity. Record model versions, dataset provenance,
patient-level splits, thresholds, and confidence intervals for every reported result.
