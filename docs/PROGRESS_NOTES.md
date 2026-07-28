# Progress Notes — Chest X-ray Tuning + MSK Fracture Expert

This document is a full, literal record of one working session: what was changed, why,
how each change was verified (or wasn't), and exactly where things are blocked. It exists
so this work can be picked up from a different machine/network/session without losing any
context — every number quoted here was actually measured, not estimated, unless explicitly
marked as unverified.

## Session summary

Started from: the chest X-ray reader (pretrained TorchXRayVision) was working but not
verified to be configured optimally, and the custom-trained chest X-ray pipeline had two
long-standing weaknesses (label imbalance during training, non-stratified validation
split). Also started building out a second body-part expert (wrist fracture detection)
following the same "pretrained-first" pattern.

Three chest X-ray changes were made and evaluated; one was reverted after real-data
testing proved it wrong. Two are implemented and mechanism-verified but not yet proven on
a full training run. One new expert module (MSK fracture) is built and wired into the
pipeline, but its real-image evaluation is currently blocked by a data/network issue
(details below — this is the part most worth revisiting from another network).

---

## 1. Chest X-ray weight-set ensemble — tried, measured, reverted

**File:** `experts/torchxrayvision.py`

**Motivation:** TorchXRayVision (the pretrained model actually running for chest X-ray)
ships several DenseNet-121 checkpoints, each trained on a different combination of public
datasets (`all`, `nih`, `chex`, `mimic_ch`, `mimic_nb`, `pc`, `rsna`). The literature
review that kicked off this work suggested averaging a few of these — the same trick
CheXpert-competition winners used with larger ensembles — as a "free" AUC win with no
training required.

**What was built:** `TorchXRayVisionExpert.weights` now accepts either a single string
(old behavior) or a sequence of them. When given multiple, it loads each DenseNet, runs
each independently, applies TorchXRayVision's own per-pathology operating-point
calibration (`op_norm`) to each, and averages the calibrated scores per pathology —
excluding, per model, any pathology that model never calibrated (rather than counting it
as a zero), so a narrower checkpoint doesn't silently drag down a label it never learned.

**How it was verified — real data, not assumption:** Built `scripts/eval_chest_xrv.py`,
which streams real labeled NIH ChestX-ray14 test images from the
`BahaaEldin0/NIH-Chest-Xray-14` HuggingFace dataset (the same real-image source already
used elsewhere in this project), runs every weight-set combination through the actual
`TorchXRayVisionExpert`, and scores per-label + macro AUC against real ground truth.

First pass used n=40 images and looked like a win for the ensemble (macro AUC 0.7727 vs.
0.7112 single). That number was noise — n=40 is too small (several labels had only 1
positive example). Re-ran at n=300 (226 total positive label instances) and the result
**reversed**:

| Config | Macro AUC (n=300) |
|---|---|
| **all (alone)** | **0.7582** ← best |
| all+nih | 0.7558 (statistically a wash, −0.0024) |
| all+nih+chex | 0.7327 |
| nih (alone) | 0.7325 |
| all+chex | 0.7296 |
| chex (alone) | 0.5960 ← degenerate |

Why: `all` is trained on the *union* including `nih` and `chex`'s own training data, so
adding those checkpoints back in isn't an independent second opinion — just dilution.
`chex` alone is actively bad because CheXpert's own label ontology doesn't include several
NIH pathologies (Fibrosis, Infiltration, Mass, Nodule, Pleural_Thickening all came back at
exactly 0.5000 AUC — a flat, uninformative score, because those labels are simply absent
from `chex`'s output and defaulted to zero in every image).

**Decision:** Reverted the default to plain `weights="densenet121-res224-all"`. The
ensembling *code* stays (it's correct and could matter for genuinely complementary
checkpoints later), but nothing ships assuming an ensemble helps. The module docstring
documents this real result so it isn't re-attempted blind.

**Lesson recorded in the code:** a small sample size can flip the sign of a real-data
comparison entirely — always re-run at a larger n before trusting a directional result.

---

## 2. Asymmetric Loss for multilabel training — built, mechanism-verified, NOT training-verified

**File:** `training/losses.py` (new `AsymmetricLoss` class, wired into `MultiTaskLoss`)

**Motivation:** the project's own custom-trained chest X-ray classifier (separate from the
pretrained TorchXRayVision reader that's actually deployed) tops out around 70% — worse on
14 simultaneous labels than on a smaller label set the same approach handled well
elsewhere. NIH ChestX-ray14's labels are heavily imbalanced per class (a common label like
Infiltration is ~18% positive; a rare one like Hernia is ~0.2%). Plain BCE loss spends
almost all its gradient on the abundant easy negatives of every label, starving the rare
positives of signal. Asymmetric Loss (Ben-Baruch et al., 2021) is a published fix:
down-weight easy negatives more aggressively than positives via separate
`gamma_neg`/`gamma_pos` focal exponents, plus a `clip` term that fully zeroes a
confidently-correct negative's contribution.

**What was built:** `AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)`, used
automatically in place of `BCEWithLogitsLoss` whenever `MultiTaskLoss(multilabel=True)` is
constructed — the exact call already used for chest X-ray training
(`training.losses.MultiTaskLoss`, invoked from `notebooks/chest_xray_training.ipynb`).
The confidence head (a separate, non-multilabel binary target — "was the classifier
correct") deliberately keeps plain BCE; ASL only replaces the classification-head loss.

**How it was verified (mechanism, not outcome):** full verification would require a
complete training run over the ~112k-image NIH dataset on a GPU (a Colab-scale job, not
run this session). What *was* verified locally, with real gradient math via PyTorch
autograd (not a hand-derived approximation): construct a batch mimicking a rare label
(Hernia-like) at a realistic mid-training snapshot — 1,996 negatives the model has already
learned to confidently reject (logit = −3, p ≈ 0.047) and 4 positives it hasn't learned yet
(same logit — the model has only learned the base rate, not what distinguishes the 4 true
cases).

| Loss | \|grad from 4 positives\| | \|grad from 1996 negatives\| | negatives/positives ratio |
|---|---|---|---|
| BCE | 0.00191 | 0.04733 | **24.84x** |
| ASL | 0.00209 | 0.00000 | **0.00x** |

Under plain BCE, the already-correct negatives still out-pull the unfixed positives by
~25x in aggregate gradient, purely by volume — confirming the exact failure mode the loss
is meant to fix. Under ASL, `clip` pushes the negative term to exactly zero once a negative
is this confidently correct, handing all gradient budget to the positives that still need
it.

**What remains unverified:** whether this actually raises the trained model's real AUC.
That requires the full Colab training run this session didn't include.

---

## 3. Multilabel-stratified train/val split — built, mechanism-verified, NOT training-verified

**File:** `data/chest_xray14.py` (new `_stratified_val_split`, `_read_rows` helpers; new
`stratify: bool = True` parameter on `load_chest_xray14`)

**Motivation:** the original train/val split (`load_chest_xray14(..., split="val")`) drew a
plain random sample from the official `train_val_list.txt`. For a label like Hernia
(~0.2% prevalence), a random 10% validation slice can by chance contain far fewer — or
more — than 0.2% Hernia-positive images, making that label's validation AUC statistically
meaningless regardless of how well the model actually learned it.

**What was built:** an implementation of iterative stratification (Sechidis, Tsoumakas &
Vlahavas, 2011), specialized to a 2-way split. It repeatedly finds the rarest label that
still has unassigned positive examples and sends each such example to whichever fold (train
or val) is furthest below its target share for that label; "No Finding" (all-zero-label)
images carry no stratification signal and are distributed last, by plain proportional
random split, purely to hit the requested fold sizes. `stratify=False` restores the old
plain-random behavior for comparison.

**How it was verified:** synthetic data matching NIH's real label-prevalence shape (14
labels, prevalence from 18% down to 0.2%, 20,000 samples). Max deviation between a label's
validation-set rate and its true overall rate, across all 14 labels: **0.0002** (i.e.
0.02 percentage points). Head-to-head on the rarest label specifically: true rate 0.27%,
stratified split gave 0.25% in validation, a plain random split on the *same data* gave
0.15% — nearly half the true rate, which is exactly the failure mode being fixed.

**What remains unverified:** same caveat as ASL — whether a better-stratified validation
signal actually changes which checkpoint gets selected as "best" during a real training
run. Untested without that real run.

---

## 4. MSK fracture detection — new expert module (built + wired; real-data eval blocked)

**Files:** `experts/msk_fracture.py` (new), `experts/__init__.py` (registers it),
`scripts/eval_msk_fracture.py` (new, currently blocked — see below)

### Why this modality, and why now

Chest X-ray has a working pretrained adapter. The earlier brain-MRI project is referenced
by this repository, but no brain-MRI expert is implemented here. Head CT (bleed detection)
and MSK fracture were identified as the next two clinically mature, open-data modalities
(a broader research pass compared these against mammography, lung nodule CT, and others —
see the earlier research digest artifact). MSK fracture was picked to build *first*
because it's a plain 2D X-ray problem (no DICOM volumetric windowing, no 3D dependency
conflicts — this project's own git history shows CT work clashing with the chest
environment's numpy/scipy versions). It was meant to be the fast, low-risk second module to
prove the "pretrained-first, evaluate honestly" pattern generalizes beyond chest X-ray.

### Model selection — verified against GitHub, not assumed from a literature summary

The earlier literature review named four candidate repos from the same author
(RuiyangJu et al., built on the GRAZPEDWRI-DX dataset — 20,327 real pediatric wrist trauma
X-rays). Each was re-checked directly against the GitHub API before picking one (this
mattered — one candidate turned out to not actually be usable):

| Repo | Stars | License | Real release weights? |
|---|---|---|---|
| `Bone_Fracture_Detection_YOLOv8` (plain YOLOv8) | 56 | MIT | **Yes** — `best.pt`, 22.5MB, 689 downloads |
| `Fracture_Detection_Improved_YOLOv8` (YOLOv8-ResCBAM) | 136 | MIT | Yes — `YOLOv8_ResCBAM.pt`, 108MB, but requires the authors' custom-modified `ultralytics` fork (adds attention-module code) to load — not a plain-`ultralytics` checkpoint |
| `YOLOv8_Global_Context_Fracture_Detection` | 6 | MIT | **No releases at all** — the literature review's headline mAP number for this variant has no downloadable weights behind it |
| `YOLOv9-Fracture-Detection` | 37 | MIT | Yes — `weights.zip`, 224MB |

**Chosen: plain YOLOv8** (`Bone_Fracture_Detection_YOLOv8`). ResCBAM scores ~2 points
higher AP50 in the paper (65.78% vs. 63.58%) but needs custom architecture code of unknown
ongoing-maintenance status — real integration risk for something meant to demo reliably.
Documented in the module docstring as a drop-in upgrade path if that extra accuracy is
ever worth the integration cost. Global-Context was eliminated outright — it doesn't
actually ship weights, despite showing up with a specific mAP figure in the earlier
research.

Weights download from
`https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8/releases/download/Trained_model/best.pt`,
cached locally at `~/.cache/doctor_assistant_weights/yolov8_fracture_best.pt` on first use
(same "download once, no network at inference" rule as the other pretrained adapters).

### Architecture

`MSKFractureExpert` follows the same `ExpertModel` contract as every other expert
(`TorchXRayVisionExpert`, `TotalSegmentatorExpert`): `predict(scan)` runs the pretrained
YOLOv8 net and stores raw detections (class id, confidence, bounding box) in
`Prediction.meta.extra["detections"]`; `findings_from_prediction(scan, pred)` — the hook
the pipeline looks for — turns each detection into a `Finding`.

Two design choices worth flagging explicitly:

- **9-class vocabulary**, from GRAZPEDWRI-DX's own `meta.yaml`: `boneanomaly`,
  `bonelesion`, `foreignbody`, **`fracture`** (the primary clinical target),
  `metal`, `periostealreaction`, `pronatorsign`, `softtissue`, `text`. The non-fracture
  classes are real dataset categories (hardware, film annotations, a positioning sign),
  kept for completeness/traceability rather than filtered out.
- **No fabricated laterality/anatomical zone.** Chest X-ray findings get a coarse
  left/right + zone from Grad-CAM because chest films follow a fixed radiographic
  convention. An isolated limb film has no such fixed convention (it could be either
  wrist, in any rotation) — inventing a side would be a fabricated fact, which this
  project's design explicitly avoids. The bounding box is preserved in `Finding.extra`
  instead, so the information is there and traceable without being oversold.

Registered via `experts.build_default_registry(include_msk=True)` under `(XRAY, BONE)`.

### Pipeline wiring — verified

Ran a full synthetic smoke test: built the registry with `include_msk=True`, routed a
random `(XRAY, BONE)` scan through the real `Pipeline`, confirmed the expert runs, produces
zero findings on pure noise (correct — no real fracture pattern in random pixels), and the
report/verification chain completes cleanly end to end (`[PASS] grounded 100% of numeric
claims`). This confirms the *integration* is correct.

### BLOCKED: real-image evaluation

This is the part most worth picking up from a different network. Full account of what was
tried:

**Attempt 1 — HF community mirror.** Found `MuhammadJazib/GRAZPEDWRI-DX_SMALL` on
HuggingFace (a public mirror of GRAZPEDWRI-DX with per-image labels as space-separated
class-index strings, e.g. `"8 3 3"` = text once, fracture twice). Pulled 300 real images,
ran the pretrained detector, got: **sensitivity 0.000** on 187 fracture-positive images
(zero true positives), and AUC exactly 0.5000 for every class — a flat, uninformative
score pattern, not "the model is mediocre."

**Root-caused, not just observed:** direct diagnostic showed these images are 16-bit
(`mode: I;16` in PIL) with abnormally high mean brightness (220–251 out of 255 — a normal
X-ray is mostly dark background). Testing with the image inverted (`255 - pixel`) measurably
increased detection activity (more boxes, higher confidence for `text`/`metal` classes),
confirming a real photometric formatting mismatch in this specific mirror — not a
generic "the model doesn't work" result. However, even inverted, `fracture` itself still
never fired confidently in the small sample checked — so simple inversion is not a
complete fix; the mirror's exact preprocessing bug (likely a mishandled DICOM
photometric-interpretation or windowing step during its creation) was not fully
characterized before time ran out on this line of investigation.

**Attempt 2 — go to the authoritative source.** The model's own repo links the original
GRAZPEDWRI-DX dataset on Figshare (article ID `14825193`). Both `curl` and Python
`requests` failed identically: `SSL: CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate` for `api.figshare.com`. The exact same failure mode was hit earlier in this
project's research phase for `zenodo.org`. This looks like a network-level TLS restriction
specific to this machine/network — `github.com` and `huggingface.co` work fine from here,
but Figshare and Zenodo do not.

**To unblock, from a network without this restriction:**
1. Download GRAZPEDWRI-DX directly from
   `https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193` (unsplit; the model
   repo's `split.py` divides it 70/20/10 by `patient_id`), or find/confirm a clean
   HF/Kaggle mirror that preserves standard 8-bit, correctly-oriented PNGs.
2. Re-run `python scripts/eval_msk_fracture.py --n 300` (already built, just needs a
   working image source) — it streams real GRAZPEDWRI images, runs the actual
   `MSKFractureExpert`, and reports per-class presence-detection AUC plus a
   sensitivity/specificity breakdown for `fracture` specifically, same methodology as the
   chest X-ray eval.
3. If the model still underperforms on genuinely correctly-formatted images, that would be
   a real result — worth comparing against the paper's own published 63.58% AP50 /
   0.62 F1 benchmark (Scientific Reports, 2023) to see if this integration reproduces it.

---

## 5. Local eval infrastructure

A CPU-only local venv now exists at `.venv/` (gitignored, not pushed) — Python 3.13, with
`torch`, `torchxrayvision`, `ultralytics`, `scikit-learn`, `datasets`, `huggingface_hub`
installed beyond `requirements.txt` (those are eval-only extras, not needed for the core
pipeline). To rebuild it elsewhere:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow scikit-learn scikit-image torchxrayvision huggingface_hub datasets ultralytics
```

Two real-data eval scripts now live in `scripts/`:

- **`scripts/eval_chest_xrv.py --n 300`** — compares any combination of TorchXRayVision
  weight sets (`all`, `nih`, `chex`, ...) against real NIH ChestX-ray14 test images.
  Caches the downloaded sample locally so repeat comparisons don't re-stream.
- **`scripts/eval_msk_fracture.py --n 300`** — same pattern for the wrist fracture
  detector, currently blocked on a clean image source (see §4 above).

Both scripts print per-label/class AUC, a head-to-head or presence/absence breakdown, and
cache their downloaded sample to avoid re-streaming on reruns.

---

## 6. Files changed this session

```
modified:   data/chest_xray14.py         — stratified split
modified:   experts/__init__.py           — registers MSKFractureExpert
modified:   experts/torchxrayvision.py    — ensemble capability; default reverted to single "all"
modified:   training/losses.py            — AsymmetricLoss
new:        experts/msk_fracture.py       — MSK fracture expert (pretrained YOLOv8)
new:        scripts/eval_chest_xrv.py     — real-data chest X-ray weight-set comparison
new:        scripts/eval_msk_fracture.py  — real-data MSK fracture eval (blocked, see §4)
new:        docs/PROGRESS_NOTES.md        — this file
```

`notebooks/system_test.ipynb` shows as modified in git status but was not intentionally
edited this session (likely IDE-side autosave/kernel metadata) — worth a `git diff` check
before assuming it's meaningful.

---

## 7. Roadmap / next steps, in priority order

1. **Unblock the MSK fracture real-data eval** (§4) — highest priority, since the code is
   done and this is purely a data-access problem.
2. **Full Colab training run** for the custom chest X-ray classifier with ASL +
   stratified split active, to finally confirm (or refute) the AUC benefit both were
   built for. Neither has been proven end-to-end yet.
3. **Head CT hemorrhage expert** (next new modality) — candidate weights identified
   (SeuTao's actual 2019 RSNA-competition-winning checkpoint, PyTorch, DenseNet121/169 +
   SE-ResNeXt101 ensemble) but not yet integrated. Known extra complexity vs. MSK fracture:
   needs DICOM brain/subdural/bone windowing (not bundled with any candidate model — has
   to be implemented), and this project's own git history shows CT/volumetric dependencies
   (`monai`, `SimpleITK`, `pydicom`) have previously conflicted with the chest X-ray
   runtime's `numpy`/`scipy` versions — budget a separate environment or careful pinning.
4. Consider ResCBAM as an MSK fracture upgrade once the base integration is fully
   validated on real data — only worth the custom-fork integration cost if the ~2-point
   AP50 gain is confirmed to matter for real cases the plain model misses.

---

## 8. Context for resuming this conversation elsewhere

If you're picking this up in a new session: the single most useful thing to know is that
**everything in §1–3 (chest X-ray) is done and either measured-and-reverted or
built-and-mechanism-verified** — nothing there is broken or waiting on you. **Everything in
§4 (MSK fracture) is built and wired correctly, but waiting on a working path to real,
correctly-formatted GRAZPEDWRI-DX images** — that's the one open thread with an actual
external blocker (network TLS restriction to Figshare/Zenodo from the original
environment), not a code problem.
