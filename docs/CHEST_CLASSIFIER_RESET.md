# Chest classifier reset

Status: implementation plan, 2026-07-28. This is a research protocol, not a
clinical-use claim.

## Decision

The legacy 14-label DenseNet checkpoint is retired as a candidate build. It can
remain in the repository as a regression fixture, but it must not drive findings
or reports.

The next chest expert will be built and accepted one finding at a time:

1. **Pneumothorax**
2. **Nodule or mass** (one combined endpoint)
3. **Airspace opacity**

The first candidate is **KAD-512**: a 512-pixel ResNet-50 image encoder, medical
knowledge encoder, and disease-query transformer. It was pretrained on MIMIC-CXR
and evaluated externally on NIH ChestX-ray14, so it is a cleaner NIH challenger
than TorchXRayVision `all`, RAD-DINO, Ark+, or other weights that were trained on
NIH images. KAD is first evaluated zero-shot. Adaptation is earned only if its
zero-shot ranking is sound but its operating point is insufficient.

Each phase-1 endpoint is a separate one-query KAD model identity. The released
decoder applies self-attention across all query tokens, so an unchanged
pneumothorax prompt produces different scores when nodule or opacity queries are
present. Combining the three prompts would therefore violate the requirement to
evaluate and refine classifiers one by one. Each endpoint has its own frozen
query-set ID, semantic tensor hash, predictions, calibration, and decision
artifact. Published KAD results obtained with a joint query context are not
directly comparable to these endpoint-isolated scores.

TorchXRayVision `all` remains an engineering control, not an independent NIH
baseline, because its training sources include NIH. RAD-DINO is a later ablation,
not the first candidate, because its released representation was pretrained on
all NIH images.

Sources:

- [KAD paper](https://www.nature.com/articles/s41467-023-40260-7)
- [KAD code and checkpoints](https://github.com/xiaoman-zhang/KAD)
- [TorchXRayVision model sources](https://github.com/mlmed/torchxrayvision)
- [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino)

Licensing is not yet cleared for company or product use. The reviewed KAD source
commit includes an MIT `LICENSE`, but neither its README nor the checkpoint
download states separate terms for the released weights. Research evaluation may
proceed; legal confirmation of checkpoint-weight rights is still required before
commercial or product use.

## Why the previous build is not repairable by threshold tuning

The 2,000-image development run showed that the legacy model still contains some
ranking information (macro AUROC about 0.73), but its probabilities collapsed
near zero. At a global threshold of 0.5 it detected the correct label in only two
of 960 weak-label abnormal studies. Only edema could meet the provisional
85%-sensitivity and 60%-specificity point-estimate constraints after per-label
threshold search.

That is not only a calibration problem. Thirteen supported labels had too much
positive/negative score overlap to meet the requested operating constraints.
Several training defects also make the checkpoint non-reproducible:

- ImageNet-pretrained features were not given their required mean/std
  normalization.
- the configured random flip could flip both image axes;
- the auxiliary “confidence” target could reward all-negative output;
- the local asymmetric-loss implementation did not match the reference
  implementation;
- the saved checkpoint lacks a complete data, preprocessing, loss, and seed
  manifest;
- its original validation split was image-level rather than patient-level.

The current Hugging Face mirror adds another protocol problem: it exposes
89,696/11,212/11,212 train/valid/test rows, not the official NIH
86,524/25,596 train+validation/test membership. Mirror split names therefore
cannot be treated as an official held-out evaluation.

## Target definitions

### Phase 1

- **Pneumothorax** uses the expert-defined pneumothorax label. It also requires a
  chest-tube versus no-chest-tube sensitivity audit to detect shortcut learning.
- **Nodule or mass** is one endpoint until a sufficiently large adjudicated
  reference standard supports separate nodule and mass claims.
- **Airspace opacity** uses the expert reference definition. NIH weak
  `Infiltration`, `Consolidation`, and `Pneumonia` labels may be used only as
  noisy training signals and must never be presented as equivalent ground truth.

### Deferred

`Hernia`, `Fibrosis`, `Pleural_Thickening`, `Emphysema`, and image-only
`Pneumonia` are deferred for inadequate or ambiguous reference standards.
`No Finding` must not be translated into “normal”: it means that none of the
limited weak-label ontology was extracted. Separate `Nodule` and `Mass` outputs
are also deferred.

`Atelectasis`, `Cardiomegaly`, `Pleural Effusion`, `Edema`, and
`Consolidation` are reasonable Phase-2 targets once licensing and an
expert-labelled evaluation source are in place.

## Data protocol

1. Start from NIH's official `train_val_list.txt` and `test_list.txt`.
2. Build an immutable canonical expert manifest containing filename, patient ID,
   official membership, adjudicated expert targets, source provenance, and
   hashes. The current implementation stops there. A validated metadata join for
   NIH weak labels, AP/PA view, age, and sex is still required and remains open
   before subgroup or shortcut-audit claims.
3. Obtain Google's additional NIH expert labels. They provide adjudicated
   three-reader development labels for 2,412 images and test labels for 1,962
   images for pneumothorax, airspace opacity, nodule/mass, and fracture.
4. Remove every expert-development patient from weak-label training. For each
   endpoint, before inspecting any candidate scores, freeze its patients once into
   four patient-disjoint roles: model selection (30%), calibration (20%),
   threshold selection (20%), and untouched acceptance (30%). The endpoint-specific
   assignment uses protocol seed `20250729` and exactly 512 label-only hash-search
   attempts; both are fixed, and the resulting membership must be reused across
   every candidate and adaptation for that endpoint. Changing the seed or search
   count to recycle the acceptance cohort is prohibited.
5. Keep the expert test locked. The notebook must refuse test mode unless the
   model, prompt set, calibrator, thresholds, and manifest hash are frozen.
6. Bootstrap confidence intervals by patient. Use paired patient resampling when
   comparing candidates on the same studies.
7. After the NIH build is frozen, use untouched VinDr-CXR as an external test for
   pneumothorax and nodule/mass. VinDr `Lung Opacity` is broader than the
   airspace-opacity endpoint and is a semantic-robustness audit unless a
   radiologist approves the mapping.

The additional NIH labels and their construction are documented by
[Google Cloud](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#additional_labels)
and the
[Radiology reference-standard study](https://pubs.rsna.org/doi/10.1148/radiol.2019191293).
[VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) requires individual
credentialing, training, and a signed data-use agreement.

[CANDID-PTX](https://doi.org/10.17608/k6.auckland.14173982) likewise requires
ethics training and a data-use agreement. Its curation and annotation protocol
is described in the
[Radiology: Artificial Intelligence dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8637219/).

For development convenience, TorchXRayVision publicly mirrors the combined Google
four-finding table as
`torchxrayvision/data/google2019_nih-chest-xray-labels.csv.gz`. Its current file
contains 2,414 rows marked `val` and 1,962 rows marked `test`, which differs by two
validation rows from Google's current documentation. The build pins the mirror's
SHA-256, reconciles every filename to the official NIH manifests, and records the
count difference. Before a locked test, replace or independently reconcile this
mirror against the direct Google download.

The Colab development path may use a filename-preserving 320-pixel NIH mirror to
avoid downloading all 48 GB of source pixels just to select a candidate. Those
images are allowed for development ranking only. Official filenames do not make
resized JPEGs equivalent to the original NIH PNGs; locked evaluation requires the
original pixels and binds their hashes into the test lock.

## One-T4 experiment ladder

### Gate 0 — pipeline sanity

- Run with augmentation disabled.
- Deliberately overfit 64–256 selected images containing positives for every
  active endpoint.
- Assert exact label order, logits (not probabilities) into the loss, positive
  batches and non-zero gradients for every head, and deterministic evaluation
  preprocessing.
- Record raw/transformed tensor statistics and positive/negative logit
  quantiles. Do not start a full run if this gate fails.

### Gate 1 — KAD-512 zero-shot

- Use the official 512-pixel PIL-BICUBIC/ToTensor preprocessing and the exact
  frozen prompt for the active endpoint. Tensor-interpolation substitutes are
  not accepted.
- Export and benchmark exactly one active-endpoint query. Never reuse or slice
  scores from the retired three-query phase-1 pack.
- Export a lean query pack containing the image encoder, disease-query network,
  one frozen text-query embedding, prompt, preprocessing contract, checkpoint
  hash, and code revision. The BERT knowledge encoder is not needed for normal
  image inference after query embeddings have been produced once.
- Canonicalize the exported text embedding through a BF16 round trip and store
  the resulting values as float32. Med-KEBERT CPU kernels can otherwise differ
  in the last float32 bits across prompt batch sizes or runtimes, making an
  exact semantic hash reject numerically equivalent exports. Query-pack format
  v3 binds this canonicalization contract.
- Phase-1 export loads the three reviewed singleton embeddings from the
  checksum-pinned `configs/chest_kad_phase1_query_features.json` asset. It does
  not rerun Med-KEBERT, because CPU kernels can still land on opposite sides of
  a BF16 rounding boundary across PyTorch/Colab runtimes. Med-KEBERT remains
  available for the noncanonical NIH-14 smoke export.

The current Colab notebook implements this zero-shot evaluation and fail-closed
decision analysis. It does not implement Gate 0 training or Gate 2 adaptation.
Those training stages should be built only after the isolated zero-shot ranking
result demonstrates that an endpoint is worth adapting.

The public Google development subset is useful for candidate ranking but is not
large enough to accept the pneumothorax operating point under this protocol. It
contains only 43 pneumothorax-positive images from 33 positive patients. Once
those patients are separated into model-selection (30%), calibration (20%),
threshold selection (20%), and untouched acceptance (30%) roles, the acceptance
role cannot contain the required minimum of 22 positive patients. The
calibration command may therefore record a diagnostic candidate threshold, but
it leaves `acceptance_complete=false` and does not produce a test-ready
threshold when the independent acceptance gate cannot be evaluated or fails.
This is separate from the pixel-provenance gate: the current 320-pixel JPEG
mirror is never decision-eligible, regardless of support or apparent
performance. Original NIH development pixels are required before freezing a
decision artifact for an original-pixel locked test.

### Gate 1b — pneumothorax shortcut and external audit

Before pneumothorax can be accepted, obtain **CANDID-PTX** through its required
ethics training and data-use agreement. It provides 19,237 frontal radiographs,
including 3,196 pneumothorax-positive images and explicit intercostal chest-tube
annotations on 1,423 images. Use untouched patient groups to report ranking and
operating behavior separately for:

- pneumothorax with a chest tube;
- pneumothorax without a chest tube;
- no pneumothorax with a chest tube; and
- no pneumothorax without a chest tube.

This is the concrete shortcut-learning audit; a saliency map alone is not a
substitute. NIH view position, age, and sex subgroup joins also remain required
before endpoint acceptance. Until those data and audits exist, the endpoint
status is “candidate evaluated, not perfected or accepted.”

### Gate 2 — controlled adaptation

Only if Gate 1 shows useful separation:

1. train the query/classification layer while the encoders are frozen;
2. optionally unfreeze the last ResNet block;
3. compare unweighted BCE with the exact reference asymmetric loss as a
   one-variable ablation;
4. early-stop on per-endpoint AUPRC/AUROC, not threshold-0.5 accuracy.

Use AMP and gradient accumulation. The notebook must benchmark a short block of
steps and report measured memory and projected runtime instead of promising a
wall-clock estimate.

### Gate 3 — calibration and operating point

After model selection, fit a separate per-endpoint calibrator on the calibration
partition. Compare Platt/vector or beta calibration using log loss, Brier score,
and reliability plots. Select the reporting threshold on the separate threshold
selection partition against a prespecified sensitivity, specificity, and
false-alert workload. Freeze that threshold before inspecting the untouched
acceptance role.

These roles are endpoint-specific, not a license to train one shared multi-target
model across another endpoint's acceptance patients. Any controlled adaptation is
an endpoint-specific copy of the frozen base candidate. A future shared encoder
adaptation requires one global patient-role manifest that protects every endpoint's
acceptance membership before training.

Acceptance retains study/radiograph-level sensitivity and specificity as the
clinical endpoint without allowing patients with many studies to dominate it.
Independently of scores, the protocol uses SHA-256 to select one adjudicated
positive study per positive patient and one adjudicated negative study per
negative patient. It gates on two-sided 95% Wilson score intervals over those
prespecified studies. The acceptance role must contain at least 22 positive
patients and 20 negative patients. Metrics over every acceptance study are
diagnostic only. The lower sensitivity and specificity bounds—not only their
point estimates—must meet the prespecified targets before
`acceptance_complete=true`. The 22-positive-patient minimum is mathematical, not
negotiable: even 21/21 sensitivity has a two-sided 95% Wilson lower bound of
about 0.845, below the 0.85 target; 22/22 reaches about 0.851.

All calibration and threshold-selection calculations made from the 320-pixel
development mirror are diagnostic only. The implementation does not inspect
untouched-acceptance scores for non-original pixels, so the mirror emits no
acceptance estimate to rationalize after the fact. Passing upstream numerical
gates cannot override `original_nih_pixels=false`; repeat the frozen protocol
through a trusted original NIH development-pixel path before any threshold can
become test-ready.

That original-pixel route is intentionally not enabled yet. Schema-1 image
provenance is only a local declaration, so the benchmark now rejects
`original_nih_pixels=true` until a trusted ingestion receipt verifies NIH
archive/source identity and binds the exact selected source bytes to the canonical
manifest. This prevents a hand-authored receipt from upgrading resized pixels into
acceptance evidence.

### Gate 4 — locked evaluation

Run the adjudicated NIH test exactly once, then the untouched external set.
Report AUROC and AUPRC with confidence intervals plus sensitivity, specificity,
PPV, NPV, false alerts on reference-negative studies, coverage/abstention, and
the planned subgroup audits.

Acceptance is per endpoint. A macro average cannot hide a failed label. A
threshold is accepted only when the lower confidence bound—not only the point
estimate—meets the declared operating requirement with an adequate number of
positive and negative acceptance patients. Neither calibrator fitting nor
threshold selection may use the untouched acceptance role.

## Colab artifact layout

Large image shards should be copied from Drive to `/content` before training;
training must not open 100,000 individual files through Drive FUSE. Drive stores
only durable inputs and outputs:

```text
doctor_assistant/
  data_manifests/
    nih_official_manifest.jsonl
    expert_development.csv
    expert_test.csv
  models/chest_xray/
    kad-512-source/
    pneumothorax/<experiment-id>/
      manifest.json
      best.pt
      last.pt
      predictions.npz
      calibration.json
      thresholds.json
      results.json
```

Every experiment uses a new immutable directory whose ID includes the endpoint,
Git commit, evaluation-config hash, and UTC timestamp. It records the source
model and checkpoint hash, canonical endpoint query-set ID and query-pack semantic hash,
data-manifest hash, prompts, target definition, preprocessing, seed,
optimizer/loss configuration, patient-role fractions, and environment versions.
The runtime contract also binds accelerator name, compute capability, memory,
CUDA/cuDNN runtime, and driver; the canonical notebook is T4-only.
Evidence artifacts are never silently overwritten, and legacy `best.pt` is
never overwritten.
