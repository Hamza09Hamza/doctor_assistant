# Stabilization notes — 2026-07-28

> Historical note: the chest-evaluation conclusions here are superseded by
> [`CHEST_CLASSIFIER_RESET.md`](CHEST_CLASSIFIER_RESET.md). The named Hugging Face
> mirror partition used below was not reconcilable to NIH's official manifests and
> cannot support held-out evidence.

This pass focused on software safety and reproducibility before adding another model.

## Corrected behavior

- If every routed expert fails, the pipeline raises `PipelineExecutionError` instead of
  drafting a false normal report.
- Partial expert failure remains isolated and is recorded in
  `AnalysisResult.expert_failures`.
- Each expert receives isolated metadata, preventing cross-reader state leakage.
- Grad-CAM localization is computed automatically for above-threshold findings produced
  by compatible trainable experts.
- Fallback routing is restricted to explicitly unknown metadata and rejects ambiguity.
- Experts registered under multiple niches execute only once.
- Verification now hard-fails unsupported, omitted, directly negated, wrongly counted,
  or wrong-unit claims.
- Recommendations are deduplicated without deleting the underlying expert evidence.
- NIH CSV fallback, checkpoint best-metric state, scheduler resume state, rotated-affine
  spacing, evaluation cache paths, and `--per-class` sample fetching were corrected.
- Deterministic fact-only reporting is now the pipeline default; no-positive studies
  never invoke an LLM, and failed generated drafts fall back to a re-verified template.
- Stale notebook outputs were cleared.

## Verification completed

The repository-local Python 3.14 environment passed:

```text
32 unit tests
scripts/smoke_report.py
scripts/smoke_system.py
```

The tests cover routing safety, metadata isolation, all/partial expert failure,
automatic Grad-CAM, report grounding, split fallback, checkpoint state, sample
selection, threshold metrics, and volume spacing.

## Real-data integration observation

The repaired chest evaluation was run with the single default TorchXRayVision `all`
checkpoint on 40 streamed NIH ChestX-ray14 test images. This sample is too small for a
performance claim: only 10 labels were scoreable and several had one positive example.
Its purpose was to exercise download, caching, preprocessing, inference, and metric
calculation. Subsequent runs pin the dataset to revision
`932bcdba9d7d9590704d4f20bc70fc2c3a1bbad7`.

Observed macro AUC was `0.7098`. More importantly, threshold-level behavior exposed a
calibration blocker:

| Threshold | Macro sensitivity | Macro specificity | Dataset-normal studies with any FP | Mean findings per normal |
|---:|---:|---:|---:|---:|
| 0.50 | 0.989 | 0.194 | 1.000 | 9.64 |
| 0.60 | 0.915 | 0.435 | 0.929 | 4.36 |
| 0.70 | 0.890 | 0.549 | 0.714 | 2.71 |
| 0.80 | 0.096 | 0.948 | 0.071 | 0.21 |

These values are debugging observations, not confirmed performance. NIH `No Finding`
labels are imperfect, and the sample is small, but the result is decisive enough to
reject a single unvalidated threshold as a basis for report generation. A verifier can
prove that prose matches model output; it cannot make overcalled model output correct.

## Product-context correction

The product explainer describes brain MRI as working. This repository has generic MRI
types and mask-to-finding geometry, but no registered brain-MRI expert, checkpoint,
dataset builder, or reproducible evaluation. Brain MRI must remain “not implemented in
this repository” unless that missing component is imported and evaluated.

The NIH labels used by the chest integration are public dataset labels, not prospective
adjudication by radiologists for this product. They are suitable for development
benchmarking but should not be described as company-specific clinical validation.

## Next scientific gate

Use a patient-separated validation set to choose **per-label** operating thresholds
against a declared target (for example, minimum sensitivity with a specificity floor).
Freeze those thresholds, then evaluate once on a separate held-out test set with
confidence intervals and calibration plots. `Pipeline` already accepts a threshold
dictionary, so no architecture change is needed.
