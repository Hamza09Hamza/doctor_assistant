# Doctor Assistant

Doctor Assistant is an **experimental medical-imaging research prototype** that explores how several specialized imaging models could be orchestrated behind one structured analysis pipeline.

The project is the next research step after [`MRI-scans`](https://github.com/Hamza09Hamza/MRI-scans), which focused on a constrained brain MRI tumour-classification task. Instead of extending that classifier into a claim that one model can diagnose everything, this repository explores a modular system in which different models handle different modalities and tasks.

## Project status

> **Work in progress — not clinically validated.**

The repository contains implemented pipeline components, model adapters, reporting logic, smoke tests, and architecture experiments. It does **not** currently provide:

- clinically valid performance metrics
- a confirmed or final system architecture
- regulatory validation
- prospective hospital evaluation
- a production-ready diagnostic service
- evidence that the combined system is safe for patient care

Any outputs produced by the code are research outputs only. They must not be used to diagnose, triage, treat, or make decisions about a real patient.

## Relationship to MRI-scans

The earlier `MRI-scans` project investigated a narrow task: classifying brain MRI images into a limited set of tumour categories and testing the model against data beyond its original training set.

Doctor Assistant begins from a different question:

> How could multiple specialized medical-imaging tools be connected without allowing a reporting model to invent findings that the vision models did not produce?

This repository therefore focuses more heavily on:

- modular expert interfaces
- routing by modality and body region
- structured findings
- failure isolation between models
- report grounding
- report verification
- explicit recommendations and urgency categories
- reproducible system-level testing

The earlier MRI classifier is the motivation for this work, but it should not be interpreted as proof that the broader Doctor Assistant architecture works.

## Current pipeline concept

The current orchestration flow is:

```text
Input scan
    |
    v
Scan loading and metadata normalization
    |
    v
Router selects compatible expert model(s)
    |
    v
Expert predictions
    |
    v
Structured findings
    |
    +--> Draft structured report
    +--> Guideline/recommendation rules
    +--> Report verification
    |
    v
AnalysisResult containing the full audit trail
```

The main entry point is [`pipeline.py`](pipeline.py).

The pipeline uses fixed data contracts such as scans, predictions, findings, and structured reports. The intention is to keep image analysis separate from language generation:

- vision or segmentation models decide what evidence was detected
- findings convert model outputs into structured records
- the reporter verbalizes those records
- the verifier checks the report against the available findings
- the recommendation layer attaches configured follow-up guidance

This architecture is still experimental. Its presence in the codebase does not mean it has been scientifically confirmed as the correct architecture for a clinical system.

## Repository structure

```text
doctor_assistant/
├── pipeline.py                  # end-to-end orchestration
├── core/
│   ├── interfaces.py           # expert and router contracts
│   ├── types.py                # scan and prediction structures
│   └── enums.py                # modality/body-region definitions
├── ingest/
│   └── loaders.py              # scan loading and metadata handling
├── models/
│   └── experts.py              # expert registration and routing support
├── experts/
│   ├── chest_xray.py           # chest X-ray model experiments
│   ├── maira2.py               # adapter for the gated MAIRA-2 model
│   └── ct_totalsegmentator.py  # CT segmentation adapter
├── preprocessing/
│   └── transforms.py           # preprocessing operations
├── reporting/
│   ├── findings.py             # structured findings extraction
│   ├── reporter.py             # structured report generation
│   ├── verifier.py             # report-grounding checks
│   └── guidelines.py           # recommendation/urgency rules
├── training/
│   ├── trainer.py
│   └── losses.py
├── evaluation/                 # evaluation utilities under development
├── scripts/
│   ├── smoke_report.py         # reporting-layer smoke checks
│   └── smoke_system.py         # end-to-end system smoke checks
└── notebooks/
    ├── system_test.ipynb                    # mixed-scenario wiring notebook
    ├── classifier_evaluation_colab.ipynb    # historical exploratory workflow
    └── chest_classifier_build_colab.ipynb   # provenance-checked KAD workflow
```

Some modules may change, move, or be replaced as the architecture is tested.

## Implemented research ideas

### Pluggable expert models

Expert models implement a shared interface so the orchestrator does not need model-specific inference logic. An expert can return classification scores, a segmentation mask, localization data, or richer model-specific output.

### Routing

The router selects experts according to scan metadata such as modality and body region. Unsupported combinations should fail clearly rather than silently selecting an unrelated model.

Routing remains an active area of work. Not every modality and body-region combination has a completed expert implementation.

### Structured findings

Model outputs are converted into `Finding` records before report generation. Depending on the expert output, findings may come from:

- thresholded class probabilities
- segmentation geometry
- heatmap localization
- an expert-specific findings adapter

This is intended to create an auditable boundary between prediction and prose.

### Failure isolation

Each expert is executed independently. A gated model repository, missing token, network failure, incompatible dependency, out-of-memory error, or model-specific exception should not automatically destroy results already produced by other experts.

A failed expert is skipped with a warning and is not represented as if it completed successfully.

### Grounded reporting

The reporting layer receives structured findings rather than unrestricted access to the original scan. The design goal is to reduce unsupported statements by limiting the reporter to facts supplied by upstream components.

This is a design constraint, not proof that hallucinations or misleading wording are impossible.

### Verification

The verifier checks the generated report against the known findings and expert label vocabulary. This is an additional guardrail, but it is not a substitute for radiologist review or clinical validation.

## Current model experiments

The repository includes experiments and adapters around tools such as:

- TorchXRayVision for multi-label chest X-ray findings
- MAIRA-2 for grounded radiology-report research, subject to gated-model access
- TotalSegmentator for CT organ segmentation and volumetry experiments
- local language models for report wording with deterministic fallback behavior

These components have different licenses, hardware requirements, datasets, calibration behavior, and dependency constraints. Their inclusion does not mean they have been validated as one combined medical product.

In particular:

- gated Hugging Face models require the appropriate access approval and token
- CT and chest environments may require separate runtimes because of scientific-Python dependency conflicts
- model thresholds are not interchangeable across pathologies
- pretrained outputs must be calibrated and tested against representative data
- low-resolution samples are not a valid substitute for native-resolution clinical images

## Running development checks

The repository is still changing, so dependency installation may differ by experiment and model. Review the imports and the notebook installation cells before running a GPU workflow.

Basic smoke checks can be launched from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/smoke_report.py
python scripts/smoke_system.py
```

Environment setup, optional expert dependencies, cache configuration, and the separate
TotalSegmentator runtime are documented in
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

Use the focused Google Colab notebook for classifier-by-classifier validation,
error analysis, calibration, and fail-closed evaluation preparation:

```text
notebooks/chest_classifier_build_colab.ipynb
```

The historical `notebooks/classifier_evaluation_colab.ipynb` uses third-party mirror
partition names and is retained only as an exploratory record; it is not the current
evidence path. The separate `notebooks/system_test.ipynb` remains a mixed-scenario
integration and wiring demonstration. Passing any workflow is **not** a
clinical-performance result.

The current chest-classifier decision, endpoint definitions, leakage controls, and
one-T4 experiment ladder are recorded in
[`docs/CHEST_CLASSIFIER_RESET.md`](docs/CHEST_CLASSIFIER_RESET.md). The legacy custom
14-label DenseNet is retired as a candidate. KAD-512 is the first replacement candidate,
and acceptance proceeds one endpoint at a time, beginning with pneumothorax. Each
endpoint uses a separate frozen one-query KAD pack because decoder self-attention makes
scores depend on the other queries present; old shared three-query scores are not valid
endpoint-isolated evidence. The
development protocol freezes patient-disjoint model-selection (30%), calibration
(20%), threshold-selection (20%), and untouched-acceptance (30%) roles. A frozen
threshold becomes test-ready only if study-level sensitivity and specificity pass
two-sided 95% Wilson bounds on score-blind, deterministic SHA-256-selected studies
(one positive per positive patient and one negative per negative patient) in the
untouched role, with at least 22 positive and 20 negative acceptance patients, using
verified original NIH development pixels. All-study acceptance metrics and the
notebook's current 320-pixel JPEG mirror cannot produce a test-ready decision.
The mirror path does not read or report untouched-acceptance scores at all.

Those memberships are endpoint-specific and fixed across candidates by protocol
seed `20250729` and a fixed 512-attempt score-blind label-support search. The
canonical CLI rejects seed/search changes; shared multi-endpoint encoder adaptation
is not supported by this split design.

The provenance reader still rejects a bare self-declared `original_nih_pixels=true`
receipt. A verified path now exists behind a multi-source consensus check instead:
`scripts/fetch_nih_original_images.py` and `scripts/benchmark_kad.py`'s schema-2
provenance loader accept original-pixel evidence only once a file's SHA-256 agrees
across at least two independently-operated sources. Running that path against the
full NIH release for real, and using it to unlock a locked-evaluation run, is still
outstanding — only the ingestion script and its validator are implemented so far.

KAD licensing also remains an explicit release blocker: its reviewed code commit has an
MIT `LICENSE`, but separate terms for the downloadable checkpoint weights are not stated
in the project README or download. Research evaluation can continue, while company or
product use requires legal confirmation of the weight rights.

## Metrics and evaluation policy

This repository intentionally does not publish a headline accuracy, AUC, sensitivity, specificity, or combined-system score at this stage.

A valid metric would require, at minimum:

- a clearly defined task and target population
- documented dataset provenance and licensing
- patient-level separation between training, validation, and testing
- checks for duplicate studies and dataset overlap
- modality-appropriate preprocessing
- fixed model versions and thresholds
- per-class sensitivity and specificity
- confidence intervals
- calibration analysis
- failure-case review
- external and preferably prospective validation
- evaluation of the complete pipeline, not only one component

Until such an evaluation is completed and reproducible, any values produced during development should be treated as debugging observations rather than confirmed performance.

## Known limitations

- The architecture is provisional and may change substantially.
- The repository does not cover all modalities, body regions, or diseases.
- Some expert adapters rely on gated or large external models.
- Different model stacks can require incompatible Python environments.
- Current rules and thresholds are not clinically validated.
- A small real-data integration run confirmed that a single global chest threshold can
  severely overcall findings; per-label thresholds must be selected on a separate
  validation set before report outputs are interpreted. See
  [`docs/STABILIZATION_NOTES.md`](docs/STABILIZATION_NOTES.md).
- Generated recommendations are not a substitute for medical guidelines applied by a qualified professional.
- Report verification cannot guarantee factual or clinical correctness.
- Dataset shift, scanner differences, acquisition protocols, and image quality can change model behavior.
- The system has not completed privacy, security, regulatory, or medical-device review.

## Research roadmap

Near-term work includes:

1. Stabilize the core interfaces and decide which architecture should remain.
2. Integrate the earlier MRI work through a documented expert adapter.
3. Define one narrow end-to-end evaluation task rather than claiming broad diagnosis.
4. Create reproducible environments for each model family.
5. Add unit and integration tests around routing, findings, verification, and failure isolation.
6. Record real measured results with dataset and configuration provenance.
7. Add calibration and uncertainty handling.
8. Review outputs with qualified medical professionals before making any clinical claims.

## Responsible use

This code is for education, research, and software experimentation. It is not medical advice and is not approved for clinical use.

Do not upload identifiable patient scans to third-party services without authorization, appropriate agreements, and compliance with the applicable privacy and health-data requirements. Remove metadata and protect all medical data used during development.
