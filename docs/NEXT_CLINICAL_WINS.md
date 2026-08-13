# Next clinical wins

Status: product/model roadmap after the Detect -> Inspect -> Compare console. “Next”
means researched, not integrated or validated. License applies separately to code,
weights, and data; a permissive repository license does not override restrictive model
weights.

## Built now

The current chest-CT path already delivers the most important system win:

```text
local DICOM CT -> automatic candidate shortlist -> source-image inspection
-> explicit MedSAM2 3D outline -> OHIF labelmap + measurements
-> prompt-matched reader comparison -> standards-valid DICOM SEG -> local Orthanc
```

It also preserves results across panel remounts, makes false-positive review explicit,
and separates candidate selection from expensive inference. The next work should add a
new clinical capability without weakening those contracts.

| Capability | State | Deployment posture |
|---|---|---|
| Chest-CT shortlist + explicit 3D outline + reader comparison + DICOM SEG | Built now | Research workflow running end to end |
| TotalSegmentator anatomy map | Next implementation | Permissive current path; endpoint evaluation still required |
| Narrow pediatric wrist-fracture review | Adapter built, real-data gate pending | Permissive candidate; do not expose before frozen evaluation |
| MedGemma structured draft | Researched | Gated terms; legal/product review before integration |
| nnInteractive, X-Raydar, VoxTell, BiomedParse | Research tracks | Non-commercial model/weight terms; not the deployable default |

## Ranked roadmap

### 1. TotalSegmentator expansion — largest deployable CT breadth win

**Why first:** this repository already proved the complete CT -> DICOM SEG -> Orthanc ->
OHIF bridge, including a standards-valid run with 91 non-empty segments and verified
source references. The default current task covers 117 anatomical structures. Expansion
can reuse the existing viewer, artifact, and measurements path instead of creating a new
product surface.

- Official project: [wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- License: Apache-2.0 for the current default project/model path already used here.
- Runtime: Colab GPU; keep its nnU-Net dependency stack isolated from the Mac and from
  the existing API environment.
- Product slice: add an **Anatomy map** workflow with search, organ visibility groups,
  organ volumes, and a compact “expected anatomy / review exceptions” summary. Do not
  present normal-organ masks as pathology findings.
- Acceptance: run on several correctly licensed CTs, verify geometry/source references,
  preserve per-structure provenance, and inspect the resulting DICOM SEG in OHIF.

This is implementation-ready, but it is an anatomy/volumetry capability—not a universal
tumour or fracture detector.

### 2. Narrow wrist-fracture workflow — strongest new-modality product win

**Why second:** it creates a visibly different, clinically understandable workflow on a
single radiograph: detect -> inspect bounding box -> record disposition. The repository
already contains a plain YOLOv8 adapter and a checksum-pinned released checkpoint trained
on pediatric wrist trauma radiographs. A YOLOv9 follow-up is available from the same
research line.

- Current adapter: `experts/msk_fracture.py`, based on
  [RuiyangJu/Bone_Fracture_Detection_YOLOv8](https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8).
- Upgrade candidate: [RuiyangJu/YOLOv9-Fracture-Detection](https://github.com/RuiyangJu/YOLOv9-Fracture-Detection),
  with released trained weights.
- License: MIT repository; a trained release is available. Record the exact release
  asset, its applicable terms, and checksum before accepting any upgrade.
- Scope: pediatric wrist trauma X-rays only. Do not relabel this as general MSK,
  adult-fracture, or whole-body fracture detection.
- Acceptance gate: finish a real held-out GRAZPEDWRI-DX evaluation, define one frozen
  operating point, measure sensitivity and false boxes per image, verify bounding boxes
  visually, then connect it to the same review/disposition rail.

Important correction: **KAD-512 is a chest-X-ray vision-language model, not an MSK
fracture model.** It must not be used as evidence for this wrist workflow.

### 3. MedGemma 1.5 4B — structured draft and evidence-linking assistant

**Why third:** once detection/segmentation evidence is stable, a small multimodal model
can turn selected evidence into a structured draft: technique, observed candidate,
measurements, uncertainty, and required review fields. It should summarize verified
facts, not originate diagnoses.

- Official model family: [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/medgemma).
- Candidate: MedGemma 1.5 4B.
- Access/license: gated and governed by the Health AI Developer Foundations terms. Treat
  it as a separate legal/product review; do not assume Apache-style deployment rights.
- Runtime: Colab GPU. The 4B candidate is intentionally preferred over a much larger
  model for the current single-worker setup.
- Product slice: generate editable draft text only from structured detector,
  segmentation, measurement, and disposition inputs; cite the source series/candidate;
  never auto-sign or silently publish a report.
- Acceptance: measure factual consistency against the structured inputs, omission rate,
  unsupported statements, and doctor edit distance on a small frozen set.

### 4. nnInteractive — biggest interactive-segmentation research win

**Why:** it is the clearest functional upgrade from one box prompt. The official remote
server/client supports positive and negative points, scribbles, lasso, and boxes, making
iterative correction much closer to how a doctor actually fixes a contour.

- Official code: [MIC-DKFZ/nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)
- Official weights: [MIC-DKFZ/nnInteractive on Hugging Face](https://huggingface.co/MIC-DKFZ/nnInteractive)
- License: Apache-2.0 code; CC BY-NC-SA weights.
- Runtime: about 10 GB GPU VRAM is recommended; use a capable Colab GPU, not the 16 GB
  Mac. Its server/client architecture fits the current remote-inference pattern.
- Product slice: replace “one box, one result” with a correction loop while retaining
  the same DICOM SEG output and Compare step.

Because the weights are non-commercial/share-alike, keep this in a clearly separated
research runtime unless and until an acceptable weight license exists. Do not make it the
only interactive segmentation path; the current MedSAM2 workflow remains the baseline.

### 5. X-Raydar — high-value chest-X-ray research benchmark

**Why:** the validated 299/512/1024 ensemble predicts 37 chest-X-ray findings and would
give the project a much broader CXR benchmark than the retired classifiers.

- Official code: [gmontana/xraydar-cv](https://github.com/gmontana/xraydar-cv)
- Weights: [dnamodel/xraydar-cv](https://huggingface.co/dnamodel/xraydar-cv)
- Study/source: the official KCL research page and linked publication should remain the
  evidence record for the 37-finding model.
- License/use: non-commercial academic use. Keep it research-only; it is not the
  deployable default for a clinic-facing workflow.
- Runtime: remote GPU; evaluate the released multi-resolution ensemble exactly as
  documented before considering a smaller serving configuration.
- Acceptance: use a frozen external set with a published label ontology, per-finding
  AUROC/sensitivity/specificity, calibration, subgroup slices when available, and a
  visible “not evaluated” state for unsupported findings.

This is a benchmark and research-demo win, not a shortcut around model licensing or
local validation.

### 6. VoxTell and BiomedParse — language-guided segmentation research tracks

These models could power “outline the left adrenal gland” or “segment this described
finding” without forcing the user to hunt for the correct tool. They are compelling UI
research, but their weight terms make them later than the deployable paths above.

#### VoxTell

- Official code: [MIC-DKFZ/VoxTell](https://github.com/MIC-DKFZ/VoxTell)
- License: Apache-2.0 code; CC BY-NC-SA weights.
- Product experiment: text-guided volumetric selection with the text prompt, model
  version, and output recorded in the artifact provenance.

#### BiomedParse

- Official code: [microsoft/BiomedParse](https://github.com/microsoft/BiomedParse)
- License: Apache-2.0 code; CC BY-NC-SA weights.
- Product experiment: promptable 2D biomedical-image parsing. Treat it as a different
  capability from full-volume CT propagation; do not advertise a 2D mask as a 3D result.

For both: use isolated Colab environments, label the feature research-only, and compare
against modality-specific references before connecting it to the main clinician mode.

## Delivery sequence

The best order balances visible product value, evidence, and reuse:

1. Finish the current chest-CT acceptance path, including durable local-Orthanc SEG save.
2. Add the TotalSegmentator anatomy map through the existing DICOM SEG contract.
3. Complete held-out wrist-fracture evaluation, then add a dedicated single-image review
   mode if it passes the frozen gate.
4. Prototype MedGemma structured drafts against machine-readable evidence only.
5. Run nnInteractive, X-Raydar, VoxTell, and BiomedParse as isolated research tracks;
   promote none of them without resolving weight terms and endpoint-specific evaluation.

## Non-negotiable evaluation language

- A detector score is a ranking value unless calibration has been demonstrated.
- A segmentation produced from a prompt follows the prompt; it does not prove the
  prompted object is abnormal.
- “No candidate” is not “normal scan.”
- Dice against a prompt-matched staged annotation is object-level overlap evidence for
  that case, not universal model accuracy.
- Anatomy segmentation is not pathology detection.
- A model that performs well in its own paper still needs a frozen evaluation for this
  project's exact endpoint, population, preprocessing, and operating point.
