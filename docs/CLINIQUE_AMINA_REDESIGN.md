# Clinique Amina clinical console — v4

Status: implemented in the Doctor Assistant OHIF mode. This document describes the
clinician-facing workflow and the boundaries of the current implementation. It is not a
clinical-validation claim.

## Product intent

Clinique Amina is a focused imaging-review console, not a generic AI dashboard and not a
replacement for the DICOM viewer. OHIF continues to own image rendering, DICOMweb,
window/level, navigation, measurements, and segmentation display. The Clinique Amina
extension adds one clear review sequence beside the images:

```text
Detect -> Inspect -> Compare
```

The viewport stays dominant. The right rail tells the doctor what has run, what is
selected, what evidence exists, and what action is available next. Candidate generation,
prompted segmentation, and reference comparison are deliberately different steps; the UI
does not collapse them into one ambiguous “AI result.”

## Information architecture

### 01 Detect — find candidates

- **Find nodule candidates** reuses a valid source/model-bound run when one exists;
  **Re-run model** explicitly requests a fresh volumetric MONAI inference.
- A visible elapsed timer and locked state explain that the remote worker permits one
  inference at a time.
- The response becomes a ranked candidate list. Scores are labeled **rank score**, with
  “not probability” shown in the result.
- Zero candidates means only that none crossed the configured threshold. The panel says
  to continue image review; it does not call the scan clear.
- The shortlist and elapsed time remain available when the right panel remounts. The API
  also stores a small atomic per-series run cache, so a compatible result can be
  recovered after an API or panel restart without repeating expensive inference.

Evidence copy is preserved exactly in substance: at the fixed 0.30 score threshold, this
project detected 21 of 23 derived consensus nodules across 27 eligible LIDC CT series and
produced 2.15 false candidates per scan. Eligible SeriesInstanceUIDs were absent from
LUNA16's published 888-series corpus. This is a small research evaluation, not clinical
validation, a calibrated cancer probability, or proof that a scan without a mark is
normal.

### 02 Inspect — review the source images first

- Selecting a candidate moves the axial viewport to its referenced source slice.
- Selection alone does **not** start MedSAM2. This prevents an accidental expensive
  inference and gives the doctor time to inspect the location in image context.
- The selected row exposes a separate **Generate 3D outline** action.
- While an outline runs, the chosen source slice and prompt are locked and the panel
  shows a working state.
- Previously outlined candidates retain their completion mark and result in the same
  viewer session. Returning to one restores its result instead of losing the review.
- **Refine with box** remains available in the toolbar for a manually chosen structure.
  Its help text states that a prompt mask is not anomaly evidence.

### 03 Compare — review, disposition, and preserve

After the outline returns, the rail shows:

- segmented slice count;
- volume in mL;
- maximum axial bounding-box diagonal in mm;
- craniocaudal extent in mm;
- detector rank score when the source was an automatic candidate;
- model provenance;
- reader-consensus Dice when valid staged references are available;
- explicit actions to compare overlays and save the DICOM SEG to the case; and
- a session-only review disposition: **Supported**, **Dismiss mark**, or **Uncertain**.

The disposition is review state, not a diagnosis or signed report. The mask is described
as following the selected box; it does not confirm a nodule, malignancy, normality, or any
other abnormality.

## Reader comparison: what the Dice means

Reader Dice is available only for cases with staged source-referenced DICOM SEG objects.
The comparison has safeguards that matter:

1. A target segment is selected independently for each reader from the seed source SOP
   Instance UID and prompt center/box **before model inference**. The prediction is not
   available during target selection, so the implementation cannot choose the reader
   contour that gives the model its best score after the fact.
2. Every reference frame is aligned through its explicit referenced source SOP Instance
   UID. Frame order, `InstanceNumber`, and proximity are never accepted as substitutes.
3. For the four-reader LIDC demo, the consensus mask requires at least three reader votes
   at a voxel. The API returns the consensus Dice plus per-reader measurements.
4. If identity checks fail, the prompt matches no reader target, or no valid consensus
   exists, the panel withholds the Dice instead of inventing an alignment.

This is an overlap measurement against the staged reference for the prompted object. It
does not convert the 27-case detector result into segmentation validation and must not be
described as general MedSAM2 accuracy.

## Persistence has two distinct meanings

The UI makes this distinction explicit:

- **Review continuity:** detector results, selected candidate, completed outlines, and
  candidate dispositions live in a series-keyed in-memory store. They survive panel-tab
  changes and React remounts in the current viewer session. They do not survive a full
  browser reload.
- **Clinical artifact:** the remote API writes a source-bound DICOM SEG and returns its
  SOP/Series UIDs, SHA-256, byte length, and download path. **Save DICOM SEG to case**
  downloads those exact bytes through the same-origin API proxy and uploads them through
  the local Orthanc REST proxy. After reloading the study, OHIF can hydrate the durable
  SEG from the case.

The download route resolves the artifact by DICOM identity inside the source series'
`derived` directory. It accepts only regular non-symlink DICOM Segmentation Storage
objects that reference that source SeriesInstanceUID.

## Visual system

The v4 console replaces the earlier patient-oriented cream/gold/serif treatment with a
low-glare clinical reading-room system:

- quiet scan-black and graphite surfaces;
- mineral teal for active workflow state;
- restrained amber for reference evidence;
- red/green only for real error/success states;
- IBM Plex Sans for interface copy and IBM Plex Mono for measurements, scores, versions,
  and identifiers;
- compact radiology controls using conventional names such as **Window / Level**; and
- a thin functional progress rail as the product's signature visual, rather than
  decorative cards, gradients, or oversized branding.

The right rail starts at 376 px, can be resized, and keeps the left series panel closed by
default so the images remain primary. The toolbar exposes conventional clinical review
tools plus one specialized **Refine with box** action.

## Architecture and ownership

The clinical UI is implemented through OHIF's supported extension/mode surfaces:

| File | Responsibility |
|---|---|
| `viewer/ohif/extensions/extension-doctor-assistant/src/ClinicalReviewRail.tsx` | Detect/Inspect/Compare presentation and clinical copy |
| `viewer/ohif/extensions/extension-doctor-assistant/src/DoctorAssistantPanel.tsx` | series resolution, API lifecycle, event subscriptions, save/compare actions |
| `viewer/ohif/extensions/extension-doctor-assistant/src/useReviewWorkflowStore.ts` | series-isolated in-session review continuity |
| `viewer/ohif/extensions/extension-doctor-assistant/src/clinicalConsole.css` | scoped clinical tokens and rail styling |
| `viewer/ohif/extensions/extension-doctor-assistant/src/lungNoduleCandidateEvents.ts` | candidate and outline lifecycle events |
| `viewer/ohif/extensions/extension-doctor-assistant/src/registerReviewWorkflowEvents.ts` | mode-lifetime result capture across panel-tab unmounts |
| `viewer/ohif/extensions/extension-doctor-assistant/src/tools/registerMedSAMBoxTool.ts` | source-slice navigation, explicit outline request, remote inference, Cornerstone labelmap |
| `viewer/ohif/extensions/extension-doctor-assistant/src/apiClient.ts` | typed detector, segmentation, artifact-download, and local-Orthanc calls |
| `viewer/ohif/modes/doctor-assistant/src/index.ts` | panel layout, tool selection, and doctor-facing mode name |
| `api/reference_evaluation.py` | prompt-anchored reader matching and Dice calculation |
| `api/lung_nodule_cache.py` | atomic source/model-bound detector run cache and restart recovery |
| `api/routes/segmentation.py` | volume segmentation, DICOM SEG metadata/download, optional reference comparison |

`onModeEnter` adds `clinique-amina-clinical` to the document body and `onModeExit`
removes it. The dark tokens therefore apply only to this review mode; they do not force a
global OHIF fork. Image rendering and the stock segmentation panel remain OHIF-owned.

## Operational constraint for the current Mac

The 16 GB Mac runs only native Orthanc and the UI. Torch/MONAI/MedSAM2 inference stays on
the single-worker Colab API. `scripts/start_ohif_with_remote_api.sh` caps the UI compiler
at a 2 GB Node heap. Do not run local volumetric inference as part of UI verification.

## Acceptance path

Use the proven LIDC-IDRI-0117 demo and verify this sequence:

1. Start native Orthanc.
2. Upload the prepared CT and four reader SEG objects.
3. Start OHIF against the active Colab/ngrok endpoint.
4. Open the printed study URL and run **Find nodule candidates**.
5. Select a candidate and confirm the viewport moves without starting segmentation.
6. Choose **Generate 3D outline** and confirm the locked/running state.
7. Confirm the labelmap, measurements, reader-consensus evidence when available, and
   review-disposition controls remain visible after switching panel tabs.
8. Choose **Save DICOM SEG to case**, reload the study, and verify the durable SEG loads
   from local Orthanc.

Success proves the complete software workflow on the demonstration case. It does not
establish clinical performance beyond the separately documented evaluation.
