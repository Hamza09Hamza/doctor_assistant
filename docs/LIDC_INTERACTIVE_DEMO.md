# LIDC detector -> MedSAM2 clinician-review demo
This is the canonical end-to-end acceptance case for the current Doctor Assistant.
It keeps all Torch/MONAI inference in Colab and uses the Mac only for native Orthanc,
OHIF, and standards-valid DICOM objects.

## Pinned public case

- Case: `LIDC-IDRI-0117`
- StudyInstanceUID: `1.3.6.1.4.1.14519.5.2.1.6279.6001.336137933660116977458622909107`
- CT SeriesInstanceUID: `1.3.6.1.4.1.14519.5.2.1.6279.6001.295958572786158575287945391206`
- CT instances: 122
- References: four individual reader DICOM SEG objects
- Source: NCI Imaging Data Commons / The Cancer Imaging Archive
- License: CC BY 3.0

The project derives a voxel consensus only during comparison by requiring at least
three of the four prompt-matched readers. The four source objects must not be described
as one pre-existing consensus SEG.

`LIDC-IDRI-0117` was selected after the frozen 27-series run as a visually clean
demonstration case. Its recorded detector result was TP=1, FP=0, FN=0 at the fixed
0.30 threshold. That is a reproduction target for this demo, not an aggregate claim.

## Start the demo

Run the Colab notebook first:

```text
notebooks/medsam2_inference_server_colab.ipynb
```

Use an L4 or another Ampere-or-newer runtime with at least 18 GiB GPU memory. When the
notebook prints `COLAB API READY`, run these in three Mac terminals from the repository
root.

Terminal A:

```bash
bash scripts/start_orthanc_macos.sh
```

Terminal B:

```bash
source .venv-demo/bin/activate
python scripts/prepare_lidc_nodule_detector_demo.py --upload-orthanc
```

If `.venv-demo` does not exist yet, create it once with Python 3.13 and install
`requirements-macos-demo.txt`. This environment only stages and uploads DICOM; it does
not contain a model runtime.

Terminal C:

```bash
bash scripts/start_ohif_with_remote_api.sh https://YOUR-NGROK-DEV-DOMAIN
```

Open the exact Clinique Amina URL printed by terminal B. No model inference runs on
the Mac in this workflow.

## Doctor workflow

Use lung window `W 1500 / L -600`, then follow the persistent review rail:

1. **Detect** — select **Find nodule candidates** and wait for the complete-volume pass.
2. **Inspect** — select a candidate to navigate to its source slice. Selection alone does
   not call MedSAM2, so the image can be reviewed first.
3. Select **Generate 3D outline**. The panel remains in a waiting state and prevents a
   second prompt until the result returns.
4. **Compare** — inspect slices, volume, maximum axial bounding-box diagonal,
   craniocaudal extent, and the prompt-matched reader consensus result.
5. Select **Compare overlays** to inspect AI and reader labelmaps in OHIF.
6. Record the candidate as **Supported**, **Dismiss mark**, or **Uncertain** for this
   viewer session. These are review dispositions, not diagnoses.
7. Select **Save DICOM SEG to case**. The viewer downloads the source-referenced object
   from Colab and uploads it to the Mac's Orthanc. Reload once to hydrate the durable SEG.

The expected live result is not a hardcoded Dice value: the detector and MedSAM2 must
actually run. Acceptance requires all of the following:

- one detector candidate is reproduced at the fixed 0.30 score threshold;
- candidate selection reaches the correct source SOP before outline generation;
- a second heavy request is rejected while the shared inference slot is occupied;
- the returned mask appears on the correct CT slices;
- four reader annotations are matched by source SOPInstanceUID, never file order;
- a >=3-of-4 consensus Dice and consensus volume appear when the prompt matches the
  annotated nodule;
- a random prompt that misses the reader targets does not receive a forced consensus;
- the downloaded DICOM SEG is non-empty, references the source CT series, and is
  accepted by local Orthanc.

## Evidence language

The current bounded project result is:

> At the fixed detector-score threshold of 0.30, 21 of 23 algorithmically derived
> >=3-of-4-reader consensus nodules were detected across 27 eligible LIDC CT series,
> with 58 false candidates (2.15 per scan). Eligible CT SeriesInstanceUIDs were absent
> from LUNA16's complete published 888-series corpus.

This is a small single-threshold research evaluation. It is not accuracy, a cancer
probability, scan clearance, external clinical validation, or a FROC result. MedSAM2
follows the selected box and does not confirm whether the outlined structure is a
nodule or malignancy.

## Historical 0686 MLX result

`LIDC-IDRI-0686` remains useful only as a local implementation smoke test for the old
prompted-segmentation path. It is present in the MONAI detector's official fold-0
training list and cannot support detector validation. The historical converted-MLX
result (full-volume Dice 0.35155, predicted volume 0.582 mL) does not describe the
current 0117 detector -> MedSAM2 workflow.

Sources:

- [TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- [IDC download documentation](https://learn.canceridc.dev/data/downloading-data)
- [MONAI lung-nodule detector](https://huggingface.co/MONAI/lung_nodule_ct_detection)
- [Official MedSAM2 repository](https://github.com/bowang-lab/MedSAM2)
