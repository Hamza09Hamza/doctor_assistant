# TotalSegmentator in Clinique Amina/OHIF

## The pathology demo to run now

The anatomy overlay below proves the viewer connection, but it does **not** highlight
disease. The next benchmark is automatic lung-nodule segmentation on a public LIDC CT
with a radiologist reference mask.

1. Open `notebooks/lung_nodule_segmentation_colab.ipynb` in Colab.
2. Select **Runtime -> Change runtime type -> L4 GPU**.
3. Run every cell from top to bottom. There is no `DICOM_INPUT` to configure.
4. Download the printed `ohif_ai_vs_expert_bundle.zip` from Google Drive.
5. On the laptop, publish it to the already-running native Orthanc:

```bash
python scripts/publish_dicom_seg_to_orthanc.py \
  --bundle /path/to/ohif_ai_vs_expert_bundle.zip
```

Open the URL printed by the publisher. The bundle contains the same CT plus two
overlays: the model prediction and one radiologist annotation. Use **SEG LOAD** in
OHIF if the overlays do not appear automatically.

The notebook deliberately fails if the model returns no encoded `lung_nodules`
segment. A successful run proves the data/model/OHIF path and enables visual
comparison; it does not prove clinical accuracy or cover tumors in other organs.

## The earlier anatomy demo

This is the shortest end-to-end path from a real CT to a colored anatomy overlay in
the project viewer. Start with the public demo below: it supplies the CT itself, so
you do **not** need to find a scan or set `DICOM_INPUT`.

## What you need

- An NVIDIA GPU with at least 6 GB VRAM. The full model has been run successfully on
  this project's RTX 3050 6 GB laptop GPU by enabling TotalSegmentator's split mode.
- Orthanc on the local computer (the native service on port 8042 is sufficient).
- The Clinique Amina/OHIF development app on port 3000.

The supplied public CT is already de-identified. The result is anatomical visual QC
only, not a diagnosis.

## 1. Run the proven local public demo

From the repository root, using the separate CT environment:

```bash
source .venv/bin/activate
python -m pip install highdicom==0.27.0
python -u scripts/run_local_totalsegmentator_ohif_demo.py
```

The script automatically:

1. Verifies that PyTorch can see the NVIDIA GPU.
2. Downloads or reuses a pinned, public, de-identified 135-slice CT from the OHIF
   test-data repository.
3. Validates its modality, Study/Series UIDs, unique instances, and
   `PatientIdentityRemoved=YES` declaration.
4. Runs the full 1.5 mm model in split mode to fit 6 GB VRAM.
5. Creates and validates a standards-based DICOM SEG and portable viewer ZIP.

The first run downloads about 1.2 GB of model weights. They remain under
`results/totalsegmentator_weights`, so later runs reuse them. The successful local run
on 2026-08-04 produced 91 non-empty anatomy segments and 3,287 SEG frames. Its final
output is:

```text
results/totalsegmentator_ohif_local_demo/study_37445c903b13/full_1p5mm/ohif_viewer_bundle.zip
```

If a different 6 GB card runs out of memory, retry with `--fast`. Do not use CPU for
this demo unless there is no GPU; it is unnecessarily slow.

## 2. Start Orthanc and Clinique Amina

Containers are optional for this local test. The native Orthanc service already
installed on this machine is sufficient and is not used for model inference.

Check the native server first:

```bash
curl http://localhost:8042/system
```

If it is stopped, start it:

```bash
sudo systemctl enable --now orthanc
```

This exposes:

- Orthanc REST: `http://localhost:8042`
- Clinique Amina/OHIF: `http://localhost:3000`

Start the OHIF development app separately on port 3000 using the repository's viewer
instructions. Its development proxy forwards DICOMweb requests to Orthanc.

## 3. Publish and open the exact study

With `.venv` still active:

```bash
python scripts/publish_dicom_seg_to_orthanc.py \
  --bundle results/totalsegmentator_ohif_local_demo/study_37445c903b13/full_1p5mm/ohif_viewer_bundle.zip
```

The publisher uploads the CT first and then the SEG, verifies both through Orthanc,
and prints the exact Clinique Amina URL. Open that URL. Select the `SEG` series in the
left study browser if it does not load automatically. The read-only Segmentation panel
on the right lets you toggle anatomy labels and opacity over the CT.

## Using your own CT later

The public demo proves the software connection without asking you for medical data.
Only after that works should you substitute your own **de-identified** DICOM CT folder
or ZIP. The scripts do not remove patient identity.

For a local run:

```bash
python -u scripts/run_totalsegmentator_dicom_seg.py \
  --dicom-input /path/to/deidentified/ct-folder-or.zip \
  --output-dir results/my_ct_seg \
  --work-dir results/my_ct_work \
  --confirm-deidentified \
  --force-split
```

The optional Colab notebook, `notebooks/totalsegmentator_dicom_seg_colab.ipynb`, is
for a computer without a suitable local NVIDIA GPU. In that notebook, `DICOM_INPUT`
means the path to the user's own de-identified CT folder or ZIP in Google Drive. It is
not needed for the public local demo.

## What success means

Success establishes a working software bridge:

```text
DICOM CT -> TotalSegmentator -> DICOM SEG -> Orthanc -> OHIF overlay
```

It does not establish clinical accuracy. A person must still inspect alignment and
organ boundaries, and the model must be evaluated separately before any clinical use.
