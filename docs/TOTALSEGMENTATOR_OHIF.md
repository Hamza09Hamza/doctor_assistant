# TotalSegmentator in Clinique Amina/OHIF

This is the shortest end-to-end path from a real CT to a colored anatomy overlay in
the project viewer.

## What you need

- One **de-identified** DICOM CT series, as a folder or ZIP.
- Google Colab with an L4 GPU for segmentation.
- Docker on the local computer for Orthanc and OHIF.

The scripts do not de-identify DICOM. Do not use identifiable patient data. The
result is anatomical visual QC only, not a diagnosis.

## 1. Create the DICOM SEG in Colab

Open `notebooks/totalsegmentator_dicom_seg_colab.ipynb` from the
`classifier-evaluation-colab` branch.

In its configuration cell:

1. Set `DICOM_INPUT` to the DICOM folder or ZIP in Google Drive.
2. Set `CONFIRM_DEIDENTIFIED = True` after checking the data.
3. Leave `USE_FAST_MODEL = False` on an L4.
4. Run all cells.

The final cell prints the path to `ohif_viewer_bundle.zip`. The ZIP contains only the
selected CT series, the generated DICOM SEG, and its run manifest. Download that ZIP
to the local computer.

If the notebook finds multiple substantial CT series, it stops and lists their
Series Instance UIDs and slice counts. Copy the desired UID into
`SERIES_INSTANCE_UID` and rerun the inference cell. It does not silently choose between
multiple diagnostic series.

## 2. Start Orthanc and Clinique Amina

From the repository root:

```bash
docker compose -f deployments/docker-compose.yml up -d orthanc ohif
```

This exposes:

- Orthanc REST: `http://localhost:8042`
- Clinique Amina/OHIF: `http://localhost:3000`

The viewer uses a same-origin Nginx proxy for authenticated DICOMweb traffic. The
browser never receives the Orthanc password.

## 3. Publish and open the exact study

Install the lightweight local dependencies if they are not already available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pydicom httpx
```

Then publish the bundle:

```bash
python scripts/publish_dicom_seg_to_orthanc.py \
  --bundle /absolute/path/to/ohif_viewer_bundle.zip
```

The publisher validates that:

- the source is a CT DICOM series;
- the generated object is DICOM Segmentation Storage;
- its Study Instance UID matches the CT;
- it references the exact CT Series Instance UID; and
- it contains non-empty segments, frames, and pixel data.

It uploads the CT first, then the SEG, verifies the study through QIDO-RS, and prints
one URL like:

```text
http://localhost:3000/doctor-assistant/orthancProxy?StudyInstanceUIDs=...
```

Open that URL. Select the `SEG` series in the left study browser if it is not hydrated
automatically. The read-only Segmentation panel on the right lets you toggle anatomy
labels and opacity over the source CT.

## What success means

Success establishes a working software bridge:

```text
DICOM CT -> TotalSegmentator -> DICOM SEG -> Orthanc -> OHIF overlay
```

It does not establish clinical accuracy. A person must still inspect alignment and
organ boundaries, and the model must be evaluated separately before any clinical use.
