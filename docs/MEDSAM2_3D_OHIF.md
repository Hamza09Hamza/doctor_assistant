# MedSAM2: one box to a full 3D DICOM SEG

## What this path does

The `3D Segment (draw box)` OHIF tool now attempts the full-volume workflow first:

```text
box on one CT/MR slice
        -> POST /v1/series/{id}/segment-volume
        -> SAM2 MLX or MedSAM2 forward + reverse propagation
        -> per-SOP masks + physical measurements
        -> standards-based semiautomatic DICOM SEG
        -> Orthanc upload + immediate OHIF labelmap
```

The response includes the segmented slice count, voxel count, volume in mL, an axial
bounding-box diagonal in mm, DICOM SEG UIDs, and the Orthanc publication status.  Masks
are keyed by source SOP Instance UID, so an ascending/descending viewer-stack reversal
cannot move a result onto the wrong slice.

If the 3D backend is not configured, OHIF falls back only on HTTP 503 to the existing 2D
MedSAM endpoint and clearly labels the result as single-slice. Geometry and inference
errors do not silently fall back.

## Apple Silicon / MLX (the local Mac path)

On an Apple-Silicon Mac, `MEDSAM2_BACKEND=auto` selects the MLX adapter. The adapter
uses `mlx-sam`, whose video predictor provides the same box-prompt and bidirectional
memory-propagation operations this workflow needs without PyTorch at inference time.
Its current release requires Python 3.14.

```bash
python3.14 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -r requirements-macos-mlx.txt

# Optional native-Metal proof before starting the API. The first run downloads an
# approximately 90 MB 16-bit tiny checkpoint; this validates runtime wiring, not
# medical accuracy.
python scripts/smoke_sam2_mlx.py

export MEDSAM2_BACKEND=mlx
export MEDSAM2_MLX_MODEL=avbiswas/sam2.1-hiera-small-mlx-16bit
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000
```

That published checkpoint is stock SAM2.1 running natively in MLX. It proves the local
3D interaction/runtime, but it is not MedSAM2 and must not be presented as medically
fine-tuned. `mlx-sam` also converts local SAM2.1-compatible Torch checkpoints. MedSAM2's
`sam2.1_hiera_t512` architecture is the tiny Hiera family, so the candidate conversion
is:

```bash
mlx-sam-convert \
  --checkpoint /path/to/MedSAM2_latest.pt \
  --model-id facebook/sam2.1-hiera-tiny \
  --output /path/to/medsam2_tiny_512_mlx.safetensors

export MEDSAM2_MLX_MODEL=/path/to/medsam2_tiny_512_mlx.safetensors
```

Treat that conversion as experimental until the same CT volume and box produce a close
mask against the official Torch MedSAM2 runtime. Record Dice, differing voxels, and the
converted checkpoint hash. The application reports `sam2-mlx:<checkpoint>` in its
result/DICOM provenance so it cannot be confused with the verified Torch backend. See
[`LIDC_INTERACTIVE_DEMO.md`](LIDC_INTERACTIVE_DEMO.md) for the pinned real CT, exact
prompt, local database, and copy-paste test commands.

## NVIDIA GPU environment

MedSAM2 is deliberately not vendored into this repository. Use the official upstream
repository and checkpoint. Its documented environment is Python 3.12, PyTorch 2.5.1,
and CUDA 12.4 on Linux.

```bash
git clone https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2
cd /opt/MedSAM2
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
bash download.sh

cd /path/to/Doctor-Assistant
pip install -r requirements.txt -r requirements-api.txt
```

Run the API from the same environment:

```bash
export DATABASE_URL=postgresql+psycopg2://doctor_assistant:doctor_assistant@localhost:5432/doctor_assistant
export ORTHANC_URL=http://localhost:8042
export ORTHANC_USERNAME=doctor_assistant
export ORTHANC_PASSWORD=doctor_assistant
export MEDSAM2_CHECKPOINT_PATH=/opt/MedSAM2/checkpoints/MedSAM2_latest.pt
export MEDSAM2_MODEL_CONFIG=configs/sam2.1_hiera_t512.yaml
export MEDSAM2_BACKEND=torch
export MEDSAM2_DEVICE=cuda

uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000
```

`MEDSAM_CHECKPOINT_PATH` or `MEDSAM_MODEL_ID` may also be set to retain the 2D fallback.

For a temporary GPU server that keeps Torch off a 16 GB Mac, see
[`COLAB_INFERENCE_SERVER.md`](COLAB_INFERENCE_SERVER.md). Its notebook starts this same
API with one worker behind an ngrok HTTPS tunnel and rejects overlapping volume jobs.

## Demo sequence

1. Start Postgres, Orthanc, and OHIF with `deployments/docker-compose.yml`.
2. Start the API in the MedSAM2 GPU environment above.
3. Put a de-identified CT or MR study in Orthanc and import its Study Instance UID with
   `POST /v1/studies/import`.
4. Open that study in the Clinique Amina OHIF mode.
5. Select `3D Segment (draw box)` and draw a tight box around one lesion on a clear
   middle slice.
6. Wait for the success notification. Scroll through the stack to inspect propagation;
   the notification reports slices, mL, axial span, and DICOM SEG publication.
7. Reload the study or select the new SEG series to confirm the persisted Orthanc object
   renders through OHIF's standard DICOM SEG path.

## Acceptance gate for the first real win

Use one clean LIDC lesion with a radiologist consensus mask and record all of:

- the box-drawing interaction and mask propagation across slices;
- a non-empty DICOM SEG whose referenced series and frames validate;
- the same SEG visible after an OHIF reload, proving Orthanc persistence;
- Dice and volume error against the consensus mask;
- GPU, checkpoint filename/hash, inference time, and source case identifier.

The code-level and synthetic-DICOM tests prove the contract and geometry plumbing. They
do not establish model accuracy or live viewer alignment; those claims begin only after
the visual LIDC run above.

## Validation completed on Apple Silicon

On 2026-08-10, the native Metal smoke command above ran the 16-bit SAM2.1 tiny model
across all five synthetic slices: 6,839 foreground voxels in 1.56 seconds with a warm
model cache. A separate integration run then exercised the real MLX model through the
FastAPI endpoint on a five-instance synthetic CT series and produced:

- HTTP 200 with masks on all five source SOP instances;
- 13,202 voxels and a spacing-derived volume of 14.786 mL;
- a readable DICOM SEG whose referenced Series Instance UID matched its CT source.

The focused automated suite passes 24 tests. The three modified OHIF TypeScript files
also pass an isolated Babel TypeScript/TSX parse.

The real LIDC-IDRI-0686 run now also proves that the converted `MedSAM2_CTLesion`
checkpoint loads in MLX, processes all 238 CT slices, and produces a valid DICOM SEG in
34.941 seconds. Its full-volume Dice against the eight-slice radiologist annotation was
0.35155, so this is an end-to-end success rather than an accuracy acceptance. Live OHIF
rendering and Orthanc persistence remain to be verified after that local infrastructure
is started.
