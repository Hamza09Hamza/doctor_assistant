# High-quality LIDC CT demo on Apple Silicon

> This page covers the original local-MLX **segmentation** case. `LIDC-IDRI-0686`
> appears in the MONAI lung-nodule detector's official fold-0 training list, so it must
> not be used as independent evidence for that detector. For the automatic
> detector -> MedSAM2 Colab demo, use `scripts/prepare_lidc_nodule_detector_demo.py`
> and `docs/COLAB_INFERENCE_SERVER.md`; that path pins `LIDC-IDRI-0117` and records its
> more limited known-overlap check explicitly.

This demo uses the de-identified `LIDC-IDRI-0686` chest CT from the NCI Imaging
Data Commons / The Cancer Imaging Archive. It is a 238-slice, 512 x 512 CT with
0.703125 mm in-plane spacing and 1.25 mm slice thickness. A radiologist DICOM SEG
for an eight-slice lung nodule is included as the reference mask.

Sources and licenses:

- [TCIA LIDC-IDRI collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
  (`CC BY 3.0`)
- [IDC download documentation](https://learn.canceridc.dev/data/downloading-data)
- [Official MedSAM2 repository](https://github.com/bowang-lab/MedSAM2)
- [MedSAM2 model files](https://huggingface.co/wanglab/MedSAM2/tree/main)
  (`CC BY-SA 4.0` on the model repository)

The raw scan, model files, generated database, and predictions are intentionally
gitignored.

## 1. Prepare or verify the study

From the repository root, in the Python 3.14 MLX environment:

```bash
source .venv-mlx/bin/activate
python scripts/prepare_lidc_interactive_demo.py
```

The command is idempotent. It verifies the study and radiologist reference, stages
the 238 CT instances, creates a local SQLite database, renders a lung-window PNG with
the radiologist mask in red and prompt box in green, and prints the exact prompt.
The local manifest is:

```text
data/validation/lidc_idri_0686/interactive_demo_manifest.json
```

The immediately viewable quality-check image is:

```text
data/validation/lidc_idri_0686/seed_instance_86_lung_window.png
```

For the currently staged copy, the key values are:

```text
API series ID:  a3b981b2abec457f82a55c8ebadccef5
Seed instance:  86
Seed SOP UID:   1.3.6.1.4.1.14519.5.2.1.6279.6001.176313405617208470254714639119
Box (x1,y1,x2,y2): [88, 247, 111, 268]
Window:         level -600, width 1500
```

## 2. Run the one-command MLX acceptance test

The official CT-lesion checkpoint has been converted locally to MLX as:

```text
checkpoints/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors
```

If that gitignored file is missing, download the official checkpoint and convert it:

```bash
mkdir -p checkpoints
curl -L \
  https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_CTLesion.pt \
  -o checkpoints/MedSAM2_CTLesion.pt
mlx-sam-convert \
  --checkpoint checkpoints/MedSAM2_CTLesion.pt \
  --model-id facebook/sam2.1-hiera-tiny \
  --output checkpoints/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors
```

The official Torch checkpoint used here has SHA-256
`78f7e125418dfd6fec22f4afe90bcd85cb1d4423d0a9df36f7a87ed63aa1a5f5`.

Run the actual MLX model through the FastAPI route, create DICOM SEG, and compare the
result with the radiologist mask:

```bash
python scripts/validate_lidc_mlx_interactive.py
```

The command takes roughly 30-40 seconds on the current Mac and writes its measurements
to `data/validation/lidc_idri_0686/mlx_interactive_result.json`. It does not require a
running web server, Orthanc, or OHIF.

## 3. Start the API with native Metal inference

Start the API from the repository root:

```bash
source .venv-mlx/bin/activate
export DATABASE_URL="sqlite:///$(pwd)/api_storage/lidc_interactive_demo.sqlite"
export STORAGE_DIR="$(pwd)/api_storage/lidc_interactive_demo"
export MEDSAM2_BACKEND=mlx
export MEDSAM2_MLX_MODEL="$(pwd)/checkpoints/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors"
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000
```

If `.venv-mlx` does not exist yet:

```bash
python3.14 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -r requirements-macos-mlx.txt
```

## 4. Test the live API-to-DICOM-SEG path

In a second terminal:

```bash
curl -X POST \
  http://localhost:8000/v1/series/a3b981b2abec457f82a55c8ebadccef5/segment-volume \
  -H 'Content-Type: application/json' \
  -d '{
    "sop_instance_uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.176313405617208470254714639119",
    "box_xyxy": [88, 247, 111, 268],
    "window_center": -600,
    "window_width": 1500,
    "segment_label": "Interactive lung nodule",
    "publish_to_orthanc": false
  }'
```

This loads all 238 source instances, runs bidirectional propagation on Apple Metal,
computes physical measurements, and writes a source-referenced DICOM SEG below the
series `derived/` directory. On the current M-series Mac the latest measured run took
34.941 seconds.

## 5. Inspect it in OHIF

Docker/Orthanc must be running before this step. Upload the CT and radiologist SEG:

```bash
python scripts/prepare_lidc_interactive_demo.py --upload-orthanc
```

The script prints the exact OHIF URL. In OHIF:

1. Select the lung window (`W 1500 / L -600`).
2. Navigate to instance 86.
3. Select **3D Segment (draw box)**.
4. Draw a tight box around the peripheral nodule at approximately
   `x=88..111, y=247..268`.
5. Scroll through adjacent slices, then reload the study to verify that the DICOM SEG
   persisted in Orthanc.

## Current accuracy result

The end-to-end mechanics are verified, but this converted checkpoint is not yet a
medical-quality acceptance result on this target:

| Measurement | Result |
|---|---:|
| Runtime | 34.941 s |
| Predicted / reference voxels | 942 / 833 |
| Predicted / reference slices | 3 / 8 |
| Full-volume Dice | 0.35155 |
| Seed-slice Dice | 0.45070 |
| Predicted volume | 0.582 mL |

Stock SAM2 produced a near-zero full-volume Dice on the same prompt. Converting the
medical checkpoint is a real improvement, but it still under-segments through the
z-axis. The next acceptance step is to run the official Torch MedSAM2 CT script on
this exact volume and prompt, then compare its mask with the MLX output. That isolates
checkpoint-conversion parity from model generalization before any accuracy claim.
