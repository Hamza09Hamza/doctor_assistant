# Colab GPU inference server through ngrok

This test mode keeps DICOM viewing on the Mac and moves both heavy CT models to one
Colab GPU runtime:

```text
Mac OHIF -> same-origin OHIF proxy -> ngrok HTTPS -> Colab FastAPI
                                                       |-> MONAI 3D nodule detector
                                                       +-> MedSAM2 volume segmenter
    |
    +-> local Orthanc for source DICOM
```

Colab downloads the pinned LIDC DICOM itself. The two copies have identical SOP
Instance UIDs, so masks returned by the remote API map onto the source images already
loaded in local OHIF. The CT volume is not uploaded from the Mac for each prompt.

This is an interactive test server. Colab-managed runtimes are temporary and may stop,
and the ngrok URL exists only while its notebook session is running.

## 1. Start local DICOM services

### Native macOS (no Docker)

Install the official universal Orthanc package once, then start it in a dedicated
terminal:

```bash
bash scripts/install_orthanc_macos.sh
bash scripts/start_orthanc_macos.sh
```

The download is about 320 MB. The server itself is lightweight and does not load any
AI model. The installer keeps the package and Orthanc database under the ignored
`data/` directory. Leave the second command running while testing.

If a network filter blocks the official Orthanc host, the Colab notebook downloads the
archive and serves it through the already-running ngrok API tunnel. Copy the printed
`ORTHANC_ARCHIVE_URL=... bash scripts/install_orthanc_macos.sh` command into the Mac
terminal. This avoids Colab's unreliable browser-file transfer and does not create a
second ngrok tunnel.

Create the lightweight, upload-only Mac environment once. It deliberately excludes
Torch, MONAI, MLX, and every inference runtime:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-demo.txt
```

Then, in another terminal, stage and upload the pinned detector demo:

```bash
source .venv-demo/bin/activate
python scripts/prepare_lidc_nodule_detector_demo.py --upload-orthanc
```

The upload command prints the LIDC study URL. No local model inference is performed.

### Docker alternative

If Docker Desktop is installed, the original route remains available:

```bash
docker compose -f deployments/docker-compose.yml up -d orthanc
source .venv-mlx/bin/activate
python scripts/prepare_lidc_interactive_demo.py --upload-orthanc
```

## 2. Start Colab

Open `notebooks/medsam2_inference_server_colab.ipynb` in Google Colab and select an L4
or other Ampere-or-newer GPU runtime with at least 18 GiB GPU memory.

Run the cells in order. The temporary test token is configured directly in the
notebook, so the Colab Secrets panel is not required. The notebook:

1. checks GPU capability and memory before installing or loading the model;
2. clones this repository's `main` branch;
3. installs the official MedSAM2 package without building its optional CUDA extension;
4. downloads `MedSAM2_CTLesion.pt` and the MONAI `lung_nodule_ct_detection` bundle once;
5. downloads and stages the pinned 122-slice `LIDC-IDRI-0117` CT plus all four reader
   DICOM SEG objects;
6. fetches the Orthanc macOS ZIP once and exposes it at a dedicated route on the same
   ngrok tunnel;
7. starts one Uvicorn worker and gives detection and segmentation one shared inference
   slot so they cannot overlap;
8. starts ngrok and prints the public HTTPS API URL plus the Mac installer command.

The token is used to create the ngrok tunnel but is not printed by the notebook.

## 3. Point local OHIF at Colab

Copy the printed ngrok URL and run on the Mac:

```bash
bash scripts/start_ohif_with_remote_api.sh https://YOUR-NGROK-DEV-DOMAIN
```

The launcher verifies `/health`, caps the OHIF Node process at 2 GiB, keeps the local
Orthanc proxy, and points only the doctor-assistant API proxy at Colab. Open the study
URL printed by the Orthanc upload step and use lung window `W 1500 / L -600`.

The primary demo is a deliberate three-stage clinician review:

1. In the AI Review panel, click **Find nodule candidates**.
2. Wait for the complete-volume detector pass. The UI intentionally blocks a second
   heavy request while Colab is working.
3. Select a candidate. OHIF jumps to its source slice without starting another model,
   so the location can be inspected first.
4. Click **Generate 3D outline** only after inspecting the source image.
5. Review the persistent measurements and the prompt-matched >=3-of-4-reader consensus
   Dice/volume. Open **Compare overlays** to use OHIF's segmentation visibility controls.
6. Choose **Supported**, **Dismiss mark**, or **Uncertain** as a session-only review
   disposition. This records how the doctor handled false candidates; it is not a diagnosis.
7. Click **Save DICOM SEG to case** to copy the generated standards-valid object from
   Colab into the Mac's local Orthanc. Reload the study once to hydrate that durable copy.

The detector produces a review shortlist, not a diagnosis. The completed 27-case run
found 21/23 consensus nodules at its fixed `0.3` threshold with 2.15 false candidates
per scan. An empty shortlist does not prove that a scan is clear. Manual **Refine with
box** remains available, but a manual mask is only a prompt-following
outline and is not evidence that the selected tissue is abnormal.

Expected `/health` fields before opening OHIF are `ready: true`,
`medsam2_configured: true`, `medsam2_loaded: true`,
`lung_nodule_detector_configured: true`, `lung_nodule_detector_loaded: true`,
`max_concurrent_inferences: 1`, and `orthanc_publication: disabled`. “Configured” means
the API registered both adapters; “loaded” means the Uvicorn subprocess built both
networks and validated their checkpoints without duplicating them in the notebook
process. The first real requests remain the end-to-end inference checks.

The remote API writes its DICOM SEG inside the temporary Colab filesystem, returns RLE
masks for immediate OHIF painting, and exposes the exact DICOM bytes through a
series-bound artifact endpoint with SHA-256 metadata. It deliberately reports direct
Orthanc publication as disabled: `localhost:8042` inside Colab is not the Mac. The
viewer therefore downloads the object through its remote API proxy and POSTs it through
the separate local-Orthanc REST proxy only when **Save DICOM SEG to case** is selected.

## Resource behavior

- The Mac runs OHIF and Orthanc only; it does not import Torch or load either model.
- Uvicorn uses one worker.
- A shared non-blocking inference lock returns HTTP 429 if detection and segmentation
  would overlap.
- The notebook refuses GPUs older than compute capability 8 or below 18 GiB.
- Do not point the detector at a local Python runtime on the 16 GiB Mac; this path was
  deliberately designed to keep Torch/MONAI inference in Colab.
- Stop the monitoring cell and run the cleanup cell to close ngrok and the API process.

ngrok's HTTPS agent tunnel is outbound-only and receives an automatically managed TLS
certificate. Its free development endpoint currently has no endpoint timeout, though
account transfer/request limits still apply. Google documents that managed free Colab
runtimes may terminate web-service-style usage; use a runtime with positive compute
units when stability matters.

Sources:

- [ngrok secure tunnels](https://ngrok.com/docs/guides/share-localhost/tunnels)
- [ngrok free-plan limits](https://ngrok.com/docs/pricing-limits/free-plan-limits)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [MONAI lung-nodule detector model card](https://huggingface.co/MONAI/lung_nodule_ct_detection/blob/main/docs/README.md)
- [Official MedSAM2 installation](https://github.com/bowang-lab/MedSAM2#installation)
