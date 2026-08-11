# Colab GPU inference server through ngrok

This test mode keeps DICOM viewing on the Mac and moves all MedSAM2 Torch inference to
a Colab GPU runtime:

```text
Mac OHIF -> same-origin OHIF proxy -> ngrok HTTPS -> Colab FastAPI -> MedSAM2 GPU
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

In another terminal, upload the already prepared LIDC study:

```bash
source .venv-mlx/bin/activate
python scripts/prepare_lidc_interactive_demo.py --upload-orthanc
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
4. downloads only `MedSAM2_CTLesion.pt`, not every upstream checkpoint;
5. downloads and stages the pinned 238-slice LIDC CT and radiologist SEG;
6. fetches the Orthanc macOS ZIP once and exposes it at a restricted route on the same
   ngrok tunnel;
7. starts one Uvicorn worker and permits only one full-volume request at a time;
8. starts ngrok and prints the public HTTPS API URL plus the Mac installer command.

The token is used to create the ngrok tunnel but is not printed by the notebook.

## 3. Point local OHIF at Colab

Copy the printed ngrok URL and run on the Mac:

```bash
bash scripts/start_ohif_with_remote_api.sh https://YOUR-NGROK-DEV-DOMAIN
```

The launcher verifies `/health`, caps the OHIF Node process at 4 GiB, keeps the local
Orthanc proxy, and points only the doctor-assistant API proxy at Colab. Open the study
URL printed by the Orthanc upload step, select **3D Segment (draw box)**, use lung window
`W 1500 / L -600`, and draw around the nodule on instance 86.

The remote API always writes its DICOM SEG inside the temporary Colab filesystem and
returns the masks to OHIF. It deliberately reports Orthanc publication as disabled:
`localhost:8042` inside Colab is not the Mac. A later bridge can download that SEG or
push it back to local Orthanc after the interactive path is accepted.

## Resource behavior

- The Mac runs OHIF and Orthanc only; it does not import Torch or load MedSAM2.
- Uvicorn uses one worker.
- A non-blocking inference lock returns HTTP 429 for overlapping volume requests.
- The notebook refuses GPUs older than compute capability 8 or below 18 GiB.
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
- [Official MedSAM2 installation](https://github.com/bowang-lab/MedSAM2#installation)
