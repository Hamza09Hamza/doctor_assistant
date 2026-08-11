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

Docker Desktop must be running:

```bash
docker compose -f deployments/docker-compose.yml up -d orthanc
source .venv-mlx/bin/activate
python scripts/prepare_lidc_interactive_demo.py --upload-orthanc
```

The upload command prints the LIDC study URL. No local model inference is performed.

## 2. Start Colab

Open `notebooks/medsam2_inference_server_colab.ipynb` in Google Colab and select an L4
or other Ampere-or-newer GPU runtime with at least 18 GiB GPU memory.

In Colab's **Secrets** panel, add a secret named `NGROK_TOKEN` and enable notebook
access. Run the cells in order. The notebook:

1. checks GPU capability and memory before installing or loading the model;
2. clones this repository's `main` branch;
3. installs the official MedSAM2 package without building its optional CUDA extension;
4. downloads only `MedSAM2_CTLesion.pt`, not every upstream checkpoint;
5. downloads and stages the pinned 238-slice LIDC CT and radiologist SEG;
6. starts one Uvicorn worker and permits only one full-volume request at a time;
7. starts ngrok and prints the public HTTPS API URL.

The token is read at runtime with `google.colab.userdata.get("NGROK_TOKEN")`; it is not
stored in this repository or printed by the notebook.

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
