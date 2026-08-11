#!/usr/bin/env python3
"""Run the staged LIDC box prompt through the real API and score its DICOM SEG.

This is the server-independent acceptance command for Apple Silicon. It uses FastAPI's
in-process client but otherwise follows the production route: load 238 DICOM slices,
run MLX propagation, calculate physical measurements, and write a DICOM SEG. The sparse
API masks are then aligned to the radiologist SEG by source SOP Instance UID for Dice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pydicom
from fastapi.testclient import TestClient

from api.main import create_app
from experts.medsam2_volume import SAM2MLXVolumeSegmenter
from experts.medsam_interactive import decode_binary_mask_rle
from routing import ExpertRegistry


def _reference_volume(expert, uid_to_index: dict[str, int], shape: tuple[int, ...]):
    reference = np.zeros(shape, dtype=bool)
    for frame_index, frame in enumerate(expert.pixel_array.astype(bool)):
        source_uid = str(
            expert.PerFrameFunctionalGroupsSequence[frame_index]
            .DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )
        if source_uid not in uid_to_index:
            raise RuntimeError(f"expert SEG references unknown source SOP {source_uid}")
        reference[uid_to_index[source_uid]] = frame
    return reference


def _dice(first, second) -> float:
    intersection = int(np.logical_and(first, second).sum())
    denominator = int(first.sum() + second.sum())
    return 2.0 * intersection / denominator if denominator else 1.0


def run(args) -> dict:
    manifest = json.loads(args.manifest.resolve().read_text())
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"converted MLX checkpoint not found: {model_path}")

    segmenter = SAM2MLXVolumeSegmenter(
        model=model_path,
        image_size=args.image_size,
        keep_prompt_component=True,
    )
    app = create_app(
        database_url=manifest["database_url"],
        storage_dir=Path(manifest["storage_dir"]),
        registry=ExpertRegistry(),
        ohif_origin="http://localhost:3000",
        medsam2=segmenter,
    )

    try:
        started = time.perf_counter()
        with TestClient(app) as client:
            response = client.post(
                f"/v1/series/{manifest['api_series_id']}/segment-volume",
                json={
                    "sop_instance_uid": manifest["seed_sop_instance_uid"],
                    "box_xyxy": manifest["recommended_box_xyxy"],
                    "segment_label": "LIDC interactive nodule",
                    "publish_to_orthanc": False,
                    "window_center": manifest["recommended_window_center"],
                    "window_width": manifest["recommended_window_width"],
                },
            )
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} {response.text}")
        body = response.json()

        source_files = sorted(Path(manifest["series_dir"]).glob("instance_*.dcm"))
        source_uids = [
            str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
            for path in source_files
        ]
        uid_to_index = {uid: index for index, uid in enumerate(source_uids)}
        prediction = np.zeros(
            (len(source_files), int(manifest["rows"]), int(manifest["columns"])),
            dtype=bool,
        )
        for item in body["masks"]:
            prediction[uid_to_index[item["sop_instance_uid"]]] = decode_binary_mask_rle(
                item["mask_rle"]
            )

        expert = pydicom.dcmread(manifest["expert_seg_path"])
        reference = _reference_volume(expert, uid_to_index, prediction.shape)
        seed_index = uid_to_index[manifest["seed_sop_instance_uid"]]

        derived_dir = Path(manifest["series_dir"]) / "derived"
        dicom_seg_path = next(
            (
                path
                for path in derived_dir.glob("*.dcm")
                if str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
                == body["dicom_seg_sop_instance_uid"]
            ),
            None,
        )
        result = {
            "status": response.status_code,
            "backend": body["model_version"],
            "elapsed_seconds": round(elapsed, 3),
            "prediction_voxels": int(prediction.sum()),
            "expert_voxels": int(reference.sum()),
            "prediction_segmented_slices": int(prediction.any(axis=(1, 2)).sum()),
            "expert_segmented_slices": int(reference.any(axis=(1, 2)).sum()),
            "volume_dice": round(_dice(prediction, reference), 5),
            "seed_slice_dice": round(
                _dice(prediction[seed_index], reference[seed_index]), 5
            ),
            "volume_ml": body["volume_ml"],
            "dicom_seg_sop_instance_uid": body["dicom_seg_sop_instance_uid"],
            "dicom_seg_path": str(dicom_seg_path) if dicom_seg_path else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        print(f"Result: {args.output.resolve()}", flush=True)
        return result
    finally:
        app.state.engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/validation/lidc_idri_0686/interactive_demo_manifest.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("checkpoints/MedSAM2_CTLesion_hiera_tiny_mlx.safetensors"),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/lidc_idri_0686/mlx_interactive_result.json"),
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
