#!/usr/bin/env python3
"""Build an OHIF-viewable bundle for one brain-tumour case: a synthetic DICOM MRI
series (via scripts/nifti_to_dicom.py) plus a real AI-prediction DICOM SEG (TC/WT/ET).

MSD Task01_BrainTumour ships as NIfTI only -- there is no real DICOM series a SEG could
reference, the same blocker documented in docs/DEVELOPMENT.md for a different NIfTI-only
demo. This produces a clearly-labeled SYNTHETIC DICOM container (fabricated UIDs and
patient tags, real pixel data and real model output) so the result can be viewed in
OHIF the same way as the lung-nodule case, through the existing bundle/publish path
(scripts/publish_dicom_seg_to_orthanc.py, unmodified, run locally afterward).

Reuses the exact bundle loading, MSD/BraTS reconciliation, and preprocessing already
verified in run_monai_pathology_experts.py -- this script does not reimplement any of
that, only adds: single-case inference with full mask capture (the eval loop there
only keeps per-case Dice, not the mask array), DICOM synthesis, and SEG construction.

Usage (Colab, same environment as run_monai_pathology_experts.py plus highdicom):

    python scripts/build_brain_tumor_seg_bundle.py \
        --data-dir /content/drive/MyDrive/doctor_assistant/monai_experts/data \
        --scratch-dir /content/monai_scratch \
        --output-dir /content/drive/MyDrive/doctor_assistant/monai_experts/results \
        --case-index 46
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_monai_pathology_experts import (  # noqa: E402
    BRAIN_TUMOR_BUNDLE,
    BRATS_CHANNELS,
    BUNDLE_MODALITY_ORDER,
    _cache_archive_to_drive,
    _derive_msd_mapping,
    _load_network_from_bundle,
    _read_msd_descriptor,
    _stage_decathlon_archive,
    download_bundle,
    log,
)
from scripts.nifti_to_dicom import build_dicom_series  # noqa: E402

# TC/WT/ET are clinically nested (WT contains TC contains ET); shown as three
# independently toggleable segments rather than merged, matching how OHIF's stock SEG
# panel already presents multi-segment objects (proven pattern, see
# scripts/run_lidc_lung_nodule_colab.py's comparison bundle).
SEGMENT_LABELS = ("Tumour core (TC)", "Whole tumour (WT)", "Enhancing tumour (ET)")


def _build_seg(
    mask_frames,
    source_datasets: list,
    output_path: Path,
    *,
    series_description: str,
    series_number: int,
    algorithm_type: str,
) -> None:
    """mask_frames: (n_frames, rows, cols, n_segments) bool array, frame order matching
    source_datasets order exactly (both indexed by the same k -- see
    scripts/nifti_to_dicom.py's docstring on why this must be identity-matched rather
    than relying on any assumed spatial sort order).

    algorithm_type: "AUTOMATIC" for the model's own prediction, "MANUAL" for the MSD/
    BraTS expert-annotated ground truth -- highdicom requires AlgorithmIdentification
    whenever it's not MANUAL (confirmed by it raising TypeError without one)."""
    import highdicom as hd
    from highdicom.seg.content import SegmentDescription
    from highdicom.sr.coding import CodedConcept
    from highdicom import AlgorithmIdentificationSequence
    from pydicom.uid import generate_uid

    present_segment_indices = [
        i for i in range(mask_frames.shape[-1]) if mask_frames[..., i].any()
    ]
    if not present_segment_indices:
        raise RuntimeError("mask is empty on every channel; nothing to encode")

    algorithm_identification = None
    if algorithm_type != "MANUAL":
        algorithm_identification = AlgorithmIdentificationSequence(
            name="brats_mri_segmentation",
            family=CodedConcept("113092", "DCM", "Deep Learning"),
            version="MONAI Model Zoo",
            source="doctor_assistant / scripts/run_monai_pathology_experts.py",
        )

    kept_mask = mask_frames[..., present_segment_indices]
    segment_descriptions = [
        SegmentDescription(
            segment_number=position + 1,
            segment_label=SEGMENT_LABELS[channel_index],
            segmented_property_category=CodedConcept("91723000", "SCT", "Anatomical Structure"),
            segmented_property_type=CodedConcept("108369006", "SCT", "Neoplasm"),
            algorithm_type=algorithm_type,
            algorithm_identification=algorithm_identification,
        )
        for position, channel_index in enumerate(present_segment_indices)
    ]

    seg = hd.seg.Segmentation(
        source_images=source_datasets,
        pixel_array=kept_mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_descriptions,
        series_instance_uid=generate_uid(),
        series_number=series_number,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        series_description=series_description,
        manufacturer="doctor_assistant",
        manufacturer_model_name="brats_mri_segmentation (MONAI Model Zoo)",
        software_versions="run_monai_pathology_experts.py",
        device_serial_number="0",
    )
    seg.save_as(str(output_path))


def run(args) -> Path:
    import numpy as np
    import torch
    from monai.apps import DecathlonDataset
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        EnsureChannelFirstd,
        Lambdad,
        LoadImaged,
        MapLabelValued,
        NormalizeIntensityd,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    bundle_dir = download_bundle(BRAIN_TUMOR_BUNDLE, args.data_dir / "bundles")
    network = _load_network_from_bundle(bundle_dir, device)

    dataset_root = _stage_decathlon_archive(args)
    dataset = DecathlonDataset(
        root_dir=str(dataset_root),
        task="Task01_BrainTumour",
        transform=None,
        section="validation",
        download=True,
        cache_rate=0.0,
        num_workers=2,
    )
    _cache_archive_to_drive(args, dataset_root)
    mapping = _derive_msd_mapping(_read_msd_descriptor(dataset_root))

    # Identical to run_brain_tumor()'s transform -- deliberately duplicated rather than
    # imported as a shared function, to keep this script independently readable and not
    # couple it to that eval loop's internals.
    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            Lambdad(
                keys="image",
                func=lambda x, perm=tuple(mapping["channel_permutation"]): x[list(perm)],
            ),
            MapLabelValued(
                keys="label",
                orig_labels=mapping["label_orig"],
                target_labels=mapping["label_target_brats18"],
                dtype=np.uint8,
            ),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    dataset.transform = transform

    if not 0 <= args.case_index < len(dataset):
        raise ValueError(f"--case-index must be in [0, {len(dataset)}); got {args.case_index}")
    case = dataset[args.case_index]
    log(f"Case {args.case_index}: input {tuple(case['image'].shape)}")

    image = case["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image, roi_size=(240, 240, 160), sw_batch_size=1, predictor=network, overlap=0.5
        )
        prediction = (torch.sigmoid(logits) > 0.5)[0].cpu().numpy()  # (3, i, j, k) TC/WT/ET
    for name, count in zip(BRATS_CHANNELS, prediction.reshape(3, -1).sum(axis=1)):
        log(f"  {name}: {int(count)} predicted voxels")

    # Display background: FLAIR, the single most tumour-conspicuous sequence for a
    # general (non-radiologist) viewer. Already z-score normalized by NormalizeIntensityd
    # -- contrast/structure is preserved, only the absolute intensity range differs from
    # raw acquisition values, which is cosmetically fine for this demo.
    flair_index = BUNDLE_MODALITY_ORDER.index("flair")
    background = case["image"][flair_index].cpu().numpy()
    affine_ras = np.asarray(case["image"].affine)

    work_dir = args.scratch_dir / f"brain_tumor_case{args.case_index:03d}_seg_bundle"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    dicom_dir = work_dir / "source_dicom"
    log(f"Synthesizing DICOM MRI series ({background.shape[2]} slices) ...")
    source_datasets = build_dicom_series(
        background,
        affine_ras,
        dicom_dir,
        series_description="Synthetic FLAIR (MSD Task01_BrainTumour, BraTS-derived) -- NOT a real patient acquisition",
        patient_id=f"SYNTHETIC-MSD-BRATS-CASE{args.case_index:03d}",
        patient_name="SYNTHETIC^MSD^BRATS",
    )

    # (channel, i, j, k) -> (k, i, j, channel): highdicom wants (frame, row, col, segment),
    # frame order matching source_datasets exactly (both are indexed by k here).
    mask_frames = np.transpose(prediction, (3, 1, 2, 0)).astype(bool)

    seg_path = work_dir / "totalsegmentator_seg.dcm"
    log("Building AI-prediction DICOM SEG (TC/WT/ET) ...")
    _build_seg(
        mask_frames, source_datasets, seg_path,
        series_description="AI prediction (brats_mri_segmentation) -- SYNTHETIC DICOM demo",
        series_number=10, algorithm_type="AUTOMATIC",
    )

    # Ground truth: case["label"] went through the identical ConvertToMultiChannelBased
    # OnBratsClassesd transform as the prediction, so it's already the same (3, i, j, k)
    # TC/WT/ET channel layout -- no extra inference, just the dataset's own expert
    # annotation (MSD/BraTS ships expert-segmented labels, not raw contours).
    ground_truth = case["label"].cpu().numpy().astype(bool)  # (3, i, j, k)
    gt_mask_frames = np.transpose(ground_truth, (3, 1, 2, 0))
    expert_dir = work_dir / "expert_reference"
    expert_dir.mkdir(parents=True, exist_ok=True)
    gt_seg_path = expert_dir / "ground_truth_seg.dcm"
    log("Building ground-truth DICOM SEG (TC/WT/ET, MSD/BraTS expert annotation) ...")
    _build_seg(
        gt_mask_frames, source_datasets, gt_seg_path,
        series_description="Ground truth (MSD Task01_BrainTumour / BraTS expert annotation)",
        series_number=11, algorithm_type="MANUAL",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / f"brain_tumor_case{args.case_index:03d}_ohif_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dcm_path in dicom_dir.glob("*.dcm"):
            archive.write(dcm_path, f"source_dicom/{dcm_path.name}")
        archive.write(seg_path, "totalsegmentator_seg.dcm")
        archive.write(gt_seg_path, f"expert_reference/{gt_seg_path.name}")

    log(f"Bundle written: {bundle_path}")
    log(
        "Locally: python scripts/publish_dicom_seg_to_orthanc.py "
        f"--bundle <downloaded {bundle_path.name}>"
    )
    return bundle_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content/monai_scratch"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case-index",
        type=int,
        default=46,
        help="MSD validation-section index (0-based); default is a case with strong "
        "Dice across TC/WT/ET in the full validation run (case 47 in 1-indexed logs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
