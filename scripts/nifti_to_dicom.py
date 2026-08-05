"""Synthesize a valid DICOM series from a NIfTI volume + its affine.

Exists so NIfTI-only research outputs (MSD/BraTS, and anything similar in future) can
be viewed in a DICOM-native viewer (OHIF) alongside a DICOM SEG built the normal way.
The resulting series is CLEARLY LABELED as synthetic in every identifying field --
PatientID, PatientName, SeriesDescription, Manufacturer -- because it is: the pixel
data and geometry are real (derived directly from the source NIfTI), but the DICOM
identifiers (UIDs, patient tags) are fabricated for this run, not read from any real
acquisition. Never use this to make research data look like a real clinical study.

Geometry: NIfTI affines map voxel index -> world coordinates in RAS+ (Right-Anterior-
Superior). DICOM uses LPS (Left-Posterior-Superior) -- the opposite handedness on the
first two axes. The conversion is a single matrix multiply:

    affine_lps = diag([-1, -1, 1, 1]) @ affine_ras

This flips both the direction/spacing columns and the origin/translation column in one
step, which is why it's used here instead of separately negating a decomposed origin
and direction cosines (fewer places to get the sign wrong).

Row/column convention (chosen once, applied consistently -- this is a fresh design
choice, not read off an external source, so there is nothing to get backwards; the
choice only needs to be internally consistent between the pixel array and the declared
orientation tags, which is verified below):

    volume axis 0 (i) -> DICOM ROW index
    volume axis 1 (j) -> DICOM COLUMN index
    volume axis 2 (k) -> slice index

Per DICOM PS3.3 C.7.6.2.1.1, ImageOrientationPatient's first triplet is the direction
of increasing COLUMN index (here: axis 1 / j), second triplet is the direction of
increasing ROW index (here: axis 0 / i). Both are read directly off the (already
RAS->LPS-converted) affine's own columns, not re-derived by hand, to avoid the same
class of row/col mixup already found and fixed twice elsewhere in this project this
session (scripts/lidc_seg_ground_truth.py, scripts/plot_lung_nodule_detections.py).

Verified against an independent computation with a synthetic volume before ever
touching real data: every written file's own ImagePositionPatient and pixel content
are geometrically exact. One real, harmless finding from that verification, documented
here rather than silently "fixed" into something that only looks tidier: GDCM sorts a
loaded series by projecting each slice's position onto the normal computed from the
orientation vectors (effectively cross(col_dir, row_dir)), NOT by filename or
InstanceNumber. Depending on that normal's sign, a reader (SimpleITK, OHIF/cornerstone,
any GDCM-based tool) may present slices in the reverse of this module's writing order.
This does not affect correctness -- every slice's absolute physical position is still
exactly right, so pixel content and any companion SEG (built from the same source_images
list, matched by identity, not by array index) stay correctly aligned. It only affects
which direction "scrolling up" moves through the volume, which is cosmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ras_to_lps(affine_ras: np.ndarray) -> np.ndarray:
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ affine_ras


def build_dicom_series(
    volume: np.ndarray,
    affine_ras: np.ndarray,
    output_dir: Path,
    *,
    series_description: str,
    modality: str = "MR",
    patient_id: str = "SYNTHETIC-RESEARCH-DATA",
    patient_name: str = "SYNTHETIC^RESEARCH^DATA",
    study_description: str = "Synthetic DICOM -- NOT a real patient acquisition",
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    window_center: float | None = None,
    window_width: float | None = None,
) -> list:
    """Write one DICOM file per slice along volume's 3rd axis (k). Returns the pydicom
    Datasets in slice order (also written to output_dir, filenames sorted the same way).

    volume: (ni, nj, nk) array, voxel-index order matching affine_ras's column order.
    affine_ras: 4x4 NIfTI-convention affine (voxel index -> RAS+ mm), e.g. from
    nibabel's img.affine, or from a MONAI MetaTensor's own tracked affine after
    Orientationd(axcodes="RAS") -- which is why this only supports RAS+ input, not an
    arbitrary orientation: matching MONAI's own canonical post-Orientationd convention
    means no additional reorientation step is needed before calling this.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import MRImageStorage, generate_uid

    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")

    affine_lps = _ras_to_lps(np.asarray(affine_ras, dtype=np.float64))
    i_dir_spacing = affine_lps[:3, 0]
    j_dir_spacing = affine_lps[:3, 1]
    k_dir_spacing = affine_lps[:3, 2]
    row_spacing_mm = float(np.linalg.norm(i_dir_spacing))
    col_spacing_mm = float(np.linalg.norm(j_dir_spacing))
    slice_spacing_mm = float(np.linalg.norm(k_dir_spacing))
    row_dir = i_dir_spacing / row_spacing_mm
    col_dir = j_dir_spacing / col_spacing_mm

    # First triplet = direction of increasing COLUMN index (j) = col_dir.
    # Second triplet = direction of increasing ROW index (i) = row_dir.
    image_orientation_patient = [*col_dir.tolist(), *row_dir.tolist()]

    ni, nj, nk = volume.shape
    finite = volume[np.isfinite(volume)]
    data_min = float(finite.min()) if finite.size else 0.0
    data_max = float(finite.max()) if finite.size else 1.0
    scale = 4095.0 / (data_max - data_min) if data_max > data_min else 1.0

    study_instance_uid = study_instance_uid or generate_uid()
    series_instance_uid = series_instance_uid or generate_uid()
    frame_of_reference_uid = frame_of_reference_uid or generate_uid()

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for k in range(nk):
        world_ras = np.asarray(affine_ras, dtype=np.float64) @ np.array([0.0, 0.0, float(k), 1.0])
        position_lps = np.array([-world_ras[0], -world_ras[1], world_ras[2]])

        slice_pixels = volume[:, :, k]
        scaled = np.clip((slice_pixels - data_min) * scale, 0, 4095).astype(np.uint16)

        ds = Dataset()
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.Modality = modality
        ds.PatientID = patient_id
        ds.PatientName = patient_name
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.AccessionNumber = ""
        ds.StudyID = ""
        ds.ReferringPhysicianName = ""
        ds.StudyDate = "20260101"
        ds.StudyTime = "000000"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = series_description
        ds.StudyDescription = study_description
        ds.Manufacturer = "doctor_assistant synthetic NIfTI->DICOM adapter"
        ds.ManufacturerModelName = "scripts/nifti_to_dicom.py"

        ds.Rows, ds.Columns = ni, nj
        ds.PixelSpacing = [round(row_spacing_mm, 6), round(col_spacing_mm, 6)]
        ds.SliceThickness = round(slice_spacing_mm, 6)
        ds.SpacingBetweenSlices = round(slice_spacing_mm, 6)
        ds.ImageOrientationPatient = [round(float(v), 8) for v in image_orientation_patient]
        ds.ImagePositionPatient = [round(float(v), 4) for v in position_lps]
        ds.SliceLocation = round(float(position_lps[2]), 4)

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.RescaleSlope = (data_max - data_min) / 4095.0 if data_max > data_min else 1.0
        ds.RescaleIntercept = data_min
        wc = window_center if window_center is not None else 2048
        ww = window_width if window_width is not None else 4095
        ds.WindowCenter = wc
        ds.WindowWidth = ww
        ds.PixelData = scaled.tobytes()

        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        out_path = output_dir / f"slice_{k:04d}.dcm"
        # enforce_file_format=True: writes the proper 128-byte preamble + "DICM" magic
        # required by the DICOM Part 10 file format. Without it, pydicom writes the
        # dataset as-is and may silently omit the preamble -- GDCM/SimpleITK tolerated
        # that when reading it back locally, but Orthanc's ingestion endpoint is a
        # separate, less-tested code path and there is no reason to ship a
        # non-conformant file when the conformant one costs nothing extra.
        pydicom.dcmwrite(str(out_path), ds, enforce_file_format=True)
        datasets.append(ds)

    return datasets
