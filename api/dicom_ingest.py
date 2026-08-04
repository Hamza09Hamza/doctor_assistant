"""Import a DICOM study from Orthanc into `Study`/`Series` rows.

Query flow: QIDO-RS (`orthanc_client.query_study`/`.query_series`) resolves the study's
own DICOM tags and its series list; each series' Orthanc-internal instance IDs are then
resolved via `.find_instance_ids` and downloaded via `.download_instance` (see
`api/orthanc_client.py`'s module docstring for why that part isn't WADO-RS). Once
downloaded, files live under `STORAGE_DIR/studies/{study_id}/series/{series_id}/` —
plain files in a directory, which `ingest.loaders.VolumeLoader` already reads
unmodified as "a DICOM series directory."
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pydicom
from sqlalchemy.orm import Session

from core.enums import BodyPart, Modality

from .models import Series, Study
from .orthanc_client import OrthancClient

# DICOM Modality tag value -> core.enums.Modality. Anything not listed here still
# creates the Series row (for visibility/audit) but with analysis_eligible=False.
_DICOM_MODALITY_MAP: dict[str, Modality] = {
    "CR": Modality.XRAY,
    "DX": Modality.XRAY,
    "CT": Modality.CT,
    "MR": Modality.MRI,
    "US": Modality.ULTRASOUND,
    "MG": Modality.MAMMOGRAPHY,
}

# DICOM BodyPartExamined tag value -> core.enums.BodyPart. This tag is notoriously
# unreliably populated in real-world DICOM data, so an absent or unrecognized value
# stays BodyPart.UNKNOWN rather than being guessed from context — the router already
# fails closed (RoutingError) on an UNKNOWN body part instead of silently misrouting.
_DICOM_BODY_PART_MAP: dict[str, BodyPart] = {
    "CHEST": BodyPart.CHEST,
    "ABDOMEN": BodyPart.ABDOMEN,
    "SPINE": BodyPart.SPINE,
    "BREAST": BodyPart.BREAST,
    "HEART": BodyPart.HEART,
    "SKULL": BodyPart.BRAIN,
    "BRAIN": BodyPart.BRAIN,
}


def _pseudonymous_patient_reference(patient_id: str) -> str:
    """One-way hash — never store or log the raw DICOM PatientID/PatientName."""
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]


def _dicom_json_value(tags: dict, tag: str) -> str | None:
    """Read one element from a DICOM JSON Model object (the standard QIDO-RS response
    shape: `{"00100020": {"vr": "LO", "Value": ["12345"]}, ...}`, PS3.18)."""
    element = tags.get(tag)
    if not element:
        return None
    values = element.get("Value")
    if not values:
        return None
    return str(values[0])


def _read_instance_header(raw_dicom_bytes: bytes) -> tuple[str | None, str | None]:
    """(ViewPosition, BodyPartExamined) from one instance's header, read once and
    reused for the whole series (both are expected constant within a series)."""
    dataset = pydicom.dcmread(io.BytesIO(raw_dicom_bytes), stop_before_pixels=True)
    view_position = getattr(dataset, "ViewPosition", None) or None
    body_part_examined = getattr(dataset, "BodyPartExamined", None) or None
    return view_position, body_part_examined


def import_study_from_orthanc(
    db: Session,
    *,
    storage_dir: Path,
    orthanc_client: OrthancClient,
    study_instance_uid: str,
) -> Study:
    study_tags = orthanc_client.query_study(study_instance_uid)
    series_tags_list = orthanc_client.query_series(study_instance_uid)
    if not series_tags_list:
        raise ValueError(f"study {study_instance_uid!r} has no series in Orthanc")

    patient_id = _dicom_json_value(study_tags, "00100020")  # PatientID
    patient_reference = _pseudonymous_patient_reference(patient_id) if patient_id else None

    study = Study(
        # A DICOM study can span series of different modalities/body parts; these
        # stay explicitly UNKNOWN rather than guessing from one series or an
        # unreliably-populated BodyPartExamined tag. Analysis is submitted per-series
        # (POST /v1/series/{id}/analyses), where the real per-series modality is known.
        modality=Modality.UNKNOWN.value,
        body_part=BodyPart.UNKNOWN.value,
        source_filename=f"orthanc-study-{study_instance_uid}",
        source="orthanc",
        dicom_study_uid=study_instance_uid,
        patient_reference=patient_reference,
    )
    db.add(study)
    db.flush()  # allocate study.id within this transaction, without committing yet

    for series_tags in series_tags_list:
        series_uid = _dicom_json_value(series_tags, "0020000E")  # SeriesInstanceUID
        if not series_uid:
            raise ValueError(f"a series under study {study_instance_uid!r} has no SeriesInstanceUID")
        dicom_modality = _dicom_json_value(series_tags, "00080060") or ""  # Modality

        instance_ids = orthanc_client.find_instance_ids(series_uid)
        series_dir = storage_dir / "studies" / study.id / "series" / series_uid.replace(".", "_")
        series_dir.mkdir(parents=True, exist_ok=True)

        view_position: str | None = None
        body_part_examined: str | None = None
        for index, instance_id in enumerate(instance_ids):
            raw = orthanc_client.download_instance(instance_id)
            (series_dir / f"instance_{index:04d}.dcm").write_bytes(raw)
            if view_position is None and body_part_examined is None:
                view_position, body_part_examined = _read_instance_header(raw)

        mapped_modality = _DICOM_MODALITY_MAP.get(dicom_modality)
        mapped_body_part = (
            _DICOM_BODY_PART_MAP.get(body_part_examined.upper()) if body_part_examined else None
        )
        eligible = mapped_modality is not None and mapped_body_part is not None
        if mapped_modality is None:
            reason = f"DICOM modality {dicom_modality!r} has no supported mapping"
        elif mapped_body_part is None:
            reason = f"BodyPartExamined {body_part_examined!r} is missing or unrecognized"
        else:
            reason = None
        db.add(
            Series(
                study_id=study.id,
                dicom_series_uid=series_uid,
                dicom_modality=dicom_modality,
                modality=(mapped_modality or Modality.UNKNOWN).value,
                body_part=(mapped_body_part or BodyPart.UNKNOWN).value,
                view_position=view_position,
                instance_count=len(instance_ids),
                storage_dir=str(series_dir),
                analysis_eligible=eligible,
                ineligible_reason=reason,
            )
        )

    db.commit()
    db.refresh(study)
    return study
