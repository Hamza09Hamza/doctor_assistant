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
import tempfile

import pydicom
from sqlalchemy import select
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
    """Import or refresh one Orthanc study, keyed by its DICOM UIDs.

    Re-importing the same StudyInstanceUID is intentionally idempotent: it reuses the
    existing Study and Series rows and refreshes their files/metadata.  The OHIF panel
    retries this endpoint during setup, so creating another row on every retry would
    make the otherwise-unique SeriesInstanceUID lookup ambiguous.
    """
    study_tags = orthanc_client.query_study(study_instance_uid)
    series_tags_list = orthanc_client.query_series(study_instance_uid)
    if not series_tags_list:
        raise ValueError(f"study {study_instance_uid!r} has no series in Orthanc")

    patient_id = _dicom_json_value(study_tags, "00100020")  # PatientID
    patient_reference = _pseudonymous_patient_reference(patient_id) if patient_id else None

    study = db.execute(
        select(Study)
        .where(Study.dicom_study_uid == study_instance_uid)
        .order_by(Study.created_at.asc(), Study.id.asc())
        .limit(1)
    ).scalar_one_or_none()
    if study is None:
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
    else:
        # Refresh mutable source metadata without replacing the stable internal ID or
        # disturbing analyses already attached to this imported study.
        study.source_filename = f"orthanc-study-{study_instance_uid}"
        study.source = "orthanc"
        study.patient_reference = patient_reference

    series_by_uid = {item.dicom_series_uid: item for item in study.series}

    for series_tags in series_tags_list:
        series_uid = _dicom_json_value(series_tags, "0020000E")  # SeriesInstanceUID
        if not series_uid:
            raise ValueError(f"a series under study {study_instance_uid!r} has no SeriesInstanceUID")
        dicom_modality = _dicom_json_value(series_tags, "00080060") or ""  # Modality

        instance_ids = orthanc_client.find_instance_ids(series_uid)
        if not instance_ids:
            raise ValueError(f"series {series_uid!r} has no instances in Orthanc")
        series_parent = storage_dir / "studies" / study.id / "series"
        series_parent.mkdir(parents=True, exist_ok=True)
        series_dir = series_parent / series_uid.replace(".", "_")

        view_position: str | None = None
        body_part_examined: str | None = None
        seen_sop_instance_uids: set[str] = set()
        # Download and validate the complete refresh in a sibling scratch directory.
        # A timeout, empty response, or wrong Orthanc object therefore cannot partially
        # overwrite the last known-good local source series.
        with tempfile.TemporaryDirectory(prefix=".orthanc-import-", dir=series_parent) as tmp:
            staged_dir = Path(tmp)
            for index, instance_id in enumerate(instance_ids):
                raw = orthanc_client.download_instance(instance_id)
                header = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                if str(getattr(header, "StudyInstanceUID", "")) != study_instance_uid:
                    raise ValueError(
                        f"Orthanc instance {instance_id!r} belongs to a different study"
                    )
                if str(getattr(header, "SeriesInstanceUID", "")) != series_uid:
                    raise ValueError(
                        f"Orthanc instance {instance_id!r} belongs to a different series"
                    )
                sop_instance_uid = str(getattr(header, "SOPInstanceUID", ""))
                if not sop_instance_uid or sop_instance_uid in seen_sop_instance_uids:
                    raise ValueError(
                        f"series {series_uid!r} contains a missing or duplicate SOPInstanceUID"
                    )
                seen_sop_instance_uids.add(sop_instance_uid)
                (staged_dir / f"instance_{index:04d}.dcm").write_bytes(raw)
                if view_position is None and body_part_examined is None:
                    view_position = getattr(header, "ViewPosition", None) or None
                    body_part_examined = getattr(header, "BodyPartExamined", None) or None

            series_dir.mkdir(parents=True, exist_ok=True)
            staged_names = {path.name for path in staged_dir.glob("instance_*.dcm")}
            for staged_path in sorted(staged_dir.glob("instance_*.dcm")):
                staged_path.replace(series_dir / staged_path.name)
            # A refreshed series can contain fewer instances. Remove stale files only
            # after the new complete set has downloaded and passed identity validation.
            for instance_path in series_dir.glob("instance_*.dcm"):
                if instance_path.name not in staged_names:
                    instance_path.unlink()

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
        values = {
            "dicom_modality": dicom_modality,
            "modality": (mapped_modality or Modality.UNKNOWN).value,
            "body_part": (mapped_body_part or BodyPart.UNKNOWN).value,
            "view_position": view_position,
            "instance_count": len(instance_ids),
            "storage_dir": str(series_dir),
            "analysis_eligible": eligible,
            "ineligible_reason": reason,
        }
        series = series_by_uid.get(series_uid)
        if series is None:
            series = Series(
                study=study,
                dicom_series_uid=series_uid,
                **values,
            )
            db.add(series)
            series_by_uid[series_uid] = series
        else:
            for name, value in values.items():
                setattr(series, name, value)

    db.commit()
    db.refresh(study)
    return study
