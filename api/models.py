"""SQLAlchemy ORM schema — a direct relational mapping of `pipeline.AnalysisResult`.

Every child table here mirrors a list already present on `AnalysisResult`
(`findings`, `recommendations`, `expert_executions`) — Phase 0's provenance work
(`core.types.ExpertExecution`, `pipeline.AnalysisResult.analysis_id`/`.status`) is what
makes this mapping close to mechanical. See `api/persistence.py` for the
`AnalysisResult` -> row translation.

Only portable column types are used (String/Integer/Float/Boolean/DateTime/Text) — no
Postgres-specific types — so the same schema works unmodified against the disposable
SQLite database the test suite uses (see `tests/test_api.py`).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid_hex() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    modality: Mapped[str] = mapped_column(String(32))
    body_part: Mapped[str] = mapped_column(String(32))
    source_filename: Mapped[str] = mapped_column(String(255))
    # Only set for a plain single-file upload (Phase 1's original path). A DICOM study
    # imported from Orthanc may have multiple series, so its analyzable path lives on
    # each `Series.storage_dir` instead — see Analysis.series_id below.
    storage_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload|orthanc
    # The real DICOM StudyInstanceUID — the standard identifier QIDO-RS itself keys
    # on, and the only study identifier this client resolves anything by (there is no
    # separate Orthanc-internal study ID tracked here; nothing in the query flow needs
    # one).
    dicom_study_uid: Mapped[str | None] = mapped_column(String(128), default=None)
    # sha256(PatientID)[:16] — never the raw DICOM PatientID/PatientName. See
    # api/dicom_ingest.py::_pseudonymous_patient_reference.
    patient_reference: Mapped[str | None] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )
    series: Mapped[list["Series"]] = relationship(
        back_populates="study", cascade="all, delete-orphan"
    )


class Series(Base):
    __tablename__ = "series"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"))
    dicom_series_uid: Mapped[str] = mapped_column(String(128))
    dicom_modality: Mapped[str] = mapped_column(String(16))  # raw DICOM tag, e.g. "CR"
    # Resolved once at ingest time (api/dicom_ingest.py) so routing never re-derives
    # the mapping — core.enums.Modality/BodyPart values, UNKNOWN when the DICOM tag is
    # absent or unrecognized (BodyPartExamined especially is often unreliable; this
    # stays UNKNOWN rather than guessing, which correctly makes the router fail closed
    # instead of silently misrouting).
    modality: Mapped[str] = mapped_column(String(32))
    body_part: Mapped[str] = mapped_column(String(32))
    view_position: Mapped[str | None] = mapped_column(String(16), default=None)
    instance_count: Mapped[int] = mapped_column(Integer, default=0)
    storage_dir: Mapped[str] = mapped_column(String(1024))
    # False when `dicom_modality` has no mapping in core.enums.Modality — the series is
    # still recorded (not silently dropped), just not submittable for analysis.
    analysis_eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    ineligible_reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    study: Mapped[Study] = relationship(back_populates="series")
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="series")


class Analysis(Base):
    __tablename__ = "analyses"

    # Allocated by the API at submission time (POST .../analyses), independent of
    # `pipeline.AnalysisResult.analysis_id` (see `pipeline_analysis_id` below) —
    # `pipeline.py` generates its own ID only once `Pipeline.analyze` actually runs,
    # which is too late to hand back to the client as the job identifier.
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid_hex)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"))
    # Set only when submitted via POST /v1/series/{id}/analyses (a DICOM import); when
    # set, api/persistence.py::run_analysis reads series.storage_dir instead of
    # study.storage_path.
    series_id: Mapped[str | None] = mapped_column(ForeignKey("series.id"), default=None)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|complete|failed
    pipeline_analysis_id: Mapped[str | None] = mapped_column(String(32), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    report_text: Mapped[str | None] = mapped_column(Text, default=None)
    verification_ok: Mapped[bool | None] = mapped_column(Boolean, default=None)
    triage_urgency: Mapped[str | None] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    study: Mapped[Study] = relationship(back_populates="analyses")
    series: Mapped["Series | None"] = relationship(back_populates="analyses")
    findings: Mapped[list["AnalysisFinding"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["AnalysisRecommendation"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    expert_executions: Mapped[list["ExpertExecutionRecord"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class AnalysisFinding(Base):
    __tablename__ = "analysis_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    label: Mapped[str] = mapped_column(String(128))
    canonical_label: Mapped[str | None] = mapped_column(String(128), default=None)
    probability: Mapped[float | None] = mapped_column(Float, default=None)
    present: Mapped[bool] = mapped_column(Boolean, default=True)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    laterality: Mapped[str | None] = mapped_column(String(16), default=None)
    location: Mapped[str | None] = mapped_column(String(64), default=None)
    size_mm: Mapped[float | None] = mapped_column(Float, default=None)
    volume_ml: Mapped[float | None] = mapped_column(Float, default=None)
    count: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[str] = mapped_column(String(64), default="")
    execution_id: Mapped[str | None] = mapped_column(String(32), default=None)

    analysis: Mapped[Analysis] = relationship(back_populates="findings")


class AnalysisRecommendation(Base):
    __tablename__ = "analysis_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    label: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    urgency: Mapped[str] = mapped_column(String(16))  # Urgency enum's .name

    analysis: Mapped[Analysis] = relationship(back_populates="recommendations")


class ExpertExecutionRecord(Base):
    __tablename__ = "expert_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    execution_id: Mapped[str] = mapped_column(String(32))
    expert: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))  # completed|failed
    expert_version: Mapped[str | None] = mapped_column(String(255), default=None)
    preprocessing_version: Mapped[str | None] = mapped_column(String(255), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    analysis: Mapped[Analysis] = relationship(back_populates="expert_executions")
