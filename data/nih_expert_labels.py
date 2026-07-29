"""Prepare Google's radiologist-adjudicated NIH ChestX-ray14 labels.

Google released expert labels for four findings on subsets of the NIH
ChestX-ray14 development and test pools.  This module converts those source
tables into one small, deterministic manifest while failing closed on the two
mistakes that would invalidate an evaluation:

* accepting anything other than an adjudicated ``YES``/``NO`` as a label; and
* allowing a development image into the official NIH test split (or vice versa).

Only the Python standard library is used so the preparation step can run before
the model environment is installed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


CORE_TARGETS: tuple[str, ...] = (
    "Pneumothorax",
    "Nodule_or_mass",
    "Airspace_opacity",
)
OPTIONAL_TARGET = "Fracture"

_TARGET_SOURCE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "Pneumothorax": ("Pneumothorax",),
    "Nodule_or_mass": (
        "Nodule or mass",
        "Nodule/Mass",
        "Nodule_or_mass",
        "Nodule or Mass",
    ),
    "Airspace_opacity": (
        "Airspace opacity",
        "Airspace Opacity",
        "Airspace_opacity",
        "Lung Opacity",
    ),
    "Fracture": ("Fracture",),
}
_IMAGE_ALIASES = (
    "Image Index",
    "image_index",
    "Image",
    "filename",
    "file_name",
    "image_id",
)
_PATIENT_ALIASES = ("Patient ID", "patient_id", "PatientID")
_SPLIT_ALIASES = ("Set Id", "set_id", "split", "dataset_split", "subset")
_MISSING_LABEL_TOKENS = frozenset(
    {
        "",
        "-",
        "N/A",
        "NA",
        "NOT ADJUDICATED",
        "NOT_ADJUDICATED",
        "UNADJUDICATED",
        "UNKNOWN",
        "UNCERTAIN",
        "NOT SURE",
        "UNREAD",
    }
)
_NIH_FILENAME_RE = re.compile(r"^(?P<patient>\d{8})_\d{3}\.png$", re.IGNORECASE)
_HEADER_NORMALIZER_RE = re.compile(r"[^a-z0-9]+")
_VALID_SPLIT_VALUES: Mapping[str, frozenset[str]] = {
    "validation": frozenset({"val", "valid", "validation", "development", "dev"}),
    "test": frozenset({"test", "testing"}),
}


@dataclass(frozen=True)
class NIHExpertRow:
    """One image and its available adjudicated binary labels."""

    filename: str
    patient_id: str
    split: str
    labels: Mapping[str, int | None]


@dataclass(frozen=True)
class NIHExpertLoad:
    """Rows and audit counts obtained from one source CSV."""

    rows: tuple[NIHExpertRow, ...]
    source_rows: int
    skipped_rows_no_labels: int
    label_counts: Mapping[str, Mapping[str, int]]
    detected_columns: Mapping[str, str]


def canonical_nih_filename(value: object) -> str:
    """Return a strict NIH image basename, normalizing only ``.PNG`` case."""

    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"NIH image filename must be text, got {type(value).__name__}")
    text = os.fspath(value).strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    match = _NIH_FILENAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            f"invalid NIH image filename {name!r}; expected '00000001_000.png'"
        )
    return name[:-4] + ".png"


def nih_patient_id(filename: object) -> str:
    """Derive the zero-padded NIH patient identifier from an image filename."""

    name = canonical_nih_filename(filename)
    return name.split("_", 1)[0]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash the exact source bytes without loading a potentially large file."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_official_nih_manifest(
    path: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Read an official NIH filename list and reject malformed/duplicate rows."""

    names: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    with open(path, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                name = canonical_nih_filename(raw)
            except ValueError as error:
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: {error}"
                ) from error
            if name in seen:
                duplicates.add(name)
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"official NIH manifest is empty: {os.fspath(path)}")
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:3])
        raise ValueError(
            f"official NIH manifest {os.fspath(path)!r} contains "
            f"{len(duplicates)} duplicate filename(s), including: {examples}"
        )
    return tuple(names)


def load_google_nih_expert_csv(
    path: str | os.PathLike[str],
    *,
    expected_split: str | None,
    include_fracture: bool = False,
) -> NIHExpertLoad:
    """Load a Google NIH expert-label CSV with strict schema/value checks.

    ``expected_split`` is ``"validation"`` or ``"test"`` for separate source
    files.  Use ``None`` for a combined source table; a documented split column
    is then required.  Blank and recognized non-adjudicated cells are retained as
    ``None`` and counted, never converted to negative labels.  Any other nonempty
    value is rejected.
    """

    if expected_split is not None and expected_split not in _VALID_SPLIT_VALUES:
        raise ValueError("expected_split must be 'validation', 'test', or None")
    targets = CORE_TARGETS + ((OPTIONAL_TARGET,) if include_fracture else ())

    with _open_csv_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"expert CSV has no header: {os.fspath(path)}")
        headers = tuple(
            "" if field is None else str(field).strip() for field in reader.fieldnames
        )
        _validate_headers(headers, path)
        image_column = _detect_column(headers, _IMAGE_ALIASES, "image filename")
        patient_column = _detect_optional_column(
            headers, _PATIENT_ALIASES, "patient ID"
        )
        split_column = _detect_optional_column(headers, _SPLIT_ALIASES, "split")
        if expected_split is None and split_column is None:
            raise ValueError(
                f"combined expert CSV {os.fspath(path)!r} requires a split column; "
                f"accepted names: {', '.join(_SPLIT_ALIASES)}"
            )
        label_columns = {
            target: _detect_column(
                headers,
                _TARGET_SOURCE_ALIASES[target],
                f"{target} label",
            )
            for target in targets
        }

        rows: list[NIHExpertRow] = []
        seen_images: set[str] = set()
        label_counts: dict[str, Counter[str]] = {
            target: Counter(yes=0, no=0, skipped_non_adjudicated=0)
            for target in targets
        }
        source_rows = 0
        skipped_rows = 0

        for line_number, source_row in enumerate(reader, start=2):
            if _row_is_blank(source_row):
                continue
            source_rows += 1
            if None in source_row:
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: row has more fields than header"
                )
            try:
                filename = canonical_nih_filename(source_row.get(image_column))
            except ValueError as error:
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: {error}"
                ) from error
            if filename in seen_images:
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: duplicate expert-label image "
                    f"{filename!r}"
                )
            seen_images.add(filename)

            patient_id = nih_patient_id(filename)
            if patient_column is not None:
                _validate_source_patient_id(
                    source_row.get(patient_column),
                    patient_id,
                    path=path,
                    line_number=line_number,
                )

            split = _parse_split(
                source_row.get(split_column) if split_column is not None else None,
                expected_split=expected_split,
                path=path,
                line_number=line_number,
            )
            labels: dict[str, int | None] = {}
            for target, column in label_columns.items():
                label, status = _parse_label(
                    source_row.get(column),
                    path=path,
                    line_number=line_number,
                    column=column,
                )
                labels[target] = label
                label_counts[target][status] += 1

            if all(label is None for label in labels.values()):
                skipped_rows += 1
                continue
            rows.append(
                NIHExpertRow(
                    filename=filename,
                    patient_id=patient_id,
                    split=split,
                    labels=labels,
                )
            )

    if source_rows == 0:
        raise ValueError(f"expert CSV contains no data rows: {os.fspath(path)}")
    if not rows:
        raise ValueError(
            f"expert CSV contains no adjudicated YES/NO labels: {os.fspath(path)}"
        )
    detected_columns: dict[str, str] = {
        "image": image_column,
        **{target: label_columns[target] for target in targets},
    }
    if patient_column is not None:
        detected_columns["patient_id"] = patient_column
    if split_column is not None:
        detected_columns["split"] = split_column
    return NIHExpertLoad(
        rows=tuple(rows),
        source_rows=source_rows,
        skipped_rows_no_labels=skipped_rows,
        label_counts={
            target: dict(sorted(counts.items()))
            for target, counts in label_counts.items()
        },
        detected_columns=detected_columns,
    )


def prepare_nih_expert_manifest(
    *,
    official_train_val_path: str | os.PathLike[str],
    official_test_path: str | os.PathLike[str],
    output_csv_path: str | os.PathLike[str],
    output_json_path: str | os.PathLike[str],
    validation_csv_path: str | os.PathLike[str] | None = None,
    test_csv_path: str | os.PathLike[str] | None = None,
    combined_csv_path: str | os.PathLike[str] | None = None,
    include_fracture: bool = False,
    output_cohort: str = "all",
) -> Mapping[str, object]:
    """Validate, reconcile, and atomically emit a canonical CSV plus metadata."""

    if output_cohort not in {"development", "test", "all"}:
        raise ValueError("output_cohort must be 'development', 'test', or 'all'")
    _validate_input_mode(
        validation_csv_path=validation_csv_path,
        test_csv_path=test_csv_path,
        combined_csv_path=combined_csv_path,
    )
    input_paths = [
        official_train_val_path,
        official_test_path,
        *(
            [combined_csv_path]
            if combined_csv_path is not None
            else [validation_csv_path, test_csv_path]
        ),
    ]
    _validate_output_paths(
        output_csv_path,
        output_json_path,
        input_paths=[path for path in input_paths if path is not None],
    )

    official_train_val = read_official_nih_manifest(official_train_val_path)
    official_test = read_official_nih_manifest(official_test_path)
    train_val_set = set(official_train_val)
    test_set = set(official_test)
    overlap = train_val_set & test_set
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise ValueError(
            f"official NIH manifests overlap by {len(overlap)} image(s), "
            f"including: {examples}"
        )
    train_patients = {nih_patient_id(name) for name in train_val_set}
    test_patients = {nih_patient_id(name) for name in test_set}
    patient_overlap = train_patients & test_patients
    if patient_overlap:
        examples = ", ".join(sorted(patient_overlap)[:3])
        raise ValueError(
            f"official NIH manifests overlap by {len(patient_overlap)} patient(s), "
            f"including: {examples}"
        )

    sources: list[tuple[str, str | os.PathLike[str], NIHExpertLoad]]
    if combined_csv_path is not None:
        combined_load = load_google_nih_expert_csv(
            combined_csv_path,
            expected_split=None,
            include_fracture=include_fracture,
        )
        validation_rows = tuple(
            row for row in combined_load.rows if row.split == "validation"
        )
        test_rows = tuple(row for row in combined_load.rows if row.split == "test")
        if not validation_rows or not test_rows:
            raise ValueError(
                "combined expert CSV must contain at least one validation row and "
                "one test row"
            )
        sources = [("combined", combined_csv_path, combined_load)]
        loads_by_split = {
            "validation": _subset_load(combined_load, validation_rows),
            "test": _subset_load(combined_load, test_rows),
        }
    else:
        assert validation_csv_path is not None and test_csv_path is not None
        validation_load = load_google_nih_expert_csv(
            validation_csv_path,
            expected_split="validation",
            include_fracture=include_fracture,
        )
        test_load = load_google_nih_expert_csv(
            test_csv_path,
            expected_split="test",
            include_fracture=include_fracture,
        )
        sources = [
            ("validation", validation_csv_path, validation_load),
            ("test", test_csv_path, test_load),
        ]
        loads_by_split = {"validation": validation_load, "test": test_load}

    validation_rows = loads_by_split["validation"].rows
    test_rows = loads_by_split["test"].rows
    _reconcile_rows(validation_rows, train_val_set, "validation", "train_val")
    _reconcile_rows(test_rows, test_set, "test", "test")
    validation_names = {row.filename for row in validation_rows}
    test_names = {row.filename for row in test_rows}
    expert_overlap = validation_names & test_names
    if expert_overlap:
        examples = ", ".join(sorted(expert_overlap)[:3])
        raise ValueError(
            f"expert validation and test tables overlap by {len(expert_overlap)} "
            f"image(s), including: {examples}"
        )

    targets = CORE_TARGETS + ((OPTIONAL_TARGET,) if include_fracture else ())
    all_ordered_rows = tuple(
        sorted(
            (*validation_rows, *test_rows),
            key=lambda row: (0 if row.split == "validation" else 1, row.filename),
        )
    )
    if output_cohort == "development":
        ordered_rows = tuple(
            row for row in all_ordered_rows if row.split == "validation"
        )
    elif output_cohort == "test":
        ordered_rows = tuple(row for row in all_ordered_rows if row.split == "test")
    else:
        ordered_rows = all_ordered_rows
    csv_bytes = _canonical_csv_bytes(ordered_rows, targets)
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()

    metadata: dict[str, object] = {
        "schema_version": 1,
        "format": "nih_google_four_findings_canonical",
        "output_cohort": output_cohort,
        "targets": list(targets),
        "label_encoding": {
            "YES": 1,
            "NO": 0,
            "non_adjudicated": "blank; exclude independently for each target",
        },
        "rows": {
            "total": len(ordered_rows),
            "validation": sum(row.split == "validation" for row in ordered_rows),
            "test": sum(row.split == "test" for row in ordered_rows),
        },
        "validated_source_rows": {
            "total": len(all_ordered_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "sources": {
            role: {
                "path": os.fspath(path),
                "sha256": sha256_file(path),
                "source_rows": load.source_rows,
                "rows_emitted": len(load.rows),
                "rows_skipped_no_adjudicated_labels": load.skipped_rows_no_labels,
                "detected_columns": dict(load.detected_columns),
                "label_counts": {
                    target: dict(counts)
                    for target, counts in load.label_counts.items()
                },
            }
            for role, path, load in sources
        },
        "official_manifests": {
            "train_val": _official_manifest_metadata(
                official_train_val_path, official_train_val
            ),
            "test": _official_manifest_metadata(official_test_path, official_test),
        },
        "reconciliation": {
            "validation_rows_in_official_train_val": len(validation_rows),
            "test_rows_in_official_test": len(test_rows),
            "cross_split_image_overlap": 0,
            "official_patient_overlap": 0,
        },
        "output_csv": {
            "path": os.fspath(output_csv_path),
            "sha256": csv_sha256,
            "columns": ["filename", "patient_id", "split", *targets],
        },
    }
    json_bytes = (
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(output_csv_path, csv_bytes)
    _atomic_write_bytes(output_json_path, json_bytes)
    return metadata


def _open_csv_text(path: str | os.PathLike[str]) -> io.TextIOBase:
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8-sig", newline="")
    return open(path, mode="r", encoding="utf-8-sig", newline="")


def _normalize_header(value: str) -> str:
    return _HEADER_NORMALIZER_RE.sub("", value.casefold())


def _validate_headers(
    headers: Sequence[str], path: str | os.PathLike[str]
) -> None:
    if not headers or all(not header for header in headers):
        raise ValueError(f"expert CSV has an empty header: {os.fspath(path)}")
    blank_count = sum(not header for header in headers)
    if blank_count:
        raise ValueError(
            f"expert CSV {os.fspath(path)!r} has {blank_count} blank header(s)"
        )
    counts = Counter(_normalize_header(header) for header in headers)
    duplicates = [header for header, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"expert CSV {os.fspath(path)!r} has duplicate/ambiguous headers after "
            f"normalization: {', '.join(sorted(duplicates))}"
        )


def _detect_column(
    headers: Sequence[str], aliases: Sequence[str], purpose: str
) -> str:
    normalized_aliases = {_normalize_header(alias) for alias in aliases}
    matches = [
        header for header in headers if _normalize_header(header) in normalized_aliases
    ]
    if not matches:
        raise ValueError(
            f"could not detect {purpose} column; accepted names: "
            f"{', '.join(aliases)}; found headers: {', '.join(headers)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous {purpose} columns: {', '.join(matches)}; keep exactly one"
        )
    return matches[0]


def _detect_optional_column(
    headers: Sequence[str], aliases: Sequence[str], purpose: str
) -> str | None:
    normalized_aliases = {_normalize_header(alias) for alias in aliases}
    matches = [
        header for header in headers if _normalize_header(header) in normalized_aliases
    ]
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous {purpose} columns: {', '.join(matches)}; keep at most one"
        )
    return matches[0] if matches else None


def _row_is_blank(row: Mapping[str | None, object]) -> bool:
    return all(
        value is None or (isinstance(value, str) and not value.strip())
        for value in row.values()
    )


def _validate_source_patient_id(
    value: object,
    derived_patient_id: str,
    *,
    path: str | os.PathLike[str],
    line_number: int,
) -> None:
    text = "" if value is None else str(value).strip()
    if not text:
        return
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(
            f"{os.fspath(path)}:{line_number}: invalid source Patient ID {value!r}"
        )
    if int(text) != int(derived_patient_id):
        raise ValueError(
            f"{os.fspath(path)}:{line_number}: source Patient ID {value!r} does "
            f"not match filename-derived patient {derived_patient_id!r}"
        )


def _parse_split(
    value: object,
    *,
    expected_split: str | None,
    path: str | os.PathLike[str],
    line_number: int,
) -> str:
    if value is None or not str(value).strip():
        if expected_split is None:
            raise ValueError(
                f"{os.fspath(path)}:{line_number}: combined CSV row has blank split"
            )
        return expected_split
    normalized = str(value).strip().casefold()
    matches = [
        canonical
        for canonical, aliases in _VALID_SPLIT_VALUES.items()
        if normalized in aliases
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{os.fspath(path)}:{line_number}: unsupported split value {value!r}; "
            "expected validation/val or test"
        )
    split = matches[0]
    if expected_split is not None and split != expected_split:
        raise ValueError(
            f"{os.fspath(path)}:{line_number}: split column says {split!r}, "
            f"but this source was declared {expected_split!r}"
        )
    return split


def _parse_label(
    value: object,
    *,
    path: str | os.PathLike[str],
    line_number: int,
    column: str,
) -> tuple[int | None, str]:
    text = "" if value is None else str(value).strip()
    if text == "YES":
        return 1, "yes"
    if text == "NO":
        return 0, "no"
    if text.upper() in _MISSING_LABEL_TOKENS:
        return None, "skipped_non_adjudicated"
    raise ValueError(
        f"{os.fspath(path)}:{line_number}: {column!r} must be exactly YES or NO "
        f"for an adjudicated label, or a recognized non-adjudicated marker; got "
        f"{value!r}"
    )


def _validate_input_mode(
    *,
    validation_csv_path: str | os.PathLike[str] | None,
    test_csv_path: str | os.PathLike[str] | None,
    combined_csv_path: str | os.PathLike[str] | None,
) -> None:
    if combined_csv_path is not None:
        if validation_csv_path is not None or test_csv_path is not None:
            raise ValueError(
                "provide either combined_csv_path or both validation/test CSV paths, "
                "not both modes"
            )
        return
    if validation_csv_path is None or test_csv_path is None:
        raise ValueError(
            "both validation_csv_path and test_csv_path are required when no "
            "combined_csv_path is provided"
        )


def _validate_output_paths(
    output_csv_path: str | os.PathLike[str],
    output_json_path: str | os.PathLike[str],
    *,
    input_paths: Iterable[str | os.PathLike[str]],
) -> None:
    output_csv = Path(output_csv_path).expanduser().resolve()
    output_json = Path(output_json_path).expanduser().resolve()
    if output_csv == output_json:
        raise ValueError("output CSV and JSON paths must be different")
    input_resolved = {Path(path).expanduser().resolve() for path in input_paths}
    for output in (output_csv, output_json):
        if output in input_resolved:
            raise ValueError(f"refusing to overwrite input file with output: {output}")


def _reconcile_rows(
    rows: Sequence[NIHExpertRow],
    official_names: set[str],
    split: str,
    official_manifest_name: str,
) -> None:
    mismatched = sorted(
        row.filename for row in rows if row.filename not in official_names
    )
    if mismatched:
        examples = ", ".join(mismatched[:3])
        raise ValueError(
            f"{len(mismatched)} expert {split} row(s) are absent from the official "
            f"NIH {official_manifest_name} manifest, including: {examples}"
        )


def _subset_load(
    source: NIHExpertLoad, rows: Sequence[NIHExpertRow]
) -> NIHExpertLoad:
    split_rows = tuple(rows)
    counts: dict[str, dict[str, int]] = {}
    targets = tuple(split_rows[0].labels) if split_rows else ()
    for target in targets:
        target_values = [row.labels[target] for row in split_rows]
        counts[target] = {
            "no": sum(value == 0 for value in target_values),
            "skipped_non_adjudicated": sum(value is None for value in target_values),
            "yes": sum(value == 1 for value in target_values),
        }
    return NIHExpertLoad(
        rows=split_rows,
        source_rows=len(split_rows),
        skipped_rows_no_labels=0,
        label_counts=counts,
        detected_columns=source.detected_columns,
    )


def _canonical_csv_bytes(
    rows: Sequence[NIHExpertRow], targets: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["filename", "patient_id", "split", *targets],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "filename": row.filename,
                "patient_id": row.patient_id,
                "split": row.split,
                **{
                    target: (
                        "" if row.labels.get(target) is None else row.labels[target]
                    )
                    for target in targets
                },
            }
        )
    return buffer.getvalue().encode("utf-8")


def _official_manifest_metadata(
    path: str | os.PathLike[str], names: Sequence[str]
) -> Mapping[str, object]:
    canonical_bytes = "".join(f"{name}\n" for name in sorted(names)).encode("utf-8")
    return {
        "path": os.fspath(path),
        "sha256": sha256_file(path),
        "canonical_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "images": len(names),
        "patients": len({nih_patient_id(name) for name in names}),
    }


def _atomic_write_bytes(path: str | os.PathLike[str], content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
