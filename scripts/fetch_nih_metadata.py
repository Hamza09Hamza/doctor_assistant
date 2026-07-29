"""Fetch and verify NIH ChestX-ray14 metadata and Google expert labels.

This downloads the three small metadata files needed to reconcile expert labels
with the original NIH split contract, plus the public TorchXRayVision mirror of
Google's four-finding adjudicated label table.  Every source is pinned to an
immutable commit and an independently verified SHA-256 digest.

Example::

    python scripts/fetch_nih_metadata.py \
        --output-dir /content/drive/MyDrive/doctor_assistant/nih_metadata

The image-fetch helper in ``fetch_nih_expert_images.py`` is deliberately
separate: that helper uses resized third-party images for development runs only.
These metadata files preserve official image identities and split membership,
but their presence does not make resized mirror images official evaluation
evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request


YEIGEN_DATASET = "yeigen/nih-chest-xray"
YEIGEN_REVISION = "c0b558ec72f1ce434f7355f0f5cf914e2d62c60a"
_RESOLVE_ROOT = (
    "https://huggingface.co/datasets/"
    f"{YEIGEN_DATASET}/resolve/{YEIGEN_REVISION}"
)
_NIH_FILENAME_RE = re.compile(r"^\d{8}_\d{3}\.png$")
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_USER_AGENT = "doctor-assistant-nih-metadata-fetch/1"

NIH_METADATA_FILES: Mapping[str, Mapping[str, Any]] = {
    "Data_Entry_2017.csv": {
        "url": f"{_RESOLVE_ROOT}/Data_Entry_2017.csv?download=true",
        "sha256": "88f75094e25ccc0c6f1f9cdfd4b2f94f9379a0ae07d5ff4dcf94242707b07462",
        "rows": 112_120,
    },
    "train_val_list.txt": {
        "url": f"{_RESOLVE_ROOT}/train_val_list.txt?download=true",
        "sha256": "61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81",
        "rows": 86_524,
    },
    "test_list.txt": {
        "url": f"{_RESOLVE_ROOT}/test_list.txt?download=true",
        "sha256": "38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc",
        "rows": 25_596,
    },
}

TORCHXRAYVISION_COMMIT = "6fa28a2fbfb1b30fcb5f8f501626d91177ef593c"
GOOGLE_EXPERT_LABEL_FILENAME = "google2019_nih-chest-xray-labels.csv.gz"
GOOGLE_EXPERT_LABEL_SPEC: Mapping[str, Any] = {
    "url": (
        "https://raw.githubusercontent.com/mlmed/torchxrayvision/"
        f"{TORCHXRAYVISION_COMMIT}/torchxrayvision/data/"
        f"{GOOGLE_EXPERT_LABEL_FILENAME}"
    ),
    "sha256": "1d1b846e463753ed0219f67e21b10c8aef381897dbd64865c8191aa5794e8048",
    "rows": 4_376,
    "split_rows": {"val": 2_414, "test": 1_962},
    "google_documented_split_rows": {"val": 2_412, "test": 1_962},
    "label_counts": {
        "Fracture": {"NO": 4_190, "YES": 186},
        "Pneumothorax": {"NO": 4_138, "YES": 238},
        "Airspace opacity": {"NO": 2_210, "YES": 2_166},
        "Nodule or mass": {"NO": 3_771, "YES": 605},
    },
}
PINNED_FILES: Mapping[str, Mapping[str, Any]] = {
    **NIH_METADATA_FILES,
    GOOGLE_EXPERT_LABEL_FILENAME: GOOGLE_EXPERT_LABEL_SPEC,
}
_GOOGLE_REQUIRED_COLUMNS = (
    "Image Index",
    "Patient ID",
    "Fracture",
    "Pneumothorax",
    "Airspace opacity",
    "Nodule or mass",
    "Set Id",
)


class MetadataFetchError(RuntimeError):
    """Raised when a download or validation check fails closed."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            pass
    return min(2.0 ** attempt, 30.0) + random.uniform(0.0, 0.25)


def _download_to_path(
    url: str,
    destination: Path,
    *,
    attempts: int = 6,
    timeout: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Download ``url`` atomically, resuming a retained ``.part`` when possible."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    last_error: BaseException | None = None

    for attempt in range(attempts):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                append = existing > 0 and status == 206
                if existing and status == 200:
                    append = False
                if status not in (200, 206):
                    raise MetadataFetchError(
                        f"unexpected HTTP {status} while fetching {url}"
                    )
                with open(partial, "ab" if append else "wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            os.replace(partial, destination)
            return
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in _RETRYABLE_STATUS or attempt + 1 >= attempts:
                break
            sleep(_retry_delay(attempt, error.headers.get("Retry-After")))
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 >= attempts:
                break
            sleep(_retry_delay(attempt))

    raise MetadataFetchError(
        f"failed to download {url} after {attempts} attempt(s): {last_error}"
    ) from last_error


def _read_filename_list(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            name = line.strip()
            if not name:
                continue
            if not _NIH_FILENAME_RE.fullmatch(name):
                raise MetadataFetchError(
                    f"{path}:{line_number}: malformed NIH filename {name!r}"
                )
            if name in seen:
                raise MetadataFetchError(
                    f"{path}:{line_number}: duplicate NIH filename {name!r}"
                )
            seen.add(name)
            names.append(name)
    return tuple(names)


def validate_metadata_files(
    output_dir: str | os.PathLike[str],
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    """Validate schemas, counts, uniqueness, and split-to-table reconciliation."""

    root = Path(output_dir)
    counts = {
        name: int(spec["rows"]) for name, spec in NIH_METADATA_FILES.items()
    }
    if expected_counts is not None:
        counts.update({name: int(value) for name, value in expected_counts.items()})

    data_entry_path = root / "Data_Entry_2017.csv"
    with data_entry_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Image Index" not in reader.fieldnames:
            raise MetadataFetchError(
                "Data_Entry_2017.csv is missing the required 'Image Index' column"
            )
        entry_names: list[str] = []
        seen_entries: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            name = (row.get("Image Index") or "").strip()
            if not _NIH_FILENAME_RE.fullmatch(name):
                raise MetadataFetchError(
                    f"{data_entry_path}:{line_number}: malformed Image Index {name!r}"
                )
            if name in seen_entries:
                raise MetadataFetchError(
                    f"{data_entry_path}:{line_number}: duplicate Image Index {name!r}"
                )
            seen_entries.add(name)
            entry_names.append(name)

    train_val = _read_filename_list(root / "train_val_list.txt")
    test = _read_filename_list(root / "test_list.txt")
    actual_counts = {
        "Data_Entry_2017.csv": len(entry_names),
        "train_val_list.txt": len(train_val),
        "test_list.txt": len(test),
    }
    for name, expected in counts.items():
        actual = actual_counts[name]
        if actual != expected:
            raise MetadataFetchError(
                f"{name} has {actual:,} row(s), expected {expected:,}"
            )

    train_set = set(train_val)
    test_set = set(test)
    overlap = train_set & test_set
    if overlap:
        examples = ", ".join(sorted(overlap)[:3])
        raise MetadataFetchError(
            f"official split lists overlap by {len(overlap)} image(s): {examples}"
        )
    split_union = train_set | test_set
    entry_set = set(entry_names)
    missing_from_splits = entry_set - split_union
    missing_from_table = split_union - entry_set
    if missing_from_splits or missing_from_table:
        raise MetadataFetchError(
            "Data_Entry_2017.csv does not exactly reconcile with the two split "
            f"lists ({len(missing_from_splits)} table-only, "
            f"{len(missing_from_table)} split-only)"
        )
    return {
        "rows": actual_counts,
        "split_overlap": 0,
        "split_union_rows": len(split_union),
        "data_entry_unique_rows": len(entry_set),
    }


def validate_google_expert_labels(
    path: str | os.PathLike[str],
    *,
    expected_rows: int | None = None,
    expected_split_rows: Mapping[str, int] | None = None,
    expected_label_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> Mapping[str, Any]:
    """Validate the pinned combined Google label mirror without normalizing it.

    The locked SHA-256 identifies the exact source bytes.  These semantic checks
    additionally prevent a valid gzip containing a truncated or misparsed table
    from entering the workflow.
    """

    label_path = Path(path)
    row_limit = (
        int(GOOGLE_EXPERT_LABEL_SPEC["rows"])
        if expected_rows is None
        else int(expected_rows)
    )
    split_limits = dict(
        GOOGLE_EXPERT_LABEL_SPEC["split_rows"]
        if expected_split_rows is None
        else expected_split_rows
    )
    label_limits = {
        str(label): {str(value): int(count) for value, count in counts.items()}
        for label, counts in (
            GOOGLE_EXPERT_LABEL_SPEC["label_counts"]
            if expected_label_counts is None
            else expected_label_counts
        ).items()
    }
    split_counts = {split: 0 for split in split_limits}
    label_counts = {
        label: {"NO": 0, "YES": 0} for label in label_limits
    }
    seen: set[str] = set()
    rows = 0
    try:
        with gzip.open(
            label_path, mode="rt", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise MetadataFetchError(
                    f"{label_path} has no CSV header"
                )
            missing = set(_GOOGLE_REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing:
                raise MetadataFetchError(
                    f"{label_path} is missing required column(s): "
                    f"{', '.join(sorted(missing))}"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise MetadataFetchError(
                        f"{label_path}:{line_number}: row has more fields than header"
                    )
                rows += 1
                filename = (row.get("Image Index") or "").strip()
                if not _NIH_FILENAME_RE.fullmatch(filename):
                    raise MetadataFetchError(
                        f"{label_path}:{line_number}: malformed Image Index "
                        f"{filename!r}"
                    )
                if filename in seen:
                    raise MetadataFetchError(
                        f"{label_path}:{line_number}: duplicate Image Index "
                        f"{filename!r}"
                    )
                seen.add(filename)
                patient = (row.get("Patient ID") or "").strip()
                if not patient.isdecimal() or int(patient) != int(
                    filename.split("_", 1)[0]
                ):
                    raise MetadataFetchError(
                        f"{label_path}:{line_number}: Patient ID {patient!r} "
                        f"does not match {filename!r}"
                    )
                split = (row.get("Set Id") or "").strip()
                if split not in split_counts:
                    raise MetadataFetchError(
                        f"{label_path}:{line_number}: unexpected Set Id {split!r}"
                    )
                split_counts[split] += 1
                for label in label_counts:
                    value = (row.get(label) or "").strip()
                    if value not in {"NO", "YES"}:
                        raise MetadataFetchError(
                            f"{label_path}:{line_number}: {label} must be "
                            f"adjudicated YES/NO, got {value!r}"
                        )
                    label_counts[label][value] += 1
    except (OSError, EOFError, UnicodeError) as error:
        raise MetadataFetchError(
            f"{label_path} is not a complete UTF-8 gzip CSV: {error}"
        ) from error

    if rows != row_limit:
        raise MetadataFetchError(
            f"{label_path.name} has {rows:,} row(s), expected {row_limit:,}"
        )
    if split_counts != split_limits:
        raise MetadataFetchError(
            f"{label_path.name} split counts are {split_counts}, expected "
            f"{split_limits}"
        )
    if label_counts != label_limits:
        raise MetadataFetchError(
            f"{label_path.name} label counts are {label_counts}, expected "
            f"{label_limits}"
        )
    return {
        "rows": rows,
        "unique_images": len(seen),
        "split_rows": split_counts,
        "label_counts": label_counts,
        "all_four_findings_adjudicated_yes_no": True,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def fetch_metadata(
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    downloader: Callable[..., None] = _download_to_path,
) -> Mapping[str, Any]:
    """Fetch all pinned files, validate them together, and write provenance."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, Mapping[str, Any]] = {}

    for name, spec in PINNED_FILES.items():
        destination = root / name
        expected_hash = str(spec["sha256"])
        if force or not destination.is_file() or sha256_file(destination) != expected_hash:
            downloader(str(spec["url"]), destination)
        actual_hash = sha256_file(destination)
        if actual_hash != expected_hash:
            raise MetadataFetchError(
                f"{name} SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
            )
        files[name] = {
            "path": str(destination),
            "url": spec["url"],
            "sha256": actual_hash,
            "bytes": destination.stat().st_size,
        }

    metadata_validation = validate_metadata_files(root)
    expert_label_validation = validate_google_expert_labels(
        root / GOOGLE_EXPERT_LABEL_FILENAME
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "doctor_assistant.nih_metadata_and_expert_labels",
        "sources": {
            "nih_metadata": {
                "dataset": YEIGEN_DATASET,
                "revision": YEIGEN_REVISION,
                "role": (
                    "byte-verified mirror of NIH metadata; image identity and "
                    "split reconciliation only"
                ),
            },
            "google_expert_labels": {
                "repository": "mlmed/torchxrayvision",
                "revision": TORCHXRAYVISION_COMMIT,
                "role": (
                    "public mirror of Google's four-finding radiologist-"
                    "adjudicated NIH table"
                ),
            },
        },
        "files": files,
        "validation": {
            "nih_metadata": metadata_validation,
            "google_expert_labels": expert_label_validation,
        },
        "known_source_discrepancy": {
            "google_documented_split_rows": dict(
                GOOGLE_EXPERT_LABEL_SPEC["google_documented_split_rows"]
            ),
            "pinned_mirror_split_rows": dict(
                GOOGLE_EXPERT_LABEL_SPEC["split_rows"]
            ),
            "validation_row_delta": 2,
            "required_action_before_locked_test": (
                "independently reconcile the pinned mirror against Google's "
                "direct release"
            ),
        },
        "warning": (
            "These manifests preserve NIH identities/splits and development "
            "expert labels. They do not certify any third-party image mirror as "
            "original NIH pixel data, and the expert-label mirror is not accepted "
            "for a locked test until its two-row documentation discrepancy is "
            "independently reconciled."
        ),
    }
    provenance_path = root / "nih_metadata.provenance.json"
    _atomic_json(provenance_path, provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "directory for NIH metadata and the pinned combined Google expert "
            "label CSV"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch even when an existing file has the pinned SHA-256",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        provenance = fetch_metadata(args.output_dir, force=args.force)
    except (OSError, MetadataFetchError) as error:
        parser.error(str(error))
    metadata = provenance["validation"]["nih_metadata"]
    rows = metadata["rows"]
    expert = provenance["validation"]["google_expert_labels"]
    print(
        "Verified NIH metadata: "
        f"{rows['Data_Entry_2017.csv']:,} table rows, "
        f"{rows['train_val_list.txt']:,} train/val, "
        f"{rows['test_list.txt']:,} test"
    )
    print(
        "Verified Google expert-label mirror: "
        f"{expert['split_rows']['val']:,} validation + "
        f"{expert['split_rows']['test']:,} test rows "
        "(development source; direct-release reconciliation still required "
        "before locked testing)"
    )
    print(Path(args.output_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
