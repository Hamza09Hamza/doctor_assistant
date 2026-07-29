"""Leakage-safe split utilities for NIH ChestX-ray14.

The NIH release defines two authoritative image manifests:

* ``train_val_list.txt`` -- the development pool
* ``test_list.txt`` -- the immutable test pool

Third-party mirrors may expose partitions named ``train``, ``valid``, and ``test``,
but those names are not evidence that the rows follow the NIH manifests.  The pure
functions in this module reconcile source filenames against the official manifests
before assigning any split.  Validation is derived only from ``train_val_list.txt``
and is grouped by the patient prefix in the original NIH filename.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


_NIH_FILENAME = re.compile(r"^\d{8}_\d{3}\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class NIHManifestReconciliation:
    """Result of matching available images to the two official NIH manifests."""

    official_train_val: frozenset[str]
    official_test: frozenset[str]
    available_train_val: frozenset[str]
    available_test: frozenset[str]
    unlisted_available: frozenset[str]
    missing_train_val: frozenset[str]
    missing_test: frozenset[str]
    patient_overlap: frozenset[str]

    @property
    def complete(self) -> bool:
        """Whether available image metadata exactly matches the official manifests."""

        return not (
            self.unlisted_available
            or self.missing_train_val
            or self.missing_test
            or self.patient_overlap
        )


@dataclass(frozen=True)
class NIHOfficialSplits:
    """Patient-disjoint development folds plus immutable official test membership."""

    train: frozenset[str]
    validation: frozenset[str]
    # `test` is the authoritative manifest membership, even when a local/mirror
    # source is missing files. Consumers must not silently replace missing members.
    test: frozenset[str]
    available_test: frozenset[str]
    missing_train_val: frozenset[str]
    missing_test: frozenset[str]
    unassigned_available: frozenset[str]
    validation_fraction: float
    seed: int


def canonical_nih_filename(value: str | os.PathLike[str]) -> str:
    """Return a source filename suitable for manifest comparison.

    Both POSIX and Windows separators are accepted because downloaded manifests and
    mirror metadata are often produced on different operating systems.
    """

    text = os.fspath(value).strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    if not name:
        raise ValueError("NIH image filename cannot be empty")
    return name


def is_nih_image_filename(value: str | os.PathLike[str]) -> bool:
    """Return whether ``value`` has the original NIH image-index shape."""

    try:
        name = canonical_nih_filename(value)
    except (TypeError, ValueError):
        return False
    return bool(_NIH_FILENAME.fullmatch(name))


def nih_patient_id(filename: str | os.PathLike[str]) -> str:
    """Extract the patient identifier from ``00000001_003.png``."""

    name = canonical_nih_filename(filename)
    if not _NIH_FILENAME.fullmatch(name):
        raise ValueError(
            f"invalid NIH image filename {name!r}; expected '00000001_000.png'"
        )
    patient_id, separator, _ = name.partition("_")
    if not separator or not patient_id:
        raise ValueError(
            f"cannot extract NIH patient ID from filename {name!r}; "
            "expected '<patient>_<image>.png'"
        )
    return patient_id


def read_nih_filename_manifest(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Read a strict NIH split manifest and reject malformed/duplicate identities."""

    names: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            name = canonical_nih_filename(line)
            if not _NIH_FILENAME.fullmatch(name):
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: invalid NIH image filename "
                    f"{name!r}; expected '00000001_000.png'"
                )
            if name in seen:
                raise ValueError(
                    f"{os.fspath(path)}:{line_number}: duplicate NIH image filename "
                    f"{name!r}"
                )
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"NIH filename manifest is empty: {os.fspath(path)}")
    return tuple(names)


def filename_from_mirror_row(row: Mapping[str, object]) -> str | None:
    """Recover an original NIH filename from mirror metadata when it is present.

    The function deliberately returns ``None`` for opaque cache/temp paths.  A
    generated row number or patient ID must never be presented as an official image
    identity.
    """

    direct_fields = (
        "Image Index",
        "image_index",
        "image_id",
        "filename",
        "file_name",
    )
    candidates: list[object] = [row.get(field) for field in direct_fields]
    image = row.get("image")
    if isinstance(image, Mapping):
        candidates.extend((image.get("path"), image.get("filename")))
    else:
        candidates.append(getattr(image, "filename", None))

    for candidate in candidates:
        if isinstance(candidate, (str, os.PathLike)) and is_nih_image_filename(candidate):
            return canonical_nih_filename(candidate)
    return None


def reconcile_nih_official_manifests(
    available_filenames: Iterable[str | os.PathLike[str]],
    official_train_val_filenames: Iterable[str | os.PathLike[str]],
    official_test_filenames: Iterable[str | os.PathLike[str]],
    *,
    require_complete: bool = False,
    require_patient_disjoint: bool = True,
) -> NIHManifestReconciliation:
    """Match available image identities to the authoritative NIH manifests.

    Images absent from both official manifests are never silently assigned to a
    development or test fold.  Missing and unlisted identities are returned as
    diagnostics; ``require_complete=True`` turns either condition into an error.
    """

    available = _unique_filenames(available_filenames, "available image metadata")
    train_val = _unique_filenames(
        official_train_val_filenames, "official train_val manifest"
    )
    test = _unique_filenames(official_test_filenames, "official test manifest")
    if not train_val:
        raise ValueError("official train_val manifest is empty")
    if not test:
        raise ValueError("official test manifest is empty")

    image_overlap = train_val & test
    if image_overlap:
        examples = ", ".join(sorted(image_overlap)[:3])
        raise ValueError(
            "official NIH manifests overlap by "
            f"{len(image_overlap)} image(s), including: {examples}"
        )

    train_patients = {nih_patient_id(name) for name in train_val}
    test_patients = {nih_patient_id(name) for name in test}
    patient_overlap = frozenset(train_patients & test_patients)
    if require_patient_disjoint and patient_overlap:
        examples = ", ".join(sorted(patient_overlap)[:3])
        raise ValueError(
            "official NIH manifests overlap by "
            f"{len(patient_overlap)} patient(s), including: {examples}"
        )

    official = train_val | test
    reconciliation = NIHManifestReconciliation(
        official_train_val=frozenset(train_val),
        official_test=frozenset(test),
        available_train_val=frozenset(available & train_val),
        available_test=frozenset(available & test),
        unlisted_available=frozenset(available - official),
        missing_train_val=frozenset(train_val - available),
        missing_test=frozenset(test - available),
        patient_overlap=patient_overlap,
    )
    if require_complete and not reconciliation.complete:
        raise ValueError(
            "available NIH metadata does not exactly match the official manifests: "
            f"{len(reconciliation.unlisted_available)} unlisted available, "
            f"{len(reconciliation.missing_train_val)} missing train_val, "
            f"{len(reconciliation.missing_test)} missing test, "
            f"{len(reconciliation.patient_overlap)} overlapping patients"
        )
    return reconciliation


def make_patient_disjoint_nih_splits(
    available_filenames: Iterable[str | os.PathLike[str]],
    official_train_val_filenames: Iterable[str | os.PathLike[str]],
    official_test_filenames: Iterable[str | os.PathLike[str]],
    *,
    validation_fraction: float = 0.1,
    seed: int = 42,
    require_complete: bool = False,
) -> NIHOfficialSplits:
    """Create reproducible train/validation folds without changing official test.

    The official test set is copied directly from reconciled manifest membership.
    Only patients in the official ``train_val`` pool participate in the development
    split.  Grouping is by patient, so multiple images from one patient cannot cross
    train and validation.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    reconciliation = reconcile_nih_official_manifests(
        available_filenames,
        official_train_val_filenames,
        official_test_filenames,
        require_complete=require_complete,
        require_patient_disjoint=True,
    )

    groups: dict[str, list[str]] = {}
    for name in sorted(reconciliation.available_train_val):
        groups.setdefault(nih_patient_id(name), []).append(name)
    if len(groups) < 2:
        raise ValueError(
            "at least two development patients are required to make train/validation"
        )

    patient_ids = sorted(groups)
    random.Random(seed).shuffle(patient_ids)
    target_images = validation_fraction * len(reconciliation.available_train_val)
    validation: set[str] = set()
    # Keep at least one complete patient group in train, even for tiny synthetic
    # manifests or an unusually high requested validation fraction.
    for patient_id in patient_ids[:-1]:
        if len(validation) >= target_images:
            break
        validation.update(groups[patient_id])

    train = set(reconciliation.available_train_val) - validation
    if not validation or not train:
        raise ValueError("patient grouping produced an empty train or validation fold")

    train_patients = {nih_patient_id(name) for name in train}
    validation_patients = {nih_patient_id(name) for name in validation}
    if train_patients & validation_patients:  # defensive invariant
        raise RuntimeError("patient leakage detected while constructing development folds")

    return NIHOfficialSplits(
        train=frozenset(train),
        validation=frozenset(validation),
        # This line is intentionally independent of validation_fraction and seed.
        test=reconciliation.official_test,
        available_test=reconciliation.available_test,
        missing_train_val=reconciliation.missing_train_val,
        missing_test=reconciliation.missing_test,
        unassigned_available=reconciliation.unlisted_available,
        validation_fraction=validation_fraction,
        seed=seed,
    )


def _unique_filenames(
    values: Iterable[str | os.PathLike[str]], source: str
) -> set[str]:
    names: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        name = canonical_nih_filename(value)
        if not _NIH_FILENAME.fullmatch(name):
            raise ValueError(
                f"{source} contains invalid NIH image filename {name!r}; "
                "expected '00000001_000.png'"
            )
        if name in names:
            duplicates.add(name)
        names.add(name)
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:3])
        raise ValueError(
            f"{source} contains {len(duplicates)} duplicate filename(s), including: "
            f"{examples}"
        )
    return names
