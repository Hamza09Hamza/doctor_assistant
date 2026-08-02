"""Ingest original NIH ChestX-ray14 pixels behind a multi-source consensus check.

THIS SCRIPT DOES NOT DOWNLOAD ANYTHING ITSELF. There is no NIH-published per-file
checksum manifest to pin against, so a single download -- however official-looking
its host -- cannot prove archive/source identity on its own. Instead this script
takes two or more already-downloaded, independently-operated local copies of the
official NIH release (e.g. a direct download from the NIH Clinical Center host plus
the NIH-attributed academictorrents release, each already extracted into a flat
directory of canonical ``00000001_000.png``-style filenames) and only accepts a file
once its SHA-256 is identical across at least ``--min-independent-sources`` of them.
Any disagreement between sources, or too few sources carrying a given file, is a
hard failure that names the exact file -- there is no silent majority-vote fallback.

Output is a schema-2 ``doctor_assistant.nih_image_provenance`` artifact
(``ORIGINAL_NIH_PIXELS.provenance.json``) that ``scripts/benchmark_kad.py``'s
``load_image_provenance`` can accept with ``original_nih_pixels=true``. Schema-1
artifacts (e.g. from ``fetch_nih_expert_images.py``) can never make that claim.

Example::

    python scripts/fetch_nih_original_images.py \
        --manifest artifacts/nih_expert/labels.csv \
        --output-images-dir /content/drive/MyDrive/doctor_assistant/nih_original \
        --source "NIH Clinical Center direct release=/mnt/nih_direct" \
        --source "academictorrents (infohash 557481faacd824c83fbf57dcf7b6da9383b3235a)=/mnt/nih_torrent"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.export_kad_query_pack import sha256_file
from scripts.fetch_nih_expert_images import ManifestTarget, load_canonical_manifest


class OriginalPixelIngestionError(RuntimeError):
    """Raised when multi-source verification or ingestion fails closed."""


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


def _verify_file_across_sources(
    filename: str,
    sources: Mapping[str, Path],
    *,
    min_independent_sources: int,
) -> tuple[str, int, Path]:
    """Return (agreed sha256, agreeing source count, a verified source path).

    Fails closed: too few sources carrying the file, or any disagreement among the
    sources that do carry it, is an error naming the exact file -- never a silent
    majority vote.
    """

    hashes_by_source: dict[str, str] = {}
    paths_by_source: dict[str, Path] = {}
    missing_sources: list[str] = []
    for name, root in sources.items():
        candidate = root / filename
        if not candidate.is_file():
            missing_sources.append(name)
            continue
        hashes_by_source[name] = sha256_file(candidate)
        paths_by_source[name] = candidate

    if len(hashes_by_source) < min_independent_sources:
        raise OriginalPixelIngestionError(
            f"{filename}: only {len(hashes_by_source)} of {len(sources)} source(s) "
            f"have this file (missing from: {', '.join(sorted(missing_sources)) or 'none'}); "
            f"need at least {min_independent_sources}"
        )
    distinct = set(hashes_by_source.values())
    if len(distinct) > 1:
        detail = ", ".join(
            f"{name}={digest[:12]}" for name, digest in sorted(hashes_by_source.items())
        )
        raise OriginalPixelIngestionError(
            f"{filename}: sources disagree on SHA-256 ({detail}); refusing to guess"
        )
    agreed_hash = next(iter(distinct))
    chosen_name = next(iter(hashes_by_source))
    return agreed_hash, len(hashes_by_source), paths_by_source[chosen_name]


def verify_and_ingest_original_images(
    manifest: str | os.PathLike[str],
    output_images_dir: str | os.PathLike[str],
    sources: Mapping[str, str | os.PathLike[str]],
    *,
    cohort: str = "development",
    acknowledge_test_is_development_only: bool = False,
    min_independent_sources: int = 2,
) -> Mapping[str, Any]:
    """Verify every selected NIH image across independent sources, then ingest it."""

    if len(sources) < 2:
        raise ValueError("at least two independently-operated sources are required")
    if min_independent_sources < 2 or min_independent_sources > len(sources):
        raise ValueError(
            "min_independent_sources must be between 2 and the number of sources "
            "supplied"
        )
    source_roots: dict[str, Path] = {}
    for name, path in sources.items():
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise OriginalPixelIngestionError(
                f"source {name!r} is not a directory: {root}"
            )
        source_roots[name] = root

    manifest_path = Path(manifest).expanduser().resolve()
    manifest_hash = sha256_file(manifest_path)
    targets: tuple[ManifestTarget, ...] = load_canonical_manifest(
        manifest_path,
        cohort=cohort,
        acknowledge_test_is_development_only=acknowledge_test_is_development_only,
    )

    images_dir = Path(output_images_dir).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, UnidentifiedImageError

    records: list[dict[str, Any]] = []
    declared_size: tuple[int, int] | None = None
    for index, target in enumerate(targets, start=1):
        agreed_hash, agreeing_count, source_file = _verify_file_across_sources(
            target.filename,
            source_roots,
            min_independent_sources=min_independent_sources,
        )
        destination = images_dir / target.filename
        if not destination.is_file() or sha256_file(destination) != agreed_hash:
            shutil.copy2(source_file, destination)
        if sha256_file(destination) != agreed_hash:
            raise OriginalPixelIngestionError(
                f"copied file failed verification: {target.filename}"
            )
        try:
            with Image.open(destination) as opened:
                size = tuple(opened.size)
        except (OSError, UnidentifiedImageError) as exc:
            raise OriginalPixelIngestionError(
                f"could not decode verified image {target.filename}: {exc}"
            ) from exc
        if declared_size is None:
            declared_size = size
        elif size != declared_size:
            raise OriginalPixelIngestionError(
                f"{target.filename} is {size}, expected {declared_size} -- every "
                "image bound to one provenance artifact must share one resolution"
            )
        records.append(
            {
                "filename": target.filename,
                "expert_split": target.expert_split,
                "output_sha256": agreed_hash,
                "agreeing_source_count": agreeing_count,
            }
        )
        if index % 500 == 0 or index == len(targets):
            print(f"Verified {index:,}/{len(targets):,} image(s) across independent sources")

    if declared_size is None:
        raise OriginalPixelIngestionError("no images were selected for this cohort")

    source_names = list(source_roots)
    provenance: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "doctor_assistant.nih_image_provenance",
        "source": (
            "Multi-source verified NIH ChestX-ray14 original pixels: "
            + "; ".join(source_names)
        ),
        "resolution": {"width": declared_size[0], "height": declared_size[1]},
        "original_nih_pixels": True,
        "development_only": False,
        "official_or_final_evidence_allowed": True,
        "canonical_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "cohort": cohort,
            "rows_selected": len(targets),
        },
        "archive_verification": {
            "method": "multi_source_sha256_consensus_v1",
            "minimum_independent_sources": min_independent_sources,
            "primary_source": {
                "name": source_names[0],
                "identifier": str(source_roots[source_names[0]]),
            },
            "corroborating_sources": [
                {"name": name, "identifier": str(source_roots[name])}
                for name in source_names[1:]
            ],
            "agreement": "all_selected_files_sha256_identical_across_all_sources",
        },
        "images": records,
    }
    provenance_path = images_dir / "ORIGINAL_NIH_PIXELS.provenance.json"
    _atomic_json(provenance_path, provenance)
    return provenance


def _parse_source_argument(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--source must be in the form NAME=PATH, e.g. "
            "'NIH Clinical Center direct release=/mnt/nih_direct'"
        )
    return name.strip(), path.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="canonical CSV emitted by prepare_nih_expert_manifest.py",
    )
    parser.add_argument(
        "--output-images-dir",
        type=Path,
        required=True,
        help="directory in which verified original NIH images will be written",
    )
    parser.add_argument(
        "--cohort",
        choices=("development", "test", "all"),
        default="development",
        help="safe default is development; test/all require an acknowledgement",
    )
    parser.add_argument(
        "--acknowledge-development-only-test-use",
        action="store_true",
        help="required to ingest locked test identities",
    )
    parser.add_argument(
        "--source",
        dest="sources",
        type=_parse_source_argument,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "an independently-operated local directory of already-obtained, "
            "extracted original NIH images (flat, canonical NIH filenames). "
            "Repeat at least twice; the first --source is recorded as the "
            "primary source, the rest as corroborating sources."
        ),
    )
    parser.add_argument(
        "--min-independent-sources",
        type=int,
        default=2,
        help="minimum number of agreeing sources required per file (default 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.sources) < 2:
        parser.error("at least two --source entries are required for multi-source consensus")
    names = [name for name, _ in args.sources]
    if len(set(names)) != len(names):
        parser.error("--source names must be unique")
    sources = dict(args.sources)
    try:
        provenance = verify_and_ingest_original_images(
            args.manifest,
            args.output_images_dir,
            sources,
            cohort=args.cohort,
            acknowledge_test_is_development_only=(
                args.acknowledge_development_only_test_use
            ),
            min_independent_sources=args.min_independent_sources,
        )
    except (OSError, ValueError, OriginalPixelIngestionError) as error:
        parser.error(str(error))
    print(
        f"Verified {len(provenance['images']):,} original NIH image(s) across "
        f"{len(sources)} independent source(s)"
    )
    print(Path(args.output_images_dir).expanduser().resolve() / "ORIGINAL_NIH_PIXELS.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
