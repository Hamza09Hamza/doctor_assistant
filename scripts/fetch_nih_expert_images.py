"""Fetch only expert-labeled NIH images from a filename-preserving 320px mirror.

THIS SCRIPT IS FOR DEVELOPMENT/PIPELINE DEBUGGING ONLY.

``arudaev/chest-xray-14-320`` preserves original NIH filenames and the official
test membership, which makes it useful for a cheap KAD smoke/development run.
Its images are third-party 320x320 JPEG re-encodes, not the original NIH PNG
pixels.  Outputs from this script are therefore *prohibited* as official or
final evaluation evidence.

The least-download lookup path is:

1. try Hugging Face dataset-server's exact ``/filter`` index;
2. if that index is unavailable/partial, stream only the Parquet ``filename``
   column (never the image column) to recover exact split/row locations;
3. request only the 100-row viewer buckets containing requested images, then
   download only those signed image assets.

Example (the safe default fetches development rows, never locked test rows)::

    python scripts/fetch_nih_expert_images.py \
        --manifest artifacts/nih_expert/labels.csv \
        --output-images-dir /content/drive/MyDrive/doctor_assistant/nih_expert_320

Fetching canonical ``test`` rows requires an explicit development-only
acknowledgement and still does not make the resulting images valid final
evidence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, UnidentifiedImageError


MIRROR_DATASET = "arudaev/chest-xray-14-320"
MIRROR_CONFIG = "default"
MIRROR_REVISION = "1c9e054e3336a473be6c01d77cdedf96442e2bad"
MIRROR_SPLIT_ROWS: Mapping[str, int] = {
    "train": 77_967,
    "validation": 8_557,
    "test": 25_596,
}
MIRROR_SPLIT_SHARDS: Mapping[str, int] = {
    "train": 12,
    "validation": 12,
    "test": 12,
}
DATASET_SERVER = "https://datasets-server.huggingface.co"
HUGGING_FACE = "https://huggingface.co"
EXPECTED_IMAGE_SIZE = (320, 320)
_NIH_FILENAME_RE = re.compile(r"^\d{8}_\d{3}\.png$")
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_USER_AGENT = "doctor-assistant-nih-expert-development-fetch/1"
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_FILTER_BATCH_SIZE = 40
_VIEWER_BUCKET_SIZE = 100
_MAX_VIEWER_WORKERS = 2
_DEVELOPMENT_WARNING = (
    "DEVELOPMENT ONLY — NOT OFFICIAL/FINAL EVIDENCE. These files are 320x320 "
    "third-party JPEG re-encodes converted to PNG, not original NIH image pixels."
)


class ExpertImageFetchError(RuntimeError):
    """Raised when identity, transport, or image validation fails closed."""


@dataclass(frozen=True)
class ManifestTarget:
    filename: str
    expert_split: str

    @property
    def candidate_mirror_splits(self) -> tuple[str, ...]:
        if self.expert_split == "test":
            return ("test",)
        return ("train", "validation")


@dataclass(frozen=True)
class MirrorLocation:
    filename: str
    expert_split: str
    mirror_split: str
    row_index: int


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_nih_filename(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("filename must be text")
    name = value.strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not _NIH_FILENAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid NIH filename {name!r}; expected '00000001_000.png'"
        )
    return name


def load_canonical_manifest(
    path: str | os.PathLike[str],
    *,
    cohort: str = "development",
    acknowledge_test_is_development_only: bool = False,
) -> tuple[ManifestTarget, ...]:
    """Read the output of ``prepare_nih_expert_manifest.py`` strictly."""

    if cohort not in {"development", "test", "all"}:
        raise ValueError("cohort must be 'development', 'test', or 'all'")
    if cohort in {"test", "all"} and not acknowledge_test_is_development_only:
        raise ValueError(
            "fetching locked expert test images requires "
            "--acknowledge-development-only-test-use; resized mirror images remain "
            "prohibited as final evidence"
        )

    manifest = Path(path)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"canonical manifest has no header: {manifest}")
        missing = {"filename", "split"} - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"canonical manifest is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )
        targets: list[ManifestTarget] = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                filename = _canonical_nih_filename(row.get("filename"))
            except ValueError as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
            split = (row.get("split") or "").strip().casefold()
            if split not in {"validation", "test"}:
                raise ValueError(
                    f"{manifest}:{line_number}: split must be 'validation' or "
                    f"'test', got {split!r}"
                )
            if filename in seen:
                raise ValueError(
                    f"{manifest}:{line_number}: duplicate filename {filename!r}"
                )
            seen.add(filename)
            if cohort == "development" and split != "validation":
                continue
            if cohort == "test" and split != "test":
                continue
            patient_id = (row.get("patient_id") or "").strip()
            if patient_id and patient_id != filename.split("_", 1)[0]:
                raise ValueError(
                    f"{manifest}:{line_number}: patient_id {patient_id!r} does not "
                    f"match filename {filename!r}"
                )
            targets.append(
                ManifestTarget(filename=filename, expert_split=split)
            )
    if not targets:
        raise ValueError(f"canonical manifest contains no rows for cohort {cohort!r}")
    return tuple(targets)


def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 310.0)
        except ValueError:
            pass
    return min(2.0 ** attempt, 30.0) + random.uniform(0.0, 0.25)


def _rate_limit_reset_seconds(headers: Mapping[str, str]) -> str | None:
    retry_after = headers.get("Retry-After")
    if retry_after:
        return retry_after
    rate_limit = headers.get("RateLimit", "")
    match = re.search(r"(?:^|[;,])\s*t=(\d+)", rate_limit)
    if match:
        return str(int(match.group(1)) + 1)
    return None


def _request_bytes(
    url: str,
    *,
    attempts: int = 6,
    timeout: float = 60.0,
    max_bytes: int,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bytes, Mapping[str, str]]:
    last_error: BaseException | None = None
    for attempt in range(attempts):
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept-Encoding": "identity",
        }
        token = os.environ.get("HF_TOKEN")
        if token and urllib.parse.urlsplit(url).hostname in {
            "huggingface.co",
            "datasets-server.huggingface.co",
        }:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url,
            headers=headers,
        )
        try:
            with opener(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                if status != 200:
                    raise ExpertImageFetchError(
                        f"unexpected HTTP {status} while fetching {url}"
                    )
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) > max_bytes:
                    raise ExpertImageFetchError(
                        f"response exceeds {max_bytes:,} byte safety limit: {url}"
                    )
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ExpertImageFetchError(
                        f"response exceeds {max_bytes:,} byte safety limit: {url}"
                    )
                return payload, dict(response.headers.items())
        except urllib.error.HTTPError as error:
            last_error = error
            error_headers = error.headers
            error.close()
            if error.code not in _RETRYABLE_STATUS or attempt + 1 >= attempts:
                break
            retry_after = (
                _rate_limit_reset_seconds(error_headers)
                if error.code == 429
                else error_headers.get("Retry-After")
            )
            if error.code == 429 and retry_after is None:
                # Hugging Face uses five-minute rate-limit windows. Dataset
                # viewer responses do not always include the standard reset
                # header, so wait through one complete anonymous window.
                retry_after = "310"
            delay = _retry_delay(attempt, retry_after)
            if error.code == 429:
                print(
                    "Hugging Face rate limit reached; preserving completed "
                    f"downloads and retrying in {delay:.0f}s",
                    flush=True,
                )
            sleep(delay)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 >= attempts:
                break
            sleep(_retry_delay(attempt))
    raise ExpertImageFetchError(
        f"failed to fetch {url} after {attempts} attempt(s): {last_error}"
    ) from last_error


def _get_json(
    endpoint: str,
    params: Mapping[str, object] | None = None,
    *,
    attempts: int = 20,
    timeout: float = 60.0,
    request_bytes: Callable[..., tuple[bytes, Mapping[str, str]]] = _request_bytes,
) -> Mapping[str, Any]:
    url = endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params)
    payload, _ = request_bytes(
        url,
        attempts=attempts,
        timeout=timeout,
        max_bytes=_MAX_JSON_BYTES,
    )
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExpertImageFetchError(f"invalid JSON from {endpoint}: {error}") from error
    if not isinstance(decoded, dict):
        raise ExpertImageFetchError(f"expected JSON object from {endpoint}")
    if decoded.get("error"):
        raise ExpertImageFetchError(f"{endpoint}: {decoded['error']}")
    return decoded


def verify_current_mirror_revision(
    expected_revision: str,
    *,
    get_json: Callable[..., Mapping[str, Any]] = _get_json,
) -> Mapping[str, Any]:
    endpoint = (
        f"{HUGGING_FACE}/api/datasets/{MIRROR_DATASET}/revision/main"
    )
    metadata = get_json(endpoint)
    current = metadata.get("sha")
    if current != expected_revision:
        raise ExpertImageFetchError(
            f"{MIRROR_DATASET} main is now {current!r}, but this workflow is pinned "
            f"to {expected_revision!r}; audit the new revision before use"
        )
    return metadata


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def try_filter_locations(
    targets: Sequence[ManifestTarget],
    *,
    get_json: Callable[..., Mapping[str, Any]] = _get_json,
) -> tuple[MirrorLocation, ...] | None:
    """Return exact locations only if every required filter result is complete.

    Dataset-server may expose ``filter=true`` while its index is still loading,
    or may return ``partial=true`` for a large split.  Either condition causes a
    clean fallback to the filename-only Parquet scan.
    """

    by_name = {target.filename: target for target in targets}
    found: dict[str, MirrorLocation] = {}
    required_splits = sorted(
        {
            split
            for target in targets
            for split in target.candidate_mirror_splits
        }
    )
    try:
        for split in required_splits:
            names = sorted(
                target.filename
                for target in targets
                if split in target.candidate_mirror_splits
            )
            for batch in _chunks(names, _FILTER_BATCH_SIZE):
                predicates = [
                    f'"filename"=\'{filename}\'' for filename in batch
                ]
                response = get_json(
                    f"{DATASET_SERVER}/filter",
                    {
                        "dataset": MIRROR_DATASET,
                        "config": MIRROR_CONFIG,
                        "split": split,
                        "where": "(" + " OR ".join(predicates) + ")",
                        "length": 100,
                    },
                    attempts=1,
                    timeout=30.0,
                )
                if response.get("partial") is True:
                    return None
                rows = response.get("rows")
                if not isinstance(rows, list):
                    raise ExpertImageFetchError("/filter response has no rows list")
                batch_set = set(batch)
                for item in rows:
                    if not isinstance(item, Mapping):
                        raise ExpertImageFetchError("/filter returned a malformed row")
                    row = item.get("row")
                    if not isinstance(row, Mapping):
                        raise ExpertImageFetchError("/filter row payload is malformed")
                    filename = _canonical_nih_filename(row.get("filename"))
                    if filename not in batch_set:
                        raise ExpertImageFetchError(
                            f"/filter returned unrequested filename {filename!r}"
                        )
                    if filename in found:
                        raise ExpertImageFetchError(
                            f"mirror contains target {filename!r} more than once"
                        )
                    row_index = item.get("row_idx")
                    if not isinstance(row_index, int) or row_index < 0:
                        raise ExpertImageFetchError(
                            f"/filter returned invalid row index for {filename!r}"
                        )
                    target = by_name[filename]
                    found[filename] = MirrorLocation(
                        filename=filename,
                        expert_split=target.expert_split,
                        mirror_split=split,
                        row_index=row_index,
                    )
    except (ExpertImageFetchError, OSError, TimeoutError):
        return None

    if set(found) != set(by_name):
        return None
    return tuple(found[name] for name in sorted(found))


def scan_filename_column_locations(
    targets: Sequence[ManifestTarget],
    *,
    revision: str = MIRROR_REVISION,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    expected_split_rows: Mapping[str, int] = MIRROR_SPLIT_ROWS,
) -> tuple[MirrorLocation, ...]:
    """Project only ``filename`` from remote Parquet and map target row indices."""

    if loader is None:
        return scan_pinned_parquet_filename_locations(
            targets,
            revision=revision,
            expected_split_rows=expected_split_rows,
        )

    by_name = {target.filename: target for target in targets}
    required_splits = sorted(
        {
            split
            for target in targets
            for split in target.candidate_mirror_splits
        }
    )
    found: dict[str, MirrorLocation] = {}
    for split in required_splits:
        eligible = {
            target.filename
            for target in targets
            if split in target.candidate_mirror_splits
        }
        stream = loader(
            MIRROR_DATASET,
            split=split,
            streaming=True,
            revision=revision,
        )
        select_columns = getattr(stream, "select_columns", None)
        if select_columns is not None:
            if not callable(select_columns):
                raise ExpertImageFetchError(
                    "streaming dataset exposes a non-callable select_columns "
                    "attribute"
                )
            try:
                stream = select_columns(["filename"])
            except (KeyError, TypeError, ValueError) as error:
                raise ExpertImageFetchError(
                    f"could not project the mirror {split!r} stream to its "
                    f"filename column: {error}"
                ) from error
        elif getattr(loader, "__module__", "").split(".", 1)[0] == "datasets":
            raise ExpertImageFetchError(
                "the installed `datasets` streaming object cannot project to the "
                "filename column; upgrade `datasets` before retrying"
            )
        count = 0
        for row_index, row in enumerate(stream):
            count += 1
            if not isinstance(row, Mapping):
                raise ExpertImageFetchError(
                    f"projected Parquet {split} row {row_index} is malformed"
                )
            raw_filename = row.get("filename")
            if raw_filename not in eligible:
                continue
            filename = _canonical_nih_filename(raw_filename)
            if filename in found:
                previous = found[filename]
                raise ExpertImageFetchError(
                    f"mirror target {filename!r} appears in both "
                    f"{previous.mirror_split}:{previous.row_index} and "
                    f"{split}:{row_index}"
                )
            target = by_name[filename]
            found[filename] = MirrorLocation(
                filename=filename,
                expert_split=target.expert_split,
                mirror_split=split,
                row_index=row_index,
            )
        expected = expected_split_rows.get(split)
        if expected is not None and count != expected:
            raise ExpertImageFetchError(
                f"filename-only scan of mirror split {split!r} yielded {count:,} "
                f"rows, expected {expected:,}; refusing a truncated/reordered source"
            )

    missing = set(by_name) - set(found)
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ExpertImageFetchError(
            f"filename-preserving mirror is missing {len(missing)} requested "
            f"expert image(s), including: {examples}"
        )
    return tuple(found[name] for name in sorted(found))


def _read_pinned_parquet_filename_shard(
    split: str,
    shard_index: int,
    shard_count: int,
    *,
    revision: str,
) -> tuple[str, ...]:
    """Read one pinned shard's filename column without decoding image bytes."""

    try:
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ExpertImageFetchError(
            "the pinned filename fallback requires `huggingface_hub` and "
            "`pyarrow`; install the notebook dependencies before retrying"
        ) from error
    path = (
        f"datasets/{MIRROR_DATASET}@{revision}/data/"
        f"{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
    )
    try:
        filesystem = HfFileSystem()
        with filesystem.open(path, "rb") as handle:
            table = parquet.ParquetFile(handle).read(columns=["filename"])
        values = table.column("filename").to_pylist()
    except Exception as error:
        raise ExpertImageFetchError(
            f"could not read pinned filename column from {path}: {error}"
        ) from error
    if not all(isinstance(value, str) for value in values):
        raise ExpertImageFetchError(
            f"pinned filename column in {path} contains non-text values"
        )
    return tuple(values)


def scan_pinned_parquet_filename_locations(
    targets: Sequence[ManifestTarget],
    *,
    revision: str = MIRROR_REVISION,
    expected_split_rows: Mapping[str, int] = MIRROR_SPLIT_ROWS,
    split_shards: Mapping[str, int] = MIRROR_SPLIT_SHARDS,
    shard_reader: Callable[..., Sequence[str]] = (
        _read_pinned_parquet_filename_shard
    ),
) -> tuple[MirrorLocation, ...]:
    """Map targets from pinned Parquet filename columns using HTTP range reads.

    This avoids ``datasets`` streaming, which can spend minutes traversing the
    image-bearing Parquet dataset even after selecting only ``filename``.
    """

    by_name = {target.filename: target for target in targets}
    required_splits = sorted(
        {
            split
            for target in targets
            for split in target.candidate_mirror_splits
        }
    )
    found: dict[str, MirrorLocation] = {}
    for split in required_splits:
        shard_count = split_shards.get(split)
        if not isinstance(shard_count, int) or shard_count < 1:
            raise ExpertImageFetchError(
                f"no pinned Parquet shard count is defined for split {split!r}"
            )
        eligible = {
            target.filename
            for target in targets
            if split in target.candidate_mirror_splits
        }
        row_offset = 0
        for shard_index in range(shard_count):
            names = shard_reader(
                split,
                shard_index,
                shard_count,
                revision=revision,
            )
            for local_index, raw_filename in enumerate(names):
                if raw_filename not in eligible:
                    continue
                filename = _canonical_nih_filename(raw_filename)
                if filename in found:
                    previous = found[filename]
                    raise ExpertImageFetchError(
                        f"mirror target {filename!r} appears in both "
                        f"{previous.mirror_split}:{previous.row_index} and "
                        f"{split}:{row_offset + local_index}"
                    )
                target = by_name[filename]
                found[filename] = MirrorLocation(
                    filename=filename,
                    expert_split=target.expert_split,
                    mirror_split=split,
                    row_index=row_offset + local_index,
                )
            row_offset += len(names)
        expected = expected_split_rows.get(split)
        if expected is not None and row_offset != expected:
            raise ExpertImageFetchError(
                f"pinned filename scan of mirror split {split!r} yielded "
                f"{row_offset:,} rows, expected {expected:,}; refusing a "
                "truncated/reordered source"
            )

    missing = set(by_name) - set(found)
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise ExpertImageFetchError(
            f"filename-preserving mirror is missing {len(missing)} requested "
            f"expert image(s), including: {examples}"
        )
    return tuple(found[name] for name in sorted(found))


def _location_cache_key(
    manifest_sha256: str,
    cohort: str,
    targets: Sequence[ManifestTarget],
) -> str:
    canonical = json.dumps(
        {
            "manifest_sha256": manifest_sha256,
            "cohort": cohort,
            "revision": MIRROR_REVISION,
            "targets": [asdict(target) for target in targets],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def _load_cached_locations(
    path: Path,
    *,
    cache_key: str,
    targets: Sequence[ManifestTarget],
) -> tuple[MirrorLocation, ...] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("cache_key") != cache_key:
            return None
        locations = tuple(
            MirrorLocation(
                filename=item["filename"],
                expert_split=item["expert_split"],
                mirror_split=item["mirror_split"],
                row_index=int(item["row_index"]),
            )
            for item in value["locations"]
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    expected = {target.filename for target in targets}
    if {location.filename for location in locations} != expected:
        return None
    if any(
        location.mirror_split
        not in next(
            target.candidate_mirror_splits
            for target in targets
            if target.filename == location.filename
        )
        for location in locations
    ):
        return None
    return locations


def resolve_locations(
    targets: Sequence[ManifestTarget],
    *,
    manifest_sha256: str,
    cohort: str,
    state_dir: Path,
    lookup: str = "auto",
    get_json: Callable[..., Mapping[str, Any]] = _get_json,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> tuple[tuple[MirrorLocation, ...], str]:
    if lookup not in {"auto", "filter", "parquet"}:
        raise ValueError("lookup must be 'auto', 'filter', or 'parquet'")
    cache_key = _location_cache_key(manifest_sha256, cohort, targets)
    cache_path = state_dir / "locations.json"
    cached = _load_cached_locations(
        cache_path, cache_key=cache_key, targets=targets
    )
    if cached is not None:
        return cached, "cached_exact_locations"

    locations: tuple[MirrorLocation, ...] | None = None
    method = ""
    if lookup in {"auto", "filter"}:
        locations = try_filter_locations(targets, get_json=get_json)
        if locations is not None:
            method = "dataset_server_exact_filter"
        elif lookup == "filter":
            raise ExpertImageFetchError(
                "dataset-server /filter is unavailable, partial, or incomplete; "
                "use --lookup parquet (or the default --lookup auto)"
            )
    if locations is None:
        locations = scan_filename_column_locations(targets, loader=loader)
        method = (
            "projected_parquet_filename_column"
            if loader is not None
            else "pinned_parquet_filename_columns"
        )

    _atomic_json(
        cache_path,
        {
            "schema_version": 1,
            "cache_key": cache_key,
            "dataset": MIRROR_DATASET,
            "revision": MIRROR_REVISION,
            "lookup_method": method,
            "locations": [asdict(location) for location in locations],
        },
    )
    return locations, method


def _validate_asset_url(
    url: object,
    *,
    location: MirrorLocation,
    expected_revision: str,
) -> str:
    if not isinstance(url, str):
        raise ExpertImageFetchError(
            f"viewer row for {location.filename!r} has no image URL"
        )
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "datasets-server.huggingface.co":
        raise ExpertImageFetchError(
            f"refusing unexpected image asset host for {location.filename!r}"
        )
    expected_fragment = (
        f"/--/{expected_revision}/--/{MIRROR_CONFIG}/"
        f"{location.mirror_split}/{location.row_index}/image/"
    )
    if expected_fragment not in parsed.path:
        raise ExpertImageFetchError(
            f"viewer asset identity/revision mismatch for {location.filename!r}"
        )
    return url


def _validate_existing_png(path: Path, expected_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        if sha256_file(path) != expected_hash:
            return False
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != EXPECTED_IMAGE_SIZE:
                return False
            image.load()
    except (OSError, UnidentifiedImageError):
        return False
    return True


def _write_png_from_mirror_bytes(payload: bytes, destination: Path) -> Mapping[str, Any]:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            source_size = tuple(source.size)
            source_format = source.format
            source_mode = source.mode
            if source_size != EXPECTED_IMAGE_SIZE:
                raise ExpertImageFetchError(
                    f"mirror image has dimensions {source_size}, expected "
                    f"{EXPECTED_IMAGE_SIZE}"
                )
            output = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ExpertImageFetchError(f"mirror payload is not a decodable image: {error}") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            output.save(handle, format="PNG")
        with Image.open(temporary_name) as check:
            if check.format != "PNG" or check.size != EXPECTED_IMAGE_SIZE:
                raise ExpertImageFetchError(
                    f"post-write PNG validation failed for {destination}"
                )
            check.load()
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "source_format": source_format,
        "source_mode": source_mode,
        "source_width": source_size[0],
        "source_height": source_size[1],
        "output_format": "PNG",
        "output_mode": "RGB",
    }


def _read_resume_record(
    state_path: Path,
    output_path: Path,
    *,
    location: MirrorLocation,
) -> Mapping[str, Any] | None:
    if not state_path.is_file():
        return None
    try:
        record = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            record.get("filename") != location.filename
            or record.get("mirror_split") != location.mirror_split
            or record.get("row_index") != location.row_index
            or record.get("mirror_revision") != MIRROR_REVISION
        ):
            return None
        output_hash = record["output_sha256"]
        if not isinstance(output_hash, str) or len(output_hash) != 64:
            return None
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not _validate_existing_png(output_path, output_hash):
        return None
    return record


def _download_location(
    location: MirrorLocation,
    viewer_item: Mapping[str, Any],
    *,
    images_dir: Path,
    state_dir: Path,
    force: bool,
    request_bytes: Callable[..., tuple[bytes, Mapping[str, str]]] = _request_bytes,
) -> Mapping[str, Any]:
    row = viewer_item.get("row")
    if not isinstance(row, Mapping):
        raise ExpertImageFetchError(
            f"viewer row payload is malformed for {location.filename!r}"
        )
    actual_filename = _canonical_nih_filename(row.get("filename"))
    if actual_filename != location.filename:
        raise ExpertImageFetchError(
            f"viewer row identity mismatch at {location.mirror_split}:"
            f"{location.row_index}: expected {location.filename!r}, got "
            f"{actual_filename!r}"
        )
    image = row.get("image")
    if not isinstance(image, Mapping):
        raise ExpertImageFetchError(
            f"viewer row has no image object for {location.filename!r}"
        )
    if (image.get("width"), image.get("height")) != EXPECTED_IMAGE_SIZE:
        raise ExpertImageFetchError(
            f"viewer metadata dimensions are not 320x320 for {location.filename!r}"
        )
    asset_url = _validate_asset_url(
        image.get("src"), location=location, expected_revision=MIRROR_REVISION
    )
    output_path = images_dir / location.filename
    state_path = state_dir / "images" / f"{location.filename}.json"
    if not force:
        resumed = _read_resume_record(
            state_path, output_path, location=location
        )
        if resumed is not None:
            return {**resumed, "run_status": "resumed"}

    payload, headers = request_bytes(
        asset_url,
        attempts=6,
        timeout=90.0,
        max_bytes=_MAX_IMAGE_BYTES,
    )
    source_hash = hashlib.sha256(payload).hexdigest()
    image_metadata = _write_png_from_mirror_bytes(payload, output_path)
    output_hash = sha256_file(output_path)
    unsigned_asset = urllib.parse.urlunsplit(
        (*urllib.parse.urlsplit(asset_url)[:3], "", "")
    )
    record: dict[str, Any] = {
        "filename": location.filename,
        "expert_split": location.expert_split,
        "mirror_split": location.mirror_split,
        "row_index": location.row_index,
        "mirror_revision": MIRROR_REVISION,
        "mirror_labels": row.get("labels"),
        "asset_route_without_signature": unsigned_asset,
        "source_content_type": headers.get("Content-Type"),
        "source_bytes": len(payload),
        "source_sha256": source_hash,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": output_hash,
        **image_metadata,
        "development_only": True,
        "official_or_final_evidence_allowed": False,
    }
    _atomic_json(state_path, record)
    return {**record, "run_status": "downloaded"}


def _fetch_viewer_bucket(
    split: str,
    offset: int,
    locations: Sequence[MirrorLocation],
    *,
    images_dir: Path,
    state_dir: Path,
    force: bool,
    get_json: Callable[..., Mapping[str, Any]] = _get_json,
    request_bytes: Callable[..., tuple[bytes, Mapping[str, str]]] = _request_bytes,
) -> tuple[Mapping[str, Any], ...]:
    response = get_json(
        f"{DATASET_SERVER}/rows",
        {
            "dataset": MIRROR_DATASET,
            "config": MIRROR_CONFIG,
            "split": split,
            "offset": offset,
            "length": _VIEWER_BUCKET_SIZE,
        },
    )
    rows = response.get("rows")
    if not isinstance(rows, list):
        raise ExpertImageFetchError("/rows response has no rows list")
    by_index: dict[int, Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping) or not isinstance(item.get("row_idx"), int):
            raise ExpertImageFetchError("/rows returned malformed row metadata")
        by_index[int(item["row_idx"])] = item
    records: list[Mapping[str, Any]] = []
    for location in locations:
        item = by_index.get(location.row_index)
        if item is None:
            raise ExpertImageFetchError(
                f"/rows omitted requested {split}:{location.row_index}"
            )
        records.append(
            _download_location(
                location,
                item,
                images_dir=images_dir,
                state_dir=state_dir,
                force=force,
                request_bytes=request_bytes,
            )
        )
    return tuple(records)


def _bucket_locations(
    locations: Sequence[MirrorLocation],
) -> Mapping[tuple[str, int], tuple[MirrorLocation, ...]]:
    buckets: dict[tuple[str, int], list[MirrorLocation]] = {}
    for location in locations:
        offset = (location.row_index // _VIEWER_BUCKET_SIZE) * _VIEWER_BUCKET_SIZE
        buckets.setdefault((location.mirror_split, offset), []).append(location)
    return {
        key: tuple(sorted(value, key=lambda item: item.row_index))
        for key, value in buckets.items()
    }


def fetch_expert_images(
    manifest: str | os.PathLike[str],
    output_images_dir: str | os.PathLike[str],
    *,
    cohort: str = "development",
    acknowledge_test_is_development_only: bool = False,
    lookup: str = "auto",
    workers: int = 6,
    force: bool = False,
    get_json: Callable[..., Mapping[str, Any]] = _get_json,
    request_bytes: Callable[..., tuple[bytes, Mapping[str, str]]] = _request_bytes,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    """Fetch, validate, and provenance-stamp the selected expert image cohort."""

    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_hash = sha256_file(manifest_path)
    targets = load_canonical_manifest(
        manifest_path,
        cohort=cohort,
        acknowledge_test_is_development_only=(
            acknowledge_test_is_development_only
        ),
    )
    images_dir = Path(output_images_dir).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)
    state_dir = images_dir / ".fetch_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    hub_metadata = verify_current_mirror_revision(
        MIRROR_REVISION, get_json=get_json
    )
    locations, lookup_method = resolve_locations(
        targets,
        manifest_sha256=manifest_hash,
        cohort=cohort,
        state_dir=state_dir,
        lookup=lookup,
        get_json=get_json,
        loader=loader,
    )
    records: list[Mapping[str, Any]] = []
    pending_locations: list[MirrorLocation] = []
    if force:
        pending_locations.extend(locations)
    else:
        for location in locations:
            resumed = _read_resume_record(
                state_dir / "images" / f"{location.filename}.json",
                images_dir / location.filename,
                location=location,
            )
            if resumed is None:
                pending_locations.append(location)
            else:
                records.append({**resumed, "run_status": "resumed"})

    buckets = _bucket_locations(pending_locations)
    futures: dict[Future[tuple[Mapping[str, Any], ...]], tuple[str, int]] = {}
    viewer_workers = min(workers, _MAX_VIEWER_WORKERS)
    with ThreadPoolExecutor(max_workers=viewer_workers) as executor:
        for (split, offset), bucket_locations in buckets.items():
            future = executor.submit(
                _fetch_viewer_bucket,
                split,
                offset,
                bucket_locations,
                images_dir=images_dir,
                state_dir=state_dir,
                force=force,
                get_json=get_json,
                request_bytes=request_bytes,
            )
            futures[future] = (split, offset)
        completed = 0
        try:
            for future in as_completed(futures):
                records.extend(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(futures):
                    print(
                        f"Resolved/downloaded {completed:,}/{len(futures):,} "
                        "viewer buckets"
                    )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if records and not buckets:
        print(f"Verified {len(records):,} cached image(s); no viewer fetch needed")

    records.sort(key=lambda item: str(item["filename"]))
    if len(records) != len(targets):
        raise ExpertImageFetchError(
            f"completed {len(records)} image record(s), expected {len(targets)}"
        )
    if {str(record["filename"]) for record in records} != {
        target.filename for target in targets
    }:
        raise ExpertImageFetchError(
            "downloaded image identities do not exactly match the canonical manifest"
        )

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "doctor_assistant.nih_image_provenance",
        "source": (
            f"Hugging Face {MIRROR_DATASET}@{MIRROR_REVISION}; viewer-served "
            "320x320 third-party JPEG re-encodes converted to RGB PNG under the "
            "original NIH basename"
        ),
        "resolution": {
            "width": EXPECTED_IMAGE_SIZE[0],
            "height": EXPECTED_IMAGE_SIZE[1],
        },
        "original_nih_pixels": False,
        "development_only": True,
        "official_or_final_evidence_allowed": False,
        "warning": _DEVELOPMENT_WARNING,
        "canonical_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "cohort": cohort,
            "rows_selected": len(targets),
        },
        "mirror": {
            "dataset": MIRROR_DATASET,
            "config": MIRROR_CONFIG,
            "revision": MIRROR_REVISION,
            "revision_verified_against_main_at_run_time": True,
            "hub_last_modified": hub_metadata.get("lastModified"),
            "license_from_hub": (
                next(
                    (
                        str(tag).split(":", 1)[1]
                        for tag in hub_metadata.get("tags", [])
                        if str(tag).startswith("license:")
                    ),
                    None,
                )
            ),
            "source_encoding": "viewer-served 320x320 JPEG",
            "output_encoding": "RGB PNG under original NIH basename",
        },
        "lookup": {
            "method": lookup_method,
            "filter_fast_path": lookup in {"auto", "filter"},
            "parquet_fallback_projects_columns": ["filename"],
            "viewer_bucket_size": _VIEWER_BUCKET_SIZE,
            "viewer_buckets_requested": len(buckets),
            "viewer_workers_requested": workers,
            "viewer_workers_effective": viewer_workers,
        },
        "validation": {
            "expected_dimensions": list(EXPECTED_IMAGE_SIZE),
            "images": len(records),
            "downloaded_this_run": sum(
                record["run_status"] == "downloaded" for record in records
            ),
            "resumed_this_run": sum(
                record["run_status"] == "resumed" for record in records
            ),
            "all_output_sha256_recorded": all(
                isinstance(record.get("output_sha256"), str)
                and len(str(record["output_sha256"])) == 64
                for record in records
            ),
        },
        "images": records,
    }
    provenance_path = images_dir / "DEVELOPMENT_ONLY.provenance.json"
    _atomic_json(provenance_path, provenance)
    warning_path = images_dir / "DEVELOPMENT_ONLY_NOT_OFFICIAL.txt"
    warning_path.write_text(
        _DEVELOPMENT_WARNING
        + "\n\nUse original, filename-reconciled NIH pixels for any locked/final "
        "evaluation.\n",
        encoding="utf-8",
    )
    return provenance


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
        help="directory in which original NIH basenames will be written as PNG",
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
        help=(
            "allow fetching locked test identities while acknowledging that resized "
            "mirror pixels are still prohibited as final evidence"
        ),
    )
    parser.add_argument(
        "--lookup",
        choices=("auto", "filter", "parquet"),
        default="auto",
        help="auto tries exact /filter then projects only filename from Parquet",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="concurrent viewer-bucket/image workers (1-16; default 6)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload even when per-image resume state and SHA-256 validate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(_DEVELOPMENT_WARNING)
    try:
        provenance = fetch_expert_images(
            args.manifest,
            args.output_images_dir,
            cohort=args.cohort,
            acknowledge_test_is_development_only=(
                args.acknowledge_development_only_test_use
            ),
            lookup=args.lookup,
            workers=args.workers,
            force=args.force,
        )
    except (OSError, ValueError, ExpertImageFetchError) as error:
        parser.error(str(error))
    validation = provenance["validation"]
    print(
        f"Verified {validation['images']:,} development-only image(s): "
        f"{validation['downloaded_this_run']:,} downloaded, "
        f"{validation['resumed_this_run']:,} resumed"
    )
    print(Path(args.output_images_dir).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
