"""Small, durable cache for expensive lung-nodule detector runs.

The cache deliberately lives beside the imported source series instead of in process
memory.  A Colab worker can therefore restart without making the doctor repeat a
multi-minute volume inference.  Entries are atomically committed JSON documents and
``latest`` is only a pointer; if that pointer is interrupted or stale, readers recover
by scanning the bounded entry set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


CACHE_SCHEMA_VERSION = 1
MAX_CACHE_ENTRIES = 8


def cache_directory(series_storage_dir: Path) -> Path:
    return series_storage_dir / "derived" / "lung-nodule-detection"


def entry_path(directory: Path, cache_key: str) -> Path:
    return directory / f"v{CACHE_SCHEMA_VERSION}-{cache_key}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_entry(directory: Path, cache_key: str) -> dict[str, Any] | None:
    """Read one exact entry, treating partial/old/malformed files as a cache miss."""

    value = _read_json(entry_path(directory, cache_key))
    if (
        value is None
        or value.get("schema_version") != CACHE_SCHEMA_VERSION
        or value.get("cache_key") != cache_key
    ):
        return None
    return value


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                value,
                handle,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_entry(directory: Path, cache_key: str, value: dict[str, Any]) -> None:
    """Commit an entry and latest pointer, then retain only a small result history."""

    target = entry_path(directory, cache_key)
    _atomic_json_write(target, value)
    _atomic_json_write(
        directory / "latest.json",
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_file": target.name,
        },
    )

    entries = sorted(
        directory.glob(f"v{CACHE_SCHEMA_VERSION}-*.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for old_entry in entries[MAX_CACHE_ENTRIES:]:
        try:
            old_entry.unlink()
        except FileNotFoundError:
            pass


def read_latest(
    directory: Path,
    *,
    series_id: str,
    source_fingerprint: str,
    model_fingerprint: str,
) -> dict[str, Any] | None:
    """Recover the newest entry for the current source and detector identities."""

    candidates: list[Path] = []
    pointer = _read_json(directory / "latest.json")
    if pointer is not None and pointer.get("schema_version") == CACHE_SCHEMA_VERSION:
        filename = pointer.get("cache_file")
        if isinstance(filename, str) and Path(filename).name == filename:
            pointed = directory / filename
            if (
                pointed.name.startswith(f"v{CACHE_SCHEMA_VERSION}-")
                and pointed.suffix == ".json"
            ):
                candidates.append(pointed)

    try:
        entries = sorted(
            directory.glob(f"v{CACHE_SCHEMA_VERSION}-*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return None
    # A crash can happen after an entry is committed but before latest.json is
    # replaced.  Filesystem recency must therefore win over a valid-but-stale pointer.
    candidates = [*entries, *candidates]

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        value = _read_json(path)
        if (
            value is not None
            and value.get("schema_version") == CACHE_SCHEMA_VERSION
            and value.get("series_id") == series_id
            and value.get("source_fingerprint") == source_fingerprint
            and value.get("model_fingerprint") == model_fingerprint
            and isinstance(value.get("response"), dict)
        ):
            return value
    return None
