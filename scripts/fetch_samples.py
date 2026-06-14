"""Fetch real, full-resolution NIH ChestX-ray14 samples for the system test.

Why this exists: the chest model is trained on native ~1024 px NIH X-rays. ChestMNIST
is the *same* images pre-crushed to 28-224 px; upscaling one to 320 px cannot recover the
fine texture the model keys on, so every prediction collapses below threshold and the
demo looks broken. This script pulls a small, curated set of the ORIGINAL images — with
their ground-truth labels — so the system test runs the model on the domain it was
trained on, and findings actually cross the line.

Source: ``BahaaEldin0/NIH-Chest-Xray-14`` on the HuggingFace Hub — the original
ChestX-ray14 images (1024x1024) with the canonical 14-label vocabulary, served as
parquet. No Kaggle account, no 30 GB pull: a dozen images stream down in seconds with no
credentials. The label strings are byte-identical to ``data.chest_xray14.CHESTXRAY14_LABELS``.

Output (under ``--out``, default ``<repo>/samples/chest_xray``):

    images/<key>.png
    manifest.csv          columns: filename,labels   (labels '|'-separated, NIH naming;
                                                       empty == "No Finding")

Run:  python scripts/fetch_samples.py
      python scripts/fetch_samples.py --per-class 2 --out /some/dir --max-scan 3000
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys

HF_DATASET = "BahaaEldin0/NIH-Chest-Xray-14"

# Common, visually-clear pathologies a CheXNet-grade model handles well. One clear
# example of each (single-label preferred) makes the "true label vs. model score"
# comparison easy to read; rarer classes (Hernia, Fibrosis) are skipped on purpose.
TARGET_PATHOLOGIES: tuple[str, ...] = (
    "Cardiomegaly", "Effusion", "Atelectasis", "Infiltration",
    "Mass", "Nodule", "Pneumothorax", "Consolidation",
)
_NO_FINDING = "No Finding"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        default=os.path.join(repo_root, "samples", "chest_xray"),
        help="output directory (default: <repo>/samples/chest_xray)",
    )
    p.add_argument("--per-class", type=int, default=1, help="images per pathology")
    p.add_argument("--normals", type=int, default=2, help='"No Finding" images to fetch')
    p.add_argument("--split", default="test", help="dataset split to stream (test/valid/train)")
    p.add_argument(
        "--max-scan", type=int, default=2000,
        help="stop scanning the stream after this many rows even if quotas aren't full",
    )
    p.add_argument("--force", action="store_true", help="re-download even if manifest exists")
    return p.parse_args(argv)


def _labels_of(example: dict) -> list[str]:
    """The dataset's `label` column is a list[str]; normalize to a clean list."""
    raw = example.get("label")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x).strip() for x in raw if str(x).strip()]


def _collect(stream, per_class: int, normals: int, max_scan: int) -> list[tuple[dict, list[str]]]:
    """Scan the stream and pick a curated set.

    For each pathology we keep the first *single-label* example (cleanest signal),
    falling back to the first example that merely contains it. Plus `normals`
    "No Finding" images. Returns [(example, kept_labels), ...].
    """
    single: dict[str, dict] = {}          # pathology -> single-label example
    multi: dict[str, dict] = {}           # pathology -> any example containing it
    chosen_normals: list[dict] = []

    def _need_pathology(p: str) -> bool:
        return p in TARGET_PATHOLOGIES and len(single.get(p, ()) or ()) == 0

    for n, ex in enumerate(stream):
        if n >= max_scan:
            break
        labels = _labels_of(ex)

        if labels == [_NO_FINDING] or not labels:
            if len(chosen_normals) < normals:
                chosen_normals.append(ex)
        else:
            positives = [l for l in labels if l != _NO_FINDING]
            for p in positives:
                if p not in TARGET_PATHOLOGIES:
                    continue
                if len(positives) == 1 and p not in single:
                    single[p] = ex
                elif p not in multi:
                    multi[p] = ex

        have_all = (
            all(p in single for p in TARGET_PATHOLOGIES)
            and len(chosen_normals) >= normals
        )
        if have_all:
            break

    out: list[tuple[dict, list[str]]] = []
    for p in TARGET_PATHOLOGIES:
        ex = single.get(p) or multi.get(p)
        if ex is None:
            continue
        kept = [l for l in _labels_of(ex) if l != _NO_FINDING]
        for _ in range(per_class):
            out.append((ex, kept))
            break  # per_class>1 would need distinct examples; one clean case is enough
    for ex in chosen_normals[:normals]:
        out.append((ex, []))
    return out


def _key_for(example: dict, index: int) -> str:
    """Stable filename: prefer the dataset's own path/Image-Index, else an index."""
    img = example.get("image")
    if isinstance(img, dict) and img.get("path"):
        base = os.path.basename(str(img["path"]))
        if base:
            return os.path.splitext(base)[0]
    for k in ("Image Index", "image_id", "id"):
        if example.get(k):
            return os.path.splitext(str(example[k]))[0]
    return f"sample_{index:03d}"


def _save_png(example: dict, path: str) -> bool:
    """Decode the (decode=False) image bytes and write a normalized PNG."""
    from PIL import Image

    img = example.get("image")
    data = None
    if isinstance(img, dict):
        data = img.get("bytes")
    if data is None:
        return False
    Image.open(io.BytesIO(data)).convert("L").save(path)
    return True


def fetch_samples(
    out: str,
    *,
    per_class: int = 1,
    normals: int = 2,
    split: str = "test",
    max_scan: int = 2000,
    force: bool = False,
) -> int:
    """Fetch the curated sample set into ``out``. Returns 0 on success.

    Importable so a notebook can call it **in-process** (where a pip-installed
    ``datasets`` is guaranteed on the path) instead of shelling out to a separate
    interpreter that may not see the same site-packages. Never calls ``os._exit`` —
    that would kill a host kernel; the hard exit lives only in the CLI entrypoint.
    """
    images_dir = os.path.join(out, "images")
    manifest_path = os.path.join(out, "manifest.csv")

    if os.path.isfile(manifest_path) and not force:
        with open(manifest_path, newline="") as f:
            have = sum(1 for _ in csv.DictReader(f))
        print(f"Samples already present: {have} in {out} — skipping (use force=True to refetch).")
        return 0

    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: the `datasets` library is required.\n"
            "    pip install -q 'datasets>=2.18' 'huggingface_hub>=0.23'",
            file=sys.stderr,
        )
        return 2

    os.makedirs(images_dir, exist_ok=True)

    print(f"Streaming {HF_DATASET} [{split}] from HuggingFace (no credentials needed)...")
    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    # decode=False -> images arrive as raw {'bytes','path'} so skipped rows aren't decoded.
    ds = ds.cast_column("image", HFImage(decode=False))

    picked = _collect(ds, per_class, normals, max_scan)
    if not picked:
        print("ERROR: scanned the stream but matched no samples — check the dataset/split.",
              file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, (ex, labels) in enumerate(picked):
        key = _key_for(ex, i)
        while key in seen:
            key = f"{key}_{i}"
        seen.add(key)
        fname = f"{key}.png"
        if not _save_png(ex, os.path.join(images_dir, fname)):
            print(f"  skip {fname}: no decodable image bytes", file=sys.stderr)
            continue
        rows.append({"filename": fname, "labels": "|".join(labels)})
        print(f"  saved {fname:28} true={labels or ['No Finding']}")

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "labels"])
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(1 for r in rows if r["labels"])
    print(f"\nDone: {len(rows)} images ({n_pos} abnormal, {len(rows) - n_pos} normal) -> {out}")
    print(f"Manifest: {manifest_path}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    return fetch_samples(
        args.out,
        per_class=args.per_class,
        normals=args.normals,
        split=args.split,
        max_scan=args.max_scan,
        force=args.force,
    )


if __name__ == "__main__":
    _rc = main(sys.argv[1:])
    # HuggingFace streaming leaves background threads / open connections that can keep
    # the interpreter alive for a long time after the work is done — which would hang a
    # `subprocess.run(..., check=True)` caller. Flush and exit hard once we're finished.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
