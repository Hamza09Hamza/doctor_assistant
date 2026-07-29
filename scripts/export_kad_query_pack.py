"""Export one frozen KAD-512 endpoint query as a BERT-free inference pack.

The released KAD checkpoint contains the image encoder, query decoder, and
Med-KEBERT text encoder.  This command uses the text encoder once, records the
source checkpoint hash, and writes a smaller runtime artifact containing the
image path plus exactly one phase-1 endpoint query.  Endpoint isolation matters
because KAD's decoder applies self-attention across every query in a pack: adding
or editing another query can otherwise change an unchanged endpoint's score.

No weights are downloaded by this script.  ``--checkpoint`` must point to a
trusted KAD-512 checkpoint already present on disk.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experts.kad import (
    KAD512_BERT_MODEL_ID,
    KAD512_BERT_REVISION,
    KAD512_EMBED_DIM,
    KAD512_LABELS,
    KAD512_PROMPTS,
    export_kad512_query_pack,
)


PHASE1_LABELS: tuple[str, ...] = (
    "Pneumothorax",
    "Nodule_or_mass",
    "Airspace_opacity",
)
PHASE1_PROMPTS: tuple[str, ...] = (
    "pneumothorax",
    "lung nodule or mass",
    "airspace opacity",
)
PHASE1_FEATURES_PATH = (
    _REPO_ROOT / "configs" / "chest_kad_phase1_query_features.json"
)
PHASE1_FEATURES_SHA256 = (
    "54c74d20a5bcecab770ce6a0d84bc0b14caa40e798693d1af6bb0b022a4cf094"
)
PHASE1_QUERY_SPECS: dict[str, dict[str, str]] = {
    "Pneumothorax": {
        "prompt": "pneumothorax",
        "query_set": "doctor_assistant.phase1.pneumothorax.v1",
        "semantic_sha256": (
            "07f9ed5ec200428c5a4e3701c61fcbdc326bd152dde300da4fbffbbc1d11b87c"
        ),
    },
    "Nodule_or_mass": {
        "prompt": "lung nodule or mass",
        "query_set": "doctor_assistant.phase1.nodule_or_mass.v1",
        "semantic_sha256": (
            "691850b79e133ed78b235f4dcfd81947dd9ac1993026bdea06a5a85faec2acd2"
        ),
    },
    "Airspace_opacity": {
        "prompt": "airspace opacity",
        "query_set": "doctor_assistant.phase1.airspace_opacity.v1",
        "semantic_sha256": (
            "b663a14cec6e4ce37829cecedbb6ed0ea406b64ff7c4cfcc5f35b39fff31ed5f"
        ),
    },
}

if tuple(PHASE1_QUERY_SPECS) != PHASE1_LABELS or tuple(
    spec["prompt"] for spec in PHASE1_QUERY_SPECS.values()
) != PHASE1_PROMPTS:
    raise RuntimeError("phase-1 query specs are not aligned with labels/prompts")


def load_phase1_query_features(
    path: str | Path = PHASE1_FEATURES_PATH,
) -> dict[str, torch.Tensor]:
    """Load the reviewed singleton embeddings without rerunning Med-KEBERT."""

    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"Could not read canonical phase-1 query features {resolved}: {exc}"
        ) from exc
    expected_metadata = {
        "format": "doctor_assistant.kad512.phase1_query_features",
        "format_version": 1,
        "dtype": "float32",
        "byte_order": "little",
        "shape": [len(PHASE1_LABELS), KAD512_EMBED_DIM],
        "labels": list(PHASE1_LABELS),
        "prompts": list(PHASE1_PROMPTS),
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Canonical phase-1 query features have invalid {key}: "
                f"expected {expected!r}, got {payload.get(key)!r}"
            )
    try:
        raw = base64.b64decode(payload["data_base64"], validate=True)
    except Exception as exc:
        raise ValueError(
            "Canonical phase-1 query features contain invalid base64 data"
        ) from exc
    actual_hash = hashlib.sha256(raw).hexdigest()
    if (
        payload.get("data_sha256") != PHASE1_FEATURES_SHA256
        or actual_hash != PHASE1_FEATURES_SHA256
    ):
        raise ValueError(
            "Canonical phase-1 query-feature bytes do not match the reviewed SHA-256"
        )
    value_count = len(PHASE1_LABELS) * KAD512_EMBED_DIM
    if len(raw) != value_count * 4:
        raise ValueError(
            "Canonical phase-1 query-feature byte length does not match its shape"
        )
    values = struct.unpack(f"<{value_count}f", raw)
    features = torch.tensor(values, dtype=torch.float32).reshape(
        len(PHASE1_LABELS), KAD512_EMBED_DIM
    )
    if not torch.equal(
        features,
        features.to(torch.bfloat16).to(torch.float32),
    ):
        raise ValueError(
            "Canonical phase-1 query features violate the BF16 round-trip contract"
        )
    return {
        label: features[index : index + 1].clone()
        for index, label in enumerate(PHASE1_LABELS)
    }


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a model file into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a frozen BERT-free KAD-512 disease-query set."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-set",
        choices=("phase1", "nih14"),
        default="phase1",
        help=(
            "phase1 exports exactly one endpoint selected by --active-target; "
            "nih14 retains the joint 14-query set only for the explicitly "
            "smoke-only mirror inference check"
        ),
    )
    parser.add_argument(
        "--active-target",
        choices=PHASE1_LABELS,
        help=(
            "required with --query-set phase1; selects the one endpoint query "
            "placed in the pack using its reviewed frozen embedding"
        ),
    )
    parser.add_argument("--device", default=None, help="for example cuda or cpu")
    parser.add_argument("--bert-model-id", default=KAD512_BERT_MODEL_ID)
    parser.add_argument("--bert-revision", default=KAD512_BERT_REVISION)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="do not contact Hugging Face for the Med-KEBERT config/tokenizer",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "allow legacy pickle loading only for a checkpoint whose source and "
            "hash you have independently verified"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"--checkpoint is not a file: {args.checkpoint}")

    if args.query_set == "phase1":
        if args.active_target is None:
            parser.error("--query-set phase1 requires --active-target")
        spec = PHASE1_QUERY_SPECS[args.active_target]
        labels = (args.active_target,)
        prompts = (spec["prompt"],)
        query_set_id = spec["query_set"]
        text_features = load_phase1_query_features()[args.active_target]
    else:
        if args.active_target is not None:
            parser.error("--active-target is accepted only with --query-set phase1")
        labels, prompts = KAD512_LABELS, KAD512_PROMPTS
        query_set_id = "doctor_assistant.nih14_smoke.v1"
        text_features = None

    checkpoint_hash = sha256_file(args.checkpoint)
    exported = export_kad512_query_pack(
        args.checkpoint,
        args.output,
        text_features=text_features,
        bert_model_id=args.bert_model_id,
        bert_revision=args.bert_revision,
        tokenizer_id=args.tokenizer_id,
        labels=labels,
        prompts=prompts,
        max_length=args.max_length,
        device=args.device,
        local_files_only=args.local_files_only,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        overwrite=args.overwrite,
        source_metadata={
            "checkpoint_sha256": checkpoint_hash,
            "query_set": query_set_id,
        },
    )
    result = {
        "query_pack": str(exported.resolve()),
        "query_pack_sha256": sha256_file(exported),
        "checkpoint_sha256": checkpoint_hash,
        "query_set": args.query_set,
        "query_set_id": query_set_id,
        "active_target": args.active_target,
        "labels": list(labels),
        "prompts": list(prompts),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
