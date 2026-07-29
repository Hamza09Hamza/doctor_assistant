"""Build a split-safe manifest from Google's NIH four-finding expert labels.

Separate-file example::

    python scripts/prepare_nih_expert_manifest.py \
      --validation-csv nih_validation_labels.csv \
      --test-csv nih_test_labels.csv \
      --official-train-val-list train_val_list.txt \
      --official-test-list test_list.txt \
      --output-csv artifacts/nih_expert/labels.csv \
      --output-json artifacts/nih_expert/labels.metadata.json

The public table bundled by TorchXRayVision combines both subsets and has a
``Set Id`` column.  It can be prepared with ``--combined-csv`` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.nih_expert_labels import prepare_nih_expert_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Google NIH adjudicated labels against the official NIH split "
            "lists and emit a canonical, hashed evaluation manifest."
        )
    )
    source = parser.add_argument_group("expert label source")
    source.add_argument(
        "--combined-csv",
        type=Path,
        help="one CSV containing both validation and test rows plus a split column",
    )
    source.add_argument(
        "--validation-csv",
        type=Path,
        help="validation/development expert-label CSV",
    )
    source.add_argument("--test-csv", type=Path, help="test expert-label CSV")

    official = parser.add_argument_group("authoritative NIH manifests")
    official.add_argument(
        "--official-train-val-list",
        type=Path,
        required=True,
        help="official NIH train_val_list.txt",
    )
    official.add_argument(
        "--official-test-list",
        type=Path,
        required=True,
        help="official NIH test_list.txt",
    )

    output = parser.add_argument_group("outputs")
    output.add_argument("--output-csv", type=Path, required=True)
    output.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--include-fracture",
        action="store_true",
        help="preserve the optional Fracture expert target",
    )
    parser.add_argument(
        "--output-cohort",
        choices=("development", "test", "all"),
        default="development",
        help=(
            "emit development labels by default so locked test labels are not "
            "materialized in the routine candidate-selection manifest"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    return dict(
        prepare_nih_expert_manifest(
            official_train_val_path=args.official_train_val_list,
            official_test_path=args.official_test_list,
            output_csv_path=args.output_csv,
            output_json_path=args.output_json,
            validation_csv_path=args.validation_csv,
            test_csv_path=args.test_csv,
            combined_csv_path=args.combined_csv,
            include_fracture=args.include_fracture,
            output_cohort=args.output_cohort,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = run(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rows = metadata["rows"]
    output = metadata["output_csv"]
    print(
        f"Wrote {output['path']} ({rows['validation']} validation + "
        f"{rows['test']} test rows; sha256={output['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
