"""Select per-label thresholds from a validation benchmark artifact.

The input must be produced by ``benchmark_chest_classifier.py`` on the validation
split.  This script never opens the test split; its output can then be supplied to a
single frozen test benchmark with ``--threshold-artifact``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from evaluation.thresholds import (
    calibrated_threshold_dict,
    select_per_label_thresholds,
)


def select_from_benchmark(
    benchmark_path: Path,
    *,
    sensitivity_target: float,
    specificity_floor: float,
    min_positives: int,
    min_negatives: int,
) -> dict:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    dataset = benchmark.get("dataset", {})
    if benchmark.get("purpose") != "development_validation" or dataset.get("split") != "valid":
        raise ValueError(
            "thresholds may only be selected from a validation benchmark artifact"
        )

    prediction_path = Path(benchmark["predictions"])
    if not prediction_path.is_absolute():
        candidate = benchmark_path.parent / prediction_path
        prediction_path = candidate if candidate.exists() else prediction_path
    if not prediction_path.is_file():
        raise FileNotFoundError(f"prediction artifact not found: {prediction_path}")
    arrays = np.load(prediction_path, allow_pickle=False)
    probabilities = arrays["probabilities"]
    labels = arrays["labels"]
    class_names = [str(value) for value in arrays["class_names"].tolist()]

    selections = select_per_label_thresholds(
        probabilities,
        labels,
        class_names,
        sensitivity_target=sensitivity_target,
        specificity_floor=specificity_floor,
        min_positives=min_positives,
        min_negatives=min_negatives,
    )
    try:
        pipeline_thresholds = calibrated_threshold_dict(selections)
        thresholds_complete = True
        blocked_labels: list[str] = []
    except ValueError:
        pipeline_thresholds = None
        thresholds_complete = False
        blocked_labels = [
            row.label
            for row in selections
            if not row.supported or not row.meets_constraints
        ]

    return {
        "schema_version": 1,
        "source_benchmark": str(benchmark_path.resolve()),
        "dataset": dataset,
        "model": benchmark.get("model"),
        "selection": {
            "sensitivity_target": sensitivity_target,
            "specificity_floor": specificity_floor,
            "min_positives": min_positives,
            "min_negatives": min_negatives,
            "per_label": [row.to_dict() for row in selections],
        },
        "thresholds_complete": thresholds_complete,
        "blocked_labels": blocked_labels,
        "pipeline_thresholds": pipeline_thresholds,
        "warning": "Research threshold selection only; not clinical validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--sensitivity-target", type=float, default=0.85)
    parser.add_argument("--specificity-floor", type=float, default=0.60)
    parser.add_argument("--min-positives", type=int, default=20)
    parser.add_argument("--min-negatives", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = select_from_benchmark(
        args.benchmark,
        sensitivity_target=args.sensitivity_target,
        specificity_floor=args.specificity_floor,
        min_positives=args.min_positives,
        min_negatives=args.min_negatives,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(artifact), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} "
        f"(thresholds_complete={artifact['thresholds_complete']})"
    )
    if artifact["blocked_labels"]:
        print("Blocked labels:", ", ".join(artifact["blocked_labels"]))


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
