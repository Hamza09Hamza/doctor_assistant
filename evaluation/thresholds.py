"""Per-label operating-threshold selection for multi-label medical classifiers.

Thresholds are selected on validation data only. The test set is reserved for one
subsequent evaluation of the frozen thresholds; selecting and scoring on the same
patients would produce optimistically biased results.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ThresholdSelection:
    label: str
    threshold: float
    sensitivity: float
    specificity: float
    positives: int
    negatives: int
    supported: bool
    meets_constraints: bool

    def to_dict(self) -> dict:
        return asdict(self)


def select_per_label_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    sensitivity_target: float = 0.85,
    specificity_floor: float = 0.0,
    min_positives: int = 10,
    min_negatives: int = 10,
    unsupported_threshold: float = 0.5,
) -> list[ThresholdSelection]:
    """Select the most specific threshold meeting declared validation constraints.

    For every supported label, candidates are the observed validation probabilities
    plus 0 and 1. Among candidates meeting both constraints, selection maximizes
    specificity, then sensitivity, then threshold. If the pair is infeasible, the
    closest candidate is returned with ``meets_constraints=False``.

    Labels with too few examples are marked unsupported. Their fallback is recorded but
    must not be presented as a calibrated threshold.
    """
    probabilities, labels = _validate_inputs(probabilities, labels, class_names)
    _validate_rate("sensitivity_target", sensitivity_target)
    _validate_rate("specificity_floor", specificity_floor)
    _validate_rate("unsupported_threshold", unsupported_threshold)

    selections: list[ThresholdSelection] = []
    for index, name in enumerate(class_names):
        truth = labels[:, index].astype(bool)
        scores = probabilities[:, index]
        positives = int(truth.sum())
        negatives = int((~truth).sum())
        supported = positives >= min_positives and negatives >= min_negatives

        if not supported:
            sensitivity, specificity = binary_sensitivity_specificity(
                truth, scores >= unsupported_threshold
            )
            selections.append(
                ThresholdSelection(
                    name,
                    float(unsupported_threshold),
                    sensitivity,
                    specificity,
                    positives,
                    negatives,
                    False,
                    False,
                )
            )
            continue

        measured: list[tuple[float, float, float]] = []
        candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
        for threshold in candidates:
            sensitivity, specificity = binary_sensitivity_specificity(
                truth, scores >= threshold
            )
            measured.append((float(threshold), sensitivity, specificity))

        feasible = [
            row
            for row in measured
            if row[1] >= sensitivity_target and row[2] >= specificity_floor
        ]
        if feasible:
            threshold, sensitivity, specificity = max(
                feasible, key=lambda row: (row[2], row[1], row[0])
            )
            meets = True
        else:
            threshold, sensitivity, specificity = min(
                measured,
                key=lambda row: (
                    max(0.0, sensitivity_target - row[1])
                    + max(0.0, specificity_floor - row[2]),
                    -row[1],
                    -row[2],
                ),
            )
            meets = False

        selections.append(
            ThresholdSelection(
                name,
                threshold,
                sensitivity,
                specificity,
                positives,
                negatives,
                True,
                meets,
            )
        )
    return selections


def evaluate_frozen_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    thresholds: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    """Evaluate already-selected thresholds on a separate held-out dataset."""
    probabilities, labels = _validate_inputs(probabilities, labels, class_names)
    missing = [name for name in class_names if name not in thresholds]
    if missing:
        raise ValueError(f"Missing frozen thresholds for: {missing}")

    results: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(class_names):
        truth = labels[:, index].astype(bool)
        predicted = probabilities[:, index] >= float(thresholds[name])
        sensitivity, specificity = binary_sensitivity_specificity(truth, predicted)
        results[name] = {
            "threshold": float(thresholds[name]),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "positives": int(truth.sum()),
            "negatives": int((~truth).sum()),
        }
    return results


def calibrated_threshold_dict(
    selections: Sequence[ThresholdSelection],
) -> dict[str, float]:
    """Return thresholds only when every label had adequate, feasible validation data."""
    invalid = [
        selection.label
        for selection in selections
        if not selection.supported or not selection.meets_constraints
    ]
    if invalid:
        raise ValueError(
            "Cannot export calibrated thresholds; unsupported/infeasible labels: "
            + ", ".join(invalid)
        )
    return {selection.label: selection.threshold for selection in selections}


def load_calibrated_thresholds(path: str | Path) -> dict[str, float]:
    """Load an explicitly export-eligible artifact for ``Pipeline(thresholds=...)``.

    Completeness alone is insufficient: exploratory mirror workflows can calculate
    diagnostic candidate thresholds but are not allowed to feed the live pipeline.
    The producing protocol must therefore opt in with
    ``threshold_export_eligible=true`` and must not carry contradictory smoke or
    official-evidence flags.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("thresholds_complete"):
        raise ValueError(f"Threshold artifact is incomplete: {path}")
    dataset = data.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    explicitly_ineligible = (
        data.get("threshold_export_eligible") is not True
        or data.get("smoke_only") is True
        or data.get("eligible_as_official_nih_test_evidence") is False
        or dataset.get("official_nih_manifest_reconciled") is False
    )
    if explicitly_ineligible:
        raise ValueError(
            f"Threshold artifact is not eligible for pipeline export: {path}"
        )
    raw = data.get("pipeline_thresholds")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Threshold artifact has no pipeline_thresholds: {path}")
    thresholds = {str(label): float(value) for label, value in raw.items()}
    for label, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid threshold for {label}: {value}")
    return thresholds


def binary_sensitivity_specificity(
    truth: np.ndarray, predicted: np.ndarray
) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    positives = truth.sum()
    negatives = (~truth).sum()
    sensitivity = float(predicted[truth].mean()) if positives else float("nan")
    specificity = float((~predicted[~truth]).mean()) if negatives else float("nan")
    return sensitivity, specificity


def _validate_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or labels.ndim != 2:
        raise ValueError("probabilities and labels must both have shape (samples, labels)")
    if probabilities.shape != labels.shape:
        raise ValueError(
            f"probabilities shape {probabilities.shape} != labels shape {labels.shape}"
        )
    if probabilities.shape[1] != len(class_names):
        raise ValueError("class_names length does not match the label dimension")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    return probabilities, labels


def _validate_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
