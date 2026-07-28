"""Detailed, patient-aware analysis for multi-label classifiers.

This module is intentionally independent of any model framework.  It consumes an
``(images, labels)`` probability matrix so the same evaluation is used for a public
baseline, a locally trained checkpoint, and every subsequent ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


def evaluate_multilabel_classifier(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    thresholds: float | Mapping[str, float] = 0.5,
    patient_ids: Sequence[str | int] | None = None,
    bootstrap_samples: int = 0,
    seed: int = 42,
) -> dict:
    """Return ranking, calibration, and operating-point metrics.

    Confidence intervals are percentile intervals from patient-clustered bootstrap
    samples.  Sampling patients rather than images prevents repeat studies from the
    same patient being treated as independent observations.
    """
    probabilities, labels = _validate(probabilities, labels, class_names)
    threshold_values = _threshold_array(thresholds, class_names)
    patient_values = _patient_array(patient_ids, len(labels))

    per_label = {
        name: _label_metrics(
            labels[:, index],
            probabilities[:, index],
            threshold_values[index],
        )
        for index, name in enumerate(class_names)
    }
    report = {
        "images": int(len(labels)),
        "patients": int(len(np.unique(patient_values))),
        "scoreable_labels": int(
            sum(row["auroc"] is not None for row in per_label.values())
        ),
        "macro": _macro_metrics(per_label),
        "study_level": _study_metrics(
            probabilities, labels, threshold_values
        ),
        "per_label": per_label,
    }
    if bootstrap_samples:
        report["confidence_intervals"] = _clustered_bootstrap(
            probabilities,
            labels,
            class_names,
            threshold_values,
            patient_values,
            bootstrap_samples,
            seed,
        )
    return report


def evaluate_subgroups(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    subgroup_values: Mapping[str, Sequence[str]],
    *,
    thresholds: float | Mapping[str, float] = 0.5,
    patient_ids: Sequence[str | int] | None = None,
    min_images: int = 20,
) -> dict[str, dict[str, dict]]:
    """Evaluate declared subgroups without selecting thresholds on those groups."""
    probabilities, labels = _validate(probabilities, labels, class_names)
    patient_values = _patient_array(patient_ids, len(labels))
    result: dict[str, dict[str, dict]] = {}
    for field, raw_values in subgroup_values.items():
        values = np.asarray([str(value) for value in raw_values], dtype=object)
        if len(values) != len(labels):
            raise ValueError(f"subgroup {field!r} length does not match samples")
        result[field] = {}
        for value in sorted(set(values.tolist())):
            mask = values == value
            if int(mask.sum()) < min_images:
                result[field][value] = {
                    "images": int(mask.sum()),
                    "status": f"insufficient_images_below_{min_images}",
                }
                continue
            result[field][value] = evaluate_multilabel_classifier(
                probabilities[mask],
                labels[mask],
                class_names,
                thresholds=thresholds,
                patient_ids=patient_values[mask],
            )
    return result


def rank_classification_errors(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    thresholds: float | Mapping[str, float] = 0.5,
    sample_ids: Sequence[str] | None = None,
    limit: int = 10,
) -> dict[str, dict[str, list[dict]]]:
    """Return the highest-confidence false positives and false negatives per label."""
    probabilities, labels = _validate(probabilities, labels, class_names)
    threshold_values = _threshold_array(thresholds, class_names)
    ids = list(sample_ids or [str(index) for index in range(len(labels))])
    if len(ids) != len(labels):
        raise ValueError("sample_ids length does not match samples")

    errors: dict[str, dict[str, list[dict]]] = {}
    for index, name in enumerate(class_names):
        scores = probabilities[:, index]
        truth = labels[:, index].astype(bool)
        predicted = scores >= threshold_values[index]
        false_positive = np.where(predicted & ~truth)[0]
        false_negative = np.where(~predicted & truth)[0]
        false_positive = false_positive[np.argsort(scores[false_positive])[::-1]][:limit]
        false_negative = false_negative[np.argsort(scores[false_negative])][:limit]
        errors[name] = {
            "false_positives": [
                {"sample_id": ids[i], "score": float(scores[i])}
                for i in false_positive
            ],
            "false_negatives": [
                {"sample_id": ids[i], "score": float(scores[i])}
                for i in false_negative
            ],
        }
    return errors


def _label_metrics(
    truth: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int | None]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = truth.astype(bool)
    predicted = scores >= threshold
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    tn = int((~predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())

    scoreable = positives > 0 and negatives > 0
    auroc = float(roc_auc_score(truth, scores)) if scoreable else None
    auprc = float(average_precision_score(truth, scores)) if positives else None
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "images": int(len(truth)),
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / len(truth) if len(truth) else None,
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((scores - truth.astype(float)) ** 2)),
        "ece": _binary_ece(scores, truth),
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _macro_metrics(per_label: Mapping[str, Mapping]) -> dict[str, float | None]:
    fields = (
        "auroc",
        "auprc",
        "brier",
        "ece",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
    )
    return {
        field: _finite_mean([row[field] for row in per_label.values()])
        for field in fields
    }


def _study_metrics(
    probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> dict[str, float | int | None]:
    predicted = probabilities >= thresholds.reshape(1, -1)
    truth = labels.astype(bool)
    normal = ~truth.any(axis=1)
    abnormal = truth.any(axis=1)
    any_prediction = predicted.any(axis=1)
    return {
        "dataset_normal_images": int(normal.sum()),
        "dataset_abnormal_images": int(abnormal.sum()),
        "normal_any_false_positive": (
            float(any_prediction[normal].mean()) if normal.any() else None
        ),
        "normal_mean_findings": (
            float(predicted[normal].sum(axis=1).mean()) if normal.any() else None
        ),
        "abnormal_any_true_label_detected": (
            float((predicted[abnormal] & truth[abnormal]).any(axis=1).mean())
            if abnormal.any()
            else None
        ),
        "exact_match_accuracy": float((predicted == truth).all(axis=1).mean()),
    }


def _clustered_bootstrap(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    thresholds: np.ndarray,
    patient_ids: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    unique_patients = np.unique(patient_ids)
    patient_rows = {
        patient: np.where(patient_ids == patient)[0] for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    macro_values: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "sensitivity": [],
        "specificity": [],
    }
    per_label_auc: dict[str, list[float]] = {name: [] for name in class_names}
    per_label_auprc: dict[str, list[float]] = {name: [] for name in class_names}

    for _ in range(samples):
        chosen = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        row_indices = np.concatenate([patient_rows[patient] for patient in chosen])
        per_label = {
            name: _label_metrics(
                labels[row_indices, index],
                probabilities[row_indices, index],
                thresholds[index],
            )
            for index, name in enumerate(class_names)
        }
        macro = _macro_metrics(per_label)
        for field in macro_values:
            value = macro[field]
            if value is not None:
                macro_values[field].append(float(value))
        for name, row in per_label.items():
            if row["auroc"] is not None:
                per_label_auc[name].append(float(row["auroc"]))
            if row["auprc"] is not None:
                per_label_auprc[name].append(float(row["auprc"]))

    return {
        "method": "patient_clustered_percentile_bootstrap",
        "samples": int(samples),
        "macro": {
            field: _percentile_interval(values)
            for field, values in macro_values.items()
        },
        "per_label_auroc": {
            name: _percentile_interval(values)
            for name, values in per_label_auc.items()
        },
        "per_label_auprc": {
            name: _percentile_interval(values)
            for name, values in per_label_auprc.items()
        },
    }


def _binary_ece(scores: np.ndarray, truth: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores > lo) & (scores <= hi)
        if mask.any():
            value += (
                abs(float(truth[mask].mean()) - float(scores[mask].mean()))
                * int(mask.sum())
                / len(scores)
            )
    return float(value)


def _threshold_array(
    thresholds: float | Mapping[str, float], class_names: Sequence[str]
) -> np.ndarray:
    if isinstance(thresholds, Mapping):
        missing = [name for name in class_names if name not in thresholds]
        if missing:
            raise ValueError(f"missing thresholds for: {missing}")
        values = np.asarray([thresholds[name] for name in class_names], dtype=float)
    else:
        values = np.full(len(class_names), float(thresholds), dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("thresholds must be finite values in [0, 1]")
    return values


def _patient_array(
    patient_ids: Sequence[str | int] | None, samples: int
) -> np.ndarray:
    if patient_ids is None:
        return np.asarray([f"image-{index}" for index in range(samples)], dtype=object)
    if len(patient_ids) != samples:
        raise ValueError("patient_ids length does not match samples")
    return np.asarray([str(value) for value in patient_ids], dtype=object)


def _validate(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)
    if probabilities.ndim != 2 or probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same two-dimensional shape")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("class_names length does not match prediction columns")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    return probabilities, labels.astype(np.uint8)


def _finite_mean(values: Sequence[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def _percentile_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return {"low": None, "high": None, "valid_samples": 0}
    low, high = np.percentile(finite, [2.5, 97.5])
    return {
        "low": float(low),
        "high": float(high),
        "valid_samples": int(len(finite)),
    }
