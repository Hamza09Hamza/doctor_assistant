"""Clinical classification metrics.

In medicine, accuracy is the least informative number. What decides whether a
model is usable is its error *profile*: a 95%-accurate screen that misses 1 in 5
cancers (low sensitivity) is dangerous, while false positives merely cost a second
read. So the evaluator reports sensitivity (recall), specificity, AUC, and
calibration — and surfaces false negatives explicitly.

`ClassificationEvaluator` accumulates predictions across batches, then `compute()`
returns everything at once. Built on numpy/sklearn; tensors are detached to CPU.
"""

from __future__ import annotations

import numpy as np
import torch


class ClassificationEvaluator:
    def __init__(self, class_names: list[str], positive_index: int | None = None) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        # For binary screens, which class counts as "disease present". Defaults to
        # the last class (convention: index 0 = normal, 1 = abnormal).
        self.positive_index = positive_index if positive_index is not None else self.num_classes - 1
        self.reset()

    def reset(self) -> None:
        self._probs: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        self._probs.append(probs)
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self) -> dict:
        from sklearn.metrics import confusion_matrix, roc_auc_score

        probs = np.concatenate(self._probs, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        preds = probs.argmax(axis=1)

        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        sens, spec = _sensitivity_specificity_per_class(cm)

        result: dict = {
            "accuracy": float((preds == labels).mean()),
            "sensitivity_per_class": {self.class_names[i]: sens[i] for i in range(self.num_classes)},
            "specificity_per_class": {self.class_names[i]: spec[i] for i in range(self.num_classes)},
            "macro_sensitivity": float(np.nanmean(sens)),
            "macro_specificity": float(np.nanmean(spec)),
            "confusion_matrix": cm.tolist(),
            "ece": _expected_calibration_error(probs, labels),
            "auc": _safe_auc(roc_auc_score, probs, labels, self.num_classes),
        }
        # Headline sensitivity = recall of the disease class; the number to watch.
        result["sensitivity"] = sens[self.positive_index]
        result["specificity"] = spec[self.positive_index]
        return result

    def false_negative_indices(self) -> list[int]:
        """Positions where a positive case was called negative — the costly errors."""
        probs = np.concatenate(self._probs, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        preds = probs.argmax(axis=1)
        is_pos = labels == self.positive_index
        return np.where(is_pos & (preds != self.positive_index))[0].tolist()


def _sensitivity_specificity_per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-vs-rest sensitivity and specificity for each class from a confusion matrix."""
    cm = cm.astype(float)
    total = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        spec = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)
    return sens, spec


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: gap between confidence and accuracy, binned by predicted confidence.

    A well-calibrated "0.9" should be right ~90% of the time. Calibration matters
    clinically because doctors act on the confidence, not just the label.
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.sum() / n
    return float(ece)


def _safe_auc(roc_auc_score, probs: np.ndarray, labels: np.ndarray, num_classes: int):
    """AUC, returning None when undefined (e.g. only one class present in a split)."""
    try:
        if num_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return None
