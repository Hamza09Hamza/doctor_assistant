"""Clinical metrics for multi-label classification (chest X-ray style).

`ClassificationEvaluator` in metrics.py assumes mutually exclusive classes and uses a
confusion matrix. That's wrong for chest X-ray where a study can be simultaneously
Cardiomegaly AND Effusion AND Pneumothorax.

`MultilabelEvaluator` drops the confusion-matrix approach and works per-label:
  - AUC per label (the number CheXNet / radiologist papers report)
  - Macro AUC across all labels (the headline number for model selection)
  - Sensitivity and specificity per label at a decision threshold (default 0.5)
  - ECE — same as single-label: gaps between sigmoid confidence and accuracy

It follows the same interface as `ClassificationEvaluator` so the trainer can use
either evaluator without changes: `.reset()`, `.update(logits, labels)`, `.compute()`.
"""

from __future__ import annotations

import numpy as np
import torch


class MultilabelEvaluator:
    """Accumulate multi-label predictions and compute per-class + macro metrics."""

    def __init__(
        self,
        class_names: list[str],
        threshold: float = 0.5,
    ) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._logits: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one batch.

        `logits`  shape (B, C) — raw pre-sigmoid outputs from the classification head.
        `labels`  shape (B, C) float32 — multi-hot ground truth (0.0 or 1.0).
        """
        self._logits.append(logits.detach().cpu().float().numpy())
        self._labels.append(labels.detach().cpu().float().numpy())

    def compute(self) -> dict:
        """Return per-label and macro metrics. Keys match ClassificationEvaluator
        where possible so the trainer's `_monitored_score` works with both."""
        from sklearn.metrics import roc_auc_score

        logits = np.concatenate(self._logits, axis=0)   # (N, C)
        labels = np.concatenate(self._labels, axis=0)   # (N, C)
        probs = _sigmoid(logits)                         # (N, C)
        preds = (probs >= self.threshold).astype(float)  # (N, C)

        auc_per_label: dict[str, float | None] = {}
        sens_per_label: dict[str, float] = {}
        spec_per_label: dict[str, float] = {}

        valid_aucs: list[float] = []
        for i, name in enumerate(self.class_names):
            gt = labels[:, i]
            pr = probs[:, i]
            pd = preds[:, i]

            # AUC: only defined when both classes appear in the split
            if gt.sum() > 0 and (1 - gt).sum() > 0:
                auc = float(roc_auc_score(gt, pr))
                valid_aucs.append(auc)
            else:
                auc = None
            auc_per_label[name] = auc

            tp = float(((pd == 1) & (gt == 1)).sum())
            fn = float(((pd == 0) & (gt == 1)).sum())
            tn = float(((pd == 0) & (gt == 0)).sum())
            fp = float(((pd == 1) & (gt == 0)).sum())
            sens_per_label[name] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            spec_per_label[name] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None
        macro_sens = float(np.nanmean(list(sens_per_label.values())))
        macro_spec = float(np.nanmean(list(spec_per_label.values())))

        # Exact-match accuracy (all labels correct per sample) — rarely useful but
        # kept for completeness / comparison with single-label accuracy.
        exact_match = float((preds == labels).all(axis=1).mean())

        return {
            # Headline — the trainer monitors "auc" by default.
            "auc": macro_auc,
            "auc_per_label": auc_per_label,
            # Sensitivity / specificity at threshold.
            "sensitivity": macro_sens,
            "specificity": macro_spec,
            "sensitivity_per_label": sens_per_label,
            "specificity_per_label": spec_per_label,
            # Macro aggregates (mirrors ClassificationEvaluator naming).
            "macro_sensitivity": macro_sens,
            "macro_specificity": macro_spec,
            # Misc.
            "exact_match_accuracy": exact_match,
            "ece": _multilabel_ece(probs, labels),
        }

    def false_negative_indices(self, label_name: str) -> list[int]:
        """Positions where a positive case was called negative for one label."""
        idx = self.class_names.index(label_name)
        logits = np.concatenate(self._logits, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        probs = _sigmoid(logits)
        preds = (probs[:, idx] >= self.threshold).astype(float)
        gt = labels[:, idx]
        return np.where((gt == 1) & (preds == 0))[0].tolist()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _multilabel_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Average ECE across all labels — treats each (sample, label) pair independently."""
    eces: list[float] = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = probs.shape[0]
    for c in range(probs.shape[1]):
        conf = probs[:, c]
        gt = labels[:, c]
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                ece += abs(gt[mask].mean() - conf[mask].mean()) * mask.sum() / n
        eces.append(ece)
    return float(np.mean(eces))
