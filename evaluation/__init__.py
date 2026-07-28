"""Evaluation layer: clinical metrics that go beyond raw accuracy."""

from .metrics import ClassificationEvaluator
from .multilabel_metrics import MultilabelEvaluator
from .thresholds import (
    ThresholdSelection,
    calibrated_threshold_dict,
    evaluate_frozen_thresholds,
    load_calibrated_thresholds,
    select_per_label_thresholds,
)
from .multilabel_analysis import (
    evaluate_multilabel_classifier,
    evaluate_subgroups,
    rank_classification_errors,
)

__all__ = [
    "ClassificationEvaluator",
    "MultilabelEvaluator",
    "ThresholdSelection",
    "select_per_label_thresholds",
    "evaluate_frozen_thresholds",
    "calibrated_threshold_dict",
    "load_calibrated_thresholds",
    "evaluate_multilabel_classifier",
    "evaluate_subgroups",
    "rank_classification_errors",
]
