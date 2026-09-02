"""Public facade for coverage plotting style helpers.

Style decisions stay beside metric resolution in ``coverage.figures`` so
categorical serving-TX plots and scalar RF metrics use the same labels, color
groups, and colorbar policy.
"""

from .figures import (
    _coverage_metric_style,
    _coverage_metric_title,
    _serving_tx_label_for_value,
    serving_tx_class_labels,
)

__all__ = [
    "_coverage_metric_style",
    "_coverage_metric_title",
    "_serving_tx_label_for_value",
    "serving_tx_class_labels",
]
