
"""Three-cluster specialist experiment framework.

The package lives outside :mod:`fusiformer_v2`: the validated V2 implementation
remains the baseline, while this package owns routing, manifests, specialist
adapters, strict checkpoints and paired evaluation.
"""

from .config import ClusterId, PredictionMode, SplitRatios
from .metrics import improvement_summary, regression_metrics
from .routing import CentroidRouter, FixedBoundaryRouter, load_router

__all__ = [
    "CentroidRouter", "ClusterId", "FixedBoundaryRouter", "PredictionMode",
    "SplitRatios", "improvement_summary", "load_router", "regression_metrics",
]
