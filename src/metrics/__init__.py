"""Attribution error metrics and 3-way error decomposition."""

from .error_metrics import (
    causal_alignment_metrics,
    normalized_l2,
    rank_metrics,
)
from .decomposition import decompose_error

__all__ = [
    "causal_alignment_metrics",
    "normalized_l2",
    "rank_metrics",
    "decompose_error",
]