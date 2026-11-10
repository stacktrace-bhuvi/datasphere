"""Attribution error metrics and 3-way error decomposition."""

from .error_metrics import (
    causal_alignment_metrics,
    importance_shares,
    normalized_l2,
    rank_metrics,
    share_l2,
    top_k_overlap,
)
from .decomposition import decompose_error
from .pipeline_runner import ERROR_COLUMNS, build_error_table

__all__ = [
    "causal_alignment_metrics",
    "importance_shares",
    "normalized_l2",
    "rank_metrics",
    "share_l2",
    "top_k_overlap",
    "decompose_error",
    "ERROR_COLUMNS",
    "build_error_table",
]
