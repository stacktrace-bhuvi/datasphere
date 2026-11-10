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
# build_error_table / ERROR_COLUMNS live in src.metrics.pipeline_runner (imported lazily so that
# `python -m src.metrics.pipeline_runner` does not trigger a double-import warning).

__all__ = [
    "causal_alignment_metrics",
    "importance_shares",
    "normalized_l2",
    "rank_metrics",
    "share_l2",
    "top_k_overlap",
    "decompose_error",
]
