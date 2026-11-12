"""Feature attribution explainers (exact SHAP, TreeSHAP, KernelSHAP, LIME, permutation)."""

from src.explainers.attribution import EXPLAINER_NAMES, Attribution, explain, supports
from src.explainers.io import (
    add_reference_to_sidecar, artifact_prefix, has_reference, load_attribution, save_attribution,
)
from src.explainers.reference import (
    MissingReferenceError, ReferenceShap, attribution_from_reference, compute_reference_shap,
    read_reference,
)

__all__ = ["EXPLAINER_NAMES", "Attribution", "explain", "supports",
           "artifact_prefix", "load_attribution", "save_attribution",
           "add_reference_to_sidecar", "has_reference",
           "MissingReferenceError", "ReferenceShap", "attribution_from_reference",
           "compute_reference_shap", "read_reference"]
