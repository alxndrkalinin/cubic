"""MicroSSIM / MicroMS3IM port from juglab/microssim@8bccb17d.

Device-agnostic (NumPy / CuPy) implementation. Public API is exposed via
:mod:`cubic.metrics`; advanced users can also import primitives directly
from this subpackage.
"""

from .ri_factor import ALPHA_MAX_DEFAULT, get_ri_factor, get_global_ri_factor
from .micro_ssim import MicroSSIM, micro_structural_similarity
from .micro_ms3im import MicroMS3IM, micro_multiscale_structural_similarity
from .ssim_elements import SSIMElements, compute_ssim_elements
from .image_processing import (
    linearize_list,
    normalize_min_max,
    compute_norm_parameters,
)

__all__ = [
    "ALPHA_MAX_DEFAULT",
    "MicroMS3IM",
    "MicroSSIM",
    "SSIMElements",
    "compute_norm_parameters",
    "compute_ssim_elements",
    "get_global_ri_factor",
    "get_ri_factor",
    "linearize_list",
    "micro_multiscale_structural_similarity",
    "micro_structural_similarity",
    "normalize_min_max",
]
