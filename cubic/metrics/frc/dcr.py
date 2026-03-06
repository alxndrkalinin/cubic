"""Backwards-compatibility shim — use ``cubic.metrics.spectral.dcr`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.dcr has been renamed to cubic.metrics.spectral.dcr. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.dcr import *  # noqa: F401, F403, E402
from cubic.metrics.spectral.dcr import (  # noqa: F401, E402
    _find_peak_in_curve,
    _dcr_curve_3d_sectioned,
    _generate_highpass_sigmas,
    _compute_decorrelation_curve_sectioned,
)
