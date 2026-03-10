"""Backwards-compatibility shim — use ``cubic.metrics.spectral.frc`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.frc has been renamed to cubic.metrics.spectral.frc. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.frc import *  # noqa: F401, F403, E402
from cubic.metrics.spectral.frc import (  # noqa: F401, E402
    preprocess_images,
    _calibration_factor,
    _calculate_fsc_sectioned_hist,
)
