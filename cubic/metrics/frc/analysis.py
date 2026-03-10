"""Backwards-compatibility shim — use ``cubic.metrics.spectral.analysis`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.analysis has been renamed to cubic.metrics.spectral.analysis. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.analysis import *  # noqa: F401, F403, E402
