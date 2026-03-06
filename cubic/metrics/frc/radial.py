"""Backwards-compatibility shim — use ``cubic.metrics.spectral.radial`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.radial has been renamed to cubic.metrics.spectral.radial. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.radial import *  # noqa: F401, F403, E402
