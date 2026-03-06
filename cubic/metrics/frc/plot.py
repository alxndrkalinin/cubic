"""Backwards-compatibility shim — use ``cubic.metrics.spectral.plot`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.plot has been renamed to cubic.metrics.spectral.plot. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.plot import *  # noqa: F401, F403, E402
