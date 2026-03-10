"""Backwards-compatibility shim — use ``cubic.metrics.spectral.iterators`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc.iterators has been renamed to cubic.metrics.spectral.iterators. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral.iterators import *  # noqa: F401, F403, E402
