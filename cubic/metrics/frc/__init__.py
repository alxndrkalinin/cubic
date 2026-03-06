"""Backwards-compatibility shim — use ``cubic.metrics.spectral`` instead."""

import warnings

warnings.warn(
    "cubic.metrics.frc has been renamed to cubic.metrics.spectral. "
    "The old path will be removed in 0.8.0.",
    DeprecationWarning,
    stacklevel=2,
)

from cubic.metrics.spectral import *  # noqa: F401, F403, E402
from cubic.metrics.spectral import __all__  # noqa: F401, E402
