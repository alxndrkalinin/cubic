"""Compatibility layer for ``skimage.io`` with optional GPU support."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import io as skio

from ..cuda import asnumpy, to_device


def imread(fname: str | bytes, *, device: str = "CPU", **kwargs: Any) -> np.ndarray:
    """Read image from ``fname`` into ``device`` memory."""
    img = skio.imread(fname, **kwargs)
    if device.upper() == "GPU":
        return to_device(img, "GPU")
    return np.asarray(img)


def imsave(fname: str | bytes, arr: np.ndarray, **kwargs: Any) -> None:
    """Save ``arr`` to ``fname`` using CPU-based ``skimage`` IO."""
    cpu_arr = asnumpy(arr)
    skio.imsave(fname, cpu_arr, **kwargs)


def imwrite(fname: str | bytes, arr: np.ndarray, **kwargs: Any) -> None:
    """Alias for :func:`imsave`."""
    imsave(fname, arr, **kwargs)
