"""Implement miscellaneous utility functions."""

import numpy as np
import numpy.typing as npt


def print_mean_std(arr: npt.ArrayLike, prefix: str = "") -> None:
    """Print mean and std of an array."""
    arr = np.asarray(arr)
    print(f"{prefix} mean={np.mean(arr):.4g} +/- {np.std(arr):.4g}")
