"""Tests for plotting helpers."""

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cubic.plot_utils import show_image_error  # noqa: E402


def test_show_image_error_symmetric_clim() -> None:
    """The diverging error map is centered on zero (symmetric color limits)."""
    img = np.zeros((4, 4), dtype=np.float32)
    fig = show_image_error(img, bit_depth=14)
    try:
        vmin, vmax = fig.axes[0].images[0].get_clim()
        assert vmin == -vmax
        assert vmax == 2**14 - 1
    finally:
        plt.close(fig)
