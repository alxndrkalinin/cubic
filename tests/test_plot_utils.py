"""Tests for plotting helpers."""

import pytest

pytest.importorskip("matplotlib")  # optional [plot] extra; skip when absent

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend for tests

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from cubic.cuda import ascupy  # noqa: E402
from cubic.plot_utils import show_2d, show_image, show_image_error  # noqa: E402


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


@pytest.mark.parametrize("show_fn", [show_image, show_image_error])
def test_show_accepts_gpu_array(show_fn, gpu_available: bool) -> None:
    """CuPy input is moved to host; ``plt.imshow`` used to raise on it."""
    if not gpu_available:
        pytest.skip("GPU not available")
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    fig = show_fn(ascupy(arr))
    try:
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), arr)
    finally:
        plt.close(fig)


def test_show_2d_accepts_gpu_array(gpu_available: bool) -> None:
    """``show_2d`` max-projects on device, then hands the host copy to matplotlib."""
    if not gpu_available:
        pytest.skip("GPU not available")
    vol = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    fig = show_2d(ascupy(vol))
    try:
        np.testing.assert_array_equal(
            fig.axes[0].images[0].get_array(), vol.max(axis=0)
        )
    finally:
        plt.close(fig)
