"""Tests for the ``Image`` container."""

import numpy as np
import pytest

from cubic.cuda import ascupy, get_device
from cubic.image import Image


def test_image_defaults_to_input_device() -> None:
    """Without an explicit ``device`` the array stays where it is."""
    img = Image(np.ones((2, 4, 4), dtype=np.float32), spacing=(0.4, 0.1, 0.1))
    assert img.device == "CPU"
    assert img.shape == (2, 4, 4)
    assert img.spacing == (0.4, 0.1, 0.1)
    assert img.filename is None


def test_image_as_float_converts_integer_input() -> None:
    """``as_float=True`` (the default) scales integers into [0, 1]."""
    img = Image(np.full((2, 2), 255, dtype=np.uint8), spacing=(1, 1))
    assert img.data.dtype == np.float32
    np.testing.assert_allclose(img.data, 1.0)

    raw = Image(np.full((2, 2), 255, dtype=np.uint8), spacing=(1, 1), as_float=False)
    assert raw.data.dtype == np.uint8


def test_image_invalid_device_raises() -> None:
    """An unknown device string raises ``ValueError`` (was a bare ``assert``)."""
    with pytest.raises(ValueError, match="Device should be 'CPU' or 'GPU'"):
        Image(np.ones((2, 2), dtype=np.float32), spacing=(1, 1), device="cpu")
    with pytest.raises(ValueError, match="Device should be 'CPU' or 'GPU'"):
        Image(np.ones((2, 2), dtype=np.float32), spacing=(1, 1), device="cuda")


def test_image_to_cpu_from_gpu(gpu_available: bool) -> None:
    """``to_cpu()`` materializes GPU data on the host.

    It used to raise ``TypeError: Implicit conversion to a NumPy array is not
    allowed`` because ``to_device(..., "CPU")`` called ``np.asarray``.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    img = Image(arr, spacing=(1, 1, 1), device="GPU")
    assert img.device == "GPU"
    assert get_device(img.data) == "GPU"

    img.to_cpu()
    assert img.device == "CPU"
    assert isinstance(img.data, np.ndarray)
    np.testing.assert_allclose(img.data, arr)


def test_image_to_cpu_is_idempotent() -> None:
    """``to_cpu()`` on host data is a no-op, not an error."""
    arr = np.arange(4, dtype=np.float32)
    img = Image(arr, spacing=(1,))
    img.to_cpu()
    img.to_cpu()
    assert img.device == "CPU"
    np.testing.assert_allclose(img.data, arr)


def test_image_round_trip_gpu(gpu_available: bool) -> None:
    """CPU → GPU → CPU preserves the data."""
    if not gpu_available:
        pytest.skip("GPU not available")
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    img = Image(arr, spacing=(1, 1))
    img.to_gpu()
    assert img.device == "GPU" and get_device(img.data) == "GPU"
    img.to_cpu()
    np.testing.assert_allclose(img.data, arr)


def test_image_accepts_gpu_input(gpu_available: bool) -> None:
    """A CuPy input is detected as a GPU image without an explicit device."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = Image(ascupy(np.ones((2, 2), dtype=np.float32)), spacing=(1, 1))
    assert img.device == "GPU"
