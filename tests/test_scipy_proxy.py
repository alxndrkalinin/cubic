"""Tests for the SciPy proxy implementation."""

import warnings
from typing import Any

import numpy as np
import pytest
from scipy import signal as sp_signal

import cubic.cuda as mc_cuda
import cubic.scipy as mc_scipy
from cubic.cuda import ascupy, asnumpy, get_device


@pytest.mark.parametrize("use_gpu", [False, True])
def test_dispatch_on_array_passed_by_keyword(
    use_gpu: bool, gpu_available: bool
) -> None:
    """A GPU array under a non-``input`` keyword routes to the GPU backend.

    ``fftconvolve(in1=..., in2=...)`` has no ``input`` argument, so the old
    first-arg-only detection saw no device, routed to CPU SciPy, and crashed
    on the raw CuPy arrays. Detection now scans every argument.
    """
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 0.5])
    cpu_res = mc_scipy.signal.fftconvolve(in1=a, in2=b)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = mc_scipy.signal.fftconvolve(in1=ascupy(a), in2=ascupy(b))
        assert get_device(gpu_res) == "GPU"
        assert np.allclose(asnumpy(gpu_res), cpu_res)
    else:
        assert np.allclose(mc_scipy.signal.fftconvolve(in1=a, in2=b), cpu_res)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_ndimage_laplace_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """Compare Laplace filter on CPU and GPU."""
    a = np.asarray([1, 2, 3], dtype=float)
    cpu_res = mc_scipy.ndimage.laplace(a)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = mc_scipy.ndimage.laplace(ascupy(a))
        assert np.allclose(cpu_res, asnumpy(gpu_res))
    else:
        gpu_res = mc_scipy.ndimage.laplace(a)
        assert np.allclose(cpu_res, gpu_res)


def test_gpu_fallback_to_cpu_when_cupyx_unavailable(
    monkeypatch: pytest.MonkeyPatch, gpu_available: bool
) -> None:
    """A missing ``cupyx.scipy`` backend computes on CPU and returns to the GPU.

    When the ``cupyx.scipy`` import fails for a GPU input, the proxy warns,
    coerces the arguments to host, runs ``scipy``, then moves the result back
    to the GPU so the caller still gets a device-consistent array.
    """
    if not gpu_available:
        pytest.skip("GPU not available")

    real_import_module = mc_cuda.import_module

    def fake_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cupyx"):
            raise ModuleNotFoundError(f"simulated missing module: {name}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(mc_cuda, "import_module", fake_import_module)

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 0.5])
    cpu_res = sp_signal.fftconvolve(a, b)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        gpu_res = mc_scipy.signal.fftconvolve(ascupy(a), ascupy(b))

    assert get_device(gpu_res) == "GPU"
    assert np.allclose(asnumpy(gpu_res), cpu_res)


def test_cpu_route_unknown_function_raises_without_warning() -> None:
    """An unknown function on the host route raises without a bogus warning.

    ``module_name`` was already ``scipy.<sub>`` on the CPU route, so the proxy
    warned ``cupyx.scipy.<sub>.<name> is unavailable, falling back to CPU`` and
    then re-raised the very same ``AttributeError`` it had just caught.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        with pytest.raises(AttributeError):
            mc_scipy.ndimage.definitely_not_a_scipy_function(np.zeros(4))


def test_missing_cupyx_function_falls_back_to_cpu(gpu_available: bool) -> None:
    """A function absent from ``cupyx.scipy`` runs on the host for GPU input."""
    if not gpu_available:
        pytest.skip("GPU not available")
    # cupyx.scipy.ndimage has no ``distance_transform_cdt``
    from scipy.ndimage import distance_transform_cdt

    img = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    cpu_res = distance_transform_cdt(img)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        gpu_res = mc_scipy.ndimage.distance_transform_cdt(ascupy(img))
    assert get_device(gpu_res) == "GPU"
    np.testing.assert_array_equal(asnumpy(gpu_res), cpu_res)
