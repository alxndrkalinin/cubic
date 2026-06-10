"""Tests for the skimage proxy module."""

import numpy as np
import pytest

import cubic.skimage as mc_skimage
from cubic.cuda import CUDAManager, ascupy, asnumpy, get_device


@pytest.mark.parametrize("use_gpu", [False, True])
def test_dispatch_on_array_passed_by_keyword(
    use_gpu: bool, gpu_available: bool
) -> None:
    """A GPU array under a non-``image`` keyword routes to the cuCIM backend.

    ``rescale(image=..., scale=...)`` works either way, but detection now
    scans all arguments so a GPU array under any keyword reaches the GPU
    implementation instead of crashing a host scikit-image call.
    """
    img = np.random.random((8, 8)).astype(np.float32)
    cpu_res = mc_skimage.transform.rescale(image=img, scale=0.5, preserve_range=True)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = mc_skimage.transform.rescale(
            image=ascupy(img), scale=0.5, preserve_range=True
        )
        assert get_device(gpu_res) == "GPU"
        assert np.allclose(asnumpy(gpu_res), cpu_res, atol=1e-5)
    else:
        assert np.allclose(
            mc_skimage.transform.rescale(image=img, scale=0.5, preserve_range=True),
            cpu_res,
        )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_filters_gaussian_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """Compare Gaussian filtering on CPU and GPU."""
    img = np.random.random((5, 5)).astype(np.float32)
    cpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        cp = CUDAManager().get_cp()
        gpu_res = mc_skimage.filters.gaussian(
            cp.asarray(img), sigma=1.0, preserve_range=True
        )
        assert np.allclose(asnumpy(gpu_res), cpu_res, atol=1e-6)
    else:
        gpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)
        assert np.allclose(gpu_res, cpu_res)
