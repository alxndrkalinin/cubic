"""Tests for cuCIM IO wrapper."""

from __future__ import annotations

import numpy as np
import pytest

import cubic.skimage as mc_skimage
from cubic.cuda import ascupy, asnumpy


@pytest.mark.parametrize("use_gpu", [False, True])
def test_io_roundtrip(tmp_path, use_gpu: bool, gpu_available: bool) -> None:
    """Write and read an image using the IO proxy."""
    img = np.random.randint(0, 255, (4, 4), dtype=np.uint8)
    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        data = ascupy(img)
    else:
        data = img

    fname = tmp_path / "img.png"
    mc_skimage.io.imsave(fname, data)
    result = mc_skimage.io.imread(fname, device="GPU" if use_gpu else "CPU")

    if use_gpu:
        assert np.allclose(asnumpy(result), img)
    else:
        assert np.allclose(result, img)
