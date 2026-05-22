"""Tests for cubic.metrics.pcc."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cubic.cuda import ascupy
from cubic.metrics import pcc


@pytest.fixture
def test_images() -> tuple[np.ndarray, np.ndarray]:
    """Return seeded image pair with non-degenerate correlation."""
    rng = np.random.default_rng(42)
    img1 = rng.random((8, 8, 8)).astype(np.float32)
    img2 = img1 + 0.1 * rng.random((8, 8, 8)).astype(np.float32)
    return img1, img2


def test_identical_inputs_score_one(test_images):
    """PCC(x, x) is exactly 1."""
    img1, _ = test_images
    assert pcc(img1, img1) == pytest.approx(1.0)


def test_anticorrelated_inputs_score_minus_one(test_images):
    """PCC(x, -x) is exactly -1."""
    img1, _ = test_images
    assert pcc(img1, -img1) == pytest.approx(-1.0)


def test_matches_numpy_corrcoef(test_images):
    """Matches np.corrcoef on the flattened arrays."""
    img1, img2 = test_images
    expected = float(np.corrcoef(img1.ravel(), img2.ravel())[0, 1])
    assert pcc(img1, img2) == pytest.approx(expected, rel=1e-6)


def test_constant_input_returns_nan():
    """Zero-variance input returns nan (not raise, not 0)."""
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    assert math.isnan(pcc(a, b))


def test_mask_restricts_correlation():
    """Masked correlation only uses the masked voxels."""
    rng = np.random.default_rng(0)
    img1 = rng.random((6, 6)).astype(np.float32)
    img2 = rng.random((6, 6)).astype(np.float32)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    expected = float(np.corrcoef(img1[mask].ravel(), img2[mask].ravel())[0, 1])
    assert pcc(img1, img2, mask=mask) == pytest.approx(expected, rel=1e-6)


def test_scale_invariant_flag_is_a_noop(test_images):
    """PCC is intrinsically affine-invariant; the flag is a no-op."""
    img1, img2 = test_images
    plain = pcc(img1, img2, scale_invariant=False)
    invariant = pcc(img1, img2, scale_invariant=True)
    assert plain == pytest.approx(invariant, abs=1e-6)


def test_shape_mismatch_raises():
    """Shape mismatch raises ValueError."""
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        pcc(a, b)


def test_returns_python_float(test_images):
    """Result is a Python float, not a numpy scalar."""
    img1, img2 = test_images
    result = pcc(img1, img2)
    assert isinstance(result, float)


def test_gpu_dispatch(test_images, gpu_available):
    """PCC on CuPy input matches the NumPy result."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img1, img2 = test_images
    expected = pcc(img1, img2)
    actual = pcc(ascupy(img1), ascupy(img2))
    assert actual == pytest.approx(expected, rel=1e-5)
