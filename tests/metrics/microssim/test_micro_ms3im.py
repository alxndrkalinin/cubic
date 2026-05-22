"""Tests for MicroMS3IM and the multi-scale convenience wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics import MicroMS3IM, micro_multiscale_structural_similarity

# MS-SSIM needs >= 2**(n_scales-1) * kernel_size = 16 * 11 = 176 px spatially.
_H = _W = 256


def _seeded_data(n: int = 4, h: int = _H, w: int = _W) -> tuple[np.ndarray, np.ndarray]:
    """Return seeded exponential GT + scaled-and-noised prediction (float32)."""
    rng = np.random.default_rng(0)
    gt = rng.exponential(0.5, (n, h, w)).astype(np.float32) * 1000
    pred = gt * 1.3 + 0.05 * float(gt.max()) * rng.standard_normal(gt.shape).astype(
        np.float32
    )
    return gt, pred


# --- score() validation ----------------------------------------------------


def test_score_before_fit_raises():
    """Score before fit raises ValueError."""
    gt = np.zeros((_H, _W), np.float32)
    with pytest.raises(ValueError, match="call `fit\\(\\)` first"):
        MicroMS3IM().score(gt, gt)


def test_score_3d_raises():
    """Score rejects 3-D inputs (upstream is 2-D only)."""
    gt, pred = _seeded_data()
    m3 = MicroMS3IM().fit(gt, pred)
    with pytest.raises(ValueError, match="Only 2D"):
        m3.score(gt, pred)


def test_score_shape_mismatch_raises():
    """Score rejects mismatched 2-D shapes."""
    gt, pred = _seeded_data()
    m3 = MicroMS3IM().fit(gt, pred)
    with pytest.raises(ValueError, match="same shape"):
        m3.score(gt[0], pred[0, : _W // 2])


def test_return_individual_components_warns():
    """The unsupported `return_individual_components` flag emits a warning."""
    gt, pred = _seeded_data()
    m3 = MicroMS3IM().fit(gt, pred)
    with pytest.warns(UserWarning, match="return_individual_components"):
        m3.score(gt[0], pred[0], return_individual_components=True)


# --- Numerical identity ----------------------------------------------------


def test_identity_score_is_one():
    """MS3IM(x, x) is 1 (within float roundoff) and ri_factor is ~1."""
    rng = np.random.default_rng(0)
    img = rng.random((4, _H, _W)).astype(np.float32)
    m3 = MicroMS3IM().fit(img, img)
    params = m3.get_parameters()
    assert abs(params["ri_factor"] - 1.0) < 1e-3
    score = m3.score(img[0], img[0])
    assert score > 1.0 - 1e-6, f"score={score}"


def test_scaled_input_invariance():
    """Uniform 3x intensity scaling leaves the score very near 1."""
    rng = np.random.default_rng(0)
    gt = rng.random((4, _H, _W)).astype(np.float32)
    pred = gt * 3.0
    m3 = MicroMS3IM().fit(gt, pred)
    score = m3.score(gt[0], pred[0])
    assert score > 0.99, f"score={score}"


# --- Statelessness ---------------------------------------------------------


def test_score_is_stateless_across_calls():
    """Repeated and reversed score() calls return identical values."""
    gt, pred = _seeded_data(n=6)
    m3 = MicroMS3IM().fit(gt, pred)
    forward = [m3.score(gt[i], pred[i]) for i in range(6)]
    reverse = [m3.score(gt[i], pred[i]) for i in reversed(range(6))]
    repeat = [m3.score(gt[0], pred[0]) for _ in range(5)]
    np.testing.assert_allclose(forward, list(reversed(reverse)), atol=1e-7)
    assert max(repeat) - min(repeat) < 1e-7


# --- get_parameters --------------------------------------------------------


def test_get_parameters_keys_match_upstream():
    """get_parameters returns the upstream key set verbatim."""
    gt, pred = _seeded_data()
    m3 = MicroMS3IM().fit(gt, pred)
    params = m3.get_parameters()
    assert set(params.keys()) == {
        "bg_percentile",
        "offset_pred",
        "offset_gt",
        "max_val",
        "ri_factor",
    }


def test_get_parameters_values_finite_after_fit():
    """All fitted params are finite floats."""
    gt, pred = _seeded_data()
    m3 = MicroMS3IM().fit(gt, pred)
    params = m3.get_parameters()
    for k in ("offset_pred", "offset_gt", "max_val", "ri_factor"):
        assert np.isfinite(params[k]), f"{k}={params[k]} not finite"


# --- Convenience function --------------------------------------------------


def test_convenience_2d_returns_float():
    """2-D input returns a single float."""
    gt = np.random.default_rng(0).random((_H, _W)).astype(np.float32)
    pred = gt + 0.05 * np.random.default_rng(1).standard_normal(gt.shape).astype(
        np.float32
    )
    out = micro_multiscale_structural_similarity(gt, pred)
    assert isinstance(out, float)


def test_convenience_3d_returns_list_of_floats():
    """3-D input returns a list of per-slice floats."""
    gt, pred = _seeded_data(n=4)
    out = micro_multiscale_structural_similarity(gt, pred)
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(x, float) for x in out)


def test_convenience_list_input_returns_list():
    """List input returns a list of per-slice floats."""
    gt_list = [
        np.random.default_rng(i).random((_H, _W)).astype(np.float32) for i in range(3)
    ]
    pred_list = [g * 1.2 for g in gt_list]
    out = micro_multiscale_structural_similarity(gt_list, pred_list)
    assert isinstance(out, list)
    assert len(out) == 3
