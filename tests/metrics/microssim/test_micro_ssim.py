"""Tests for MicroSSIM and the single-image convenience wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics import MicroSSIM, micro_structural_similarity


def _seeded_data(n: int = 4, h: int = 64, w: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return seeded exponential GT + scaled-and-noised prediction (float32)."""
    rng = np.random.default_rng(0)
    gt = rng.exponential(0.5, (n, h, w)).astype(np.float32) * 1000
    pred = gt * 1.3 + 0.05 * float(gt.max()) * rng.standard_normal(gt.shape).astype(
        np.float32
    )
    return gt, pred


# --- Construction ----------------------------------------------------------


def test_default_construction_is_uninitialized():
    """Default constructor leaves the instance uninitialized."""
    ms = MicroSSIM()
    assert ms._initialized is False


def test_direct_init_with_all_params_is_initialized():
    """Supplying all four params lets score() skip fit()."""
    ms = MicroSSIM(offset_gt=0.0, offset_pred=0.0, max_val=1.0, ri_factor=1.0)
    assert ms._initialized is True


def test_ri_factor_without_norm_params_raises():
    """ri_factor alone is rejected (matches upstream micro_ssim.py:270-279)."""
    with pytest.raises(ValueError, match="offset_pred, offset_gt and max_val"):
        MicroSSIM(ri_factor=1.0)


def test_ri_factor_with_partial_norm_params_raises():
    """Partial norm params + ri_factor is rejected."""
    with pytest.raises(ValueError, match="offset_pred, offset_gt and max_val"):
        MicroSSIM(offset_gt=0.0, ri_factor=1.0)


# --- fit() validation ------------------------------------------------------


def test_fit_mixed_types_raises():
    """Fit rejects gt/pred of different types."""
    gt = np.zeros((4, 64, 64), np.float32)
    pred = [np.zeros((64, 64), np.float32) for _ in range(4)]
    with pytest.raises(ValueError, match="same type"):
        MicroSSIM().fit(gt, pred)


def test_fit_unequal_list_lengths_raises():
    """Fit rejects mismatched list lengths."""
    gt = [np.zeros((64, 64), np.float32) for _ in range(3)]
    pred = [np.zeros((64, 64), np.float32) for _ in range(4)]
    with pytest.raises(ValueError, match="same length"):
        MicroSSIM().fit(gt, pred)


def test_fit_unequal_shapes_raises():
    """Fit rejects arrays of different shapes."""
    gt = np.zeros((4, 64, 64), np.float32)
    pred = np.zeros((4, 32, 32), np.float32)
    with pytest.raises(ValueError, match="same shape"):
        MicroSSIM().fit(gt, pred)


def test_fit_ndim_1_raises():
    """Fit rejects 1-D arrays."""
    gt = np.zeros(64, np.float32)
    pred = np.zeros(64, np.float32)
    with pytest.raises(ValueError, match="2D or 3D"):
        MicroSSIM().fit(gt, pred)


def test_fit_ndim_4_raises():
    """Fit rejects 4-D arrays."""
    gt = np.zeros((2, 2, 64, 64), np.float32)
    pred = np.zeros((2, 2, 64, 64), np.float32)
    with pytest.raises(ValueError, match="2D or 3D"):
        MicroSSIM().fit(gt, pred)


# --- score() validation ----------------------------------------------------


def test_score_before_fit_raises():
    """Score before fit raises ValueError."""
    gt = np.zeros((64, 64), np.float32)
    with pytest.raises(ValueError, match="call `fit\\(\\)` first"):
        MicroSSIM().score(gt, gt)


def test_score_3d_raises():
    """Score rejects 3-D inputs (upstream is 2-D only)."""
    gt, pred = _seeded_data()
    ms = MicroSSIM().fit(gt, pred)
    with pytest.raises(ValueError, match="Only 2D"):
        ms.score(gt, pred)


def test_score_shape_mismatch_raises():
    """Score rejects mismatched 2-D shapes."""
    gt, pred = _seeded_data()
    ms = MicroSSIM().fit(gt, pred)
    with pytest.raises(ValueError, match="same shape"):
        ms.score(gt[0], pred[0, :32])


def test_fit_returns_self():
    """Fit returns self to enable chaining."""
    gt, pred = _seeded_data()
    ms = MicroSSIM()
    assert ms.fit(gt, pred) is ms
    assert ms._initialized is True


# --- Numerical identity ----------------------------------------------------


def test_identity_score_is_one():
    """SSIM(x, x) is 1 (within float roundoff) and ri_factor is ~1."""
    rng = np.random.default_rng(0)
    img = rng.random((4, 128, 128)).astype(np.float32)
    ms = MicroSSIM().fit(img, img)
    params = ms.get_parameters()
    assert abs(params["ri_factor"] - 1.0) < 1e-3
    score = ms.score(img[0], img[0])
    assert score > 1.0 - 1e-6, f"score={score}"


def test_scaled_input_invariance():
    """Uniform 3x intensity scaling leaves the score very near 1."""
    rng = np.random.default_rng(0)
    gt = rng.random((4, 128, 128)).astype(np.float32)
    pred = gt * 3.0
    ms = MicroSSIM().fit(gt, pred)
    score = ms.score(gt[0], pred[0])
    assert score > 0.99, f"score={score}"


# --- Statelessness ---------------------------------------------------------


def test_score_is_stateless_across_calls():
    """Repeated and reversed score() calls return identical values."""
    gt, pred = _seeded_data(n=6, h=64, w=64)
    ms = MicroSSIM().fit(gt, pred)
    forward = [ms.score(gt[i], pred[i]) for i in range(6)]
    reverse = [ms.score(gt[i], pred[i]) for i in reversed(range(6))]
    repeat = [ms.score(gt[0], pred[0]) for _ in range(5)]
    np.testing.assert_allclose(forward, list(reversed(reverse)), atol=1e-7)
    assert max(repeat) - min(repeat) < 1e-7


# --- get_parameters --------------------------------------------------------


def test_get_parameters_keys_match_upstream():
    """get_parameters returns the upstream key set verbatim."""
    gt, pred = _seeded_data()
    ms = MicroSSIM().fit(gt, pred)
    params = ms.get_parameters()
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
    ms = MicroSSIM().fit(gt, pred)
    params = ms.get_parameters()
    for k in ("offset_pred", "offset_gt", "max_val", "ri_factor"):
        assert np.isfinite(params[k]), f"{k}={params[k]} not finite"


def test_get_parameters_bg_percentile_passes_through():
    """Constructor bg_percentile is preserved in get_parameters output."""
    gt, pred = _seeded_data()
    ms = MicroSSIM(bg_percentile=7).fit(gt, pred)
    assert ms.get_parameters()["bg_percentile"] == 7


# --- kwargs forwarding -----------------------------------------------------


def test_score_kwargs_forwarded_to_elements():
    """Explicit K1 override changes the score (kwargs reach compute_ssim_elements)."""
    gt, pred = _seeded_data()
    ms = MicroSSIM().fit(gt, pred)
    default_score = ms.score(gt[0], pred[0])
    bigger_K1_score = ms.score(gt[0], pred[0], K1=0.5)
    assert default_score != bigger_K1_score


# --- alpha_max kwarg -------------------------------------------------------


def test_alpha_max_constructor_propagates_to_fit():
    """Low ``alpha_max`` on the constructor surfaces as a fit-time RuntimeError.

    Heavy down-scaled input (alpha* well past 1e3); the bumped default
    ``alpha_max=1e6`` would handle this, but an explicit low cap must
    fail loudly instead of silently clipping.
    """
    rng = np.random.default_rng(12)
    gt = rng.random((3, 64, 64)).astype(np.float64)
    pred = gt * 1e-4
    with pytest.raises(RuntimeError, match="failed to bracket on the right"):
        MicroSSIM(alpha_max=1e3).fit(gt, pred)


# --- pinned ri_factor (external calibration) -------------------------------


def test_score_with_pinned_ri_factor_skips_fit():
    """Constructor-supplied params + ri_factor enable scoring without fit().

    Use case: load a per-(model, organelle) RI factor calibrated once
    upstream and reuse it across all evaluation calls without re-fitting.
    """
    gt, pred = _seeded_data()
    fitted = MicroSSIM().fit(gt, pred)
    params = fitted.get_parameters()

    pinned = MicroSSIM(
        offset_gt=params["offset_gt"],
        offset_pred=params["offset_pred"],
        max_val=params["max_val"],
        ri_factor=params["ri_factor"],
    )
    # No fit() call — score works immediately.
    assert pinned._initialized
    score_fitted = fitted.score(gt[0], pred[0])
    score_pinned = pinned.score(gt[0], pred[0])
    assert abs(score_fitted - score_pinned) < 1e-12


# --- Convenience function --------------------------------------------------


def test_convenience_2d_returns_float():
    """2-D input returns a single float."""
    gt = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    pred = gt + 0.05 * np.random.default_rng(1).standard_normal(gt.shape).astype(
        np.float32
    )
    out = micro_structural_similarity(gt, pred)
    assert isinstance(out, float)


def test_convenience_3d_returns_list_of_floats():
    """3-D input returns a list of per-slice floats."""
    gt, pred = _seeded_data(n=4)
    out = micro_structural_similarity(gt, pred)
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(x, float) for x in out)


def test_convenience_list_input_returns_list():
    """List input returns a list of per-slice floats."""
    gt_list = [
        np.random.default_rng(i).random((64, 64)).astype(np.float32) for i in range(3)
    ]
    pred_list = [g * 1.2 for g in gt_list]
    out = micro_structural_similarity(gt_list, pred_list)
    assert isinstance(out, list)
    assert len(out) == 3


# --- GPU dispatch ----------------------------------------------------------


def test_xp_dispatch():
    """End-to-end fit+score on CuPy input matches the NumPy result.

    Exercises MicroSSIM's full pipeline (compute_norm_parameters,
    normalize_min_max, compute_ssim_elements, get_global_ri_factor,
    score) through cubic's device-agnostic dispatch on a CuPy array.
    Auto-skips on CPU-only hosts via ``pytest.importorskip``.
    """
    cp = pytest.importorskip("cupy")
    gt, pred = _seeded_data()
    score_np = MicroSSIM().fit(gt, pred).score(gt[0], pred[0])

    gt_cp, pred_cp = cp.asarray(gt), cp.asarray(pred)
    ms_cp = MicroSSIM().fit(gt_cp, pred_cp)
    score_cp = ms_cp.score(gt_cp[0], pred_cp[0])

    assert abs(float(score_np) - float(score_cp)) < 1e-4
