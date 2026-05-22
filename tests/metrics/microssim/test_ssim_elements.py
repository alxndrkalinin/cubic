"""Tests for ``cubic.metrics.microssim.ssim_elements``."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics.microssim.ssim_elements import (
    SSIMElements,
    compute_ssim_elements,
)

# -- SSIMElements dataclass ---------------------------------------------------


def test_field_order_matches_upstream() -> None:
    """Positional construction must follow upstream field order.

    Upstream field order (``ssim_utils.py:23-42``):
    ``ux, uy, vxy, vx, vy, C1, C2``.
    """
    ux = np.full((4, 4), 1.0)
    uy = np.full((4, 4), 2.0)
    vxy = np.full((4, 4), 3.0)
    vx = np.full((4, 4), 4.0)
    vy = np.full((4, 4), 5.0)
    e_pos = SSIMElements(ux, uy, vxy, vx, vy, 0.1, 0.2)
    e_kw = SSIMElements(ux=ux, uy=uy, vxy=vxy, vx=vx, vy=vy, C1=0.1, C2=0.2)

    assert np.array_equal(e_pos.ux, e_kw.ux)
    assert np.array_equal(e_pos.uy, e_kw.uy)
    assert np.array_equal(e_pos.vxy, e_kw.vxy)
    assert np.array_equal(e_pos.vx, e_kw.vx)
    assert np.array_equal(e_pos.vy, e_kw.vy)
    assert e_pos.C1 == e_kw.C1
    assert e_pos.C2 == e_kw.C2

    # All five arrays must be distinct (catches an accidental swap such as
    # vx <-> vxy).
    assert e_pos.ux[0, 0] == 1.0
    assert e_pos.uy[0, 0] == 2.0
    assert e_pos.vxy[0, 0] == 3.0
    assert e_pos.vx[0, 0] == 4.0
    assert e_pos.vy[0, 0] == 5.0


def test_dataclass_is_frozen() -> None:
    """SSIMElements is frozen, so attribute assignment must raise."""
    e = SSIMElements(
        ux=np.zeros((2, 2)),
        uy=np.zeros((2, 2)),
        vxy=np.zeros((2, 2)),
        vx=np.zeros((2, 2)),
        vy=np.zeros((2, 2)),
        C1=0.0,
        C2=0.0,
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        e.C1 = 1.0  # type: ignore[misc]


def test_no_win_size_field() -> None:
    """SSIMElements must not carry a ``win_size`` field (matches upstream)."""
    fields = set(SSIMElements.__dataclass_fields__.keys())
    assert "win_size" not in fields
    assert fields == {"ux", "uy", "vxy", "vx", "vy", "C1", "C2"}


# -- Identical-input behaviour ------------------------------------------------


def test_identical_constant_input_uniform() -> None:
    """A constant image has ux == image1 everywhere (filter is mean-preserving)."""
    img = np.full((16, 16), 0.7, dtype=np.float64)
    e = compute_ssim_elements(
        img, img, data_range=1.0, gaussian_weights=False, win_size=7, crop=True
    )
    # Cropped shape: 16 - 2*((7-1)//2) = 16 - 6 = 10
    assert e.ux.shape == (10, 10)
    np.testing.assert_allclose(e.ux, 0.7, atol=1e-12)
    np.testing.assert_allclose(e.uy, 0.7, atol=1e-12)
    # vx / vy / vxy should be ~0 for a constant image.
    np.testing.assert_allclose(e.vx, 0.0, atol=1e-12)
    np.testing.assert_allclose(e.vy, 0.0, atol=1e-12)
    np.testing.assert_allclose(e.vxy, 0.0, atol=1e-12)


def test_identical_constant_input_gaussian() -> None:
    """A constant image preserves its value under Gaussian filtering too."""
    img = np.full((32, 32), 0.42, dtype=np.float64)
    e = compute_ssim_elements(
        img,
        img,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        truncate=3.5,
        crop=True,
    )
    # win_size = 2*int(3.5*1.5 + 0.5) + 1 = 2*5 + 1 = 11; pad = 5
    # cropped shape: 32 - 10 = 22
    assert e.ux.shape == (22, 22)
    np.testing.assert_allclose(e.ux, 0.42, atol=1e-12)
    np.testing.assert_allclose(e.uy, 0.42, atol=1e-12)


# -- Shape and crop behaviour ------------------------------------------------


def test_shape_uniform_cropped_vs_uncropped() -> None:
    """Uniform-filter mode: crop trims (win_size-1)//2 from each spatial edge."""
    rng = np.random.default_rng(0)
    img1 = rng.random((20, 24)).astype(np.float64)
    img2 = img1 + 0.01 * rng.standard_normal((20, 24))

    e_crop = compute_ssim_elements(
        img1, img2, data_range=1.0, gaussian_weights=False, win_size=7, crop=True
    )
    e_full = compute_ssim_elements(
        img1, img2, data_range=1.0, gaussian_weights=False, win_size=7, crop=False
    )
    assert e_full.ux.shape == (20, 24)
    assert e_crop.ux.shape == (14, 18)


def test_shape_gaussian_cropped() -> None:
    """Gaussian-filter mode: derived win_size=11 gives pad=5."""
    rng = np.random.default_rng(1)
    img1 = rng.random((32, 32)).astype(np.float32)
    img2 = img1 + 0.05 * rng.standard_normal((32, 32)).astype(np.float32)
    e = compute_ssim_elements(
        img1,
        img2,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        truncate=3.5,
        crop=True,
    )
    assert e.ux.shape == (22, 22)
    assert e.uy.shape == (22, 22)
    assert e.vxy.shape == (22, 22)
    assert e.vx.shape == (22, 22)
    assert e.vy.shape == (22, 22)


# -- Batched (N, H, W) input -------------------------------------------------


def test_batched_uniform_matches_per_slice() -> None:
    """Slice-i of a batched call must equal a single-slice call on slice i."""
    rng = np.random.default_rng(2)
    batch = rng.random((3, 16, 16)).astype(np.float64)
    pred = batch + 0.1 * rng.standard_normal((3, 16, 16))

    # data_range matched across the two paths so C1 / C2 are identical.
    dr = 1.0
    e_batch = compute_ssim_elements(
        batch, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    assert e_batch.ux.shape == (3, 10, 10)

    for i in range(3):
        e_i = compute_ssim_elements(
            batch[i],
            pred[i],
            data_range=dr,
            gaussian_weights=False,
            win_size=7,
            crop=True,
        )
        np.testing.assert_allclose(e_batch.ux[i], e_i.ux, atol=1e-12)
        np.testing.assert_allclose(e_batch.uy[i], e_i.uy, atol=1e-12)
        np.testing.assert_allclose(e_batch.vxy[i], e_i.vxy, atol=1e-12)
        np.testing.assert_allclose(e_batch.vx[i], e_i.vx, atol=1e-12)
        np.testing.assert_allclose(e_batch.vy[i], e_i.vy, atol=1e-12)


def test_batched_gaussian_matches_per_slice() -> None:
    """Same independence check for Gaussian mode."""
    rng = np.random.default_rng(3)
    batch = rng.random((3, 32, 32)).astype(np.float64)
    pred = batch + 0.05 * rng.standard_normal((3, 32, 32))
    dr = 1.0
    e_batch = compute_ssim_elements(
        batch,
        pred,
        data_range=dr,
        gaussian_weights=True,
        sigma=1.5,
        truncate=3.5,
        crop=True,
    )
    for i in range(3):
        e_i = compute_ssim_elements(
            batch[i],
            pred[i],
            data_range=dr,
            gaussian_weights=True,
            sigma=1.5,
            truncate=3.5,
            crop=True,
        )
        # Tiny float drift is acceptable here — different array shapes hit
        # different vectorization paths.
        np.testing.assert_allclose(e_batch.ux[i], e_i.ux, atol=1e-10)
        np.testing.assert_allclose(e_batch.uy[i], e_i.uy, atol=1e-10)
        np.testing.assert_allclose(e_batch.vxy[i], e_i.vxy, atol=1e-10)
        np.testing.assert_allclose(e_batch.vx[i], e_i.vx, atol=1e-10)
        np.testing.assert_allclose(e_batch.vy[i], e_i.vy, atol=1e-10)


# -- Dtype promotion ---------------------------------------------------------


def test_dtype_promotion_int_to_float64() -> None:
    """Integer input promotes to float64 (mirrors ``_supported_float_type``)."""
    rng = np.random.default_rng(4)
    img1 = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
    img2 = rng.integers(0, 255, size=(16, 16), dtype=np.uint8)
    e = compute_ssim_elements(
        img1, img2, data_range=255.0, gaussian_weights=False, win_size=7
    )
    assert e.ux.dtype == np.float64
    assert e.uy.dtype == np.float64
    assert e.vx.dtype == np.float64
    assert e.vy.dtype == np.float64
    assert e.vxy.dtype == np.float64


def test_dtype_promotion_bool_to_float64() -> None:
    """Boolean input promotes to float64."""
    rng = np.random.default_rng(5)
    img1 = rng.integers(0, 2, size=(16, 16)).astype(bool)
    img2 = rng.integers(0, 2, size=(16, 16)).astype(bool)
    e = compute_ssim_elements(
        img1, img2, data_range=1.0, gaussian_weights=False, win_size=7
    )
    assert e.ux.dtype == np.float64


def test_dtype_promotion_float16_to_float32() -> None:
    """float16 input promotes to float32."""
    rng = np.random.default_rng(6)
    img1 = rng.random((16, 16)).astype(np.float16)
    img2 = img1 + np.float16(0.01)
    e = compute_ssim_elements(
        img1, img2, data_range=1.0, gaussian_weights=False, win_size=7
    )
    assert e.ux.dtype == np.float32


def test_dtype_preserved_float32_float64() -> None:
    """float32 and float64 are preserved."""
    rng = np.random.default_rng(7)
    img1_32 = rng.random((16, 16)).astype(np.float32)
    img2_32 = img1_32 + 0.01
    e32 = compute_ssim_elements(
        img1_32, img2_32, data_range=1.0, gaussian_weights=False, win_size=7
    )
    assert e32.ux.dtype == np.float32

    img1_64 = rng.random((16, 16)).astype(np.float64)
    img2_64 = img1_64 + 0.01
    e64 = compute_ssim_elements(
        img1_64, img2_64, data_range=1.0, gaussian_weights=False, win_size=7
    )
    assert e64.ux.dtype == np.float64


# -- ValueError fail-fast paths ---------------------------------------------


def test_raises_on_shape_mismatch() -> None:
    """Shape mismatch raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 17))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_ssim_elements(a, b, data_range=1.0)


def test_raises_on_ndim_1() -> None:
    """1-D input raises ValueError."""
    a = np.zeros(16)
    b = np.zeros(16)
    with pytest.raises(ValueError, match="ndim"):
        compute_ssim_elements(a, b, data_range=1.0)


def test_raises_on_ndim_4() -> None:
    """4-D input raises ValueError."""
    a = np.zeros((2, 3, 16, 16))
    b = np.zeros((2, 3, 16, 16))
    with pytest.raises(ValueError, match="ndim"):
        compute_ssim_elements(a, b, data_range=1.0)


def test_raises_on_even_win_size() -> None:
    """Even ``win_size`` raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="win_size must be odd"):
        compute_ssim_elements(a, b, data_range=1.0, gaussian_weights=False, win_size=6)


def test_raises_on_zero_win_size() -> None:
    """``win_size=0`` raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="win_size must be odd"):
        compute_ssim_elements(a, b, data_range=1.0, gaussian_weights=False, win_size=0)


def test_raises_on_win_size_larger_than_image() -> None:
    """``win_size`` exceeding the spatial extent raises ValueError."""
    a = np.zeros((8, 8))
    b = np.zeros((8, 8))
    with pytest.raises(ValueError, match="Spatial dims must be"):
        compute_ssim_elements(a, b, data_range=1.0, gaussian_weights=False, win_size=11)


def test_raises_on_nan_data_range() -> None:
    """Non-finite ``data_range`` raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="data_range"):
        compute_ssim_elements(a, b, data_range=float("nan"))


def test_raises_on_negative_data_range() -> None:
    """Negative ``data_range`` raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="data_range"):
        compute_ssim_elements(a, b, data_range=-1.0)


def test_raises_on_zero_data_range() -> None:
    """Zero ``data_range`` raises ValueError (would zero C1 / C2)."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="data_range"):
        compute_ssim_elements(a, b, data_range=0.0)


def test_raises_on_inf_data_range() -> None:
    """Infinite ``data_range`` raises ValueError."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError, match="data_range"):
        compute_ssim_elements(a, b, data_range=float("inf"))


# -- Sample vs population covariance ----------------------------------------


def test_use_sample_covariance_scales_variance() -> None:
    """``use_sample_covariance=False`` produces strictly smaller variances.

    The sample covariance multiplies by ``NP / (NP - 1) > 1``; turning it
    off divides through by that factor, so every per-pixel variance
    estimate shrinks by the same constant.
    """
    rng = np.random.default_rng(8)
    img1 = rng.random((24, 24)).astype(np.float64)
    img2 = img1 + 0.1 * rng.standard_normal((24, 24))

    win_size = 7
    e_sample = compute_ssim_elements(
        img1,
        img2,
        data_range=1.0,
        gaussian_weights=False,
        win_size=win_size,
        use_sample_covariance=True,
    )
    e_pop = compute_ssim_elements(
        img1,
        img2,
        data_range=1.0,
        gaussian_weights=False,
        win_size=win_size,
        use_sample_covariance=False,
    )
    NP = win_size * win_size
    expected_ratio = NP / (NP - 1)
    # Sample = pop * ratio, both per-pixel; vx is generally positive here.
    mask = np.abs(e_pop.vx) > 1e-10
    np.testing.assert_allclose(
        e_sample.vx[mask] / e_pop.vx[mask], expected_ratio, atol=1e-12
    )
    np.testing.assert_allclose(
        e_sample.vy[mask] / e_pop.vy[mask], expected_ratio, atol=1e-12
    )
    # And — at least somewhere — the population variance is strictly smaller
    # in magnitude than the sample variance.
    assert np.all(np.abs(e_pop.vx[mask]) < np.abs(e_sample.vx[mask]))


# -- C1 / C2 derivation ------------------------------------------------------


def test_C1_C2_formulas() -> None:
    """``C1 = (K1 * data_range)**2``, ``C2 = (K2 * data_range)**2``."""
    rng = np.random.default_rng(9)
    img1 = rng.random((16, 16)).astype(np.float64)
    img2 = img1.copy()
    e = compute_ssim_elements(
        img1,
        img2,
        data_range=2.5,
        gaussian_weights=False,
        win_size=7,
        K1=0.01,
        K2=0.03,
    )
    assert e.C1 == pytest.approx((0.01 * 2.5) ** 2)
    assert e.C2 == pytest.approx((0.03 * 2.5) ** 2)


# -- No-clamp invariant ------------------------------------------------------


def test_no_variance_clamp_allows_small_negative() -> None:
    """``vx`` is not clamped — float round-off can yield negative values.

    Construct a float32 input with a large mean and small variation around
    it so that ``uniform_filter(image*image)`` suffers catastrophic
    cancellation against ``ux*ux``: ``uxx - ux*ux`` drifts below zero in
    many pixels. Upstream microssim leaves this signed (``ssim_utils.py
    :235-237``); we must too.
    """
    rng = np.random.default_rng(123)
    # Mean ~1e7, std ~100 in float32 — single-pixel `image*image` has only
    # ~7 digits of precision, while `ux*ux` accumulates differently → the
    # subtraction is genuinely signed.
    img = (1e7 + rng.standard_normal((64, 64)).astype(np.float32) * 100).astype(
        np.float32
    )
    e = compute_ssim_elements(
        img,
        img,
        data_range=1.0,
        gaussian_weights=False,
        win_size=7,
        crop=True,
    )
    # We do NOT assert vx >= 0; we explicitly verify it can go negative
    # without being clamped.
    assert float(e.vx.min()) < 0.0, (
        "Expected catastrophic-cancellation round-off to drive vx below "
        "zero; the no-clamp invariant requires we preserve the signed "
        "result. Got min(vx) = "
        f"{float(e.vx.min())}."
    )
    # vy mirrors vx for identical inputs.
    assert float(e.vy.min()) < 0.0


def test_no_clamp_invariant_against_explicit_clamp() -> None:
    """A clamped reference is everywhere >= our (signed) ``vx``.

    Independent check that we are returning the raw signed difference:
    ``np.maximum(vx, 0)`` differs from ``vx`` somewhere when the input
    triggers catastrophic cancellation.
    """
    rng = np.random.default_rng(456)
    img = (1e7 + rng.standard_normal((64, 64)).astype(np.float32) * 100).astype(
        np.float32
    )
    e = compute_ssim_elements(
        img,
        img,
        data_range=1.0,
        gaussian_weights=False,
        win_size=7,
        crop=True,
    )
    clamped = np.maximum(e.vx, 0.0)
    # Clamped values are >= our (possibly-signed) values everywhere…
    assert np.all(clamped >= e.vx)
    # …and strictly greater somewhere (proves we did NOT clamp internally).
    assert np.any(clamped > e.vx)


# -- check_same_device hook --------------------------------------------------


def test_check_same_device_passes_for_cpu_pair() -> None:
    """Two NumPy arrays must not trigger the device check."""
    a = np.zeros((16, 16))
    b = np.zeros((16, 16))
    # Should simply not raise.
    compute_ssim_elements(a, b, data_range=1.0, gaussian_weights=False, win_size=7)
