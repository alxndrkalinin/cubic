"""Tests for ``cubic.metrics.microssim.ri_factor``."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics.microssim.ri_factor import (
    get_ri_factor,
    _compute_S_mean,
    _compute_dS_mean,
    _flatten_elements,
    get_global_ri_factor,
)
from cubic.metrics.microssim.ssim_elements import (
    SSIMElements,
    compute_ssim_elements,
)

# -- Identity: gt == pred --------------------------------------------------


def test_identity_alpha_is_one() -> None:
    """When gt == pred, the optimal alpha is 1.0 to high precision.

    For identical inputs ``ux == uy``, ``vx == vy == vxy`` everywhere, so
    ``S(1) = 1`` exactly (modulo round-off) and ``dS/dalpha(1) = 0``.
    """
    rng = np.random.default_rng(0)
    gt = rng.random((32, 32)).astype(np.float64)
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, gt, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )

    alpha = get_ri_factor(elements)
    assert abs(alpha - 1.0) < 1e-6


# -- Synthetic linear scaling ----------------------------------------------


def test_linear_scaling_optimum_exists() -> None:
    """A uniformly scaled prediction yields an optimum that beats alpha=1.

    With ``pred = scale * gt`` and ``scale != 1``, the RI factor pulls the
    prediction back toward the ground truth so ``mean(S(alpha*)) >=
    mean(S(1))``. We do not pin the recovered ``alpha`` to ``1 / scale``
    exactly — the optimum of mean-SSIM is not the same as least-squares
    inversion of the scaling — but we do verify the ascent invariant and
    that ``alpha*`` lies in a sensible (positive) range.
    """
    rng = np.random.default_rng(1)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = 2.5 * gt
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha_star = get_ri_factor(elements)
    flat = _flatten_elements(elements)
    s_star = _compute_S_mean(alpha_star, flat)
    s_one = _compute_S_mean(1.0, flat)
    # Ascent (with slack); positive alpha.
    assert alpha_star > 0.0
    assert s_star >= s_one - 1e-12


# -- Pathological inputs ----------------------------------------------------


def test_constant_gt_noisy_pred_no_silent_nan() -> None:
    """Constant gt + non-constant pred is handled safely.

    With gt constant, ``ux*uy != 0`` but ``vx == vxy == 0`` (or nearly so);
    ``vy`` is small but positive. Empirically the derivative still changes
    sign inside the default bracket window for this configuration, so we
    accept either a finite positive ``alpha*`` (with the ascent invariant
    satisfied) or a clean ``RuntimeError`` from bracket failure — never a
    silent NaN / inf.
    """
    rng = np.random.default_rng(2)
    gt = np.full((16, 16), 5.0, dtype=np.float64)
    pred = gt + 0.1 * rng.standard_normal((16, 16))
    # data_range = gt.max() - gt.min() == 0; use the pred range so
    # compute_ssim_elements accepts it.
    dr = float(pred.max() - pred.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    try:
        alpha = get_ri_factor(elements)
    except RuntimeError:
        return  # bracket failure is an acceptable outcome
    assert np.isfinite(alpha)
    assert alpha > 0.0


def test_extreme_scaling_bracket_failure() -> None:
    """Tight ``alpha_min`` raises cleanly when the optimum falls below it.

    ``pred = 1e5 * gt`` puts the optimum at alpha ~ 1e-5. With an explicit
    ``alpha_min = 1e-3`` (the pre-PR default), the bracket cannot reach
    the optimum and raises a clean ``RuntimeError`` instead of converging
    to a nonsensical value or NaN.
    """
    rng = np.random.default_rng(10)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = 1e5 * gt
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    with pytest.raises(RuntimeError, match="RI factor failed to bracket"):
        get_ri_factor(elements, alpha_min=1e-3)


def test_all_zero_pred_no_silent_nan() -> None:
    """All-zero prediction: either a sensible result or a clean RuntimeError.

    What we do NOT tolerate is silent NaN / inf / -1 / 0 returns. With
    ``pred = 0`` we have ``uy = vy = vxy = 0`` everywhere, so
    ``dS/dalpha`` is identically zero — the solver returns alpha=1 from
    the early-exit branch (``|f(1)| < 1e-14``).
    """
    rng = np.random.default_rng(3)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = np.zeros_like(gt)
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    try:
        alpha = get_ri_factor(elements)
    except RuntimeError:
        return  # acceptable outcome
    # If we got an alpha, it must be finite. The optimum need not be > 0
    # in any meaningful sense for this degenerate input, but it must not be
    # NaN/inf/negative.
    assert np.isfinite(alpha)
    assert alpha > 0.0


# -- SciPy parity -----------------------------------------------------------


def test_scipy_parity() -> None:
    """Bisection result matches ``scipy.optimize.minimize`` to 1e-5.

    The objective is smooth and unimodal in alpha on ``(0, +inf)``, so BFGS
    from ``x0=[1.0]`` converges to the same optimum. We verify our root of
    ``dS/dalpha`` matches the BFGS argmin of ``-S(alpha)``.
    """
    pytest.importorskip("scipy")
    from scipy.optimize import minimize

    rng = np.random.default_rng(4)
    gt = rng.random((16, 16)).astype(np.float64)
    pred = gt + 0.1 * rng.standard_normal((16, 16))
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )

    flat = _flatten_elements(elements)
    alpha_bisect = get_ri_factor(elements)

    def neg_s(alpha_arr: np.ndarray) -> float:
        return -_compute_S_mean(float(alpha_arr[0]), flat)

    res = minimize(neg_s, x0=np.array([1.0]))
    alpha_bfgs = float(res.x[0])
    assert abs(alpha_bisect - alpha_bfgs) < 1e-5


# -- Convergence speed ------------------------------------------------------


def test_bisection_terminates_quickly() -> None:
    """Bisection converges in fewer than 100 iterations on a normal input.

    We replicate the inner loop manually so we can count iterations.
    """
    from cubic.metrics.microssim.ri_factor import _bracket_root  # local

    rng = np.random.default_rng(5)
    gt = rng.random((24, 24)).astype(np.float64)
    pred = gt + 0.05 * rng.standard_normal((24, 24))
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    flat = _flatten_elements(elements)
    f1 = _compute_dS_mean(1.0, flat)
    # The setup above should yield a non-trivial root — skip if the input
    # accidentally already satisfies |f(1)| ~ 0.
    if abs(f1) < 1e-14:
        pytest.skip("f(1) already at machine zero; iter count not meaningful")
    lo, hi, f_lo, f_hi = _bracket_root(flat, f1, alpha_min=1e-3, alpha_max=1e3)

    iters = 0
    mid = 0.5 * (lo + hi)
    f_mid = _compute_dS_mean(mid, flat)
    while iters < 200:
        if abs(f_mid) < 1e-10 and abs(hi - lo) < 1e-8:
            break
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        mid = 0.5 * (lo + hi)
        f_mid = _compute_dS_mean(mid, flat)
        iters += 1

    assert iters < 100, f"bisection took {iters} iterations"


# -- get_global_ri_factor ---------------------------------------------------


def test_global_ri_factor_3d_stack() -> None:
    """``get_global_ri_factor`` on (3, 32, 32) is finite, positive, and matches manual pool."""
    rng = np.random.default_rng(6)
    gt = rng.random((3, 32, 32)).astype(np.float64)
    pred = gt + 0.1 * rng.standard_normal((3, 32, 32))

    alpha = get_global_ri_factor(gt, pred)
    assert np.isfinite(alpha)
    assert alpha > 0.0

    # Manual pool — must match exactly (same code path internally).
    ux_l: list[np.ndarray] = []
    uy_l: list[np.ndarray] = []
    vxy_l: list[np.ndarray] = []
    vx_l: list[np.ndarray] = []
    vy_l: list[np.ndarray] = []
    C1_last = 0.0
    C2_last = 0.0
    for i in range(gt.shape[0]):
        dr = float(gt[i].max() - gt[i].min())
        e = compute_ssim_elements(
            gt[i],
            pred[i],
            data_range=dr,
            gaussian_weights=False,
            win_size=7,
            crop=True,
        )
        ux_l.append(e.ux.ravel())
        uy_l.append(e.uy.ravel())
        vxy_l.append(e.vxy.ravel())
        vx_l.append(e.vx.ravel())
        vy_l.append(e.vy.ravel())
        C1_last, C2_last = e.C1, e.C2
    pooled = SSIMElements(
        ux=np.concatenate(ux_l),
        uy=np.concatenate(uy_l),
        vxy=np.concatenate(vxy_l),
        vx=np.concatenate(vx_l),
        vy=np.concatenate(vy_l),
        C1=C1_last,
        C2=C2_last,
    )
    alpha_manual = get_ri_factor(pooled)
    assert abs(alpha - alpha_manual) < 1e-12


def test_global_ri_factor_2d_input_treated_as_n1() -> None:
    """2-D input to ``get_global_ri_factor`` matches the direct single-image call."""
    rng = np.random.default_rng(7)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt + 0.1 * rng.standard_normal((32, 32))

    alpha_global = get_global_ri_factor(gt, pred)

    dr = float(gt.max() - gt.min())
    e = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha_direct = get_ri_factor(e)
    assert abs(alpha_global - alpha_direct) < 1e-12


def test_global_ri_factor_shape_mismatch_raises() -> None:
    """Mismatched shapes raise ValueError."""
    gt = np.zeros((3, 32, 32))
    pred = np.zeros((3, 32, 33))
    with pytest.raises(ValueError, match="same shape"):
        get_global_ri_factor(gt, pred)


def test_global_ri_factor_rejects_ndim_1() -> None:
    """1-D input raises ValueError."""
    gt = np.zeros(32)
    pred = np.zeros(32)
    with pytest.raises(ValueError, match="ndim"):
        get_global_ri_factor(gt, pred)


def test_global_ri_factor_rejects_ndim_4() -> None:
    """4-D input raises ValueError."""
    gt = np.zeros((2, 3, 32, 32))
    pred = np.zeros((2, 3, 32, 32))
    with pytest.raises(ValueError, match="ndim"):
        get_global_ri_factor(gt, pred)


# -- alpha_max kwarg --------------------------------------------------------


def test_alpha_max_below_one_raises() -> None:
    """``alpha_max <= 1`` is degenerate (bracket starts at 1) — must raise."""
    rng = np.random.default_rng(11)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt.copy()
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    with pytest.raises(ValueError, match="alpha_max"):
        get_ri_factor(elements, alpha_max=1.0)
    with pytest.raises(ValueError, match="alpha_max"):
        get_ri_factor(elements, alpha_max=0.5)


def test_alpha_max_default_lifts_right_bracket() -> None:
    """Heavy down-scaling (alpha* ~ 1e4) now succeeds at the default cap.

    With the old ``alpha_max = 1e3`` default this would have raised
    ``RuntimeError("RI factor failed to bracket on the right ...")``;
    the bumped default (``1e6``) keeps the fit working on this case.
    """
    rng = np.random.default_rng(12)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt * 1e-4  # optimum sits near alpha ~ 1e4
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha = get_ri_factor(elements)  # default alpha_max=1e6
    assert np.isfinite(alpha)
    assert alpha > 1e3  # would have hit the old 1e3 cap


def test_alpha_max_below_optimum_raises() -> None:
    """Explicit low cap forces a clean bracket failure (no silent clip).

    Same heavy down-scaling input as the previous test; passing
    ``alpha_max=1e3`` reproduces the failure mode of the old default.
    """
    rng = np.random.default_rng(12)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt * 1e-4
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    with pytest.raises(RuntimeError, match="failed to bracket on the right"):
        get_ri_factor(elements, alpha_max=1e3)


def test_global_ri_factor_forwards_alpha_max() -> None:
    """``get_global_ri_factor`` forwards ``alpha_max`` to ``get_ri_factor``.

    Same heavy down-scaled stack used for the unit test; explicit low cap
    must propagate and raise.
    """
    rng = np.random.default_rng(12)
    gt = rng.random((3, 32, 32)).astype(np.float64)
    pred = gt * 1e-4
    with pytest.raises(RuntimeError, match="failed to bracket on the right"):
        get_global_ri_factor(gt, pred, alpha_max=1e3)
    # Default cap recovers a finite positive alpha.
    alpha = get_global_ri_factor(gt, pred)
    assert np.isfinite(alpha) and alpha > 1e3


def test_global_ri_factor_rejects_bad_alpha_max_before_per_slice_pass() -> None:
    """``get_global_ri_factor`` validates ``alpha_max`` before the element loop.

    A bad ``alpha_max`` should surface immediately, not after N expensive
    ``compute_ssim_elements`` calls. Use a stack large enough that a
    deferred check would be observable in wallclock; here we just confirm
    the error type / message matches the eager-validation contract.
    """
    gt = np.zeros((3, 32, 32))
    pred = np.zeros((3, 32, 32))
    for bad in (0.5, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha_max"):
            get_global_ri_factor(gt, pred, alpha_max=bad)


# -- alpha_min kwarg (mirror of alpha_max) ---------------------------------


def test_alpha_min_outside_unit_interval_raises() -> None:
    """``alpha_min`` outside ``(0, 1)`` is rejected up-front."""
    rng = np.random.default_rng(20)
    gt = rng.random((32, 32)).astype(np.float64)
    elements = compute_ssim_elements(
        gt,
        gt,
        data_range=float(gt.max() - gt.min()),
        gaussian_weights=False,
        win_size=7,
        crop=True,
    )
    for bad in (0.0, 1.0, -1.0, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha_min"):
            get_ri_factor(elements, alpha_min=bad)


def test_alpha_min_default_lifts_left_bracket() -> None:
    """Heavy up-scaling (alpha* ~ 1e-5) now succeeds at the default floor.

    With the old ``_ALPHA_MIN = 1e-3`` this would have raised
    ``RuntimeError("RI factor failed to bracket on the left ...")``;
    the bumped default (``1e-6``) keeps the fit working on this case.
    """
    rng = np.random.default_rng(21)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt * 1e5  # optimum sits near alpha ~ 1e-5
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha = get_ri_factor(elements)  # default alpha_min=1e-6
    assert np.isfinite(alpha)
    assert alpha < 1e-3  # would have hit the old 1e-3 floor


def test_alpha_min_above_optimum_raises() -> None:
    """Explicit high floor forces a clean bracket failure (no silent clip)."""
    rng = np.random.default_rng(21)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt * 1e5
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    with pytest.raises(RuntimeError, match="failed to bracket on the left"):
        get_ri_factor(elements, alpha_min=1e-3)


def test_global_ri_factor_forwards_alpha_min() -> None:
    """``get_global_ri_factor`` forwards ``alpha_min`` to ``get_ri_factor``."""
    rng = np.random.default_rng(21)
    gt = rng.random((3, 32, 32)).astype(np.float64)
    pred = gt * 1e5
    with pytest.raises(RuntimeError, match="failed to bracket on the left"):
        get_global_ri_factor(gt, pred, alpha_min=1e-3)
    # Default floor recovers a finite positive alpha.
    alpha = get_global_ri_factor(gt, pred)
    assert np.isfinite(alpha) and alpha < 1e-3


def test_global_ri_factor_rejects_bad_alpha_min_before_per_slice_pass() -> None:
    """``get_global_ri_factor`` validates ``alpha_min`` before the element loop."""
    gt = np.zeros((3, 32, 32))
    pred = np.zeros((3, 32, 32))
    for bad in (0.0, 1.0, -0.5, 2.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="alpha_min"):
            get_global_ri_factor(gt, pred, alpha_min=bad)
