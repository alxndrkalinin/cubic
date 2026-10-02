"""Tests for ``cubic.metrics.microssim.ri_factor``."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.cuda import ascupy
from cubic.metrics.microssim import ri_factor as ri
from cubic.metrics.microssim.ri_factor import (
    get_ri_factor,
    _compute_S_mean,
    _compute_dS_mean,
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
    s_star = _compute_S_mean(alpha_star, elements)
    s_one = _compute_S_mean(1.0, elements)
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

    alpha_bisect = get_ri_factor(elements)

    def neg_s(alpha_arr: np.ndarray) -> float:
        return -_compute_S_mean(float(alpha_arr[0]), elements)

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
    f1 = _compute_dS_mean(1.0, elements)
    # The setup above should yield a non-trivial root — skip if the input
    # accidentally already satisfies |f(1)| ~ 0.
    if abs(f1) < 1e-14:
        pytest.skip("f(1) already at machine zero; iter count not meaningful")
    lo, hi, f_lo = _bracket_root(elements, f1, alpha_min=1e-3, alpha_max=1e3)

    iters = 0
    mid = 0.5 * (lo + hi)
    f_mid = _compute_dS_mean(mid, elements)
    while iters < 200:
        if abs(f_mid) < 1e-10 and abs(hi - lo) < 1e-8:
            break
        if (f_lo > 0.0) != (f_mid > 0.0):
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        mid = 0.5 * (lo + hi)
        f_mid = _compute_dS_mean(mid, elements)
        iters += 1

    assert iters < 100, f"bisection took {iters} iterations"


def test_bisection_stops_at_its_fixed_point(monkeypatch) -> None:
    """float32 elements: same alpha as running to the cap, far fewer evaluations.

    |f| stays above ``_F_TOL`` on float32 maps, so the loop without the
    fixed-point exit spun to ``_MAX_BISECT_ITERS``; the replay below checks
    that this input is such a case.
    """
    from cubic.metrics.microssim.ri_factor import _bracket_root  # local

    rng = np.random.default_rng(56)
    gt = rng.random((6, 64, 64), dtype=np.float32)
    pred = (1.6 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    e = compute_ssim_elements(gt, pred, data_range=float(gt.max() - gt.min()))
    assert e.ux.dtype == np.float32

    # The loop as it was, run to the iteration cap.
    lo, hi, f_lo = _bracket_root(e, _compute_dS_mean(1.0, e), 1e-6, 1e6)
    mid = 0.5 * (lo + hi)
    f_mid = _compute_dS_mean(mid, e)
    converged = False
    for _ in range(ri._MAX_BISECT_ITERS):
        if abs(f_mid) < ri._F_TOL and abs(hi - lo) < ri._X_TOL:
            converged = True
            break
        if (f_lo > 0.0) != (f_mid > 0.0):
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        mid = 0.5 * (lo + hi)
        f_mid = _compute_dS_mean(mid, e)
    assert not converged

    evals = []
    inner = ri._compute_dS_mean
    monkeypatch.setattr(
        ri, "_compute_dS_mean", lambda a, el: evals.append(a) or inner(a, el)
    )
    assert get_ri_factor(e) == mid
    assert len(evals) < 100


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


# -- Bracket boundary: cap probed once before raising ----------------------


def test_alpha_max_cap_probed_when_loop_overshoots() -> None:
    """``alpha_max`` itself is probed once when the powers-of-2 schedule overshoots.

    Setup: ``pred = gt * 0.833`` puts ``alpha*`` near ``1.2``; with
    ``alpha_max = 1.5``, the doubling loop's first probe ``alpha = 2.0``
    already overshoots ``alpha_max`` and the loop exits without entering
    its body. Pre-fix, this raised ``RuntimeError`` despite a root
    existing in ``(1, 1.5]``. Post-fix, the cap probe at ``1.5`` covers
    the gap and the bracket succeeds.
    """
    rng = np.random.default_rng(30)
    gt = rng.random((48, 48)).astype(np.float64)
    pred = gt * 0.833  # alpha* ~ 1.2 (between 1.0 and the first would-be probe at 2.0)
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha = get_ri_factor(elements, alpha_max=1.5)
    assert np.isfinite(alpha)
    assert 1.0 <= alpha <= 1.5, f"expected root in [1.0, 1.5]; got {alpha}"


def test_element_layout_does_not_change_alpha() -> None:
    """2-D, 3-D batched, and pre-raveled elements all yield the same alpha.

    Pins the removal of the ``_flatten_elements`` no-op: ``.mean()`` already
    reduces over every element pixel, so the solver must be layout-agnostic
    without an explicit ravel (which cost five array copies per call, since
    cropped element arrays are non-contiguous views).
    """
    rng = np.random.default_rng(40)
    gt = rng.random((3, 32, 32)).astype(np.float64)
    pred = gt * 1.4 + 0.05 * rng.standard_normal((3, 32, 32))
    dr = float(gt.max() - gt.min())
    batched = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    raveled = SSIMElements(
        ux=batched.ux.ravel(),
        uy=batched.uy.ravel(),
        vxy=batched.vxy.ravel(),
        vx=batched.vx.ravel(),
        vy=batched.vy.ravel(),
        C1=batched.C1,
        C2=batched.C2,
    )
    assert get_ri_factor(batched) == get_ri_factor(raveled)

    # A single 2-D slice pooled on its own must also work unchanged.
    single = compute_ssim_elements(
        gt[0], pred[0], data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    assert get_ri_factor(single) == get_ri_factor(
        SSIMElements(
            ux=single.ux.ravel(),
            uy=single.uy.ravel(),
            vxy=single.vxy.ravel(),
            vx=single.vx.ravel(),
            vy=single.vy.ravel(),
            C1=single.C1,
            C2=single.C2,
        )
    )


@pytest.mark.parametrize("scale", [1e-4, 0.1, 0.5, 1.5, 3.0, 100.0, 1e5])
def test_matches_brute_force_argmax(scale: float) -> None:
    """``get_ri_factor`` recovers the brute-force argmax of ``mean S(alpha)``.

    End-to-end guard on the solver after the ``_terms`` extraction, the
    ``_flatten_elements`` removal, and the switch from a ``f_lo * f_mid``
    product test to a direct sign comparison. A dense log-spaced sweep
    around the returned root must not find a better objective value.
    """
    rng = np.random.default_rng(41)
    gt = rng.random((32, 32)).astype(np.float64)
    # Noise scales with the signal so the optimum stays near 1 / scale
    # instead of being swamped at small scale factors.
    pred = scale * (gt + 0.02 * rng.standard_normal((32, 32)))
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha = get_ri_factor(elements)
    s_alpha = _compute_S_mean(alpha, elements)

    grid = np.logspace(np.log10(alpha) - 1.0, np.log10(alpha) + 1.0, 401)
    best = max(_compute_S_mean(float(a), elements) for a in grid)
    assert s_alpha >= best - 1e-12, (
        f"scale={scale}: solver S={s_alpha} below grid best S={best} at alpha={alpha}"
    )


def test_exact_zero_derivative_probe_returns_that_alpha(monkeypatch) -> None:
    """A probe landing exactly on the root returns it via a degenerate bracket.

    ``_bracket_root`` returns ``lo == hi`` when ``f(alpha) == 0.0`` exactly.
    This is the one case where the direct sign comparison in the bisection
    loop would misbehave if the bracket kept a zero-valued endpoint, so pin
    the degenerate-bracket contract.
    """
    rng = np.random.default_rng(42)
    gt = rng.random((16, 16)).astype(np.float64)
    elements = compute_ssim_elements(
        gt,
        gt,
        data_range=float(gt.max() - gt.min()),
        gaussian_weights=False,
        win_size=7,
        crop=True,
    )

    def fake_dS(alpha: float, _elements: SSIMElements) -> float:
        return 0.0 if alpha == 2.0 else 1.0

    monkeypatch.setattr(ri, "_compute_dS_mean", fake_dS)
    monkeypatch.setattr(ri, "_compute_S_mean", lambda alpha, _e: float(alpha))

    lo, hi, f_lo = ri._bracket_root(elements, 1.0, 1e-6, 1e6)
    assert (lo, hi, f_lo) == (2.0, 2.0, 0.0)
    assert ri.get_ri_factor(elements) == 2.0


def test_non_ascent_raises_value_error(monkeypatch) -> None:
    """The ascent invariant raises ``ValueError``, not ``AssertionError``.

    ``assert`` is stripped under ``python -O``, which would silently return
    a descending iterate. Force a violation by monkeypatching the objective
    so ``S(alpha*) < S(1)``.
    """
    rng = np.random.default_rng(43)
    gt = rng.random((32, 32)).astype(np.float64)
    pred = gt * 1.4
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    # S(1) = 1.0, S(anything else) = 0.0 -> guaranteed non-ascent.
    monkeypatch.setattr(
        ri, "_compute_S_mean", lambda alpha, _e: 1.0 if alpha == 1.0 else 0.0
    )
    with pytest.raises(ValueError, match="non-ascent"):
        ri.get_ri_factor(elements)


def test_alpha_min_cap_probed_when_loop_undershoots() -> None:
    """Mirror: ``alpha_min`` itself is probed once when halving undershoots.

    ``pred = gt * 1.2`` puts ``alpha*`` near ``0.833``; with
    ``alpha_min = 0.6``, the halving loop's first probe ``alpha = 0.5``
    already undershoots ``alpha_min``. Pre-fix this raised; post-fix the
    cap probe at ``0.6`` covers the gap.
    """
    rng = np.random.default_rng(31)
    gt = rng.random((48, 48)).astype(np.float64)
    pred = gt * 1.2  # alpha* ~ 0.833 (between 0.5 and 1.0)
    dr = float(gt.max() - gt.min())
    elements = compute_ssim_elements(
        gt, pred, data_range=dr, gaussian_weights=False, win_size=7, crop=True
    )
    alpha = get_ri_factor(elements, alpha_min=0.6)
    assert np.isfinite(alpha)
    assert 0.6 <= alpha <= 1.0, f"expected root in [0.6, 1.0]; got {alpha}"


# -- Chunked objective -----------------------------------------------------


def _chunk_layouts(e: SSIMElements) -> dict[str, SSIMElements]:
    """Return ``e`` as a 3-D batch, a single 2-D map, and a 1-D pooled array."""
    return {
        "3d": e,
        "2d": ri._map_arrays(e, lambda a: a[0]),
        "1d": ri._map_arrays(e, lambda a: a.ravel()),
    }


@pytest.mark.parametrize("chunk", [1, 37, 1000])
def test_chunked_objective_matches_full_map(monkeypatch, chunk: int) -> None:
    """Chunked ``S`` / ``dS`` means equal the whole-map means in every layout.

    ``chunk=1`` and ``37`` are smaller than one image row, so rows split
    recursively (``37`` also does not divide a row); ``1000`` spans rows.
    """
    rng = np.random.default_rng(50)
    gt = rng.random((4, 40, 40))
    pred = 0.8 * gt + 0.05 * rng.standard_normal(gt.shape)
    e = compute_ssim_elements(gt, pred, data_range=float(gt.max() - gt.min()))
    monkeypatch.setattr(ri, "_CHUNK_ELEMS", chunk)
    for name, layout in _chunk_layouts(e).items():
        for alpha in (0.5, 1.0, 1.7):
            want_S = float(ri._S_map(alpha, layout).mean())
            want_dS = float(ri._dS_map(alpha, layout).mean())
            assert _compute_S_mean(alpha, layout) == pytest.approx(want_S, rel=1e-12), (
                name
            )
            assert _compute_dS_mean(alpha, layout) == pytest.approx(
                want_dS, rel=1e-10, abs=1e-15
            ), name


def test_chunked_objective_accumulates_float32_in_float64(monkeypatch) -> None:
    """float32 chunk sums accumulate in float64, matching a float64 mean.

    The per-pixel maps stay float32; only the reduction is upcast, so the
    chunked mean agrees with the float64 mean of the same float32 map far
    below float32 rounding (~1e-7).
    """
    rng = np.random.default_rng(51)
    gt = rng.random((4, 128, 128), dtype=np.float32)
    pred = (0.8 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    e = compute_ssim_elements(gt, pred, data_range=float(gt.max() - gt.min()))
    assert e.ux.dtype == np.float32
    monkeypatch.setattr(ri, "_CHUNK_ELEMS", 4096)
    for alpha in (0.5, 1.0, 1.7):
        want_S = ri._S_map(alpha, e).astype(np.float64).mean()
        want_dS = ri._dS_map(alpha, e).astype(np.float64).mean()
        assert _compute_S_mean(alpha, e) == pytest.approx(want_S, rel=1e-12)
        assert _compute_dS_mean(alpha, e) == pytest.approx(want_dS, rel=1e-10)


def test_chunked_objective_accepts_0d_elements() -> None:
    """Scalar (0-D) element arrays reduce as one chunk instead of indexing axis 0."""
    e = SSIMElements(
        ux=np.asarray(0.5),
        uy=np.asarray(0.4),
        vxy=np.asarray(0.01),
        vx=np.asarray(0.02),
        vy=np.asarray(0.015),
        C1=1e-4,
        C2=9e-4,
    )
    assert _compute_S_mean(1.2, e) == pytest.approx(float(ri._S_map(1.2, e)))
    assert _compute_dS_mean(1.2, e) == pytest.approx(float(ri._dS_map(1.2, e)))


def test_global_ri_factor_chunked_matches_unchunked(monkeypatch) -> None:
    """Forcing many chunks leaves the fitted alpha unchanged."""
    rng = np.random.default_rng(51)
    gt = rng.random((6, 48, 48))
    pred = 1.3 * gt + 0.05 * rng.standard_normal(gt.shape)
    alpha_full = get_global_ri_factor(gt, pred)
    monkeypatch.setattr(ri, "_CHUNK_ELEMS", 500)
    alpha_chunked = get_global_ri_factor(gt, pred)
    assert alpha_chunked == pytest.approx(alpha_full, rel=1e-9)


def test_global_ri_factor_empty_stack_raises() -> None:
    """A zero-slice stack raises ``ValueError`` before any element compute."""
    empty = np.zeros((0, 16, 16))
    with pytest.raises(ValueError, match="at least one slice"):
        get_global_ri_factor(empty, empty)


@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 5, 5)])
def test_ri_factor_empty_elements_raises(shape: tuple[int, ...]) -> None:
    """Empty element arrays raise ``ValueError`` instead of dividing by zero."""
    z = np.zeros(shape)
    e = SSIMElements(ux=z, uy=z, vxy=z, vx=z, vy=z, C1=1e-4, C2=9e-4)
    with pytest.raises(ValueError, match="at least one pixel"):
        get_ri_factor(e)


def _gpu_peak_above_base(fn) -> tuple[object, int]:
    """Run ``fn()`` and return its result and the CuPy pool peak above entry."""
    import cupy as cp
    from cupy.cuda import memory_hook

    class PeakHook(memory_hook.MemoryHook):
        name = "PeakHook"

        def __init__(self, pool) -> None:
            self.pool = pool
            self.peak = pool.used_bytes()

        def malloc_postprocess(self, **kwargs) -> None:
            self.peak = max(self.peak, self.pool.used_bytes())

    pool = cp.get_default_memory_pool()
    base = pool.used_bytes()
    with PeakHook(pool) as hook:
        result = fn()
    return result, hook.peak - base


def test_global_ri_factor_gpu_peak_memory(monkeypatch, gpu_available: bool) -> None:
    """GPU fit peak stays under 9 pooled-element arrays above the input.

    Pooling via per-slice lists + ``concatenate`` and unchunked objective
    evaluation peaked at ~27 element-sized arrays (MEASURED on 640x960
    float32 stacks), which OOMed 48 GB GPUs on 576 slices.
    """
    if not gpu_available:
        pytest.skip("GPU not available")

    n, h, w = 16, 256, 256
    rng = np.random.default_rng(52)
    gt = rng.random((n, h, w), dtype=np.float32)
    pred = (0.7 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    # Force the chunked path on a test-sized input.
    monkeypatch.setattr(ri, "_CHUNK_ELEMS", 1 << 16)
    gt_cp, pred_cp = ascupy(gt), ascupy(pred)
    alpha, peak = _gpu_peak_above_base(lambda: get_global_ri_factor(gt_cp, pred_cp))
    assert np.isfinite(alpha)
    assert peak / (n * (h - 6) * (w - 6) * 4) < 9


def test_ri_factor_gpu_single_large_slice_is_chunked(
    monkeypatch, gpu_available: bool
) -> None:
    """A single slice larger than a chunk is split, not reduced whole.

    Chunking only along axis 0 left a ``(1, h, w)`` pool as one chunk, which
    built ~15 element-sized temporaries (MEASURED); splitting inside the
    slice bounds them to ~15 chunk-sized ones.
    """
    if not gpu_available:
        pytest.skip("GPU not available")

    rng = np.random.default_rng(53)
    gt = rng.random((1024, 1024), dtype=np.float32)
    pred = (0.7 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    e = compute_ssim_elements(
        ascupy(gt), ascupy(pred), data_range=float(gt.max() - gt.min())
    )
    pooled = ri._map_arrays(e, lambda a: a[None].copy())
    monkeypatch.setattr(ri, "_CHUNK_ELEMS", 1 << 16)
    alpha, peak = _gpu_peak_above_base(lambda: get_ri_factor(pooled))
    assert np.isfinite(alpha)
    assert peak / (pooled.ux.size * 4) < 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_gpu_objective_matches_array_maps(
    monkeypatch, dtype: type, gpu_available: bool
) -> None:
    """The fused GPU reduction sums the array path's per-pixel maps.

    Its kernel repeats the NumPy operation order with FMA contraction off, so
    only the float64 summation order differs from summing the CuPy maps.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    rng = np.random.default_rng(54)
    gt = rng.random((5, 64, 72)).astype(dtype)
    pred = (0.8 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(dtype)
    e = compute_ssim_elements(
        ascupy(gt), ascupy(pred), data_range=float(gt.max() - gt.min())
    )
    assert e.ux.dtype == dtype
    fused = ri._fused_mean
    calls: list[str] = []
    monkeypatch.setattr(
        ri, "_fused_mean", lambda name, *args: calls.append(name) or fused(name, *args)
    )
    for name, layout in _chunk_layouts(e).items():
        for alpha in (0.5, 1.0, 1.7):
            want_S = float(ri._S_map(alpha, layout).astype(np.float64).mean())
            want_dS = float(ri._dS_map(alpha, layout).astype(np.float64).mean())
            assert _compute_S_mean(alpha, layout) == pytest.approx(want_S, rel=1e-13), (
                name
            )
            assert _compute_dS_mean(alpha, layout) == pytest.approx(
                want_dS, rel=1e-11, abs=1e-15
            ), name
    assert calls.count("ri_s") == calls.count("ri_ds") == 9


def test_global_ri_factor_gpu_matches_cpu(gpu_available: bool) -> None:
    """The fused GPU fit lands on the CPU alpha within the bisection tolerance."""
    if not gpu_available:
        pytest.skip("GPU not available")
    rng = np.random.default_rng(55)
    gt = rng.random((6, 96, 80), dtype=np.float32)
    pred = (1.6 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    cpu = get_global_ri_factor(gt, pred)
    gpu = get_global_ri_factor(ascupy(gt), ascupy(pred))
    assert cpu != pytest.approx(1.0, abs=0.05)
    assert gpu == pytest.approx(cpu, abs=2 * ri._X_TOL)


def test_mixed_dtype_gpu_elements_take_the_array_path(gpu_available: bool) -> None:
    """Elements of mixed float dtypes skip the single-dtype fused kernel."""
    if not gpu_available:
        pytest.skip("GPU not available")
    rng = np.random.default_rng(57)
    gt = rng.random((3, 40, 40), dtype=np.float32)
    pred = (0.8 * gt + 0.05 * rng.standard_normal(gt.shape)).astype(np.float32)
    e = compute_ssim_elements(
        ascupy(gt), ascupy(pred), data_range=float(gt.max() - gt.min())
    )
    mixed = SSIMElements(
        ux=e.ux,
        uy=e.uy,
        vxy=e.vxy.astype(np.float64),
        vx=e.vx.astype(np.float64),
        vy=e.vy.astype(np.float64),
        C1=e.C1,
        C2=e.C2,
    )
    assert not ri._fused_eligible(mixed)
    assert np.isfinite(_compute_dS_mean(1.3, mixed))
