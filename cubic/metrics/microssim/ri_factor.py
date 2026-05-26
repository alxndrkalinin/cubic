"""Range-invariant factor solver for MicroSSIM.

Ported from juglab/microssim@8bccb17d ``ri_factor/ri_factor.py``. Upstream
solves the 1-D optimum of ``mean_n S_n(alpha)`` with ``scipy.optimize.minimize``;
this module replaces that with a dependency-free bracket-then-bisection root
finder on the analytical derivative ``f(alpha) = mean_n dS_n/dalpha``.

Definitions (per-pixel, with ``e = SSIMElements``):

* ``A1(alpha) = 2*alpha*ux*uy + C1``
* ``A2(alpha) = 2*alpha*vxy + C2``
* ``B1(alpha) = ux**2 + alpha**2 * uy**2 + C1``
* ``B2(alpha) = vx + alpha**2 * vy + C2``
* ``S_n(alpha) = (A1 * A2) / (B1 * B2)``

Analytical derivative (per-pixel):

* ``dA1/dalpha = 2*ux*uy``
* ``dA2/dalpha = 2*vxy``
* ``dB1/dalpha = 2*alpha*uy**2``
* ``dB2/dalpha = 2*alpha*vy``
* ``dN/dalpha = (2*ux*uy)*A2 + A1*(2*vxy)``
* ``dD/dalpha = (2*alpha*uy**2)*B2 + B1*(2*alpha*vy)``
* ``dS_n/dalpha = (dN*D - N*dD) / D**2``

``f(alpha)`` is the mean of ``dS_n/dalpha`` over all flattened element pixels.
MicroSSIM normalization places the optimum near ``alpha = 1``; we bracket by
doubling outwards from ``alpha = 1`` (default range ``1e-6 <= alpha <= 1e6``;
both bounds are configurable via the ``alpha_min`` / ``alpha_max`` kwargs) and
refine with bisection terminated on both ``|f(mid)| < 1e-10`` AND
``|hi - lo| < 1e-8``.
"""

from __future__ import annotations

import numpy as np

from .ssim_elements import SSIMElements, compute_ssim_elements

# Bracket and bisection tunables. The bracket caps mirror upstream
# MicroSSIM's normalization regime (alpha ~ 1 by construction); the
# conjunctive termination guards both flat-region stalls (pure |f| tol)
# and tiny-slope spinning (pure x tol).
_ALPHA_INIT = 1.0
ALPHA_MIN_DEFAULT = 1e-6
ALPHA_MAX_DEFAULT = 1e6
_F_TOL = 1e-10
_X_TOL = 1e-8
_INIT_F_TOL = 1e-14
_ASCENT_SLACK = 1e-12
_MAX_BISECT_ITERS = 200


def _validate_alpha_bounds(alpha_min: float, alpha_max: float) -> None:
    """Validate that ``(alpha_min, alpha_max)`` brackets ``alpha = 1`` strictly.

    The bracket starts at ``alpha = 1`` and expands by halving leftward /
    doubling rightward, so any ``alpha_min`` not in ``(0, 1)`` or any
    ``alpha_max`` not in ``(1, +inf)`` produces a degenerate window.
    """
    if not (np.isfinite(alpha_min) and 0.0 < alpha_min < 1.0):
        raise ValueError(f"alpha_min must be a finite float in (0, 1); got {alpha_min}")
    if not (np.isfinite(alpha_max) and alpha_max > 1.0):
        raise ValueError(f"alpha_max must be a finite float > 1; got {alpha_max}")


def _compute_S_mean(alpha: float, elements: SSIMElements) -> float:
    """Mean per-pixel SSIM at a given ``alpha``.

    Parameters
    ----------
    alpha : float
        Scalar multiplier applied to the prediction.
    elements : SSIMElements
        Precomputed SSIM elements (flattened element arrays accepted).

    Returns
    -------
    float
        Mean of ``S_n(alpha) = (A1*A2) / (B1*B2)`` over all element pixels.
    """
    ux = elements.ux
    uy = elements.uy
    vxy = elements.vxy
    vx = elements.vx
    vy = elements.vy
    C1 = elements.C1
    C2 = elements.C2

    A1 = 2.0 * alpha * ux * uy + C1
    A2 = 2.0 * alpha * vxy + C2
    B1 = ux * ux + (alpha * alpha) * uy * uy + C1
    B2 = vx + (alpha * alpha) * vy + C2
    return float(((A1 * A2) / (B1 * B2)).mean())


def _compute_dS_mean(alpha: float, elements: SSIMElements) -> float:
    """Mean per-pixel derivative ``dS/dalpha`` at a given ``alpha``.

    Parameters
    ----------
    alpha : float
        Scalar multiplier applied to the prediction.
    elements : SSIMElements
        Precomputed SSIM elements (flattened element arrays accepted).

    Returns
    -------
    float
        Mean of ``dS_n/dalpha`` over all element pixels, computed via the
        analytical quotient rule on ``(A1*A2)/(B1*B2)``.
    """
    ux = elements.ux
    uy = elements.uy
    vxy = elements.vxy
    vx = elements.vx
    vy = elements.vy
    C1 = elements.C1
    C2 = elements.C2

    A1 = 2.0 * alpha * ux * uy + C1
    A2 = 2.0 * alpha * vxy + C2
    B1 = ux * ux + (alpha * alpha) * uy * uy + C1
    B2 = vx + (alpha * alpha) * vy + C2

    dA1 = 2.0 * ux * uy
    dA2 = 2.0 * vxy
    dB1 = 2.0 * alpha * uy * uy
    dB2 = 2.0 * alpha * vy

    N = A1 * A2
    D = B1 * B2
    dN = dA1 * A2 + A1 * dA2
    dD = dB1 * B2 + B1 * dB2

    dS = (dN * D - N * dD) / (D * D)
    return float(dS.mean())


def _flatten_elements(elements: SSIMElements) -> SSIMElements:
    """Return a copy of ``elements`` with all element arrays flattened.

    Flattening lets the derivative / SSIM evaluators take a single mean over
    all element pixels regardless of how the caller laid them out (2-D map,
    3-D batched map, or pre-concatenated 1-D pool from
    :func:`get_global_ri_factor`).

    Parameters
    ----------
    elements : SSIMElements
        Source elements.

    Returns
    -------
    SSIMElements
        Same scalars, with ``ux, uy, vxy, vx, vy`` reshaped to 1-D.
    """
    return SSIMElements(
        ux=elements.ux.ravel(),
        uy=elements.uy.ravel(),
        vxy=elements.vxy.ravel(),
        vx=elements.vx.ravel(),
        vy=elements.vy.ravel(),
        C1=elements.C1,
        C2=elements.C2,
    )


def _bracket_root(
    elements: SSIMElements, f1: float, alpha_min: float, alpha_max: float
) -> tuple[float, float, float, float]:
    """Find ``(lo, hi)`` with opposite-sign ``f`` values by expanding from 1.

    On entry ``f1 = f(1)`` is already known to be non-zero (the caller
    returns 1 directly when ``|f1| < _INIT_F_TOL``). Doubling expansion to
    the right is used when ``f1 > 0`` (gradient points to larger alpha);
    halving expansion to the left when ``f1 < 0``.

    Parameters
    ----------
    elements : SSIMElements
        Flattened SSIM elements (variances / covariance).
    f1 : float
        Value of ``f(1)``; sign decides direction.
    alpha_min : float
        Lower bracket cap. Halving expansion stops once ``alpha`` falls
        below this value without a sign change.
    alpha_max : float
        Upper bracket cap. Doubling expansion stops once ``alpha`` exceeds
        this value without a sign change.

    Returns
    -------
    lo, hi, f_lo, f_hi : float
        Bracket endpoints and their ``f`` values, with
        ``f_lo * f_hi <= 0``.

    Raises
    ------
    RuntimeError
        If no sign change is found within ``[alpha_min, alpha_max]``.
    """
    if f1 > 0.0:
        # f(1) > 0: root is to the right of 1 — expand by doubling.
        lo, f_lo = _ALPHA_INIT, f1
        alpha = _ALPHA_INIT * 2.0
        while alpha <= alpha_max:
            f_alpha = _compute_dS_mean(alpha, elements)
            if f_alpha == 0.0 or (f_alpha < 0.0):
                return lo, alpha, f_lo, f_alpha
            lo, f_lo = alpha, f_alpha
            alpha *= 2.0
        # Powers-of-2 schedule could miss a root in (lo, alpha_max] when
        # alpha_max isn't itself a power of 2 (e.g., default 1e6 →
        # last probe is 2**19 = 524288). Probe alpha_max itself before
        # giving up so the full requested interval is actually covered.
        if lo < alpha_max:
            f_cap = _compute_dS_mean(alpha_max, elements)
            if f_cap <= 0.0:
                return lo, alpha_max, f_lo, f_cap
        raise RuntimeError(
            "RI factor failed to bracket on the right; input may violate "
            f"fit assumptions or alpha_max={alpha_max} is too small. "
            f"ux shape={elements.ux.shape}"
        )
    # f(1) < 0: root is to the left of 1 — expand by halving.
    hi, f_hi = _ALPHA_INIT, f1
    alpha = _ALPHA_INIT * 0.5
    while alpha >= alpha_min:
        f_alpha = _compute_dS_mean(alpha, elements)
        if f_alpha == 0.0 or (f_alpha > 0.0):
            return alpha, hi, f_alpha, f_hi
        hi, f_hi = alpha, f_alpha
        alpha *= 0.5
    # Mirror of the right-side fix: probe alpha_min itself before giving
    # up so a root in [alpha_min, hi) isn't missed by the powers-of-2
    # schedule.
    if hi > alpha_min:
        f_floor = _compute_dS_mean(alpha_min, elements)
        if f_floor >= 0.0:
            return alpha_min, hi, f_floor, f_hi
    raise RuntimeError(
        "RI factor failed to bracket on the left; input may violate "
        f"fit assumptions or alpha_min={alpha_min} is too large. "
        f"ux shape={elements.ux.shape}"
    )


def get_ri_factor(
    elements: SSIMElements,
    *,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
) -> float:
    """Compute the range-invariant factor by bisection on ``dS/dalpha = 0``.

    The MicroSSIM range-invariant factor ``alpha`` is the scalar multiplier
    applied to the prediction that maximizes the mean per-pixel SSIM. This
    function locates the unique optimum on ``(0, +inf)`` by bracketing
    outwards from ``alpha = 1`` and refining with bisection. No SciPy
    dependency; no Newton iteration (no second derivative required).

    Parameters
    ----------
    elements : SSIMElements
        Per-pixel SSIM elements. Element arrays may be 2-D, 3-D batched, or
        pre-flattened; they are reshaped to 1-D internally so the mean is
        taken over all pixels. ``C1`` and ``C2`` are taken from the
        ``elements`` object (callers using :func:`get_global_ri_factor` get
        the last-slice values, matching upstream).
    alpha_min : float, default=:data:`ALPHA_MIN_DEFAULT` (``1e-6``)
        Lower bracket cap for the halving expansion. Used when ``pred`` is
        scaled larger than ``gt`` so the optimum sits below ``1``. The
        default safely covers most reasonable fits; lower it for heavily
        up-scaled predictions (``pred ~ 1e6 * gt``).
    alpha_max : float, default=:data:`ALPHA_MAX_DEFAULT` (``1e6``)
        Upper bracket cap for the doubling expansion. The default safely
        covers most reasonable MicroSSIM fits where the optimum sits near
        ``alpha = 1``; raise it for pathological inputs (very small
        calibration sets, heavily mis-normalized predictions).

        Note: bracket probes are powers of 2 starting at 2 (rightward) or
        0.5 (leftward); if the schedule overshoots ``alpha_max`` /
        undershoots ``alpha_min`` without finding a sign change, the cap
        itself is probed once before raising so a root in
        ``(last_probe, alpha_max]`` (or ``[alpha_min, last_probe)``) is
        not missed.

    Returns
    -------
    float
        The optimal ``alpha`` such that ``mean(dS/dalpha) ~= 0``.

    Raises
    ------
    RuntimeError
        If the bracketing phase fails to find a sign change within
        ``[alpha_min, alpha_max]`` — typical for pathological inputs
        (constant ground truth with non-constant prediction, or all-zero
        prediction).
    ValueError
        If ``alpha_min`` is outside ``(0, 1)`` or ``alpha_max`` is outside
        ``(1, +inf)``. The bracket starts at ``alpha = 1`` and expands
        outward, so any cap on the wrong side of 1 is degenerate.

    Notes
    -----
    Termination: bisection stops as soon as both ``|f(mid)| < 1e-10`` AND
    ``|hi - lo| < 1e-8`` hold. After the iterate ``alpha*`` is returned, the
    invariant ``mean(S(alpha*)) >= mean(S(1)) - 1e-12`` is asserted (with
    slack to allow ``alpha = 1`` itself being the optimum).
    """
    _validate_alpha_bounds(alpha_min, alpha_max)
    flat = _flatten_elements(elements)

    f1 = _compute_dS_mean(_ALPHA_INIT, flat)
    if abs(f1) < _INIT_F_TOL:
        return float(_ALPHA_INIT)

    lo, hi, f_lo, f_hi = _bracket_root(flat, f1, alpha_min, alpha_max)

    # Bisection refinement. The conjunction of |f| and x tolerances guards
    # both stalling in flat regions and spinning on near-zero slope.
    mid = 0.5 * (lo + hi)
    f_mid = _compute_dS_mean(mid, flat)
    for _ in range(_MAX_BISECT_ITERS):
        if abs(f_mid) < _F_TOL and abs(hi - lo) < _X_TOL:
            break
        # Sign decides which half retains the root.
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        mid = 0.5 * (lo + hi)
        f_mid = _compute_dS_mean(mid, flat)

    alpha_star = float(mid)

    s_star = _compute_S_mean(alpha_star, flat)
    s_one = _compute_S_mean(_ALPHA_INIT, flat)
    assert s_star >= s_one - _ASCENT_SLACK, (
        f"RI bisection produced a non-ascent: S(alpha*={alpha_star})={s_star} "
        f"< S(1)={s_one} - {_ASCENT_SLACK}"
    )

    return alpha_star


def get_global_ri_factor(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    alpha_min: float = ALPHA_MIN_DEFAULT,
    alpha_max: float = ALPHA_MAX_DEFAULT,
    **ssim_kwargs: object,
) -> float:
    """Compute the range-invariant factor on a stack of images.

    Mirrors upstream's per-slice element pooling
    (``ri_factor/ri_factor.py:84-132``): every slice contributes its own
    ``compute_ssim_elements`` call using its own ``data_range``; the
    flattened ``ux, uy, vxy, vx, vy`` are concatenated and ``C1, C2`` are
    taken from the **last** slice. The pooled elements are passed to
    :func:`get_ri_factor`.

    Parameters
    ----------
    gt : numpy.ndarray
        Ground-truth image stack with shape ``(H, W)`` or ``(N, H, W)``.
        A 2-D input is treated as ``N = 1``. Ragged lists are NOT supported
        here; pool such inputs at the ``MicroSSIM.fit`` layer.
    pred : numpy.ndarray
        Prediction stack; same shape as ``gt``.
    alpha_min : float, default=:data:`ALPHA_MIN_DEFAULT` (``1e-6``)
        Lower bracket cap forwarded to :func:`get_ri_factor`.
    alpha_max : float, default=:data:`ALPHA_MAX_DEFAULT` (``1e6``)
        Upper bracket cap forwarded to :func:`get_ri_factor`.
    **ssim_kwargs : object
        Additional keyword arguments forwarded to
        :func:`compute_ssim_elements`. Defaults match the upstream RI-factor
        fit path (``gaussian_weights=False``, ``win_size=7``, ``crop=True``).
        ``data_range`` is computed per slice and must not be supplied.

    Returns
    -------
    float
        The optimal ``alpha`` from :func:`get_ri_factor` on the pooled
        elements.

    Raises
    ------
    ValueError
        If ``alpha_min`` / ``alpha_max`` are out of range (see
        :func:`get_ri_factor`), ``gt`` and ``pred`` differ in shape, or
        either has ``ndim`` not in ``{2, 3}``.
    """
    # Validate bracket bounds up-front so a bad value fails before the
    # potentially-expensive per-slice compute_ssim_elements loop.
    _validate_alpha_bounds(alpha_min, alpha_max)
    if gt.shape != pred.shape:
        raise ValueError(
            f"Ground-truth and prediction arrays must have the same shape "
            f"(got {gt.shape} and {pred.shape})."
        )
    if gt.ndim not in (2, 3):
        raise ValueError(
            f"Only (H, W) or (N, H, W) input is supported; got ndim={gt.ndim}"
        )

    if gt.ndim == 2:
        gt = gt[None]
        pred = pred[None]

    # Upstream's fit path uses uniform filter with win_size=7, crop=True.
    # Allow callers to override via ssim_kwargs.
    defaults: dict[str, object] = {
        "gaussian_weights": False,
        "win_size": 7,
        "crop": True,
    }
    for key, value in defaults.items():
        ssim_kwargs.setdefault(key, value)
    # Per-slice data_range — callers cannot override.
    if "data_range" in ssim_kwargs:
        raise ValueError(
            "data_range is computed per slice and must not be supplied to "
            "get_global_ri_factor."
        )

    ux_list: list[np.ndarray] = []
    uy_list: list[np.ndarray] = []
    vxy_list: list[np.ndarray] = []
    vx_list: list[np.ndarray] = []
    vy_list: list[np.ndarray] = []
    C1_last = 0.0
    C2_last = 0.0
    for i in range(gt.shape[0]):
        dr = float(gt[i].max() - gt[i].min())
        e_i = compute_ssim_elements(gt[i], pred[i], data_range=dr, **ssim_kwargs)  # type: ignore[arg-type]
        ux_list.append(e_i.ux.ravel())
        uy_list.append(e_i.uy.ravel())
        vxy_list.append(e_i.vxy.ravel())
        vx_list.append(e_i.vx.ravel())
        vy_list.append(e_i.vy.ravel())
        C1_last = e_i.C1
        C2_last = e_i.C2

    pooled = SSIMElements(
        ux=np.concatenate(ux_list),
        uy=np.concatenate(uy_list),
        vxy=np.concatenate(vxy_list),
        vx=np.concatenate(vx_list),
        vy=np.concatenate(vy_list),
        C1=C1_last,
        C2=C2_last,
    )
    return get_ri_factor(pooled, alpha_min=alpha_min, alpha_max=alpha_max)
