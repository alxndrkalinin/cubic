"""Range-invariant factor solver for MicroSSIM.

Ported from juglab/microssim@8bccb17d ``ri_factor/ri_factor.py``. Upstream
solves the 1-D optimum of ``mean_n S_n(alpha)`` with ``scipy.optimize.minimize``;
this module replaces that with a dependency-free bracket-then-bisection root
finder on the analytical derivative ``f(alpha) = mean_n dS_n/dalpha``. The
per-pixel algebra lives in :func:`_terms`, :func:`_compute_S_mean` and
:func:`_compute_dS_mean`.

MicroSSIM normalization places the optimum near ``alpha = 1``, so the bracket
expands outward from ``alpha = 1`` by doubling / halving within the default
window ``1e-6 <= alpha <= 1e6`` (both bounds configurable via the
``alpha_min`` / ``alpha_max`` kwargs). Bisection then refines until both
``|f(mid)| < 1e-10`` AND ``|hi - lo| < 1e-8`` hold.
"""

from __future__ import annotations

import numpy as np

from .ssim_elements import SSIMElements, compute_ssim_elements

# Bracket and bisection tunables. The bracket caps mirror upstream
# MicroSSIM's normalization regime (alpha ~ 1 by construction); the
# conjunctive termination guards both flat-region stalls (pure |f| tol)
# and tiny-slope spinning (pure x tol).
ALPHA_MIN_DEFAULT = 1e-6
ALPHA_MAX_DEFAULT = 1e6
_F_TOL = 1e-10
_X_TOL = 1e-8
_INIT_F_TOL = 1e-14
_ASCENT_SLACK = 1e-12
_MAX_BISECT_ITERS = 200


def validate_alpha_bounds(alpha_min: float, alpha_max: float) -> None:
    """Validate that ``(alpha_min, alpha_max)`` brackets ``alpha = 1`` strictly.

    The bracket starts at ``alpha = 1`` and expands by halving leftward /
    doubling rightward, so any ``alpha_min`` not in ``(0, 1)`` or any
    ``alpha_max`` not in ``(1, +inf)`` produces a degenerate window.

    Raises
    ------
    ValueError
        If ``alpha_min`` is not a finite float in ``(0, 1)`` or
        ``alpha_max`` is not a finite float ``> 1``.
    """
    if not (np.isfinite(alpha_min) and 0.0 < alpha_min < 1.0):
        raise ValueError(f"alpha_min must be a finite float in (0, 1); got {alpha_min}")
    if not (np.isfinite(alpha_max) and alpha_max > 1.0):
        raise ValueError(f"alpha_max must be a finite float > 1; got {alpha_max}")


def _terms(
    alpha: float, elements: SSIMElements
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-pixel SSIM numerator / denominator terms at ``alpha``.

    ``S(alpha) = (A1 * A2) / (B1 * B2)`` with

    * ``A1 = 2*alpha*ux*uy + C1``
    * ``A2 = 2*alpha*vxy + C2``
    * ``B1 = ux**2 + alpha**2 * uy**2 + C1``
    * ``B2 = vx + alpha**2 * vy + C2``

    Parameters
    ----------
    alpha : float
        Scalar multiplier applied to the prediction.
    elements : SSIMElements
        Precomputed SSIM elements.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A1, A2, B1, B2)``, each shaped like the element arrays.
    """
    alpha_sq = alpha * alpha
    A1 = 2.0 * alpha * elements.ux * elements.uy + elements.C1
    A2 = 2.0 * alpha * elements.vxy + elements.C2
    B1 = elements.ux * elements.ux + alpha_sq * elements.uy * elements.uy + elements.C1
    B2 = elements.vx + alpha_sq * elements.vy + elements.C2
    return A1, A2, B1, B2


def _compute_S_mean(alpha: float, elements: SSIMElements) -> float:
    """Mean per-pixel SSIM at a given ``alpha``.

    Parameters
    ----------
    alpha : float
        Scalar multiplier applied to the prediction.
    elements : SSIMElements
        Precomputed SSIM elements. Any layout is accepted (2-D map, 3-D
        batched map, or a pre-pooled 1-D array) — ``.mean()`` reduces over
        every element pixel regardless.

    Returns
    -------
    float
        Mean of ``S_n(alpha) = (A1*A2) / (B1*B2)`` over all element pixels.
    """
    A1, A2, B1, B2 = _terms(alpha, elements)
    return float(((A1 * A2) / (B1 * B2)).mean())


def _compute_dS_mean(alpha: float, elements: SSIMElements) -> float:
    """Mean per-pixel derivative ``dS/dalpha`` at a given ``alpha``.

    The analytical quotient rule on ``S = N / D = (A1*A2) / (B1*B2)`` with
    ``dA1 = 2*ux*uy``, ``dA2 = 2*vxy``, ``dB1 = 2*alpha*uy**2`` and
    ``dB2 = 2*alpha*vy``.

    Parameters
    ----------
    alpha : float
        Scalar multiplier applied to the prediction.
    elements : SSIMElements
        Precomputed SSIM elements; any layout (see :func:`_compute_S_mean`).

    Returns
    -------
    float
        Mean of ``dS_n/dalpha`` over all element pixels.
    """
    A1, A2, B1, B2 = _terms(alpha, elements)

    dA1 = 2.0 * elements.ux * elements.uy
    dA2 = 2.0 * elements.vxy
    dB1 = 2.0 * alpha * elements.uy * elements.uy
    dB2 = 2.0 * alpha * elements.vy

    N = A1 * A2
    D = B1 * B2
    dN = dA1 * A2 + A1 * dA2
    dD = dB1 * B2 + B1 * dB2

    dS = (dN * D - N * dD) / (D * D)
    return float(dS.mean())


def _bracket_root(
    elements: SSIMElements, f1: float, alpha_min: float, alpha_max: float
) -> tuple[float, float, float]:
    """Find ``(lo, hi)`` with opposite-sign ``f`` values by expanding from 1.

    On entry ``f1 = f(1)`` is already known to be non-zero (the caller
    returns 1 directly when ``|f1| < _INIT_F_TOL``). Doubling expansion to
    the right is used when ``f1 > 0`` (gradient points to larger alpha);
    halving expansion to the left when ``f1 < 0``.

    Parameters
    ----------
    elements : SSIMElements
        SSIM elements (variances / covariance); any layout.
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
    lo, hi, f_lo : float
        Bracket endpoints and ``f(lo)``. ``f(lo)`` is strictly positive and
        ``f(hi)`` is strictly negative, so the caller's bisection can pick
        the half retaining the root from ``f_lo`` alone. When a probe lands
        *exactly* on the root, the degenerate bracket ``lo == hi`` is
        returned with ``f_lo = 0.0``, which makes bisection terminate on
        its first check.

    Raises
    ------
    RuntimeError
        If no sign change is found within ``[alpha_min, alpha_max]``.
    """
    if f1 > 0.0:
        # f(1) > 0: root is to the right of 1 — expand by doubling.
        lo, f_lo = 1.0, f1
        alpha = 2.0
        while alpha <= alpha_max:
            f_alpha = _compute_dS_mean(alpha, elements)
            if f_alpha == 0.0:
                return alpha, alpha, 0.0
            if f_alpha < 0.0:
                return lo, alpha, f_lo
            lo, f_lo = alpha, f_alpha
            alpha *= 2.0
        # Powers-of-2 schedule could miss a root in (lo, alpha_max] when
        # alpha_max isn't itself a power of 2 (e.g., default 1e6 →
        # last probe is 2**19 = 524288). Probe alpha_max itself before
        # giving up so the full requested interval is actually covered.
        if lo < alpha_max:
            f_cap = _compute_dS_mean(alpha_max, elements)
            if f_cap == 0.0:
                return alpha_max, alpha_max, 0.0
            if f_cap < 0.0:
                return lo, alpha_max, f_lo
        raise RuntimeError(
            "RI factor failed to bracket on the right; input may violate "
            f"fit assumptions or alpha_max={alpha_max} is too small. "
            f"ux shape={elements.ux.shape}"
        )
    # f(1) < 0: root is to the left of 1 — expand by halving.
    hi = 1.0
    alpha = 0.5
    while alpha >= alpha_min:
        f_alpha = _compute_dS_mean(alpha, elements)
        if f_alpha == 0.0:
            return alpha, alpha, 0.0
        if f_alpha > 0.0:
            return alpha, hi, f_alpha
        hi = alpha
        alpha *= 0.5
    # Mirror of the right-side fix: probe alpha_min itself before giving
    # up so a root in [alpha_min, hi) isn't missed by the powers-of-2
    # schedule.
    if hi > alpha_min:
        f_floor = _compute_dS_mean(alpha_min, elements)
        if f_floor == 0.0:
            return alpha_min, alpha_min, 0.0
        if f_floor > 0.0:
            return alpha_min, hi, f_floor
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
        pre-flattened; the objective reduces with ``.mean()`` over every
        element pixel, so layout is irrelevant. ``C1`` and ``C2`` are taken
        from the ``elements`` object (callers using
        :func:`get_global_ri_factor` get the last-slice values, matching
        upstream).
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
        outward, so any cap on the wrong side of 1 is degenerate. Also
        raised if the returned iterate violates the ascent invariant
        (see Notes).

    Notes
    -----
    Termination: bisection stops as soon as both ``|f(mid)| < 1e-10`` AND
    ``|hi - lo| < 1e-8`` hold. Before ``alpha*`` is returned, the invariant
    ``mean(S(alpha*)) >= mean(S(1)) - 1e-12`` is checked and a violation
    raises ``ValueError`` (the slack allows ``alpha = 1`` itself being the
    optimum). ``ValueError`` rather than ``assert`` because asserts are
    stripped under ``python -O``.
    """
    validate_alpha_bounds(alpha_min, alpha_max)

    f1 = _compute_dS_mean(1.0, elements)
    if abs(f1) < _INIT_F_TOL:
        return 1.0

    lo, hi, f_lo = _bracket_root(elements, f1, alpha_min, alpha_max)

    # Bisection refinement. The conjunction of |f| and x tolerances guards
    # both stalling in flat regions and spinning on near-zero slope.
    mid = 0.5 * (lo + hi)
    f_mid = _compute_dS_mean(mid, elements)
    for _ in range(_MAX_BISECT_ITERS):
        if abs(f_mid) < _F_TOL and abs(hi - lo) < _X_TOL:
            break
        # Sign decides which half retains the root. Compare signs directly
        # rather than testing the product ``f_lo * f_mid <= 0``: the product
        # underflows to +0.0 when both factors are tiny, which would take
        # the wrong branch.
        if (f_lo > 0.0) != (f_mid > 0.0):
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        mid = 0.5 * (lo + hi)
        f_mid = _compute_dS_mean(mid, elements)

    alpha_star = float(mid)

    s_star = _compute_S_mean(alpha_star, elements)
    s_one = _compute_S_mean(1.0, elements)
    if s_star < s_one - _ASCENT_SLACK:
        raise ValueError(
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
    validate_alpha_bounds(alpha_min, alpha_max)
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
