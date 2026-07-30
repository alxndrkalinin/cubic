"""Unmatched back-projector PSF generation for Richardson-Lucy deconvolution.

This module is a device-agnostic (NumPy/CuPy) port of the reference MATLAB
``BackProjector.m`` from Guo et al. (2020), "Rapid image deconvolution and
multiview fusion for optical microscopy" (Nature Biotechnology). It builds a
spatial-domain back-projector PSF from a forward PSF, enabling the
Wiener-Butterworth (WB) unmatched back projector that accelerates RL
convergence.

The same function works on NumPy arrays (CPU) or CuPy arrays (GPU) based on the
input array's device location; no code changes are required to switch devices.

Intentional divergences from the reference are listed in the "Notes" section of
:func:`create_backprojector`.

Reference: Guo, M. et al. Nat Biotechnol 38, 1337-1346 (2020).
"""

import numpy as np

from cubic.cuda import asnumpy, get_array_module

__all__ = ["create_backprojector"]

_VALID_BP_TYPES = (
    "traditional",
    "gaussian",
    "butterworth",
    "wiener",
    "wiener-butterworth",
)


def _fwhm_1d(y: np.ndarray) -> float:
    """Full-width at half-maximum of a 1-D profile, in pixel units.

    Port of the MATLAB ``fwhm`` helper (Patrick Egan, Rev 1.2). ``x`` is the
    pixel-index axis ``0..N-1``; since the returned width is a difference of two
    interpolated crossing positions, the 0- vs 1-indexed origin is irrelevant.

    The polarity branch mirrors the reference: a peaked profile (``y[0] < 0.5``
    after normalization) searches outward from the argmax; otherwise it tracks a
    trough from the argmin.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y / y.max()
    n = y.size
    lev = 0.5

    # Center index and polarity (peaked vs trough), matching the reference.
    if y[0] < lev:
        center = int(np.argmax(y))
    else:
        center = int(np.argmin(y))

    # Leading crossing: first sign change of (y - lev) scanning from index 1.
    i = 1
    while i < n and np.sign(y[i] - lev) == np.sign(y[i - 1] - lev):
        i += 1
    if i >= n:
        return float("nan")
    interp = (lev - y[i - 1]) / (y[i] - y[i - 1])
    tlead = (i - 1) + interp

    # Trailing crossing: search outward starting just past the center.
    i = center + 1
    while i < n and np.sign(y[i] - lev) == np.sign(y[i - 1] - lev):
        i += 1
    if i >= n:
        # No second edge found (step-like / unresolved pulse).
        return float("nan")
    interp = (lev - y[i - 1]) / (y[i] - y[i - 1])
    ttrail = (i - 1) + interp
    return float(ttrail - tlead)


def _fwhm_psf(psf: np.ndarray) -> tuple[float, ...]:
    """Per-axis FWHM of ``psf`` in pixels, in array-axis order.

    Port of the MATLAB ``fwhm_PSF`` with ``cFlag=0`` (use the global-max voxel
    as the PSF center), ``fitFlag=0`` (no fitting), and ``pixelSize=1``. The
    1-D line profiles are moved to the host before the FWHM computation to avoid
    CuPy scalar-indexing pitfalls; the result is plain Python floats.
    """
    peak = np.unravel_index(int(np.argmax(psf)), psf.shape)
    widths: list[float] = []
    for axis in range(psf.ndim):
        # Index the peak on all other axes, full range on this axis.
        index = tuple(
            slice(None) if d == axis else int(peak[d]) for d in range(psf.ndim)
        )
        profile = asnumpy(psf[index])
        widths.append(_fwhm_1d(profile))
    return tuple(widths)


def _endpoint_indices(size: int, t: float) -> tuple[int, int]:
    """MATLAB-style cutoff endpoint indices ``(idx_minus, idx_plus)``.

    Indices address a DC-centered (``fftshift``-ed) axis, whose origin sits at
    ``size // 2``; the cutoff offset ``t`` is in Fourier pixels. MATLAB uses
    half-away-from-zero rounding (``round``), reproduced here via
    ``floor(x + 0.5)`` rather than NumPy's banker's rounding. Indices are
    clipped into ``[0, size - 1]``.
    """
    center = size // 2
    idx_minus = int(np.floor(center - t + 0.5))
    idx_plus = int(np.floor(center + t + 0.5))
    idx_minus = min(max(idx_minus, 0), size - 1)
    idx_plus = min(max(idx_plus, 0), size - 1)
    return idx_minus, idx_plus


def _cutoff_gain(magnitude_shifted: np.ndarray, axis: int, t: float) -> float:
    """Average OTF gain at the resolution cutoff along ``axis``.

    ``magnitude_shifted`` is a DC-centered magnitude array (``fftshift`` of an
    OTF magnitude). Max-project over all axes except ``axis`` to get a 1-D
    profile, then average the two endpoint values at the cutoff.
    """
    other_axes = tuple(d for d in range(magnitude_shifted.ndim) if d != axis)
    profile = (
        magnitude_shifted.max(axis=other_axes) if other_axes else magnitude_shifted
    )
    idx_minus, idx_plus = _endpoint_indices(magnitude_shifted.shape[axis], t)
    return float(asnumpy(profile[idx_minus]) + asnumpy(profile[idx_plus])) / 2.0


def _centered_grid(shape: tuple[int, ...], reference: np.ndarray) -> list[np.ndarray]:
    """Per-axis DC-centered coordinate arrays, broadcastable to ``shape``.

    The coordinate of index ``i`` along axis ``d`` is ``i - shape[d] // 2``,
    which is the convention of ``np.fft.fftshift`` / ``np.fft.ifftshift``: they
    place index ``shape[d] // 2`` at DC. The arrays are created on
    ``reference``'s device with ``reference``'s dtype.
    """
    xp = get_array_module(reference)
    grids: list[np.ndarray] = []
    for d, size in enumerate(shape):
        axis_shape = [1] * len(shape)
        axis_shape[d] = size
        grids.append(
            (xp.arange(size) - size // 2).reshape(axis_shape).astype(reference.dtype)
        )
    return grids


def _flip_about_center(array: np.ndarray) -> np.ndarray:
    """Reverse ``array`` about the fftshift origin (index ``shape[d] // 2``).

    A plain reverse sends centered index ``j`` to ``-j - 1`` on even-sized axes,
    so those axes are rolled by one. The result satisfies
    ``fftn(ifftshift(out)) == conj(fftn(ifftshift(array)))`` exactly, for both
    odd and even shapes.
    """
    flipped = array[(slice(None, None, -1),) * array.ndim]
    even_axes = tuple(d for d in range(array.ndim) if array.shape[d] % 2 == 0)
    return np.roll(flipped, 1, axis=even_axes) if even_axes else flipped


def _butterworth_mask(
    shape: tuple[int, ...],
    kc: tuple[float, ...],
    ee: float,
    n: int,
    reference: np.ndarray,
) -> np.ndarray:
    """DC-centered Butterworth mask ``1 / sqrt(1 + ee * w**n)``.

    ``w = sum_d (q_d / kc_d)**2`` is the squared ellipsoidal radius over the
    centered grid; ``reference`` supplies the output device and dtype.

    The radius is accumulated in float64 regardless of ``reference``'s dtype:
    with the default ``n = 10``, ``w ** n`` overflows float32 once ``kc`` drops
    below about 1 Fourier pixel, and ``ee == 0`` then evaluates ``0 * inf`` and
    yields NaN across the overflowing shell. The result is cast back, so a
    float32 PSF still gets a float32 mask.
    """
    grids = _centered_grid(shape, reference)
    w = (grids[0].astype(np.float64) / kc[0]) ** 2
    for d in range(1, len(shape)):
        w = w + (grids[d].astype(np.float64) / kc[d]) ** 2
    if ee == 0.0:
        # beta == 1 (or an auto beta_fp of 1) makes the mask identically one;
        # short-circuit rather than evaluate 0 * w**n.
        mask = np.ones_like(w)
    else:
        mask = 1.0 / np.sqrt(1.0 + ee * w**n)
    return mask.astype(reference.dtype)


def create_backprojector(
    psf: np.ndarray,
    bp_type: str = "wiener-butterworth",
    *,
    alpha: float = 0.05,
    beta: float = 1.0,
    n: int = 10,
    res_flag: int = 1,
    i_res: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Generate an unmatched back-projector PSF from a forward PSF.

    Device-agnostic port of Guo et al. (2020) ``BackProjector.m``. The returned
    PSF lives in the spatial domain, has the same shape, device, and float dtype
    as the input, and is normalized so that ``out.sum() == 1`` (DC gain of 1).

    Parameters
    ----------
    psf:
        Forward-projector PSF, 2-D (YX) or 3-D (ZYX). NumPy (CPU) or CuPy (GPU).
    bp_type:
        One of ``"traditional"``, ``"gaussian"``, ``"butterworth"``,
        ``"wiener"``, ``"wiener-butterworth"``.
    alpha:
        Wiener regularization parameter; must be ``> 0``, since ``alpha == 0``
        divides by zero at exact OTF nulls. A value of ``1.0`` means "auto": use
        the forward projector's average cutoff gain ``beta_fp``. Used by
        ``"wiener"`` and ``"wiener-butterworth"`` only.
    beta:
        Butterworth cutoff parameter, ``0 < beta <= 1``; ``1.0`` means "auto"
        (use ``beta_fp``). For ``"butterworth"`` it is exactly the mask gain at
        the cutoff frequency. For ``"wiener-butterworth"`` the mask is built
        from ``ee = beta_wiener / beta**2 - 1``, so the *combined* gain at the
        cutoff is ``beta * sqrt(beta_wiener)`` rather than ``beta``. Used by
        ``"butterworth"`` and ``"wiener-butterworth"`` only.
    n:
        Order (slope) of the Butterworth filter. Used by ``"butterworth"`` and
        ``"wiener-butterworth"`` only.
    res_flag:
        Resolution-limit mode. ``0``: use ``FWHM / sqrt(2)`` (e.g. iSIM);
        ``1``: use ``FWHM``; ``2``: use ``i_res`` directly. Unused by
        ``"traditional"``, which needs no resolution limit.
    i_res:
        Resolution limits in pixels per axis, in array-axis order (ZYX for 3-D).
        Required (and must be finite and positive) when ``res_flag == 2``.

    Returns
    -------
    np.ndarray
        Spatial-domain back-projector PSF (same shape/device/float dtype as
        ``psf``), normalized to sum to 1.

    Raises
    ------
    ValueError
        If ``psf.ndim`` is not 2 or 3; ``bp_type`` is unknown; ``res_flag`` is
        not 0, 1 or 2; ``res_flag == 2`` without a valid ``i_res``; ``alpha`` or
        ``beta`` is out of range; the PSF cannot be sum-normalized; the PSF FWHM
        cannot be measured when it is needed; or the assembled back projector is
        not finite.

    Notes
    -----
    Intentional divergences from the reference ``BackProjector.m``:

    * ``alpha`` defaults to ``0.05`` instead of the reference's ``0.001``. The
      stronger Wiener regularization keeps the back projector stable on
      measured (noisy) PSFs.
    * The returned PSF is sum-normalized to a DC gain of 1; the reference
      returns the raw inverse FFT. The unmatched RL loop in
      :func:`~cubic.preprocessing.richardson_lucy_xp.richardson_lucy_xp`
      divides by the reblurred estimate with no epsilon, which is only safe
      because both the forward PSF and the back projector sum to one, so this
      normalization must be kept.
    * Centered coordinate grids place the origin at index ``shape[d] // 2`` (the
      ``fftshift`` convention) instead of the reference's ``(shape[d] - 1) / 2``,
      and the flipped PSF is rolled by one voxel on even-sized axes. For even
      sizes the reference grid is not symmetric about DC, which makes the
      assembled ``otf_bp`` non-Hermitian (``real(ifftn(...))`` then discards a
      large imaginary part) and makes ``OTF_flip != conj(OTF_f)`` (shifting the
      deconvolution by one voxel per even axis). Odd-shaped PSFs are unaffected
      and stay bit-identical to the reference.
    """
    if psf.ndim not in (2, 3):
        raise ValueError(f"psf must be 2-D or 3-D, got ndim={psf.ndim}.")
    if bp_type not in _VALID_BP_TYPES:
        raise ValueError(
            f"Unknown bp_type {bp_type!r}; expected one of {_VALID_BP_TYPES}."
        )
    if res_flag not in (0, 1, 2):
        raise ValueError(f"res_flag must be 0, 1, or 2, got {res_flag}.")
    if not 0.0 < beta <= 1.0:
        raise ValueError(
            f"beta must satisfy 0 < beta <= 1 (1.0 means 'auto'); got {beta}. "
            "Larger values make the Butterworth ee term negative, which turns "
            "the mask into all-NaN."
        )
    if not alpha > 0.0:
        raise ValueError(
            f"alpha must be > 0 (1.0 means 'auto'); got {alpha}. Zero Wiener "
            "regularization divides by zero at exact OTF nulls."
        )

    # Float dtype to preserve on output (keep float32 in, float32 out).
    out_dtype = psf.dtype if np.issubdtype(psf.dtype, np.floating) else np.dtype(float)
    f = psf.astype(out_dtype)
    psf_sum = float(asnumpy(f.sum()))
    if not np.isfinite(psf_sum) or psf_sum == 0.0:
        raise ValueError(
            f"psf must have a finite, non-zero sum to be normalized; got {psf_sum}."
        )
    f = f / psf_sum  # normalized forward PSF, sum == 1

    shape = f.shape
    ndim = f.ndim

    # --- Resolution limits and frequency cutoffs (computed only when used) ---
    # ``traditional`` needs neither, and ``res_flag=2`` supplies the resolution
    # limits directly, so the FWHM is measured only for res_flag 0 and 1.
    res: tuple[float, ...] = ()
    t: tuple[float, ...] = ()
    if bp_type != "traditional":
        if res_flag == 2:
            if i_res is None or len(i_res) != ndim:
                raise ValueError(
                    "res_flag=2 requires i_res with one entry per axis "
                    f"({ndim} entries for ndim={ndim})."
                )
            res = tuple(float(r) for r in i_res)
            if not all(np.isfinite(r) and r > 0.0 for r in res):
                raise ValueError(f"i_res entries must be finite and > 0; got {res}.")
        else:
            fwhm = _fwhm_psf(f)
            # A profile with no half-maximum crossing (NaN) means the PSF is not
            # resolved within its array, so fail with a clear message rather
            # than feed NaN cutoffs downstream.
            if not all(np.isfinite(fwhm)):
                raise ValueError(
                    f"Could not estimate the PSF FWHM (got {fwhm}); the PSF may be "
                    "too wide for its array. Pass explicit cutoffs via res_flag=2 "
                    "and i_res."
                )
            res = tuple(w / np.sqrt(2.0) for w in fwhm) if res_flag == 0 else fwhm
        # Frequency cutoff in Fourier pixels per axis: t_d = S_d / res_d
        # (this is also the Butterworth kc).
        t = tuple(shape[d] / res[d] for d in range(ndim))

    # --- Assemble the back-projector PSF by type ---------------------------
    # ``gaussian`` needs neither the flipped PSF nor the Wiener filter, and
    # ``traditional`` needs no OTF at all, so those (full-size, complex)
    # intermediates are built only inside the branches that consume them.
    if bp_type == "traditional":
        # OTF_bp = OTF_flip; round-trip returns the flipped PSF.
        psf_bp = _flip_about_center(f)

    elif bp_type == "gaussian":
        # Centered Gaussian PSF with sigma_d = res_d / 2.3548.
        sigma = tuple(r / 2.3548 for r in res)
        grids = _centered_grid(shape, f)
        d2 = (grids[0] * grids[0]) / (2.0 * sigma[0] ** 2)
        for d in range(1, ndim):
            d2 = d2 + (grids[d] * grids[d]) / (2.0 * sigma[d] ** 2)
        psf_bp = np.exp(-d2)

    else:
        # wiener, butterworth, and wiener-butterworth all start from the
        # normalized flipped-PSF OTF and the forward-projector cutoff gain.
        otf_flip = np.fft.fftn(np.fft.ifftshift(_flip_about_center(f)))
        m = float(asnumpy(np.abs(otf_flip).max()))
        otf_flip_norm = otf_flip / m
        mag = np.abs(otf_flip_norm)
        flip_mag_shifted = np.fft.fftshift(mag)
        beta_fp = float(
            np.mean([_cutoff_gain(flip_mag_shifted, d, t[d]) for d in range(ndim)])
        )
        # Auto-substitution: 1.0 means "use beta_fp" (matches the reference).
        if (beta == 1.0 or alpha == 1.0) and not beta_fp > 0.0:
            raise ValueError(
                f"The forward projector has no gain (beta_fp={beta_fp}) at the "
                f"cutoff {t}, so 'auto' alpha/beta cannot be derived. Pass "
                "explicit alpha/beta values, or cutoffs that avoid the OTF nulls."
            )
        if beta == 1.0:
            beta = beta_fp
        if alpha == 1.0:
            alpha = beta_fp

        if bp_type == "butterworth":
            ee = 1.0 / beta**2 - 1.0
            otf_bp = np.fft.ifftshift(_butterworth_mask(shape, t, ee, n, f))
        else:
            # wiener / wiener-butterworth need the (complex) Wiener filter.
            otf_wiener = otf_flip_norm / (mag**2 + alpha)
            if bp_type == "wiener":
                otf_bp = otf_wiener
            else:  # wiener-butterworth
                # beta_wiener: single X-axis scalar from the Wiener OTF magnitude.
                aw = np.fft.fftshift(np.abs(otf_wiener))
                aw_plane = aw[shape[0] // 2] if ndim == 3 else aw  # central-Z -> (Y, X)
                xprof = aw_plane.max(axis=0)  # max over Y -> X profile
                idx_minus, idx_plus = _endpoint_indices(shape[-1], t[-1])
                beta_wiener = (
                    float(asnumpy(xprof[idx_minus]) + asnumpy(xprof[idx_plus])) / 2.0
                )
                ee = beta_wiener / beta**2 - 1.0
                if ee < 0.0:
                    raise ValueError(
                        f"Inconsistent cutoff: beta_wiener={beta_wiener:.6g} is "
                        f"below beta**2={beta**2:.6g}, making the Butterworth ee "
                        "term negative (all-NaN mask). Lower beta or lower alpha."
                    )
                mask = _butterworth_mask(shape, t, ee, n, f)
                otf_bp = np.fft.ifftshift(mask) * otf_wiener

        psf_bp = np.fft.fftshift(np.real(np.fft.ifftn(otf_bp)))

    # Normalize to sum == 1 (DC gain 1) and preserve the input float dtype.
    bp_sum = float(asnumpy(psf_bp.sum()))
    if not np.isfinite(bp_sum) or bp_sum == 0.0:
        raise ValueError(
            f"The assembled {bp_type!r} back projector has a sum of {bp_sum} and "
            "cannot be normalized; check the PSF and the alpha/beta/n settings."
        )
    psf_bp = psf_bp / bp_sum
    return psf_bp.astype(out_dtype)
