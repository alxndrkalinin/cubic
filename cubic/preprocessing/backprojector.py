"""Unmatched back-projector PSF generation for Richardson-Lucy deconvolution.

This module is a device-agnostic (NumPy/CuPy) port of the reference MATLAB
``BackProjector.m`` from Guo et al. (2020), "Rapid image deconvolution and
multiview fusion for optical microscopy" (Nature Biotechnology). It builds a
spatial-domain back-projector PSF from a forward PSF, enabling the
Wiener-Butterworth (WB) unmatched back projector that accelerates RL
convergence.

The same function works on NumPy arrays (CPU) or CuPy arrays (GPU) based on the
input array's device location; no code changes are required to switch devices.

Reference: Guo, M. et al. Nat Biotechnol 38, 1337-1346 (2020).
"""

from typing import Any

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

    Centered coordinate of index ``idx`` is ``idx - (size - 1) / 2``; the cutoff
    offset ``t`` is in Fourier pixels. MATLAB uses half-away-from-zero rounding
    (``round``), reproduced here via ``floor(x + 0.5)`` rather than NumPy's
    banker's rounding. Indices are clipped into ``[0, size - 1]``.
    """
    center = (size - 1) / 2.0
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
    # max is order-independent, so a single multi-axis reduction is exact.
    profile = (
        magnitude_shifted.max(axis=other_axes) if other_axes else magnitude_shifted
    )
    idx_minus, idx_plus = _endpoint_indices(magnitude_shifted.shape[axis], t)
    return float(asnumpy(profile[idx_minus]) + asnumpy(profile[idx_plus])) / 2.0


def _butterworth_mask(
    shape: tuple[int, ...],
    kc: tuple[float, ...],
    ee: float,
    n: int,
    xp: Any,
    dtype: Any,
) -> np.ndarray:
    """DC-centered Butterworth mask ``1 / sqrt(1 + ee * w**n)``.

    ``w = sum_d (q_d / kc_d)**2`` is the squared ellipsoidal radius with centered
    coordinates ``q_d = idx - (S_d - 1) / 2`` built on the input device via ``xp``.
    """
    ndim = len(shape)
    w = xp.zeros(shape, dtype=dtype)
    for d in range(ndim):
        axis_shape = [1] * ndim
        axis_shape[d] = shape[d]
        q = (
            (xp.arange(shape[d]) - (shape[d] - 1) / 2.0)
            .reshape(axis_shape)
            .astype(dtype)
        )
        w = w + (q / kc[d]) ** 2
    return 1.0 / np.sqrt(1.0 + ee * w**n)


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
        Wiener regularization parameter. A value of ``1.0`` means "auto": use the
        forward projector's average cutoff gain ``beta_fp``. The default
        ``0.05`` is used as-is (it is not ``1.0``).
    beta:
        Target cutoff gain of the back projector. A value of ``1.0`` means
        "auto": use ``beta_fp`` (the default behavior).
    n:
        Order (slope) of the Butterworth filter.
    res_flag:
        Resolution-limit mode. ``0``: use ``FWHM / sqrt(2)`` (e.g. iSIM);
        ``1``: use ``FWHM``; ``2``: use ``i_res`` directly.
    i_res:
        Resolution limits in pixels per axis, in array-axis order (ZYX for 3-D).
        Required when ``res_flag == 2``.

    Returns
    -------
    np.ndarray
        Spatial-domain back-projector PSF (same shape/device/float dtype as
        ``psf``), normalized to sum to 1.

    Raises
    ------
    ValueError
        If ``psf.ndim`` is not 2 or 3, ``bp_type`` is unknown, or
        ``res_flag == 2`` without a valid ``i_res``.
    """
    if psf.ndim not in (2, 3):
        raise ValueError(f"psf must be 2-D or 3-D, got ndim={psf.ndim}.")
    if bp_type not in _VALID_BP_TYPES:
        raise ValueError(
            f"Unknown bp_type {bp_type!r}; expected one of {_VALID_BP_TYPES}."
        )

    xp = get_array_module(psf)
    # Float dtype to preserve on output (keep float32 in, float32 out).
    out_dtype = psf.dtype if np.issubdtype(psf.dtype, np.floating) else np.dtype(float)
    f = psf.astype(out_dtype)
    f = f / f.sum()  # normalized forward PSF, sum == 1

    shape = f.shape
    ndim = f.ndim

    # --- Per-axis FWHM (host floats) and resolution cutoffs ----------------
    fwhm = _fwhm_psf(f)
    # FWHM is unused only for res_flag=2 with a non-Gaussian projector; otherwise
    # a profile with no half-maximum crossing (NaN) means the PSF is not resolved
    # within its array, so fail with a clear message rather than downstream NaN.
    if (bp_type == "gaussian" or res_flag != 2) and not all(np.isfinite(fwhm)):
        raise ValueError(
            f"Could not estimate the PSF FWHM (got {fwhm}); the PSF may be too "
            "wide for its array. Pass explicit cutoffs via res_flag=2 and i_res."
        )
    if res_flag == 0:
        res = tuple(w / np.sqrt(2.0) for w in fwhm)
    elif res_flag == 1:
        res = fwhm
    elif res_flag == 2:
        if i_res is None or len(i_res) != ndim:
            raise ValueError(
                "res_flag=2 requires i_res with one entry per axis "
                f"({ndim} entries for ndim={ndim})."
            )
        res = tuple(float(r) for r in i_res)
    else:
        raise ValueError(f"res_flag must be 0, 1, or 2, got {res_flag}.")

    # Frequency cutoff in Fourier pixels per axis: t_d = S_d / res_d (= Butterworth kc).
    t = tuple(shape[d] / res[d] for d in range(ndim))

    flipped = f[(slice(None, None, -1),) * ndim]

    # --- Assemble the back-projector PSF by type ---------------------------
    # ``traditional`` and ``gaussian`` use neither the flipped OTF nor the
    # Wiener filter, so those (full-size, complex) intermediates are built only
    # inside the branches that consume them.
    if bp_type == "traditional":
        # OTF_bp = OTF_flip; round-trip returns the flipped PSF.
        psf_bp = flipped.copy()

    elif bp_type == "gaussian":
        # Centered Gaussian PSF with sigma_d = FWHM_d / 2.3548.
        sigma = tuple(w / 2.3548 for w in fwhm)
        d2 = xp.zeros(shape, dtype=out_dtype)
        for d in range(ndim):
            axis_shape = [1] * ndim
            axis_shape[d] = shape[d]
            q = (xp.arange(shape[d]) - (shape[d] - 1) / 2.0).reshape(axis_shape)
            q = q.astype(out_dtype)
            d2 = d2 + (q * q) / (2.0 * sigma[d] ** 2)
        psf_bp = np.exp(-d2)

    else:
        # wiener, butterworth, and wiener-butterworth all start from the
        # normalized flipped-PSF OTF and the forward-projector cutoff gain.
        otf_flip = np.fft.fftn(np.fft.ifftshift(flipped))
        m = float(asnumpy(np.abs(otf_flip).max()))
        otf_flip_norm = otf_flip / m
        mag = np.abs(otf_flip_norm)
        flip_mag_shifted = np.fft.fftshift(mag)
        beta_fp = float(
            np.mean([_cutoff_gain(flip_mag_shifted, d, t[d]) for d in range(ndim)])
        )
        # Auto-substitution: 1.0 means "use beta_fp" (matches the reference).
        if beta == 1.0:
            beta = beta_fp
        if alpha == 1.0:
            alpha = beta_fp

        if bp_type == "butterworth":
            ee = 1.0 / beta**2 - 1.0
            otf_bp = np.fft.ifftshift(_butterworth_mask(shape, t, ee, n, xp, out_dtype))
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
                mask = _butterworth_mask(shape, t, ee, n, xp, out_dtype)
                otf_bp = np.fft.ifftshift(mask) * otf_wiener

        psf_bp = np.fft.fftshift(np.real(np.fft.ifftn(otf_bp)))

    # Normalize to sum == 1 (DC gain 1) and preserve the input float dtype.
    psf_bp = psf_bp / psf_bp.sum()
    return psf_bp.astype(out_dtype)
