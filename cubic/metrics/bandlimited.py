"""Band-limited similarity metrics for virtual staining evaluation.

Virtual staining models predict fluorescence from label-free inputs.
Fluorescence targets contain measurement noise and deconvolution artefacts,
so direct pixel-wise metrics (PCC, SSIM) over-penalise models for *not*
reproducing noise.  The solution is to compare images only over spatial
frequencies that carry reliable biological information, by applying a soft
low-pass filter before computing metrics.

This module provides:

* **Butterworth low-pass filter** in the frequency domain.
* **OTF / Nyquist / data-driven cutoff estimation**.
* **Radial power spectrum** and **noise-floor estimation**.
* **Band-limited PCC and SSIM** (hard-cutoff filtering).
* **Spectral PCC** (soft per-frequency weighting).

All functions are device-agnostic (NumPy / CuPy).

References
----------
Descloux, A., et al. (2019). Parameter-free image resolution estimation
    based on decorrelation analysis. *Nature Methods* 16, 918-924.
"""

from __future__ import annotations

import warnings
from typing import Literal
from collections.abc import Callable, Sequence

import numpy as np

from cubic.cuda import asnumpy, to_same_device, check_same_device
from cubic.image_utils import tukey_window, hamming_window

from .spectral.dcr import dcr_resolution
from .spectral.frc import frc_resolution, fsc_resolution
from .skimage_metrics import ssim as _ssim
from .spectral.radial import (
    radial_edges,
    reduce_power,
    radial_bin_id,
    radial_k_grid,
)

__all__ = [
    "butterworth_lowpass",
    "otf_cutoff",
    "nyquist_cutoff",
    "estimate_cutoff",
    "radial_power_spectrum",
    "estimate_noise_floor",
    "spectral_weights",
    "smooth_spectral_weights",
    "frc_weights",
    "spectral_pcc_frcw",
    "percentile_band_taper",
    "bandpass_spectral_pcc",
    "band_limited_pcc",
    "band_limited_ssim",
    "spectral_pcc",
]

# ---------------------------------------------------------------------------
# 1  Core building blocks
# ---------------------------------------------------------------------------


def butterworth_lowpass(
    shape: tuple[int, ...],
    cutoff: float,
    spacing: Sequence[float] | None = None,
    order: int = 2,
) -> np.ndarray:
    """Frequency-domain Butterworth low-pass filter.

    Parameters
    ----------
    shape : tuple of int
        Spatial dimensions of the image (2-D or 3-D).
    cutoff : float
        Cutoff frequency in the same units as *spacing* (cycles / length).
        If *spacing* is ``None``, cutoff is in index-frequency units.
    spacing : sequence of float, optional
        Physical pixel/voxel spacing per axis.
    order : int
        Filter order (steepness).  Default 2.

    Returns
    -------
    H : np.ndarray
        Real-valued filter in **unshifted** FFT layout.  H(DC) = 1 and
        values are in [0, 1].
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    if order < 1:
        raise ValueError("order must be >= 1")

    k_radius, _ = radial_k_grid(shape, spacing=spacing)
    H = 1.0 / (1.0 + (k_radius / cutoff) ** (2 * order))
    return H.astype(np.float32)


# ---------------------------------------------------------------------------
# 2  Cutoff estimation helpers
# ---------------------------------------------------------------------------


def otf_cutoff(
    numerical_aperture: float,
    wavelength_emission: float,
    modality: str = "widefield",
    medium_refractive_index: float = 1.515,
) -> float:
    """Lateral OTF cutoff frequency.

    Parameters
    ----------
    numerical_aperture : float
        Objective NA.
    wavelength_emission : float
        Emission wavelength in the **same length unit** as *spacing*
        (typically micrometres).
    modality : ``"widefield"`` | ``"confocal"`` | ``"lightsheet"``
        Imaging modality.  ``"widefield"`` and ``"lightsheet"`` use
        ``2 NA / λ``; ``"confocal"`` uses ``4 NA / λ``.
    medium_refractive_index : float
        Immersion medium refractive index (unused in current formula but
        reserved for future axial-OTF extension).

    Returns
    -------
    float
        Cutoff frequency in cycles / length.
    """
    modality = modality.lower()
    if modality in ("widefield", "lightsheet"):
        multiplier = 2.0
    elif modality == "confocal":
        multiplier = 4.0
    else:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            "Choose from 'widefield', 'confocal', 'lightsheet'."
        )
    return multiplier * numerical_aperture / wavelength_emission


def nyquist_cutoff(spacing: float | Sequence[float]) -> float:
    """Most conservative (coarsest-axis) Nyquist cutoff.

    Parameters
    ----------
    spacing : float or sequence of float
        Physical pixel / voxel spacing.

    Returns
    -------
    float
        ``0.5 / max(spacing)`` — the Nyquist frequency for the coarsest
        axis.
    """
    if isinstance(spacing, (int, float)):
        return 0.5 / float(spacing)
    return 0.5 / float(max(spacing))


def estimate_cutoff(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float],
    numerical_aperture: float | None = None,
    wavelength_emission: float | None = None,
    modality: str = "widefield",
    method: str = "dcr",
    dcr_safety: float = 1.0,
    otf_safety: float = 0.95,
    nyquist_safety: float = 0.9,
    frc_safety: float = 1.0,
    dcr_kwargs: dict | None = None,
    frc_kwargs: dict | None = None,
) -> float:
    """Data-driven cutoff frequency estimation.

    Computes up to four independent bounds and returns their minimum:

    * **DCR bound** — ``dcr_safety / dcr_resolution(image)`` (data-driven).
    * **FRC/FSC bound** — ``frc_safety / frc_resolution(image)`` (data-driven).
    * **OTF bound** — ``otf_safety * otf_cutoff(NA, λ)`` (physics).
    * **Nyquist bound** — ``nyquist_safety * nyquist_cutoff(spacing)``.

    Bounds whose required parameters are absent are silently skipped.
    Data-driven bounds (DCR, FRC/FSC) that raise ``ValueError``,
    ``RuntimeError``, or ``TypeError`` emit a warning and are excluded;
    other exceptions propagate.  At least one bound must be computable.

    Parameters
    ----------
    image : np.ndarray
        2-D or 3-D image used for data-driven estimates.
    spacing : float or sequence of float
        Physical pixel / voxel spacing.
    numerical_aperture, wavelength_emission : float, optional
        Objective NA and emission wavelength (same length unit as
        *spacing*).  Both must be given for the OTF bound.
    modality : str
        Passed to :func:`otf_cutoff`.
    method : ``"dcr"`` | ``"frc"`` | ``"both"``
        Data-driven estimation method.  ``"dcr"`` (default) uses
        decorrelation analysis; ``"frc"`` uses FRC (2-D) or FSC (3-D);
        ``"both"`` computes both and takes the minimum.
    dcr_safety, otf_safety, nyquist_safety, frc_safety : float
        Safety factors applied to each bound.
    dcr_kwargs : dict, optional
        Extra keyword arguments forwarded to ``dcr_resolution``.
    frc_kwargs : dict, optional
        Extra keyword arguments forwarded to ``frc_resolution`` /
        ``fsc_resolution``.  For 3-D inputs, ``zero_padding=True`` and
        ``resample_isotropic=True`` (when spacing is anisotropic) are set
        by default unless explicitly overridden here.

    Returns
    -------
    float
        Recommended low-pass cutoff frequency.

    Raises
    ------
    ValueError
        If no bound can be computed (should not happen — Nyquist is always
        available).
    """
    bounds: list[float] = []

    # --- DCR (data-driven) bound ---
    if method in ("dcr", "both"):
        try:
            dcr_kw = dict(dcr_kwargs) if dcr_kwargs else {}
            dcr_res = dcr_resolution(image, spacing=spacing, **dcr_kw)
            if isinstance(dcr_res, dict):
                # 3-D: use XY resolution
                dcr_val = dcr_res.get("xy", float("inf"))
            else:
                dcr_val = dcr_res
            if np.isfinite(dcr_val) and dcr_val > 0:
                bounds.append(dcr_safety / dcr_val)
        except (ValueError, RuntimeError, TypeError) as exc:
            warnings.warn(f"DCR resolution estimation failed: {exc}", stacklevel=2)

    # --- FRC / FSC (data-driven) bound ---
    if method in ("frc", "both"):
        try:
            frc_kw = dict(frc_kwargs) if frc_kwargs else {}
            if "spacing" not in frc_kw:
                frc_kw["spacing"] = spacing
            if image.ndim == 2:
                frc_res = frc_resolution(image, **frc_kw)
            elif image.ndim == 3:
                # FSC defaults: hist backend uses zero_padding=False, but
                # padding improves shell-binning accuracy; isotropic
                # resampling helps anisotropic volumes (e.g. Z >> XY spacing).
                if "zero_padding" not in frc_kw:
                    frc_kw["zero_padding"] = True
                if "resample_isotropic" not in frc_kw:
                    frc_kw["resample_isotropic"] = _is_anisotropic(spacing)

                frc_res = fsc_resolution(image, **frc_kw).get("xy", float("nan"))
            else:
                frc_res = float("nan")
            if np.isfinite(frc_res) and frc_res > 0:
                bounds.append(frc_safety / frc_res)
        except (ValueError, RuntimeError, TypeError) as exc:
            warnings.warn(f"FRC/FSC resolution estimation failed: {exc}", stacklevel=2)

    # --- OTF (physics) bound ---
    if numerical_aperture is not None and wavelength_emission is not None:
        f_otf = otf_cutoff(numerical_aperture, wavelength_emission, modality=modality)
        bounds.append(otf_safety * f_otf)

    # --- Nyquist bound ---
    f_nyq = nyquist_cutoff(spacing)
    bounds.append(nyquist_safety * f_nyq)

    if not bounds:
        raise ValueError("Could not compute any cutoff bound.")

    return float(min(bounds))


# ---------------------------------------------------------------------------
# 3  Spectral analysis helpers
# ---------------------------------------------------------------------------


def radial_power_spectrum(
    image: np.ndarray,
    spacing: float | Sequence[float] | None = None,
    bin_delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial (azimuthally-averaged) power spectrum.

    Parameters
    ----------
    image : np.ndarray
        2-D or 3-D input.
    spacing : float or sequence of float, optional
        Physical spacing.
    bin_delta : float
        Bin width in index-frequency units.

    Returns
    -------
    radii : np.ndarray
        Bin centres (physical frequency if *spacing* given).
    mean_power : np.ndarray
        Mean |F|² per radial bin (DC excluded).
    """
    spacing_seq: Sequence[float] | None
    if spacing is None:
        spacing_seq = None
    elif isinstance(spacing, (int, float)):
        spacing_seq = [float(spacing)] * image.ndim
    else:
        spacing_seq = list(spacing)

    F = np.fft.fftn(image.astype(np.float32))
    edges_cpu, radii = radial_edges(
        image.shape, bin_delta=bin_delta, spacing=spacing_seq
    )

    # Move edges to same device as image for correct bin_id device placement
    edges = to_same_device(edges_cpu, image)
    bid = radial_bin_id(image.shape, edges, spacing=spacing_seq)
    S2, N = reduce_power(F, bid)

    # Mean power per bin (avoid /0)
    N_safe = np.maximum(N.astype(np.float64), 1.0)
    mean_power = (S2 / N_safe).astype(np.float32)

    return asnumpy(radii).astype(np.float32), asnumpy(mean_power).astype(np.float32)


def estimate_noise_floor(
    radii: np.ndarray,
    power: np.ndarray,
    tail_fraction: float = 0.2,
) -> float:
    """Estimate the spectral noise floor from the high-frequency tail.

    Parameters
    ----------
    radii : np.ndarray
        Radial-bin centres (from :func:`radial_power_spectrum`).
    power : np.ndarray
        Mean power per bin.
    tail_fraction : float
        Fraction of the highest-frequency bins to average.

    Returns
    -------
    float
        Estimated noise-floor power.
    """
    n = len(power)
    n_tail = max(1, int(np.ceil(n * tail_fraction)))
    return float(np.mean(power[-n_tail:]))


def spectral_weights(
    radii: np.ndarray,
    power: np.ndarray,
    noise_floor: float,
    cutoff: float | None = None,
) -> np.ndarray:
    """Per-bin spectral weights: signal power above noise floor.

    ``w[i] = max(0, P[i] - noise) / max_j(max(0, P[j] - noise))``.
    Bins above *cutoff* (if given) are set to zero.

    Parameters
    ----------
    radii : np.ndarray
        Radial-bin centres.
    power : np.ndarray
        Mean power per bin.
    noise_floor : float
        Noise-floor estimate from :func:`estimate_noise_floor`.
    cutoff : float, optional
        Hard cutoff frequency.

    Returns
    -------
    np.ndarray
        Weights in [0, 1] with ``max(w) == 1``.
    """
    w = np.maximum(power - noise_floor, 0.0)
    if cutoff is not None:
        w[radii > cutoff] = 0.0
    w_max = float(np.max(w))
    if w_max > 0:
        w = w / w_max
    return w.astype(np.float32)


def smooth_spectral_weights(
    radii: np.ndarray,
    power: np.ndarray,
    noise_floor: float,
    cutoff: float | None = None,
    sg_window: int = 15,
    sg_polyorder: int = 3,
) -> np.ndarray:
    """Smooth Wiener-style weights from SG-filtered log-power spectrum.

    Savitzky-Golay-filters ``log(P)`` to remove per-bin variance, then
    applies Wiener weighting: ``P_smooth² / (P_smooth² + N²)``.

    Parameters
    ----------
    radii : np.ndarray
        Radial-bin centres.
    power : np.ndarray
        Mean power per bin.
    noise_floor : float
        Noise-floor estimate from :func:`estimate_noise_floor`.
    cutoff : float, optional
        Hard cutoff frequency.
    sg_window : int
        Savitzky-Golay window length (must be odd; clamped to array size).
    sg_polyorder : int
        Savitzky-Golay polynomial order.

    Returns
    -------
    np.ndarray
        Weights in [0, 1].
    """
    from cubic.scipy import signal as csignal

    log_p = np.log(np.maximum(power, 1e-30))
    # DC bin (index 0) is always zero after mean subtraction, mapping to
    # log(1e-30) ≈ -69.  Replace with neighbor before SG smoothing to
    # prevent the extreme outlier from poisoning the first few fitted values.
    if len(log_p) > 1:
        log_p[0] = log_p[1]
    n = len(log_p)
    # Clamp window to array length (must be odd, > polyorder, <= n)
    min_wlen = sg_polyorder + 2
    if n < min_wlen:
        # Array too short for SG filter — skip smoothing
        p_smooth = power.copy()
    else:
        wlen = min(sg_window, n)
        if wlen % 2 == 0:
            wlen -= 1
        wlen = max(wlen, min_wlen)
        log_p_smooth = csignal.savgol_filter(log_p, wlen, sg_polyorder)
        p_smooth = np.exp(log_p_smooth)

    n2 = noise_floor**2
    w = p_smooth**2 / (p_smooth**2 + n2)

    if cutoff is not None:
        w[radii > cutoff] = 0.0
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# 3b  Baseline-corrected spectral weights
# ---------------------------------------------------------------------------


def _running_quantile_1d(
    arr: np.ndarray,
    window: int = 11,
    q: float = 0.1,
) -> np.ndarray:
    """Sliding-window quantile for a 1-D array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (typically ~100 elements).
    window : int
        Window size (must be odd and >= 3).
    q : float
        Quantile in [0, 1].

    Returns
    -------
    np.ndarray
        Same length as *arr*, with per-element local quantile.
    """
    if window % 2 == 0:
        window += 1
    window = max(window, 3)
    n = len(arr)
    half = window // 2
    out = np.empty(n, dtype=arr.dtype)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.quantile(arr[lo:hi], q)
    return out


def _estimate_noise_baseline(
    power: np.ndarray,
    sg_window: int = 15,
    sg_polyorder: int = 3,
    quantile_window: int = 11,
    quantile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-dependent noise baseline from smoothed power spectrum.

    Parameters
    ----------
    power : np.ndarray
        Mean power per radial bin (from :func:`radial_power_spectrum`).
    sg_window : int
        Savitzky-Golay window for power smoothing.
    sg_polyorder : int
        Savitzky-Golay polynomial order.
    quantile_window : int
        Window size for running low-quantile (must be odd, >= 3).
    quantile : float
        Quantile for baseline estimation (e.g. 0.1 = 10th percentile).

    Returns
    -------
    p_smooth : np.ndarray
        SG-smoothed power spectrum.
    noise_baseline : np.ndarray
        Frequency-dependent noise floor N(k), same length as *power*.
    """
    from cubic.scipy import signal as csignal

    n = len(power)

    # SG window: must be odd, > polyorder, <= array length
    min_wlen = sg_polyorder + 2
    if n < min_wlen:
        # Array too short for SG filter — return raw power as baseline
        raw = np.maximum(power, 1e-30).astype(np.float32)
        return raw, raw.copy()
    wlen = min(sg_window, n)
    if wlen % 2 == 0:
        wlen -= 1
    wlen = max(wlen, min_wlen)

    log_p = np.log(np.maximum(power, 1e-30))
    # DC bin (index 0) is always zero after mean subtraction → log(1e-30) ≈ -69.
    # Replace with neighbor before SG to prevent boundary poisoning.
    if len(log_p) > 1:
        log_p[0] = log_p[1]
    log_p_smooth = csignal.savgol_filter(log_p, wlen, sg_polyorder)

    # Running low-quantile of smoothed log-power
    log_n = _running_quantile_1d(log_p_smooth, window=quantile_window, q=quantile)

    # Guardrail: baseline must not exceed smoothed spectrum
    log_n = np.minimum(log_n, log_p_smooth)

    # Monotone non-increasing constraint: baseline should not rise with k.
    # Propagate maximum from high-k backward so earlier bins are >= later bins.
    log_n = np.maximum.accumulate(log_n[::-1])[::-1]

    return np.exp(log_p_smooth).astype(np.float32), np.exp(log_n).astype(np.float32)


def _baseline_snr2_weights(
    radii: np.ndarray,
    p_smooth: np.ndarray,
    noise_baseline: np.ndarray,
    nbins_low: int = 3,
    cap_quantile: float = 0.99,
    cutoff: float | None = None,
) -> np.ndarray:
    """SNR² weights with frequency-dependent noise baseline.

    Parameters
    ----------
    radii : np.ndarray
        Radial-bin centres.
    p_smooth : np.ndarray
        Smoothed power spectrum from :func:`_estimate_noise_baseline`.
    noise_baseline : np.ndarray
        Frequency-dependent noise floor N(k) from :func:`_estimate_noise_baseline`.
    nbins_low : int
        Number of lowest-frequency bins to exclude (DC/background).
    cap_quantile : float
        Soft cap: clamp weights above this quantile of nonzero weights.
    cutoff : float, optional
        Hard cutoff frequency.

    Returns
    -------
    np.ndarray
        Unnormalized SNR² weights (zero where excluded).
    """
    if len(radii) <= nbins_low:
        return np.zeros_like(radii, dtype=np.float32)

    snr = p_smooth / np.maximum(noise_baseline, 1e-30)
    # Cap SNR to prevent overflow in squaring (SNR > 1e4 is extreme)
    snr = np.minimum(snr, 1e4)
    w = np.maximum(snr - 1.0, 0.0) ** 2

    # Exclude lowest bins (DC / background / autofluorescence)
    w[:nbins_low] = 0.0

    # Hard cutoff
    if cutoff is not None:
        w[radii > cutoff] = 0.0

    # Soft cap to prevent single ultra-low bin from dominating
    nonzero = w[w > 0]
    if len(nonzero) > 0:
        cap = float(np.quantile(nonzero, cap_quantile))
        w = np.minimum(w, cap)

    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# 3c  Percentile-band taper
# ---------------------------------------------------------------------------


def percentile_band_taper(
    w_bins: np.ndarray,
    n_bins: np.ndarray,
    radii: np.ndarray,
    k_nyquist: float,
    p_low: float = 0.0,
    p_high: float = 0.99,
    taper_width: int = 3,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply percentile-based band selection with cosine taper to spectral weights.

    Selects a frequency band by finding the radial bins where the
    cumulative weight mass reaches *p_low* and *p_high*, then applies a
    soft cosine taper outside the band to avoid Gibbs-like ringing.

    The weight mass per bin is ``w_bins[i] * n_bins[i]``, reflecting
    each ring's actual contribution to the weighted PCC sum (rings at
    higher *k* contain more Fourier pixels).

    Parameters
    ----------
    w_bins : np.ndarray
        1-D per-bin weights (from any weight law).
    n_bins : np.ndarray
        1-D per-bin Fourier pixel counts (same length as *w_bins*).
    radii : np.ndarray
        1-D radial-bin centres (same length as *w_bins*).
    k_nyquist : float
        Nyquist frequency for normalising diagnostics.
    p_low : float
        Lower percentile of cumulative weight mass (default 0.0 = no
        low-frequency exclusion).
    p_high : float
        Upper percentile of cumulative weight mass (default 0.99).
    taper_width : int
        Number of bins for cosine ramp on each side of the band
        (clamped to available bins at boundaries).

    Returns
    -------
    w_tapered : np.ndarray
        Tapered weights (float32, same length as *w_bins*).
    diagnostics : dict[str, float]
        Band diagnostic info:

        - ``k_low``, ``k_high``: band edges as fraction of Nyquist.
        - ``k_low_phys``, ``k_high_phys``: band edges in physical
          frequency units (same units as *radii*).
        - ``k50``, ``k90``: 50th / 90th percentile of cumulative mass
          as fraction of Nyquist.
        - ``i_low``, ``i_high``: band-edge bin indices.
    """
    nbins = len(w_bins)
    nan_diag: dict[str, float] = {
        "k_low": float("nan"),
        "k_high": float("nan"),
        "k_low_phys": float("nan"),
        "k_high_phys": float("nan"),
        "k50": float("nan"),
        "k90": float("nan"),
        "i_low": float("nan"),
        "i_high": float("nan"),
    }

    # Cumulative weight mass
    m = np.asarray(w_bins, dtype=np.float64) * np.asarray(n_bins, dtype=np.float64)
    total = float(np.sum(m))
    if total < 1e-30:
        warnings.warn(
            "Total weight mass is ~0; returning zero weights.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(nbins, dtype=np.float32), nan_diag

    cum = np.cumsum(m) / total

    # Cut indices
    i_low = int(np.searchsorted(cum, p_low))
    i_high = int(np.searchsorted(cum, p_high))
    i_low = max(0, min(i_low, nbins - 1))
    i_high = max(0, min(i_high, nbins - 1))

    if i_low >= i_high:
        warnings.warn(
            f"Empty band: i_low={i_low} >= i_high={i_high}; returning zero weights.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros(nbins, dtype=np.float32), nan_diag

    # Build cosine taper
    taper = np.zeros(nbins, dtype=np.float64)
    taper[i_low : i_high + 1] = 1.0

    # Low-side ramp (clamp width to available bins below i_low)
    lo_w = min(taper_width, i_low)
    if lo_w > 0:
        t = np.arange(1, lo_w + 1, dtype=np.float64)
        taper[i_low - lo_w : i_low] = 0.5 * (1.0 - np.cos(np.pi * t / lo_w))

    # High-side ramp (clamp width to available bins above i_high)
    hi_w = min(taper_width, nbins - 1 - i_high)
    if hi_w > 0:
        t = np.arange(1, hi_w + 1, dtype=np.float64)
        taper[i_high + 1 : i_high + 1 + hi_w] = 0.5 * (1.0 + np.cos(np.pi * t / hi_w))

    w_tapered = (np.asarray(w_bins, dtype=np.float64) * taper).astype(np.float32)

    # Diagnostics
    radii_f = np.asarray(radii, dtype=np.float64)
    k_nyq = max(k_nyquist, 1e-30)

    def _cum_percentile_k(q: float) -> float:
        idx = int(np.searchsorted(cum, q))
        idx = max(0, min(idx, nbins - 1))
        return float(radii_f[idx] / k_nyq)

    diagnostics: dict[str, float] = {
        "k_low": float(radii_f[i_low] / k_nyq),
        "k_high": float(radii_f[i_high] / k_nyq),
        "k_low_phys": float(radii_f[i_low]),
        "k_high_phys": float(radii_f[i_high]),
        "k50": _cum_percentile_k(0.50),
        "k90": _cum_percentile_k(0.90),
        "i_low": float(i_low),
        "i_high": float(i_high),
    }

    return w_tapered, diagnostics


# ---------------------------------------------------------------------------
# 3b  FRC-weighted spectral PCC
# ---------------------------------------------------------------------------


def frc_weights(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    threshold: float = 0.143,
    alpha: float = 2.0,
    nbins_low: int = 3,
    smooth_window: int = 5,
    split_type: Literal["checkerboard", "binomial"] = "binomial",
    n_repeats: int = 3,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Per-bin weights derived from single-image FRC reproducibility.

    Uses binomial splitting (same-shape, full frequency coverage) by
    default, falling back to checkerboard if requested.

    Parameters
    ----------
    image : np.ndarray
        2-D image (must be square).
    bin_delta : int
        Radial-bin width in index units.
    threshold : float
        FRC threshold (default 0.143 = 1-bit).
    alpha : float
        Weight exponent (default 2.0 — sharpens the transition).
    nbins_low : int
        Number of lowest bins to zero (DC / background exclusion).
    smooth_window : int
        Median-filter window (clamped to odd, >= 3).
    split_type : str
        ``"binomial"`` (default, same-shape, Rieger et al. 2024) or
        ``"checkerboard"`` (subsampled halves, Koho et al. 2019).
    n_repeats : int
        Number of independent binomial splits to average (default 3
        for stability; lower-level ``calculate_frc`` defaults to 1).
        Ignored for checkerboard.
    rng : Generator, int, or None
        Random seed for binomial split reproducibility.

    Returns
    -------
    np.ndarray
        1-D float32 weight array (one per radial bin, index-unit
        binning matching ``radial_edges(image.shape, bin_delta,
        spacing=None)``).
    """
    from cubic.scipy import ndimage as cndimage

    # --- lazy import to avoid circular deps ---
    from .spectral.frc import calculate_frc as _calculate_frc

    if image.ndim != 2:
        raise ValueError("frc_weights currently supports 2-D images only.")
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"frc_weights requires square images, got {image.shape}.")

    # 1. FRC curve via public API (no spacing → index units)
    frc_kwargs = dict(
        image2=None,
        backend="hist",
        bin_delta=bin_delta,
        zero_padding=False,
        disable_hamming=False,
        split_type=split_type,
    )
    if split_type == "binomial":
        frc_kwargs.update(
            counts_mode="poisson_thinning",
            n_repeats=n_repeats,
            rng=rng,  # type: ignore[arg-type]
            average=False,  # no checkerboard reverse averaging
        )
    else:
        frc_kwargs["average"] = True  # checkerboard forward+reverse

    result = _calculate_frc(image, **frc_kwargs)  # type: ignore[arg-type]
    frc_curve = np.clip(
        np.asarray(result.correlation["correlation"], dtype=np.float64),
        -1.0,
        1.0,
    )
    freq_norm = np.asarray(
        result.correlation["frequency"],
        dtype=np.float64,
    )

    # 2. Full-resolution radii in index units
    _, radii_full_idx = radial_edges(
        image.shape,
        bin_delta=bin_delta,
        spacing=None,
    )
    freq_nyq_full = float(np.floor(image.shape[0] / 2.0))
    freq_full_norm = radii_full_idx / freq_nyq_full

    # 3. Map FRC frequency axis onto full-resolution bins
    if split_type == "checkerboard":
        # Checkerboard halves are shape//2 → FRC [0,1] covers half Nyquist
        freq_nyq_half = float(np.floor(image.shape[0] // 2 / 2.0))
        freq_scale = freq_nyq_full / freq_nyq_half
        interp_x = np.clip(freq_full_norm * freq_scale, 0.0, 1.0)
    else:
        # Binomial split: same shape → FRC and full bins share the same axis
        interp_x = freq_full_norm

    frc_full = np.interp(
        interp_x,
        freq_norm,
        frc_curve,
        left=float(frc_curve[0]),
        right=float(frc_curve[-1]),
    )

    # 4. Convert to weights
    w = (
        np.clip(
            (frc_full - threshold) / (1.0 - threshold),
            0.0,
            1.0,
        )
        ** alpha
    )

    # 5. Smooth + monotone non-increasing envelope
    sw = smooth_window | 1  # clamp to odd
    sw = max(3, min(sw, len(w) | 1))
    w = cndimage.median_filter(w, size=sw).astype(np.float64)
    w = np.maximum.accumulate(w[::-1])[::-1].copy()

    # 6. Low-k exclusion
    w[:nbins_low] = 0.0

    if not ((w >= 0).all() and (w <= 1.0 + 1e-7).all()):
        raise ValueError(f"Weights out of range [0, 1]: min={w.min()}, max={w.max()}")
    return w.astype(np.float32)


def spectral_pcc_frcw(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    bin_delta: int = 1,
    apodization: str = "tukey",
    threshold: float = 0.143,
    alpha: float = 2.0,
    nbins_low: int = 3,
    smooth_window: int = 5,
    split_type: Literal["checkerboard", "binomial"] = "binomial",
    n_repeats: int = 3,
    rng: np.random.Generator | int | None = None,
    frozen_weights: np.ndarray | None = None,
) -> float:
    """Spectral PCC weighted by single-image FRC reproducibility.

    FRC weights are computed in index-frequency units (spacing is not
    needed — the threshold crossing in normalised frequency is
    independent of pixel size).

    Parameters
    ----------
    prediction, target : np.ndarray
        Images to compare (same shape, 2-D).
    bin_delta : int
        Radial-bin width in index units.
    apodization : str
        Window function for edge apodisation (``"tukey"`` or ``"hamming"``).
    threshold : float
        FRC threshold for weight conversion.
    alpha : float
        Weight exponent.
    nbins_low : int
        Number of lowest bins to exclude.
    smooth_window : int
        Median-filter window for weight smoothing.
    split_type : str
        ``"binomial"`` or ``"checkerboard"`` — passed to :func:`frc_weights`.
    n_repeats : int
        Number of binomial splits to average (default 3 for stability;
        lower-level ``calculate_frc`` defaults to 1).
    rng : Generator, int, or None
        Random seed for binomial split.
    frozen_weights : np.ndarray, optional
        Pre-computed 1-D radial-bin weights. If given, skip FRC weight
        estimation.  Must match index-unit binning for the target shape.

    Returns
    -------
    float
        Weighted Pearson *r* in [-1, 1].
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )

    # 1. Compute or reuse per-bin weights
    if frozen_weights is not None:
        w_bins = frozen_weights
    else:
        w_bins = frc_weights(
            target,
            bin_delta=bin_delta,
            threshold=threshold,
            alpha=alpha,
            nbins_low=nbins_low,
            smooth_window=smooth_window,
            split_type=split_type,
            n_repeats=n_repeats,
            rng=rng,
        )

    # 2. Zero-weight-mass guard
    if float(np.sum(w_bins)) < 1e-6:
        return 0.0

    # 3. Mean-subtract + apodise → FFT
    apo_fn = _APODIZATION_FNS.get(apodization)
    if apo_fn is None:
        raise ValueError(
            f"Unknown apodization '{apodization}'. "
            f"Choose from {list(_APODIZATION_FNS)}."
        )
    pred = prediction.astype(np.float32) - np.mean(prediction)
    targ = target.astype(np.float32) - np.mean(target)
    pred = apo_fn(pred)
    targ = apo_fn(targ)

    F_pred = np.fft.fftn(pred)
    F_targ = np.fft.fftn(targ)

    # 4. Map per-bin weights → per-voxel weight volume (index units)
    edges_cpu, _ = radial_edges(
        prediction.shape,
        bin_delta=bin_delta,
        spacing=None,
    )
    edges = to_same_device(edges_cpu, prediction)
    bid = radial_bin_id(prediction.shape, edges, spacing=None)

    n_bins_needed = int(asnumpy(bid[bid >= 0].max())) + 1 if np.any(bid >= 0) else 0
    if len(w_bins) < n_bins_needed:
        raise ValueError(
            f"frozen_weights has {len(w_bins)} bins but binning requires "
            f"{n_bins_needed}; check that bin_delta and image shape match."
        )
    w_bins_dev = to_same_device(np.asarray(w_bins, dtype=np.float32), prediction)

    W = np.zeros_like(bid, dtype=np.float32)
    valid = bid >= 0
    W[valid] = w_bins_dev[bid[valid]]

    # 5. Weighted cross-spectrum correlation
    cross = np.real(F_pred.ravel() * np.conj(F_targ.ravel()))
    num = float(asnumpy(np.sum(W * cross)))
    denom_pred = float(asnumpy(np.sum(W * np.abs(F_pred.ravel()) ** 2)))
    denom_targ = float(asnumpy(np.sum(W * np.abs(F_targ.ravel()) ** 2)))
    denom = np.sqrt(denom_pred * denom_targ)

    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 3e  Percentile-band spectral PCC
# ---------------------------------------------------------------------------

_WEIGHT_METHODS = frozenset({"simple", "smooth_wiener", "snr2"})


def bandpass_spectral_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: float | Sequence[float],
    bin_delta: float = 1.0,
    apodization: str = "tukey",
    p_low: float = 0.0,
    p_high: float = 0.99,
    taper_width: int = 3,
    frozen_weights: np.ndarray | None = None,
    weight_method: Literal["simple", "smooth_wiener", "snr2"] = "smooth_wiener",
    return_diagnostics: bool = False,
    **weight_kwargs,
) -> float | tuple[float, dict[str, float]]:
    r"""Spectral PCC within a percentile-defined frequency band.

    Computes per-bin weights (or accepts pre-computed ones), selects a
    frequency band via cumulative weight-mass percentiles, applies a
    soft cosine taper at the band edges, and returns the weighted
    Fourier-domain PCC.

    The band is defined by *p_low* / *p_high* on the cumulative
    distribution of ``w_i * n_i`` (weight times Fourier-pixel count
    per ring), which measures each ring's actual contribution to the
    PCC sum.

    Parameters
    ----------
    prediction, target : np.ndarray
        Images to compare (same shape, 2-D or 3-D).
    spacing : float or sequence of float
        Physical pixel / voxel spacing.
    bin_delta : float
        Radial-bin width (index units).
    apodization : str
        Window function for edge apodisation (``"tukey"`` or
        ``"hamming"``).
    p_low : float
        Lower percentile of cumulative weight mass.  Default 0.0
        (no low-frequency exclusion).
    p_high : float
        Upper percentile of cumulative weight mass.  Default 0.99.
    taper_width : int
        Number of bins for cosine ramp on each side of the band.
    frozen_weights : np.ndarray, optional
        Pre-computed 1-D per-bin weights.  If given, *weight_method*
        and *weight_kwargs* are ignored.  Must match the binning
        defined by *bin_delta* and *spacing* for the target shape.
        FRC-derived weights (from :func:`frc_weights`) should be passed
        here since they use index-unit binning.
    weight_method : ``"simple"`` | ``"smooth_wiener"`` | ``"snr2"``
        How to compute per-bin weights from the target power spectrum.
        ``"simple"`` uses :func:`spectral_weights`;
        ``"smooth_wiener"`` uses :func:`smooth_spectral_weights`;
        ``"snr2"`` uses :func:`_estimate_noise_baseline` +
        :func:`_baseline_snr2_weights`.
    return_diagnostics : bool
        If True, return ``(pcc, diagnostics)`` instead of just *pcc*.
    **weight_kwargs
        Extra keyword arguments forwarded to the selected weight
        function (e.g. ``tail_fraction``, ``sg_window``, ``cutoff``).

    Returns
    -------
    float or tuple[float, dict[str, float]]
        Weighted Pearson *r* in [-1, 1].  When *return_diagnostics* is
        True, also returns band diagnostics from
        :func:`percentile_band_taper`.
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )
    if weight_method not in _WEIGHT_METHODS:
        raise ValueError(
            f"Unknown weight_method {weight_method!r}. "
            f"Choose from {sorted(_WEIGHT_METHODS)}."
        )

    spacing_seq = _normalize_spacing(spacing, prediction.ndim)

    apo_fn = _APODIZATION_FNS.get(apodization)
    if apo_fn is None:
        raise ValueError(
            f"Unknown apodization '{apodization}'. "
            f"Choose from {list(_APODIZATION_FNS)}."
        )

    # Mean-subtract + apodise → FFT
    pred = prediction.astype(np.float32) - np.mean(prediction)
    targ = target.astype(np.float32) - np.mean(target)
    pred = apo_fn(pred)
    targ = apo_fn(targ)

    F_pred = np.fft.fftn(pred)
    F_targ = np.fft.fftn(targ)

    # --- Per-bin weights ---
    if frozen_weights is not None:
        w_bins = np.asarray(frozen_weights, dtype=np.float32)
    elif weight_method == "simple":
        radii, power = radial_power_spectrum(
            target, spacing=spacing_seq, bin_delta=bin_delta
        )
        kw = {k: v for k, v in weight_kwargs.items() if k in ("cutoff",)}
        noise = estimate_noise_floor(
            radii, power, tail_fraction=weight_kwargs.get("tail_fraction", 0.2)
        )
        w_bins = spectral_weights(radii, power, noise, **kw)
    elif weight_method == "smooth_wiener":
        radii, power = radial_power_spectrum(
            target, spacing=spacing_seq, bin_delta=bin_delta
        )
        kw = {
            k: v
            for k, v in weight_kwargs.items()
            if k in ("cutoff", "sg_window", "sg_polyorder")
        }
        noise = estimate_noise_floor(
            radii, power, tail_fraction=weight_kwargs.get("tail_fraction", 0.2)
        )
        w_bins = smooth_spectral_weights(radii, power, noise, **kw)
    else:  # snr2
        radii, power = radial_power_spectrum(
            target, spacing=spacing_seq, bin_delta=bin_delta
        )
        bl_kw = {
            k: v
            for k, v in weight_kwargs.items()
            if k in ("sg_window", "sg_polyorder", "quantile_window", "quantile")
        }
        p_smooth, noise_bl = _estimate_noise_baseline(power, **bl_kw)
        snr_kw = {
            k: v for k, v in weight_kwargs.items() if k in ("cap_quantile", "cutoff")
        }
        w_bins = _baseline_snr2_weights(
            radii, p_smooth, noise_bl, nbins_low=0, **snr_kw
        )

    # --- Radial binning ---
    edges_cpu, radii_cpu = radial_edges(
        prediction.shape, bin_delta=bin_delta, spacing=spacing_seq
    )
    edges = to_same_device(edges_cpu, prediction)
    bid = radial_bin_id(prediction.shape, edges, spacing=spacing_seq)

    # Validate weight vector length
    valid = bid >= 0
    n_bins_needed = int(asnumpy(bid[valid].max())) + 1 if np.any(valid) else 0
    if len(w_bins) < n_bins_needed:
        raise ValueError(
            f"frozen_weights has {len(w_bins)} bins but binning requires "
            f"{n_bins_needed}; check that bin_delta and image shape match."
        )

    # Bin counts and Nyquist for taper
    n_per_bin = np.bincount(
        asnumpy(bid[valid]).astype(np.intp), minlength=len(w_bins)
    ).astype(np.float32)
    k_nyquist = float(edges_cpu[-1])

    # --- Percentile band taper ---
    w_tapered, diagnostics = percentile_band_taper(
        asnumpy(w_bins),
        n_per_bin,
        asnumpy(radii_cpu).astype(np.float32),
        k_nyquist,
        p_low=p_low,
        p_high=p_high,
        taper_width=taper_width,
    )

    # Zero-weight guard
    if float(np.sum(w_tapered)) < 1e-6:
        if return_diagnostics:
            return 0.0, diagnostics
        return 0.0

    # Map tapered weights → per-voxel weight volume
    w_dev = to_same_device(w_tapered, prediction)

    W = np.zeros_like(bid, dtype=np.float32)
    W[valid] = w_dev[bid[valid]]

    # Weighted cross-spectrum correlation
    cross = np.real(F_pred.ravel() * np.conj(F_targ.ravel()))
    num = float(asnumpy(np.sum(W * cross)))
    denom_pred = float(asnumpy(np.sum(W * np.abs(F_pred.ravel()) ** 2)))
    denom_targ = float(asnumpy(np.sum(W * np.abs(F_targ.ravel()) ** 2)))
    denom = np.sqrt(denom_pred * denom_targ)

    if denom < 1e-12:
        pcc = 0.0
    else:
        pcc = float(np.clip(num / denom, -1.0, 1.0))

    if return_diagnostics:
        return pcc, diagnostics
    return pcc


# ---------------------------------------------------------------------------
# 4  Internal filtering helper
# ---------------------------------------------------------------------------

_APODIZATION_FNS: dict[str, Callable[..., np.ndarray]] = {
    "tukey": tukey_window,
    "hamming": hamming_window,
}


def _apply_lowpass(
    image: np.ndarray,
    cutoff: float,
    spacing: Sequence[float] | None = None,
    order: int = 2,
    apodization: str = "tukey",
) -> np.ndarray:
    """Mean-subtract, apodise, FFT, Butterworth multiply, IFFT → real.

    Parameters
    ----------
    image : np.ndarray
        Input image (2-D or 3-D).
    cutoff : float
        Butterworth cutoff frequency.
    spacing : sequence of float, optional
        Physical spacing.
    order : int
        Butterworth order.
    apodization : ``"tukey"`` | ``"hamming"``
        Window function for edge apodisation.

    Returns
    -------
    np.ndarray
        Low-pass filtered image (real-valued, float32).
    """
    img = image.astype(np.float32)
    img = img - np.mean(img)

    apo_fn = _APODIZATION_FNS.get(apodization)
    if apo_fn is None:
        raise ValueError(
            f"Unknown apodization '{apodization}'. Choose from {list(_APODIZATION_FNS)}."
        )
    img = apo_fn(img)  # type: ignore[operator]

    F = np.fft.fftn(img)
    H = butterworth_lowpass(img.shape, cutoff, spacing=spacing, order=order)

    # Move filter to same device as data
    H = to_same_device(H, img)

    F *= H
    return np.fft.ifftn(F).real.astype(np.float32)


# ---------------------------------------------------------------------------
# 5  Band-limited metrics
# ---------------------------------------------------------------------------


def band_limited_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    cutoff: float | None = None,
    spacing: float | Sequence[float],
    numerical_aperture: float | None = None,
    wavelength_emission: float | None = None,
    modality: str = "widefield",
    method: str = "dcr",
    filter_order: int = 2,
    apodization: str = "tukey",
    dcr_kwargs: dict | None = None,
    frc_kwargs: dict | None = None,
) -> float:
    """Band-limited Pearson correlation coefficient.

    Applies a Butterworth low-pass to both images, then computes PCC on
    the filtered versions.  The cutoff frequency is either provided
    explicitly or estimated automatically from the *target*.

    Parameters
    ----------
    prediction, target : np.ndarray
        Images to compare (same shape, 2-D or 3-D).
    cutoff : float, optional
        Explicit cutoff frequency.  If ``None``, estimated via
        :func:`estimate_cutoff`.
    spacing : float or sequence of float
        Physical pixel / voxel spacing.
    numerical_aperture, wavelength_emission : float, optional
        Passed to :func:`estimate_cutoff` when *cutoff* is ``None``.
    modality : str
        Passed to :func:`estimate_cutoff`.
    method : str
        Data-driven estimation method (``"dcr"``, ``"frc"``, or
        ``"both"``).  Passed to :func:`estimate_cutoff`.
    filter_order : int
        Butterworth order.
    apodization : str
        Window function (``"tukey"`` or ``"hamming"``).
    dcr_kwargs : dict, optional
        Extra keyword arguments for DCR resolution estimation.
    frc_kwargs : dict, optional
        Extra keyword arguments for FRC/FSC resolution estimation.

    Returns
    -------
    float
        Pearson *r* in [-1, 1].
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )

    spacing_seq = _normalize_spacing(spacing, prediction.ndim)

    if cutoff is None:
        cutoff = estimate_cutoff(
            target,
            spacing=spacing_seq,
            numerical_aperture=numerical_aperture,
            wavelength_emission=wavelength_emission,
            modality=modality,
            method=method,
            dcr_kwargs=dcr_kwargs,
            frc_kwargs=frc_kwargs,
        )

    pred_f = _apply_lowpass(
        prediction,
        cutoff,
        spacing=spacing_seq,
        order=filter_order,
        apodization=apodization,
    )
    targ_f = _apply_lowpass(
        target, cutoff, spacing=spacing_seq, order=filter_order, apodization=apodization
    )

    return float(_pearson(pred_f, targ_f))


def band_limited_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    cutoff: float | None = None,
    spacing: float | Sequence[float],
    numerical_aperture: float | None = None,
    wavelength_emission: float | None = None,
    modality: str = "widefield",
    method: str = "dcr",
    filter_order: int = 2,
    apodization: str = "tukey",
    win_size: int | None = None,
    data_range: float | None = None,
    dcr_kwargs: dict | None = None,
    frc_kwargs: dict | None = None,
) -> float:
    """Band-limited structural similarity index (SSIM).

    Filters both images with a Butterworth low-pass, then delegates to
    :func:`cubic.metrics.ssim`.

    Parameters
    ----------
    prediction, target : np.ndarray
        Images to compare (same shape, 2-D or 3-D).
    cutoff : float, optional
        Explicit cutoff frequency.
    spacing : float or sequence of float
        Physical pixel / voxel spacing.
    numerical_aperture, wavelength_emission : float, optional
        Passed to :func:`estimate_cutoff`.
    modality : str
        Passed to :func:`estimate_cutoff`.
    method : str
        Data-driven estimation method (``"dcr"``, ``"frc"``, or
        ``"both"``).  Passed to :func:`estimate_cutoff`.
    filter_order : int
        Butterworth order.
    apodization : str
        Window function.
    win_size : int, optional
        SSIM window size.
    data_range : float, optional
        SSIM data range.  If ``None``, computed from filtered target.
    dcr_kwargs : dict, optional
        Extra keyword arguments for DCR resolution estimation.
    frc_kwargs : dict, optional
        Extra keyword arguments for FRC/FSC resolution estimation.

    Returns
    -------
    float
        SSIM value.
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )

    spacing_seq = _normalize_spacing(spacing, prediction.ndim)

    if cutoff is None:
        cutoff = estimate_cutoff(
            target,
            spacing=spacing_seq,
            numerical_aperture=numerical_aperture,
            wavelength_emission=wavelength_emission,
            modality=modality,
            method=method,
            dcr_kwargs=dcr_kwargs,
            frc_kwargs=frc_kwargs,
        )

    pred_f = _apply_lowpass(
        prediction,
        cutoff,
        spacing=spacing_seq,
        order=filter_order,
        apodization=apodization,
    )
    targ_f = _apply_lowpass(
        target, cutoff, spacing=spacing_seq, order=filter_order, apodization=apodization
    )

    if data_range is None:
        data_range = float(targ_f.max() - targ_f.min())

    kwargs: dict = {"data_range": data_range}
    if win_size is not None:
        kwargs["win_size"] = win_size

    return float(_ssim(pred_f, targ_f, **kwargs))


def spectral_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: float | Sequence[float],
    bin_delta: float = 1.0,
    tail_fraction: float = 0.2,
    cutoff: float | None = None,
    apodization: str = "tukey",
    smooth: bool = False,
    sg_window: int = 15,
    sg_polyorder: int = 3,
    nbins_low: int = 0,
    taper_low: int = 0,
    weighting: Literal["simple", "smooth_wiener", "snr2"] | None = None,
) -> float:
    r"""Spectrally-weighted Pearson correlation coefficient.

    Instead of a hard Butterworth cutoff, this metric applies *soft*
    per-frequency weights derived from the target's radial power spectrum,
    down-weighting frequency bins whose power does not exceed the
    estimated noise floor.

    .. math::

        r = \frac{\sum_k W(k)\,\operatorname{Re}\{F_{pred}(k)\,
            F_{target}^*(k)\}}{
            \sqrt{\sum_k W(k)\,|F_{pred}(k)|^2\;
                   \sum_k W(k)\,|F_{target}(k)|^2}}

    Parameters
    ----------
    prediction, target : np.ndarray
        Images to compare (same shape, 2-D or 3-D).
    spacing : float or sequence of float
        Physical pixel / voxel spacing.
    bin_delta : float
        Radial-bin width (index units).
    tail_fraction : float
        Fraction of high-frequency bins for noise-floor estimation.
    cutoff : float, optional
        Hard cutoff zeroing bins above this frequency.
    apodization : str
        Window function for edge apodisation.
    weighting : ``"simple"`` | ``"smooth_wiener"`` | ``"snr2"`` or None
        Per-bin weight law.  ``None`` (default) auto-selects based on
        *smooth*.  ``"simple"`` uses subtract-normalize weights;
        ``"smooth_wiener"`` uses SG-filtered Wiener weights; ``"snr2"``
        uses a frequency-dependent SNR² model.
    smooth : bool
        Deprecated sugar: when *weighting* is ``None``, ``smooth=True``
        selects ``"smooth_wiener"``.  Ignored when *weighting* is set
        explicitly.
    sg_window : int
        Savitzky-Golay window length (used for ``"smooth_wiener"`` and
        ``"snr2"``).
    sg_polyorder : int
        Savitzky-Golay polynomial order.
    nbins_low : int
        Number of lowest-frequency bins to zero after weight computation
        (DC / background / autofluorescence exclusion).  Ignored when
        *taper_low* > 0.
    taper_low : int
        Soft low-frequency exclusion: apply a half-cosine ramp from 0 to 1
        over the first *taper_low* bins.  When > 0, takes precedence over
        *nbins_low*.

    Returns
    -------
    float
        Weighted Pearson *r* in [-1, 1].
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )

    spacing_seq = _normalize_spacing(spacing, prediction.ndim)

    apo_fn = _APODIZATION_FNS.get(apodization)
    if apo_fn is None:
        raise ValueError(
            f"Unknown apodization '{apodization}'. Choose from {list(_APODIZATION_FNS)}."
        )

    # Mean-subtract + apodise → FFT
    pred = prediction.astype(np.float32) - np.mean(prediction)
    targ = target.astype(np.float32) - np.mean(target)
    pred = apo_fn(pred)  # type: ignore[operator]
    targ = apo_fn(targ)  # type: ignore[operator]

    F_pred = np.fft.fftn(pred)
    F_targ = np.fft.fftn(targ)

    # Resolve effective weighting (smooth= is deprecated sugar)
    if weighting is not None:
        _weighting = weighting
    elif smooth:
        _weighting = "smooth_wiener"
    else:
        _weighting = "simple"

    if _weighting not in _WEIGHT_METHODS:
        raise ValueError(
            f"Unknown weighting {_weighting!r}. Choose from {sorted(_WEIGHT_METHODS)}."
        )

    # Radial power spectrum of target → per-bin weights
    radii, power = radial_power_spectrum(
        target, spacing=spacing_seq, bin_delta=bin_delta
    )
    if _weighting == "snr2":
        p_smooth, noise_bl = _estimate_noise_baseline(
            power, sg_window=sg_window, sg_polyorder=sg_polyorder
        )
        w_bins = _baseline_snr2_weights(radii, p_smooth, noise_bl, nbins_low=0)
        if cutoff is not None:
            w_bins[radii > cutoff] = 0.0
    elif _weighting == "smooth_wiener":
        noise = estimate_noise_floor(radii, power, tail_fraction=tail_fraction)
        w_bins = smooth_spectral_weights(
            radii,
            power,
            noise,
            cutoff=cutoff,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
        )
    else:
        noise = estimate_noise_floor(radii, power, tail_fraction=tail_fraction)
        w_bins = spectral_weights(radii, power, noise, cutoff=cutoff)

    # Low-k exclusion (DC / illumination / background)
    if taper_low > 0:
        _n = min(taper_low, len(w_bins))
        _ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, _n + 1) / _n))
        w_bins[:_n] *= _ramp.astype(w_bins.dtype)
    elif nbins_low > 0:
        nbins_low = min(nbins_low, len(w_bins))
        w_bins[:nbins_low] = 0.0
    if float(w_bins.max().item()) == 0.0:
        return 0.0

    # Map per-bin weights → per-voxel weight volume
    edges_cpu, _ = radial_edges(
        prediction.shape,
        bin_delta=bin_delta,
        spacing=spacing_seq,
    )
    edges = to_same_device(edges_cpu, prediction)
    bid = radial_bin_id(prediction.shape, edges, spacing=spacing_seq)

    n_bins_needed = int(asnumpy(bid[bid >= 0].max())) + 1 if np.any(bid >= 0) else 0
    if len(w_bins) < n_bins_needed:
        raise ValueError(
            f"Weight vector has {len(w_bins)} bins but binning requires "
            f"{n_bins_needed}; check that bin_delta and image shape match."
        )
    w_bins_dev = to_same_device(np.asarray(w_bins, dtype=np.float32), prediction)

    # Build weight volume: map bin weights through bin_id
    W = np.zeros_like(bid, dtype=np.float32)
    valid = bid >= 0
    W[valid] = w_bins_dev[bid[valid]]

    # Weighted cross-spectrum correlation
    cross = np.real(F_pred.ravel() * np.conj(F_targ.ravel()))
    num = float(asnumpy(np.sum(W * cross)))
    denom_pred = float(asnumpy(np.sum(W * np.abs(F_pred.ravel()) ** 2)))
    denom_targ = float(asnumpy(np.sum(W * np.abs(F_targ.ravel()) ** 2)))
    denom = np.sqrt(denom_pred * denom_targ)

    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_spacing(
    spacing: float | Sequence[float],
    ndim: int,
) -> list[float]:
    """Ensure spacing is a list of length *ndim*."""
    if isinstance(spacing, (int, float)):
        return [float(spacing)] * ndim
    sp = [float(s) for s in spacing]
    if len(sp) != ndim:
        raise ValueError(f"spacing length {len(sp)} != image ndim {ndim}")
    return sp


def _is_anisotropic(
    spacing: float | Sequence[float],
    ratio_threshold: float = 1.5,
) -> bool:
    """Return True if max/min spacing ratio exceeds *ratio_threshold*."""
    if isinstance(spacing, (int, float)):
        return False
    values = [float(s) for s in spacing]
    if len(values) <= 1:
        return False
    return max(values) / min(values) > ratio_threshold


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays (device-agnostic)."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    a_c = a_flat - np.mean(a_flat)
    b_c = b_flat - np.mean(b_flat)
    num = float(asnumpy(np.sum(a_c * b_c)))
    denom = float(np.sqrt(asnumpy(np.sum(a_c**2)) * asnumpy(np.sum(b_c**2))))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(num / denom, -1.0, 1.0))
