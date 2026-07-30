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

All functions **accept** either NumPy (CPU) or CuPy (GPU) arrays and keep
the computation on the input's device.  Two of them always return host
(NumPy) arrays regardless of the input device:
:func:`butterworth_lowpass`, because its frequency grid is built with
plain ``np.fft.fftfreq``, and :func:`radial_power_spectrum`, which
transfers its per-bin results to the host before returning.

References
----------
Descloux, A., et al. (2019). Parameter-free image resolution estimation
    based on decorrelation analysis. *Nature Methods* 16, 918-924.
"""

from __future__ import annotations

import warnings
from typing import Any
from collections.abc import Callable, Sequence

import numpy as np

from cubic.cuda import asnumpy, to_same_device, check_same_device
from cubic.image_utils import tukey_window, hamming_window

from .pcc import pcc as _pcc
from .spectral.dcr import dcr_resolution
from .spectral.frc import frc_resolution, fsc_resolution
from .skimage_metrics import ssim as _ssim
from .spectral.radial import (
    radial_edges,
    reduce_power,
    radial_bin_id,
    radial_k_grid,
    _normalize_spacing,
)

_CUTOFF_METHODS = ("dcr", "frc", "both")

#: Default per-bound safety factors for :func:`estimate_cutoff`.
_DEFAULT_SAFETY: dict[str, float] = {
    "dcr": 1.0,
    "frc": 1.0,
    "otf": 0.95,
    "nyquist": 0.9,
}

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
    safety: dict[str, float] | None = None,
    dcr_kwargs: dict | None = None,
    frc_kwargs: dict | None = None,
) -> float:
    """Data-driven cutoff frequency estimation.

    Computes up to four independent bounds and returns their minimum:

    * **DCR bound** — ``safety["dcr"] / dcr_resolution(image)`` (data-driven).
    * **FRC/FSC bound** — ``safety["frc"] / frc_resolution(image)`` (data-driven).
    * **OTF bound** — ``safety["otf"] * otf_cutoff(NA, λ)`` (physics).
    * **Nyquist bound** — ``safety["nyquist"] * nyquist_cutoff(spacing)``.

    Bounds whose required parameters are absent are silently skipped.
    Data-driven bounds (DCR, FRC/FSC) that raise ``ValueError``,
    ``RuntimeError``, or ``TypeError`` emit a warning and are excluded.
    The Nyquist bound is always computable, so a value is always returned.

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
    safety : dict, optional
        Per-bound safety factors, overriding the defaults
        ``{"dcr": 1.0, "frc": 1.0, "otf": 0.95, "nyquist": 0.9}``.
        Unknown keys raise ``ValueError``.
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
        If *method* is not one of ``"dcr"``, ``"frc"``, ``"both"``, or if
        *safety* contains an unknown key.
    """
    if method not in _CUTOFF_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(_CUTOFF_METHODS)} "
            "(lower-case)."
        )

    factors = dict(_DEFAULT_SAFETY)
    if safety:
        unknown = set(safety) - set(factors)
        if unknown:
            raise ValueError(
                f"Unknown safety keys {sorted(unknown)}. Choose from {sorted(factors)}."
            )
        factors.update({k: float(v) for k, v in safety.items()})

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
                bounds.append(factors["dcr"] / dcr_val)
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
                bounds.append(factors["frc"] / frc_res)
        except (ValueError, RuntimeError, TypeError) as exc:
            warnings.warn(f"FRC/FSC resolution estimation failed: {exc}", stacklevel=2)

    # --- OTF (physics) bound ---
    if numerical_aperture is not None and wavelength_emission is not None:
        f_otf = otf_cutoff(numerical_aperture, wavelength_emission, modality=modality)
        bounds.append(factors["otf"] * f_otf)

    # --- Nyquist bound (always available) ---
    bounds.append(factors["nyquist"] * nyquist_cutoff(spacing))

    return float(min(bounds))


# ---------------------------------------------------------------------------
# 3  Spectral analysis helpers
# ---------------------------------------------------------------------------


def _radial_power_from_spectrum(
    F: np.ndarray,
    shape: tuple[int, ...],
    spacing: Sequence[float] | None,
    bin_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthally average a **precomputed** FFT into radial bins.

    Callers that already hold the spectrum they intend to correlate must
    weight it with power derived from *that* spectrum, not from a second,
    differently-preprocessed transform of the same image.

    Returns
    -------
    radii : np.ndarray
        Bin centres, on the host.
    mean_power : np.ndarray
        Mean |F|² per radial bin (DC excluded), on the host.
    bin_id : np.ndarray
        Flat per-voxel bin index on *F*'s device, reusable by the caller
        to map per-bin values back onto the frequency grid.
    """
    edges_cpu, radii = radial_edges(shape, bin_delta=bin_delta, spacing=spacing)

    # Move edges to same device as the spectrum for correct bin_id placement
    edges = to_same_device(edges_cpu, F)
    bid = radial_bin_id(shape, edges, spacing=spacing)
    # Pass nbins so the sums always align with ``radii`` positionally; without
    # it an empty trailing bin would return a shorter curve than ``radii``.
    S2, N = reduce_power(F, bid, nbins=len(radii))

    # Mean power per bin (avoid /0)
    N_safe = np.maximum(N.astype(np.float64), 1.0)
    mean_power = (S2 / N_safe).astype(np.float32)

    return (
        asnumpy(radii).astype(np.float32),
        asnumpy(mean_power).astype(np.float32),
        bid,
    )


def radial_power_spectrum(
    image: np.ndarray,
    spacing: float | Sequence[float] | None = None,
    bin_delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial (azimuthally-averaged) power spectrum.

    The image is transformed as-is: no mean subtraction and no
    apodisation, so the DC pedestal and edge discontinuities contribute.

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
        Bin centres (physical frequency if *spacing* given).  Always a
        host (NumPy) array.
    mean_power : np.ndarray
        Mean |F|² per radial bin (DC excluded).  Always a host array.
    """
    spacing_seq = _normalize_spacing(spacing, image.ndim)
    F = np.fft.fftn(image.astype(np.float32))
    radii, mean_power, _ = _radial_power_from_spectrum(
        F, image.shape, spacing_seq, bin_delta
    )
    return radii, mean_power


def estimate_noise_floor(
    power: np.ndarray,
    tail_fraction: float = 0.2,
) -> float:
    """Estimate the spectral noise floor from the high-frequency tail.

    Parameters
    ----------
    power : np.ndarray
        Mean power per bin, ordered from low to high frequency (as
        returned by :func:`radial_power_spectrum`).
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
        Weights in [0, 1] with ``max(w) == 1``, or all-zero when no bin
        exceeds the noise floor (nothing is left to rescale by).
    """
    w = np.maximum(power - noise_floor, 0.0)
    if cutoff is not None:
        w[radii > cutoff] = 0.0
    w_max = float(np.max(w))
    if w_max > 0:
        w = w / w_max
    return w.astype(np.float32)


# ---------------------------------------------------------------------------
# 4  Internal filtering helper
# ---------------------------------------------------------------------------

_APODIZATION_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
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


def _filtered_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    cutoff: float | None,
    spacing: float | Sequence[float],
    filter_order: int,
    apodization: str,
    **cutoff_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a metric's inputs and low-pass both images identically.

    Shared front end of :func:`band_limited_pcc` and
    :func:`band_limited_ssim`.  When *cutoff* is ``None`` it is estimated
    from *target* with :func:`estimate_cutoff`, to which
    *cutoff_kwargs* is forwarded.
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )

    spacing_seq = _require_spacing(spacing, prediction.ndim)

    if cutoff is None:
        cutoff = estimate_cutoff(target, spacing=spacing_seq, **cutoff_kwargs)

    return (
        _apply_lowpass(
            prediction,
            cutoff,
            spacing=spacing_seq,
            order=filter_order,
            apodization=apodization,
        ),
        _apply_lowpass(
            target,
            cutoff,
            spacing=spacing_seq,
            order=filter_order,
            apodization=apodization,
        ),
    )


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
    safety: dict[str, float] | None = None,
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
    safety : dict, optional
        Per-bound safety factors passed to :func:`estimate_cutoff`.
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
        Pearson *r* in [-1, 1], or ``nan`` when either filtered image has
        zero variance (e.g. a constant input), matching
        :func:`cubic.metrics.pcc`.
    """
    pred_f, targ_f = _filtered_pair(
        prediction,
        target,
        cutoff=cutoff,
        spacing=spacing,
        filter_order=filter_order,
        apodization=apodization,
        numerical_aperture=numerical_aperture,
        wavelength_emission=wavelength_emission,
        modality=modality,
        method=method,
        safety=safety,
        dcr_kwargs=dcr_kwargs,
        frc_kwargs=frc_kwargs,
    )
    return float(_pcc(pred_f, targ_f))


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
    safety: dict[str, float] | None = None,
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
    safety : dict, optional
        Per-bound safety factors passed to :func:`estimate_cutoff`.
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
        SSIM value, or ``nan`` when the filtered target has zero dynamic
        range (e.g. a constant input, which mean-subtracts to all zeros)
        and no explicit *data_range* was given.  This matches the
        degenerate-input convention of :func:`band_limited_pcc` and
        :func:`cubic.metrics.pcc`.
    """
    pred_f, targ_f = _filtered_pair(
        prediction,
        target,
        cutoff=cutoff,
        spacing=spacing,
        filter_order=filter_order,
        apodization=apodization,
        numerical_aperture=numerical_aperture,
        wavelength_emission=wavelength_emission,
        modality=modality,
        method=method,
        safety=safety,
        dcr_kwargs=dcr_kwargs,
        frc_kwargs=frc_kwargs,
    )

    if data_range is None:
        data_range = float(targ_f.max() - targ_f.min())
        # data_range == 0 zeroes SSIM's c1/c2 stabilisers, so skimage would
        # evaluate 0/0 and return nan with a RuntimeWarning. Report the
        # undefined result directly instead.
        if data_range <= 0:
            return float("nan")

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
    nbins_low: int = 0,
    taper_low: int = 0,
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
    nbins_low : int
        Number of lowest-frequency bins to zero (DC / background exclusion).
        Default 0 (include all bins).  Ignored when *taper_low* > 0.
    taper_low : int
        Soft low-frequency exclusion: apply a half-cosine ramp from 0 → 1
        over the first *taper_low* bins.  Takes precedence over *nbins_low*.
        Default 0 (no taper).

    Returns
    -------
    float
        Weighted Pearson *r* in [-1, 1], or ``nan`` when the weighting
        leaves nothing to correlate — every bin excluded, or a filtered
        input with zero energy.  Matches :func:`cubic.metrics.pcc`.
    """
    check_same_device(prediction, target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )
    if nbins_low < 0:
        raise ValueError(f"nbins_low must be >= 0, got {nbins_low}")
    if taper_low < 0:
        raise ValueError(f"taper_low must be >= 0, got {taper_low}")

    spacing_seq = _require_spacing(spacing, prediction.ndim)

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

    # Radial power spectrum of the *same* target spectrum that is correlated
    # below → noise floor → per-bin weights. Deriving the power from a raw
    # (un-centred, un-apodised) transform instead would let the DC pedestal
    # and edge discontinuities leak into the low-frequency bins and inflate
    # their weights. ``bid`` is reused so the radial grid is built once.
    radii, power, bid = _radial_power_from_spectrum(
        F_targ, prediction.shape, spacing_seq, bin_delta
    )
    noise = estimate_noise_floor(power, tail_fraction=tail_fraction)
    w_bins = spectral_weights(radii, power, noise, cutoff=cutoff)

    # Low-frequency exclusion (DC / background / autofluorescence)
    if taper_low > 0:
        _n = min(taper_low, len(w_bins))
        if _n == 1:
            _ramp = np.array([0.0], dtype=w_bins.dtype)
        else:
            _ramp = 0.5 * (
                1.0 - np.cos(np.pi * np.arange(_n, dtype=np.float32) / (_n - 1))
            )
        w_bins[:_n] *= _ramp.astype(w_bins.dtype)
    elif nbins_low > 0:
        _nb = min(nbins_low, len(w_bins))
        w_bins[:_nb] = 0.0

    # Guard: no bin survives the noise floor / exclusions — nothing to correlate
    if float(w_bins.max()) == 0.0:
        return float("nan")

    # Map per-bin weights → per-voxel weight volume, reusing ``bid``
    w_bins_dev = to_same_device(w_bins, prediction)
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
        return float("nan")
    return float(np.clip(num / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_spacing(
    spacing: float | Sequence[float] | None,
    ndim: int,
) -> list[float]:
    """Expand *spacing* to a list of length *ndim*.

    Wraps the shared helper from :mod:`cubic.metrics.spectral.radial` and
    adds the two constraints the band-limited metrics need: *spacing* is
    required (index units are meaningless for a physical cutoff), and it
    must have one entry per axis.
    """
    sp = _normalize_spacing(spacing, ndim)
    if sp is None:
        raise ValueError(
            "spacing is required for band-limited metrics, got None. Pass a "
            "scalar or one value per axis, in the same length unit as cutoff."
        )
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
