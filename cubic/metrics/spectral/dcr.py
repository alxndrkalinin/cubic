"""Decorrelation Analysis (DCR) - Descloux et al. 2019."""

from collections.abc import Sequence

import numpy as np
from scipy.signal import savgol_filter

from cubic.cuda import asnumpy, to_same_device
from cubic.skimage import filters
from cubic.image_utils import tukey_window

from .radial import (
    _kmax_phys,
    radial_edges,
    radial_k_grid,
    _kmax_phys_max,
    sectioned_bin_id,
    _normalize_spacing,
)


def _preprocess_dcr_image(image: np.ndarray, windowing: bool = True) -> np.ndarray:
    """Center and optionally window an image for DCR analysis.

    Consolidates the repeated pattern of float32 cast, mean subtraction, and
    optional Tukey windowing used at the start of dcr_curve and
    _dcr_curve_3d_sectioned.
    """
    image = image.astype(np.float32)
    image -= np.mean(image)
    if windowing:
        image = tukey_window(image, alpha=0.1)
    return image


def _kc_to_resolution(k_c_norm: float, k_max: float) -> float:
    """Convert normalized cutoff frequency to physical resolution.

    Parameters
    ----------
    k_c_norm : float
        Normalized cutoff frequency (0-1 range).
    k_max : float
        Maximum physical frequency (Nyquist limit).

    Returns
    -------
    float
        Physical resolution = 1 / (k_c_norm * k_max), or inf if either is zero.
    """
    if k_c_norm > 0 and k_max > 0:
        return 1.0 / (k_c_norm * k_max)
    return float("inf")


def _refinement_ranges(
    peaks: np.ndarray,
    sigmas: np.ndarray,
    *,
    r_max_pad: float = 0.3,
) -> tuple[float, float, float, float] | None:
    """Compute narrowed frequency and sigma ranges for the DCR refinement pass.

    Implements the two-scoring-strategy logic (best_score = k_c * A, max_score =
    max k_c) from NanoPyx/ImDecorr and returns the narrowed ranges for a second
    pass.

    Parameters
    ----------
    peaks : np.ndarray, shape (N, 2)
        Array of [r_peak, amplitude] pairs from the coarse pass.
    sigmas : np.ndarray
        High-pass sigma array used in the coarse pass.
    r_max_pad : float
        Padding added above the maximum peak frequency for the refined range.

    Returns
    -------
    (r_min, r_max, sigma_min, sigma_max) or None if no valid peaks exist.
    """
    r_peaks = peaks[:, 0]
    a_peaks = peaks[:, 1]
    valid = r_peaks > 0
    if not np.any(valid):
        return None

    # Two scoring strategies to pick anchor points
    gm = np.where(valid, r_peaks * a_peaks, 0.0)
    gm_idx = int(np.argmax(gm))
    kc_gm = r_peaks[gm_idx]
    max_idx = int(np.argmax(np.where(valid, r_peaks, 0.0)))
    kc_max = r_peaks[max_idx]

    # Narrow frequency range
    r_min = max(0.0, min(kc_gm, kc_max) - 0.05)
    r_max = min(1.0, max(kc_gm, kc_max) + r_max_pad)

    # Narrow sigma range
    idx_lo = max(0, min(gm_idx, max_idx) - 1)
    idx_hi = max(gm_idx, max_idx)
    if idx_hi < len(sigmas):
        sigma_min = float(sigmas[idx_lo])
        sigma_max = float(sigmas[idx_hi])
    else:
        sigma_min = float(sigmas[0]) if len(sigmas) > 0 else 1.0
        sigma_max = float(sigmas[-1]) if len(sigmas) > 0 else 1.0

    return r_min, r_max, sigma_min, sigma_max


def _smooth_curve(d_curve: np.ndarray, window: int | None) -> np.ndarray:
    """Apply Savitzky-Golay smoothing if window is specified and valid."""
    if window is None or window <= 1:
        return d_curve
    win = min(window, len(d_curve))
    if win % 2 == 0:
        win -= 1
    if win >= 3:
        return savgol_filter(d_curve, window_length=win, polyorder=3)
    return d_curve


def _find_peak_in_curve(
    radii: np.ndarray,
    d_curve: np.ndarray,
    r_min: float = 0.0,
    r_max: float = 0.9,
    min_prominence: float = 0.001,
    boundary_threshold: float = 0.8,
    min_amplitude: float = 0.0,
) -> tuple[float, float]:
    """
    Find peak in decorrelation curve (Descloux et al. 2019, Supplementary Note 1.1).

    Iteratively finds global maximum, excluding boundary artifacts and checking
    prominence. Rejects peaks near boundary if curve is monotonically increasing.

    Parameters
    ----------
    min_amplitude : float
        Reject peaks with amplitude below this value (SNR gate).
        ImDecorr uses 0.05. Default 0.0 (disabled).
    """
    d_work = d_curve.copy()
    n = len(d_work)

    # Mask invalid frequency regions
    d_work[(radii < r_min) | (radii >= r_max)] = -np.inf

    for _ in range(n):  # Match ImDecorr: iterate until peak found or all exhausted
        peak_idx = int(np.argmax(d_work))
        if d_work[peak_idx] == -np.inf:
            return 0.0, 0.0

        r_peak = radii[peak_idx]
        a_peak = d_curve[peak_idx]

        # Reject if at boundary
        if peak_idx >= n - 1 or r_peak >= r_max - 0.01:
            d_work[peak_idx] = -np.inf
            continue

        # Local maximum check: curve must actually decrease after the peak.
        # Without this, a point on a monotonically increasing slope can be
        # accepted as a "peak" when all higher-frequency candidates are
        # rejected by the boundary check.
        lookahead = max(3, n // 20)
        end_idx = min(peak_idx + lookahead, n)
        if peak_idx + 1 < end_idx:
            if np.all(d_curve[peak_idx + 1 : end_idx] >= a_peak):
                d_work[peak_idx] = -np.inf
                continue

        # For peaks near boundary, reject if curve is mostly increasing (>80%)
        if r_peak > boundary_threshold:
            valid_start = int(np.searchsorted(radii, r_min))
            if peak_idx > valid_start:
                diffs = np.diff(d_curve[valid_start : peak_idx + 1])
                if np.sum(diffs < 0) / len(diffs) < 0.2:
                    d_work[peak_idx] = -np.inf
                    continue

        # Prominence check: peak must exceed subsequent local minimum
        if peak_idx < n - 1:
            prom_end = min(peak_idx + max(10, n // 5), n)
            if a_peak - np.min(d_curve[peak_idx + 1 : prom_end]) < min_prominence:
                d_work[peak_idx] = -np.inf
                continue

        # SNR gate: reject peaks with amplitude below threshold
        # (ImDecorr: kc(SNR < 0.05) = 0)
        if min_amplitude > 0 and a_peak < min_amplitude:
            d_work[peak_idx] = -np.inf
            continue

        return r_peak, a_peak

    return 0.0, 0.0


def dcr_curve(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    smoothing: int | None = None,
    windowing: bool = True,
    refine: bool = True,
    quantize: bool = False,
    min_amplitude: float = 0.0,
) -> tuple[float, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Compute decorrelation curve using algorithm from Descloux et al. 2019.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D input image (numpy or cupy array)
    spacing : float or sequence of floats, optional
        Physical spacing per axis. If None, uses index units.
    num_radii : int
        Number of radial sampling points for d(r) (default: 100)
    num_highpass : int
        Number of Gaussian high-pass filters to apply (default: 10).
        Following NanoPyx convention, uses logarithmically-spaced sigmas.
    smoothing : int or None
        Savitzky-Golay filter window length for curve smoothing.
        Default: None (disabled). The decorrelation curve is intrinsically
        smooth (Descloux et al. 2019, Supplementary Note 1).
    windowing : bool
        If True (default), apply internal Tukey window for edge apodization.
        Set to False when windowing is applied externally for consistent
        preprocessing across different methods.
    refine : bool
        If True, run a second refinement pass with narrowed frequency and
        sigma ranges around the coarse peaks. This follows the NanoPyx
        two-pass strategy and often yields a higher k_c (better resolution
        estimate). Default: True.
    quantize : bool
        If True, truncate d(r) values to 3 decimal places via
        ``floor(1000 * d) / 1000`` (ImDecorr convention). Default: False.
    min_amplitude : float
        Reject peaks with amplitude below this value (ImDecorr SNR gate
        uses 0.05). Default: 0.0 (disabled).

    Returns
    -------
    resolution : float
        Estimated resolution (k_c = max of all peak positions)
    radii : np.ndarray
        Normalized frequency values for mask radii
    all_curves : list of np.ndarray
        List of d(r) curves for each high-pass filtered version.
        When refine=True, contains coarse + refined curves (2 * num_highpass).
    all_peaks : np.ndarray
        Array of [r_i, A_i] pairs (peak position and amplitude), shape (N, 2).
        When refine=True, contains coarse + refined peaks.

    Notes
    -----
    Implements the canonical decorrelation analysis:
    1. Compute FFT and normalize: I_n(k) = I(k) / |I(k)|
    2. For each radius r, compute Pearson correlation d(r)
    3. Apply N_g high-pass filters and repeat
    4. Resolution = max(r_0, r_1, ..., r_Ng)

    The high-pass filter sigmas are logarithmically spaced from 1.0 pixels
    to min(image_shape)/2 pixels, following the NanoPyx/ImageJ DecorrAnalysis
    convention from Descloux et al. 2019.

    When refine=True, a second pass narrows both the frequency range and
    sigma range around the coarse peaks (following NanoPyx convention),
    then recomputes with finer effective sampling.
    """
    # Validate dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")

    image = _preprocess_dcr_image(image, windowing=windowing)

    # Normalize spacing
    spacing_list = _normalize_spacing(spacing, image.ndim)
    if spacing_list is not None:
        spacing_arr = np.array(spacing_list, dtype=np.float32)
    else:
        spacing_arr = np.ones(image.ndim, dtype=np.float32)

    # Generate log-spaced high-pass filter sigmas (in pixels)
    # Following NanoPyx convention: sigma from 0.5 to min(shape)/2
    sigmas = _generate_highpass_sigmas(image.shape, num_highpass)

    # Storage for all curves and peaks
    all_curves = []
    all_peaks = []
    k_max = None  # Will be set from first call

    # Process each high-pass filtered version
    for sigma_hp in sigmas:
        # Apply high-pass filter
        if sigma_hp > 0:
            filtered_image = _highpass_filter(image, sigma_hp)
        else:
            filtered_image = image.copy()

        # Compute decorrelation curve for this filtered image
        # Pass spacing for correct anisotropy handling
        radii, d_curve, k_max = _compute_decorrelation_curve(
            filtered_image,
            num_radii,
            spacing=spacing_list,
            smoothing=smoothing,
            quantize=quantize,
        )

        # Find peak using algorithm from Descloux et al. 2019 Supplementary Note 1.1
        # Finds global maximum with iterative boundary exclusion and prominence check
        r_peak, a_peak = _find_peak_in_curve(
            radii, d_curve, min_amplitude=min_amplitude
        )

        all_curves.append(d_curve)
        all_peaks.append([r_peak, a_peak])

    # Extract peak positions from coarse pass
    all_peaks_arr = np.array(all_peaks)
    r_peaks = all_peaks_arr[:, 0]

    # --- Two-pass refinement (NanoPyx convention) ---
    refined = _refinement_ranges(all_peaks_arr, sigmas) if refine else None
    if refined is not None:
        r_min2, r_max2, sigma_min_r, sigma_max_r = refined
        sigmas_refined = _generate_highpass_sigmas(
            image.shape,
            num_highpass,
            sigma_min=sigma_min_r,
            sigma_max=sigma_max_r,
        )

        # Second pass with narrowed ranges
        for sigma_hp in sigmas_refined:
            filtered = (
                _highpass_filter(image, sigma_hp) if sigma_hp > 0 else image.copy()
            )
            radii_ref, d_ref, _ = _compute_decorrelation_curve(
                filtered,
                num_radii,
                spacing=spacing_list,
                smoothing=smoothing,
                r_min=r_min2,
                r_max=r_max2,
                quantize=quantize,
            )
            r_peak, a_peak = _find_peak_in_curve(
                radii_ref, d_ref, min_amplitude=min_amplitude
            )
            all_curves.append(d_ref)
            all_peaks.append([r_peak, a_peak])

        # Rebuild peaks array with coarse + refined
        all_peaks_arr = np.array(all_peaks)
        r_peaks = all_peaks_arr[:, 0]

        # Use max_score from refined pass for final k_c
        refined_r_peaks = all_peaks_arr[num_highpass:, 0]
        if np.any(refined_r_peaks > 0):
            k_c_norm = float(np.max(refined_r_peaks))
        else:
            k_c_norm = float(np.max(r_peaks))

        # Use refined radii for return value
        radii = radii_ref
    else:
        k_c_norm = float(np.max(r_peaks))

    resolution = (
        _kc_to_resolution(k_c_norm, k_max) if k_max is not None else float("inf")
    )

    return resolution, radii, all_curves, all_peaks_arr


def _compute_decorrelation_curve(
    image: np.ndarray,
    num_radii: int = 100,
    spacing: Sequence[float] | None = None,
    smoothing: int | None = None,
    r_min: float = 0.0,
    r_max: float = 1.0,
    quantize: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute decorrelation curve d(r) for a single image (vectorized).

    Implements Equation 1 from the paper using radial binning and cumulative sums
    for GPU-efficient computation:
    d(r) = Pearson correlation between I(k) and I_n(k)·M(k,r)

    Parameters
    ----------
    image : np.ndarray
        Input image (pre-filtered and centered, numpy or cupy array)
    num_radii : int
        Number of radial sampling points
    spacing : np.ndarray, optional
        Physical spacing per axis. If None, uses index units.
    smoothing : int or None
        Savitzky-Golay filter window length for curve smoothing.
        Default: None (disabled). The decorrelation curve is intrinsically
        smooth (Descloux et al. 2019, Supplementary Note 1).
    quantize : bool
        If True, truncate d(r) values to 3 decimal places via
        ``floor(1000 * d) / 1000`` (ImDecorr convention). This suppresses
        spurious peaks on monotonically-increasing curves where per-step
        increments are smaller than 0.001. Default: False.

    Returns
    -------
    radii : np.ndarray
        Normalized frequency values (0 to 1) - on CPU
    d_curve : np.ndarray
        Decorrelation values d(r) - on CPU
    k_max : float
        Maximum frequency (Nyquist limit) in physical or index units
    """
    # Compute FFT (dispatches to cupy.fft if input is on GPU)
    F = np.fft.fftn(image)

    # Absolute value |F(k)|
    absF = np.abs(F)

    # Use shared infrastructure for frequency grid with correct anisotropy handling
    k_radius, k_max = radial_k_grid(image.shape, spacing=spacing)

    # Transfer k_radius to same device as image
    k_radius = to_same_device(k_radius, image)
    k_radius_norm = k_radius / k_max

    # Build radial bins
    radii_cpu = np.linspace(r_min, r_max, num_radii, dtype=np.float32)
    edges_cpu = np.concatenate([[r_min], radii_cpu]).astype(np.float32)
    edges = to_same_device(edges_cpu, image)

    # Assign pixels to radial bins.
    # Low-end clip (raw < 0 → 0): frequencies below r_min contribute to the
    # cumulative baseline in bin 0, preserving d(r)'s cumulative semantics.
    # High-end exclude (raw >= num_radii): frequencies above r_max are dropped
    # to avoid a boundary spike in the last bin.
    k_flat = k_radius_norm.ravel()
    raw_bin_id = np.digitize(k_flat, edges) - 1
    below_max = raw_bin_id < num_radii
    bin_id = np.clip(raw_bin_id[below_max], 0, num_radii - 1).astype(np.int32)
    absF_flat = absF.ravel()

    # Compute d(r) = cumsum(|F|) / sqrt(total_power * cumsum(count))
    cross_cumsum = np.cumsum(
        np.bincount(bin_id, weights=absF_flat[below_max], minlength=num_radii)
    )
    count_cumsum = np.cumsum(
        np.bincount(bin_id, minlength=num_radii).astype(np.float32)
    )
    power = np.sum(absF_flat**2)

    d_curve = cross_cumsum / (np.sqrt(power * count_cumsum) + 1e-10)
    d_curve = asnumpy(d_curve).astype(np.float32)
    if quantize:
        d_curve = np.floor(1000.0 * d_curve) / 1000.0

    return radii_cpu, _smooth_curve(d_curve, smoothing), k_max


def _compute_decorrelation_curve_sectioned(
    image: np.ndarray,
    num_radii: int = 100,
    angle_delta: int = 45,
    bin_delta: int = 1,
    spacing: Sequence[float] | None = None,
    exclude_axis_angle: float = 0.0,
    smoothing: int | None = None,
    quantize: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute sectioned decorrelation curves for 3D images.

    Uses polar angle from Z axis (0-90°):
    theta ≈ 0° → Z resolution, theta ≈ 90° → XY resolution.
    """
    if image.ndim != 3:
        raise ValueError("Sectioned DCR requires 3D images")

    shape = image.shape

    F = np.fft.fftn(image)
    absF = np.abs(F)
    absF_flat = absF.ravel()

    # Use max Nyquist so radial bins extend to XY frequencies.
    # Per-sector k_max handles the different normalization for Z vs XY.
    r_edges_raw, radii_raw = radial_edges(
        shape,
        bin_delta=bin_delta,
        spacing=spacing,
        use_max_nyquist=True,
    )
    n_radial_raw = len(radii_raw)
    r_edges = to_same_device(r_edges_raw, image)

    # Angular edges (polar 0-90°)
    n_angle = 90 // angle_delta
    angle_edges_cpu = np.array(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )
    angle_edges = to_same_device(angle_edges_cpu, image)

    # Per-sector k_max for resolution conversion
    # XY sector uses XY-Nyquist (max), Z sector uses Z-Nyquist (min)
    if spacing is not None:
        k_max_z = _kmax_phys(shape, spacing)
        k_max_xy = _kmax_phys_max(shape, spacing)
    else:
        k_max_z = min(n // 2 for n in shape)
        k_max_xy = max(n // 2 for n in shape)

    # Get sectioned bin IDs
    radial_id, angle_id = sectioned_bin_id(
        shape,  # type: ignore[arg-type]
        r_edges,
        angle_edges,
        spacing=list(spacing) if spacing is not None else None,
        exclude_axis_angle=exclude_axis_angle,
    )

    results = {}

    # Process each angular sector
    # Sectors with center < 45° are Z-dominated, >= 45° are XY-dominated
    for aid in range(n_angle):
        center_angle = (aid + 0.5) * angle_delta
        sector_name = "z" if center_angle < 45 else "xy"
        sector_k_max = k_max_z if sector_name == "z" else k_max_xy
        sector_mask = (angle_id == aid) & (radial_id >= 0)

        if not np.any(sector_mask):
            radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)
            results[sector_name] = (
                radii_norm,
                np.zeros(num_radii, dtype=np.float32),
                sector_k_max,
            )
            continue

        sector_radial_id = radial_id[sector_mask]
        sector_absF = absF_flat[sector_mask]

        # Compute d(r) = cumsum(|F|) / sqrt(power * cumsum(count))
        cross_cumsum = np.cumsum(
            np.bincount(sector_radial_id, weights=sector_absF, minlength=n_radial_raw)
        )
        count_cumsum = np.cumsum(
            np.bincount(sector_radial_id, minlength=n_radial_raw).astype(np.float32)
        )
        power = np.sum(sector_absF**2)
        d_curve_raw = cross_cumsum / (np.sqrt(power * count_cumsum) + 1e-10)

        # Resample to num_radii points, normalized to sector-specific k_max
        radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)
        d_curve = np.interp(
            radii_norm,
            radii_raw / sector_k_max,
            asnumpy(d_curve_raw).astype(np.float32),
        )
        if quantize:
            d_curve = np.floor(1000.0 * d_curve) / 1000.0

        results[sector_name] = (
            radii_norm,
            _smooth_curve(d_curve, smoothing),
            sector_k_max,
        )

    return results


def _dcr_curve_3d_sectioned(
    image: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    angle_delta: int = 45,
    bin_delta: int = 1,
    smoothing: int | None = None,
    exclude_axis_angle: float = 0.0,
    windowing: bool = True,
    refine: bool = True,
    return_curves: bool = False,
    quantize: bool = False,
    min_amplitude: float = 0.0,
) -> dict[str, float] | dict[str, dict]:
    """Compute 3D DCR with angular sectoring for XY and Z directions.

    Follows Descloux et al. 2019 algorithm:
    1. Compute d₀ (unfiltered) per sector to find initial peak r₀
    2. Use r₀ to set adaptive HP sigma range (ImDecorr convention)
    3. Sweep HP filters, find peaks per sector
    4. Optional 2-pass refinement (default: on)

    When *return_curves* is True, returns per-sector dicts with full curve data.
    """
    if image.ndim != 3:
        raise ValueError("3D sectioned DCR requires 3D images")

    image = _preprocess_dcr_image(image, windowing=windowing)

    common_kwargs = dict(
        num_radii=num_radii,
        angle_delta=angle_delta,
        bin_delta=bin_delta,
        spacing=spacing,
        exclude_axis_angle=exclude_axis_angle,
        smoothing=smoothing,
        quantize=quantize,
    )

    # --- Step 1: Compute d₀ (unfiltered) to anchor sigma range ---
    d0_results = _compute_decorrelation_curve_sectioned(image, **common_kwargs)  # type: ignore[arg-type]

    r0 = {}
    all_peaks: dict[str, list[tuple[float, float]]] = {"xy": [], "z": []}
    all_curves: dict[str, list[np.ndarray]] = {"xy": [], "z": []}
    for sector_name in ["xy", "z"]:
        radii, d_curve, _ = d0_results[sector_name]
        r_peak, a_peak = _find_peak_in_curve(
            radii, d_curve, min_amplitude=min_amplitude
        )
        r0[sector_name] = r_peak
        all_peaks[sector_name].append((r_peak, a_peak))
        if return_curves:
            all_curves[sector_name].append(asnumpy(d_curve))

    # --- Step 2: Adaptive sigma range per sector ---
    # ImDecorr: g from gMax=2/r₀ (weak HP) to 0.15 (strong HP)
    # Conversion: σ_cubic ≈ 2g/π
    # Plus extra weak-HP entry at g=size(im)/4 → σ≈2*(max(shape)/4)/π
    sector_sigmas: dict[str, np.ndarray] = {}
    for sector_name in ["xy", "z"]:
        r0_val = r0[sector_name]
        if r0_val > 0:
            g_max = 2.0 / r0_val
            sigma_max_adaptive = 2.0 * g_max / np.pi
        else:
            # Fallback if no d₀ peak
            sigma_max_adaptive = max(image.shape) / 4.0
        # Spatial Gaussians with σ < 1 pixel are degenerate on discrete grids.
        # ImDecorr's g=0.15 maps to σ≈0.096, but we floor at 1.0 pixel.
        sigma_min_adaptive = 1.0
        # Extra weak-HP entry (ImDecorr: g=size(im)/4)
        sigma_extra_weak = 2.0 * (max(image.shape) / 4.0) / np.pi
        # Generate log-spaced sigmas from strong to weak
        base_sigmas = _generate_highpass_sigmas(
            image.shape,
            num_highpass,
            sigma_min=sigma_min_adaptive,
            sigma_max=max(sigma_max_adaptive, sigma_min_adaptive + 0.1),
        )
        # Prepend extra weak-HP entry (like ImDecorr line 111)
        sector_sigmas[sector_name] = np.concatenate([[sigma_extra_weak], base_sigmas])

    # --- Step 3: Coarse pass with adaptive sigmas ---
    # Use the union of both sector sigma sets
    all_sigmas_set = set(sector_sigmas["xy"].tolist() + sector_sigmas["z"].tolist())
    all_sigmas = np.array(sorted(all_sigmas_set), dtype=np.float32)

    for sigma_hp in all_sigmas:
        filtered_image = _highpass_filter(image, sigma_hp)
        sector_results = _compute_decorrelation_curve_sectioned(
            filtered_image,
            **common_kwargs,  # type: ignore[arg-type]
        )
        for sector_name in ["xy", "z"]:
            radii, d_curve, _ = sector_results[sector_name]
            r_peak, a_peak = _find_peak_in_curve(
                radii, d_curve, min_amplitude=min_amplitude
            )
            all_peaks[sector_name].append((r_peak, a_peak))
            if return_curves:
                all_curves[sector_name].append(asnumpy(d_curve))
        del filtered_image

    # --- Step 4: Optional 2-pass refinement ---
    if refine:
        for sector_name in ["xy", "z"]:
            peaks = np.array(all_peaks[sector_name])
            s_arr = sector_sigmas[sector_name]
            refined = _refinement_ranges(peaks, s_arr, r_max_pad=0.4)
            if refined is None:
                continue

            r_min2, r_max2, sigma_min_r, sigma_max_r = refined
            refined_sigmas = _generate_highpass_sigmas(
                image.shape,
                num_highpass,
                sigma_min=sigma_min_r,
                sigma_max=sigma_max_r,
            )

            for sigma_hp in refined_sigmas:
                filtered = _highpass_filter(image, sigma_hp)
                sr = _compute_decorrelation_curve_sectioned(filtered, **common_kwargs)  # type: ignore[arg-type]
                radii, d_curve, _ = sr[sector_name]
                r_peak, a_peak = _find_peak_in_curve(
                    radii, d_curve, min_amplitude=min_amplitude
                )
                all_peaks[sector_name].append((r_peak, a_peak))
                del filtered

    # --- Step 5: Compute resolution for each sector ---
    results: dict[str, dict] = {}
    for sector_name in ["xy", "z"]:
        peaks_arr = np.array(all_peaks[sector_name])
        r_peaks = peaks_arr[:, 0]
        k_c_norm = float(np.max(r_peaks))

        radii_cpu, _, k_max = d0_results[sector_name]
        radii_cpu = asnumpy(radii_cpu)

        resolution = _kc_to_resolution(k_c_norm, k_max)

        results[sector_name] = {
            "resolution": resolution,
            "radii": radii_cpu,
            "peaks": peaks_arr,
            "k_max": float(k_max),
        }
        if return_curves:
            results[sector_name]["curves"] = all_curves[sector_name]

    if not return_curves:
        # Backwards-compatible: return dict[str, float]
        return {k: v["resolution"] for k, v in results.items()}

    return results


def dcr_curve_3d_sectioned(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    angle_delta: int = 45,
    exclude_axis_angle: float = 0.0,
    windowing: bool = True,
    refine: bool = True,
    quantize: bool = False,
    min_amplitude: float = 0.0,
) -> dict[str, dict]:
    """Compute 3D DCR with angular sectoring, returning full curve data.

    Like :func:`dcr_resolution` with ``use_sectioned=True``, but returns
    per-sector curves and peaks for plotting.

    Parameters
    ----------
    image : np.ndarray
        3D input image (numpy or cupy array).
    spacing : float or sequence of floats, optional
        Physical spacing per axis [z, y, x]. If None, uses index units.
    num_radii : int
        Number of radial sampling points (default: 100).
    num_highpass : int
        Number of high-pass filters (default: 10).
    angle_delta : int
        Angular sector width in degrees (default: 45).
    exclude_axis_angle : float
        Exclude frequencies within this angle from Z axis (default: 0.0).
    windowing : bool
        Apply Tukey window for edge apodization (default: True).
    refine : bool
        Two-pass refinement (default: True).
    quantize : bool
        Truncate d(r) to 3 decimal places (ImDecorr convention).
        Default: False.
    min_amplitude : float
        Reject peaks below this amplitude (ImDecorr SNR gate: 0.05).
        Default: 0.0 (disabled).

    Returns
    -------
    dict[str, dict]
        Keys ``"xy"`` and ``"z"``, each containing:

        - ``"resolution"`` : float — estimated resolution in physical units
        - ``"radii"`` : np.ndarray — normalized frequencies (0..1)
        - ``"curves"`` : list[np.ndarray] — decorrelation curves (d₀ + highpass)
        - ``"peaks"`` : np.ndarray, shape (N, 2) — [r_peak, amplitude] per curve
        - ``"k_max"`` : float — physical k_max for the sector
    """
    spacing_list = _normalize_spacing(spacing, 3)

    return _dcr_curve_3d_sectioned(  # type: ignore[return-value]
        image,
        spacing=spacing_list,
        num_radii=num_radii,
        num_highpass=num_highpass,
        angle_delta=angle_delta,
        exclude_axis_angle=exclude_axis_angle,
        windowing=windowing,
        refine=refine,
        return_curves=True,
        quantize=quantize,
        min_amplitude=min_amplitude,
    )


def _highpass_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian high-pass filter (device-agnostic)."""
    if sigma <= 0:
        return image.copy()
    return image - filters.gaussian(image, sigma=sigma, preserve_range=True)


def _generate_highpass_sigmas(
    image_shape: tuple[int, ...],
    num_highpass: int,
    sigma_min: float = 1.0,
    sigma_max: float | None = None,
) -> np.ndarray:
    """Generate log-spaced high-pass sigmas (NanoPyx convention).

    The floor is 1.0 pixel because spatial-domain Gaussian subtraction
    (image - gaussian(image, σ)) becomes degenerate when σ < ~1 px: the
    discrete kernel collapses to near-identity and the high-pass filter
    removes almost no signal.

    Parameters
    ----------
    image_shape : tuple of int
        Shape of the image.
    num_highpass : int
        Number of high-pass sigmas to generate.
    sigma_min : float
        Minimum sigma (floor at 1.0 pixel).
    sigma_max : float or None
        Maximum sigma. If None, uses min(image_shape) / 2.
    """
    sigma_min = max(sigma_min, 1.0)
    if sigma_max is None:
        sigma_max = min(image_shape) / 2.0
    if num_highpass <= 1:
        return np.array([sigma_min], dtype=np.float32)
    log_min = np.log(sigma_min)
    log_max = np.log(sigma_max)
    return np.exp(np.linspace(log_min, log_max, num_highpass)).astype(np.float32)


def dcr_resolution(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    exclude_axis_angle: float = 0.0,
    use_sectioned: bool = True,
    windowing: bool = True,
    refine: bool = True,
    quantize: bool = False,
    min_amplitude: float = 0.0,
) -> float | dict[str, float]:
    """
    Calculate resolution using DCR algorithm (Descloux et al. 2019).

    For 3D images, supports full 3D analysis with angular sectioning (similar to FSC)
    following the methodology from Koho et al. 2019 for directional resolution.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D input image (numpy or cupy array)
    spacing : float or sequence of floats, optional
        Physical spacing per axis in nm/μm. If None, returns in index units.
        For 3D: [z_spacing, y_spacing, x_spacing]
    num_radii : int
        Number of radial sampling points (default: 100)
    num_highpass : int
        Number of high-pass filters (default: 10).
        Uses logarithmically-spaced sigmas following NanoPyx convention.
    exclude_axis_angle : float
        Exclude frequencies within this angle (degrees) from the Z axis.
        Following Koho et al. 2019 to avoid artifacts from piezo stage motion.
        Default: 0.0 (no exclusion). Typical value: 5.0. Only used for 3D.
    use_sectioned : bool
        If True (default), use full 3D angular sectioning for 3D images.
        If False, use legacy 2D slice-based approach.
    windowing : bool
        If True (default), apply internal Tukey window for edge apodization.
        Set to False when windowing is applied externally for consistent
        preprocessing across different methods.
    refine : bool
        If True, run a second refinement pass with narrowed frequency and
        sigma ranges around the coarse peaks (NanoPyx two-pass strategy).
        Default: True.
    quantize : bool
        If True, truncate d(r) values to 3 decimal places via
        ``floor(1000 * d) / 1000`` (ImDecorr convention). This suppresses
        spurious peaks on monotonically-increasing curves. Default: False.
    min_amplitude : float
        Reject peaks with amplitude below this value (ImDecorr SNR gate
        uses 0.05). Default: 0.0 (disabled).

    Returns
    -------
    resolution : float or dict
        For 2D: Estimated resolution in physical units (if spacing provided)
        For 3D: Dict with 'xy' and 'z' resolutions

    Examples
    --------
    >>> from cubic.metrics.spectral import dcr_resolution
    >>> import numpy as np
    >>> # 2D image
    >>> image_2d = np.random.randn(64, 64)
    >>> res = dcr_resolution(image_2d, spacing=0.065)  # 0.065 μm/pixel
    >>> print(f"Resolution: {res:.2f} μm")
    >>> # 3D image with full sectioned analysis
    >>> image_3d = np.random.randn(30, 64, 64)
    >>> res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065],
    ...                      exclude_axis_angle=5.0)
    >>> print(f"XY: {res['xy']:.2f} μm, Z: {res['z']:.2f} μm")
    """
    if image.ndim == 2:
        # 2D: return single resolution
        resolution, _, _, _ = dcr_curve(
            image,
            spacing=spacing,
            num_radii=num_radii,
            num_highpass=num_highpass,
            windowing=windowing,
            refine=refine,
            quantize=quantize,
            min_amplitude=min_amplitude,
        )
        return resolution

    elif image.ndim == 3:
        spacing_list = _normalize_spacing(spacing, 3)

        if use_sectioned:
            # Full 3D analysis with angular sectoring
            return _dcr_curve_3d_sectioned(  # type: ignore[return-value]
                image,
                spacing=spacing_list,
                num_radii=num_radii,
                num_highpass=num_highpass,
                angle_delta=45,
                exclude_axis_angle=exclude_axis_angle,
                windowing=windowing,
                refine=refine,
                quantize=quantize,
                min_amplitude=min_amplitude,
            )
        else:
            # Legacy: compute directional resolutions from 2D slices
            if spacing_list is None:
                spacing_z = spacing_y = spacing_x = 1.0
            else:
                spacing_z, spacing_y, spacing_x = spacing_list

            # XY resolution: analyze middle XY slice
            mid_z = image.shape[0] // 2
            xy_slice = image[mid_z, :, :]
            xy_spacing = [spacing_y, spacing_x]
            res_xy, _, _, _ = dcr_curve(
                xy_slice,
                spacing=xy_spacing,
                num_radii=num_radii,
                num_highpass=num_highpass,
                windowing=windowing,
                quantize=quantize,
                min_amplitude=min_amplitude,
            )

            # Z resolution: analyze middle XZ slice
            mid_y = image.shape[1] // 2
            xz_slice = image[:, mid_y, :]
            xz_spacing = [spacing_z, spacing_x]
            res_xz, _, _, _ = dcr_curve(
                xz_slice,
                spacing=xz_spacing,
                num_radii=num_radii,
                num_highpass=num_highpass,
                windowing=windowing,
                quantize=quantize,
                min_amplitude=min_amplitude,
            )

            return {"xy": res_xy, "z": res_xz}

    else:
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")
