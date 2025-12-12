"""Decorrelation Analysis (DCR) - Descloux et al. 2019.

Implementation following the algorithm described in:
"Parameter-free image resolution estimation based on decorrelation analysis"
Nature Methods 16:918-924 (2019)
"""

from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks

from cubic.cuda import asnumpy
from cubic.skimage import filters


def dcr_curve(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 30,
    sigma_range: tuple[float, float] = (0.0, 0.5),
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
        Number of Gaussian high-pass filters to apply (default: 30)
    sigma_range : tuple of float
        Range of high-pass filter sigmas in normalized frequency units (default: (0.0, 0.5))

    Returns
    -------
    resolution : float
        Estimated resolution (k_c = max of all peak positions)
    radii : np.ndarray
        Normalized frequency values for mask radii
    all_curves : list of np.ndarray
        List of d(r) curves for each high-pass filtered version
    all_peaks : np.ndarray
        Array of [r_i, A_i] pairs (peak position and amplitude), shape (N, 2)

    Notes
    -----
    Implements the canonical decorrelation analysis:
    1. Compute FFT and normalize: I_n(k) = I(k) / |I(k)|
    2. For each radius r, compute Pearson correlation d(r)
    3. Apply N_g high-pass filters and repeat
    4. Resolution = max(r_0, r_1, ..., r_Ng)
    """
    # Validate dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")

    # Ensure float32 and center
    image = image.astype(np.float32, copy=False)
    image = image - np.mean(image)

    # Optional: Apply Tukey window for edge apodization
    image = _apply_tukey_window(image, alpha=0.1)

    # Normalize spacing
    if spacing is None:
        spacing_arr = np.ones(image.ndim, dtype=np.float32)
    elif isinstance(spacing, (int, float)):
        spacing_arr = np.full(image.ndim, float(spacing), dtype=np.float32)
    else:
        spacing_arr = np.array(spacing, dtype=np.float32)

    # Generate high-pass filter sigmas (normalized frequency units)
    sigma_min, sigma_max = sigma_range
    if num_highpass > 1:
        sigmas = np.linspace(sigma_min, sigma_max, num_highpass)
    else:
        sigmas = np.array([sigma_min])

    # Storage for all curves and peaks
    all_curves = []
    all_peaks = []
    k_max = None  # Will be set from first call

    # Process each high-pass filtered version
    for sigma_hp in sigmas:
        # Apply high-pass filter
        if sigma_hp > 0:
            filtered_image = _highpass_filter(image, sigma_hp, spacing_arr)
        else:
            filtered_image = image.copy()

        # Compute decorrelation curve for this filtered image
        # Pass spacing for correct anisotropy handling
        radii, d_curve, k_max = _compute_decorrelation_curve(
            filtered_image,
            num_radii,
            spacing=spacing_arr if spacing is not None else None,
        )

        # Find local maximum
        peaks, properties = find_peaks(d_curve, plateau_size=(1, None))

        if len(peaks) > 0:
            # Take first peak (highest correlation maximum)
            peak_idx = peaks[0]
            r_peak = radii[peak_idx]
            a_peak = d_curve[peak_idx]
        else:
            # No peak found - too much filtering
            r_peak = 0.0
            a_peak = 0.0

        all_curves.append(d_curve)
        all_peaks.append([r_peak, a_peak])

    # Extract peak positions
    all_peaks = np.array(all_peaks)
    r_peaks = all_peaks[:, 0]

    # Resolution is the MAXIMUM peak frequency (Equation 2 in paper)
    # k_c is in normalized units (0 to 1), where 1 = Nyquist = k_max
    k_c_norm = np.max(r_peaks)

    # Convert normalized k_c to physical frequency, then to resolution
    # k_c_physical = k_c_norm * k_max (in cycles per unit length)
    # Resolution = 1 / k_c_physical
    if k_c_norm > 0 and k_max is not None:
        k_c_physical = k_c_norm * k_max
        resolution = 1.0 / k_c_physical
    else:
        resolution = float("inf")

    return resolution, radii, all_curves, all_peaks


def _compute_decorrelation_curve(
    image: np.ndarray,
    num_radii: int = 100,
    spacing: np.ndarray | None = None,
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

    Returns
    -------
    radii : np.ndarray
        Normalized frequency values (0 to 1) - on CPU
    d_curve : np.ndarray
        Decorrelation values d(r) - on CPU
    k_max : float
        Maximum frequency (Nyquist limit) in physical or index units
    """
    from cubic.cuda import get_array_module

    from .radial import radial_k_grid

    xp = get_array_module(image)

    # Compute FFT (dispatches to cupy.fft if input is on GPU)
    F = np.fft.fftn(image)

    # Absolute value |F(k)|
    absF = np.abs(F)

    # Use shared infrastructure for frequency grid with correct anisotropy handling
    k_radius, k_max = radial_k_grid(image.shape, spacing=spacing)

    # Transfer k_radius to same device as image
    if xp is not np:
        k_radius = xp.asarray(k_radius)

    # Normalize frequencies to [0, 1] range
    k_radius_norm = k_radius / k_max

    # Create radii from 0 to 1 (same as original loop-based version)
    radii_cpu = np.linspace(0, 1, num_radii, dtype=np.float32)

    # Create bin edges: each radius r defines a bin [0, r]
    # We need edges that correspond to the radii values
    # Use radii as right edges of bins, with 0 as first left edge
    edges_cpu = np.concatenate([[0], radii_cpu]).astype(np.float32)
    edges = xp.asarray(edges_cpu) if xp is not np else edges_cpu

    # Assign each pixel to a radial bin (vectorized)
    k_flat = k_radius_norm.ravel()
    bin_id = np.digitize(k_flat, edges) - 1
    bin_id = np.clip(bin_id, 0, num_radii - 1).astype(np.int32)

    # Flatten |F| for bincount
    absF_flat = absF.ravel()

    # Per-bin sums using bincount (vectorized, GPU-accelerated)
    # The decorrelation formula simplifies because |F_normalized| = 1:
    # d(r) = sum_{k<=r}(|F(k)|) / sqrt(sum_all(|F|²) * count(k<=r))
    #
    # Derivation:
    # - F_normalized = F / |F|, so |F_normalized| = 1
    # - numerator = Re{sum(F * conj(F_n * M))} = sum(|F| * M)
    # - denom1 = sum(|F|²) [total power, constant]
    # - denom2 = sum(|F_n * M|²) = sum(M) = count [since |F_n|=1]

    # Sum |F| per bin
    cross_bin = np.bincount(bin_id, weights=absF_flat, minlength=num_radii)
    # Count pixels per bin
    count_bin = np.bincount(bin_id, minlength=num_radii).astype(np.float32)

    # Cumulative sums for d(r) which uses all frequencies <= r
    cross_cumsum = np.cumsum(cross_bin)
    count_cumsum = np.cumsum(count_bin)

    # Total power (constant denominator term)
    power = np.sum(absF_flat**2)

    # Compute d(r) = cross_cumsum / sqrt(power * count_cumsum)
    eps = 1e-10
    d_curve = cross_cumsum / (np.sqrt(power * count_cumsum) + eps)

    radii = radii_cpu

    # Transfer to CPU for peak finding (scipy doesn't support GPU)
    d_curve = asnumpy(d_curve).astype(np.float32)

    return radii, d_curve, k_max


def _highpass_filter(
    image: np.ndarray,
    sigma: float,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Apply Gaussian high-pass filter.

    Parameters
    ----------
    image : np.ndarray
        Input image (numpy or cupy array)
    sigma : float
        High-pass cutoff in normalized frequency units (0-1)
    spacing : np.ndarray
        Physical spacing per axis

    Returns
    -------
    filtered : np.ndarray
        High-pass filtered image (same device as input)
    """
    if sigma <= 0:
        return image.copy()

    # Convert sigma from normalized frequency to real space pixels
    # sigma in freq domain → 1/sigma in spatial domain
    # Normalized freq 1 = Nyquist = 0.5 cycles/pixel
    # So sigma=0.1 in norm freq ≈ cutoff at 0.05 cycles/pixel ≈ 20 pixels
    sigma_spatial = 1.0 / (2.0 * np.pi * sigma + 1e-6)

    # Compute low-pass (Gaussian blur) - cubic.skimage.filters.gaussian is device-agnostic
    lowpass = filters.gaussian(image, sigma=sigma_spatial, preserve_range=True)

    # High-pass = original - low-pass
    highpass = image - lowpass

    return highpass


def _apply_tukey_window(image: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply Tukey (tapered cosine) window for edge apodization.

    Parameters
    ----------
    image : np.ndarray
        Input image (numpy or cupy array)
    alpha : float
        Taper fraction (0=rectangular, 1=Hann)

    Returns
    -------
    windowed : np.ndarray
        Windowed image (same device as input)
    """
    from scipy.signal.windows import tukey

    from cubic.cuda import get_array_module

    # Create separable Tukey windows for each dimension (on CPU from scipy)
    windows = []
    for axis_size in image.shape:
        w = tukey(axis_size, alpha=alpha).astype(image.dtype)
        windows.append(w)

    # Create N-D window by outer product (on CPU)
    if image.ndim == 2:
        window_nd = np.outer(windows[0], windows[1])
    elif image.ndim == 3:
        window_nd = np.einsum("i,j,k->ijk", windows[0], windows[1], windows[2])
    else:
        return image

    # Transfer window to same device as image (scipy only works on CPU)
    xp = get_array_module(image)
    window_nd = xp.asarray(window_nd)

    return image * window_nd


def dcr_resolution(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 30,
    sigma_range: tuple[float, float] = (0.0, 0.5),
) -> float | dict[str, float]:
    """
    Calculate resolution using DCR algorithm (Descloux et al. 2019).

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
        Number of high-pass filters (default: 30)
    sigma_range : tuple of float
        Range of high-pass sigmas in normalized frequency (default: (0.0, 0.5))

    Returns
    -------
    resolution : float or dict
        For 2D: Estimated resolution in physical units (if spacing provided)
        For 3D: Dict with 'xy' and 'z' resolutions, computed from 2D slices

    Examples
    --------
    >>> from cubic.metrics.frc import dcr_resolution
    >>> import numpy as np
    >>> # 2D image
    >>> image_2d = np.random.randn(64, 64)
    >>> res = dcr_resolution(image_2d, spacing=0.065)  # 0.065 μm/pixel
    >>> print(f"Resolution: {res:.2f} μm")
    >>> # 3D image
    >>> image_3d = np.random.randn(30, 64, 64)
    >>> res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065])
    >>> print(f"XY: {res['xy']:.2f} μm, Z: {res['z']:.2f} μm")
    """
    if image.ndim == 2:
        # 2D: return single resolution
        resolution, _, _, _ = dcr_curve(
            image,
            spacing=spacing,
            num_radii=num_radii,
            num_highpass=num_highpass,
            sigma_range=sigma_range,
        )
        return resolution

    elif image.ndim == 3:
        # 3D: compute directional resolutions from 2D slices
        # Normalize spacing
        if spacing is None:
            spacing_z = 1.0
            spacing_y = 1.0
            spacing_x = 1.0
        elif isinstance(spacing, (int, float)):
            spacing_z = spacing_y = spacing_x = float(spacing)
        else:
            spacing_z, spacing_y, spacing_x = spacing[0], spacing[1], spacing[2]

        # XY resolution: analyze middle XY slice
        mid_z = image.shape[0] // 2
        xy_slice = image[mid_z, :, :]
        xy_spacing = [spacing_y, spacing_x]
        res_xy, _, _, _ = dcr_curve(
            xy_slice,
            spacing=xy_spacing,
            num_radii=num_radii,
            num_highpass=num_highpass,
            sigma_range=sigma_range,
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
            sigma_range=sigma_range,
        )

        return {"xy": res_xy, "z": res_xz}

    else:
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")
