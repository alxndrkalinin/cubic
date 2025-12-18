"""Decorrelation Analysis (DCR) - Descloux et al. 2019."""

from collections.abc import Sequence

import numpy as np
from scipy.signal import find_peaks

from cubic.cuda import asnumpy, get_array_module
from cubic.skimage import filters
from cubic.image_utils import rescale_isotropic

from .radial import _kmax_phys, radial_edges, radial_k_grid, sectioned_bin_id


def dcr_curve(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
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

    The high-pass filter sigmas are logarithmically spaced from 0.5 pixels
    to min(image_shape)/2 pixels, following the NanoPyx/ImageJ DecorrAnalysis
    convention from Descloux et al. 2019.
    """
    # Validate dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")

    # Ensure float32 and center (copy to avoid modifying input)
    image = image.astype(np.float32)
    image -= np.mean(image)

    # Optional: Apply Tukey window for edge apodization
    image = _apply_tukey_window(image, alpha=0.1)

    # Normalize spacing
    if spacing is None:
        spacing_arr = np.ones(image.ndim, dtype=np.float32)
    elif isinstance(spacing, (int, float)):
        spacing_arr = np.full(image.ndim, float(spacing), dtype=np.float32)
    else:
        spacing_arr = np.array(spacing, dtype=np.float32)

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


def _compute_decorrelation_curve_sectioned(
    image: np.ndarray,
    num_radii: int = 100,
    angle_delta: int = 45,
    spacing: np.ndarray | None = None,
    exclude_axis_angle: float = 0.0,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute sectioned decorrelation curves d(r,θ) for 3D images.

    Uses angular sectioning similar to FSC to compute separate DCR curves
    for XY (lateral) and Z (axial) directions.

    Parameters
    ----------
    image : np.ndarray
        3D input image (pre-filtered and centered, numpy or cupy array)
    num_radii : int
        Number of radial sampling points
    angle_delta : int
        Angular bin width in degrees. Default 45 gives 2 bins (XY and Z).
    spacing : np.ndarray, optional
        Physical spacing per axis [z, y, x]. If None, uses index units.
    exclude_axis_angle : float
        Exclude frequencies within this angle (degrees) from Z axis.
        Following Koho et al. 2019. Default: 0.0 (no exclusion).

    Returns
    -------
    dict mapping 'xy' and 'z' to (radii, d_curve, k_max) tuples
    """
    if image.ndim != 3:
        raise ValueError("Sectioned DCR requires 3D images")

    xp = get_array_module(image)

    # Compute FFT
    F = np.fft.fftn(image)
    absF = np.abs(F)

    # Build radial edges using shared infrastructure
    # Use bin_delta=1 for fine radial resolution, then resample to num_radii
    shape = image.shape
    r_edges_raw, radii_raw = radial_edges(shape, bin_delta=1, spacing=spacing)

    # Angular edges: 0°=Z axis, 90°=XY plane (internal convention)
    # With angle_delta=45: bin 0 covers 0-45° (Z-like), bin 1 covers 45-90° (XY-like)
    n_angle = 90 // angle_delta
    angle_edges_cpu = np.array(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )

    # Transfer to device
    r_edges = xp.asarray(r_edges_raw) if xp is not np else r_edges_raw
    angle_edges = xp.asarray(angle_edges_cpu) if xp is not np else angle_edges_cpu

    # Get sectioned bin IDs
    radial_id, angle_id = sectioned_bin_id(
        shape,
        r_edges,
        angle_edges,
        spacing=list(spacing) if spacing is not None else None,
        exclude_axis_angle=exclude_axis_angle,
    )

    # Flatten arrays
    absF_flat = absF.ravel()
    n_radial_raw = len(radii_raw)

    # Compute k_max for resolution conversion
    if spacing is not None:
        k_max = _kmax_phys(shape, spacing)
    else:
        k_max = min(n // 2 for n in shape)

    results = {}

    # Process each angular sector
    for aid in range(n_angle):
        # Map to output name following Koho convention:
        # aid=0 (polar 0-45°, near Z axis) → 'z'
        # aid=1 (polar 45-90°, near XY plane) → 'xy'
        if aid == 0:
            sector_name = "z"
        else:
            sector_name = "xy"

        # Get mask for this angular sector (valid pixels only)
        sector_mask = (angle_id == aid) & (radial_id >= 0)

        if not np.any(sector_mask):
            # No valid pixels in this sector
            radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)
            d_curve = np.zeros(num_radii, dtype=np.float32)
            results[sector_name] = (radii_norm, d_curve, k_max)
            continue

        # Extract data for this sector
        sector_radial_id = radial_id[sector_mask]
        sector_absF = absF_flat[sector_mask]

        # Per-bin sums within this sector
        cross_bin = np.bincount(
            sector_radial_id, weights=sector_absF, minlength=n_radial_raw
        )
        count_bin = np.bincount(sector_radial_id, minlength=n_radial_raw).astype(
            np.float32
        )

        # Cumulative sums for d(r) which uses all frequencies <= r
        cross_cumsum = np.cumsum(cross_bin)
        count_cumsum = np.cumsum(count_bin)

        # Total power within sector
        power = np.sum(sector_absF**2)

        # Compute d(r) = cross_cumsum / sqrt(power * count_cumsum)
        eps = 1e-10
        d_curve_raw = cross_cumsum / (np.sqrt(power * count_cumsum) + eps)

        # Resample to num_radii points for consistent output
        # radii_raw is in physical or index units, normalize to [0, 1]
        radii_norm_raw = radii_raw / k_max

        # Create output radii
        radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)

        # Interpolate d_curve to new radii
        d_curve_raw_cpu = asnumpy(d_curve_raw).astype(np.float32)
        d_curve = np.interp(radii_norm, radii_norm_raw, d_curve_raw_cpu)

        results[sector_name] = (radii_norm, d_curve, k_max)

    return results


def _dcr_curve_3d_sectioned(
    image: np.ndarray,
    *,
    spacing: np.ndarray | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    angle_delta: int = 45,
    exclude_axis_angle: float = 0.0,
) -> dict[str, float]:
    """
    Compute 3D DCR with angular sectioning for XY and Z directions.

    Parameters
    ----------
    image : np.ndarray
        3D input image (numpy or cupy array)
    spacing : np.ndarray, optional
        Physical spacing per axis [z, y, x]
    num_radii : int
        Number of radial sampling points
    num_highpass : int
        Number of high-pass filters (default: 10).
        Uses logarithmically-spaced sigmas following NanoPyx convention.
    angle_delta : int
        Angular bin width in degrees (default: 45)
    exclude_axis_angle : float
        Exclude frequencies near Z axis (degrees)

    Returns
    -------
    dict with 'xy' and 'z' resolution values
    """
    if image.ndim != 3:
        raise ValueError("3D sectioned DCR requires 3D images")

    # Ensure float32 and center (copy to avoid modifying input)
    image = image.astype(np.float32)
    image -= np.mean(image)

    # Apply Tukey window for edge apodization
    image = _apply_tukey_window(image, alpha=0.1)

    # Normalize spacing
    if spacing is None:
        spacing_arr = None
    else:
        spacing_arr = np.array(spacing, dtype=np.float32)

    # Generate log-spaced high-pass filter sigmas (in pixels)
    sigmas = _generate_highpass_sigmas(image.shape, num_highpass)

    # Storage for peaks per sector
    all_peaks = {"xy": [], "z": []}

    # Process each high-pass filtered version
    for sigma_hp in sigmas:
        # Apply high-pass filter
        if sigma_hp > 0:
            filtered_image = _highpass_filter(image, sigma_hp)
        else:
            filtered_image = image.copy()

        # Compute sectioned decorrelation curves
        sector_results = _compute_decorrelation_curve_sectioned(
            filtered_image,
            num_radii=num_radii,
            angle_delta=angle_delta,
            spacing=spacing_arr,
            exclude_axis_angle=exclude_axis_angle,
        )

        # Find peaks for each sector
        for sector_name in ["xy", "z"]:
            radii, d_curve, k_max = sector_results[sector_name]

            # Find local maximum, excluding boundary artifacts
            # Peaks at r > 0.9 are typically boundary artifacts from the
            # cumulative sum reaching the edge of the frequency range
            peaks, _ = find_peaks(d_curve, plateau_size=(1, None))

            r_peak = 0.0
            for peak_idx in peaks:
                r_candidate = radii[peak_idx]
                if r_candidate < 0.9:  # Exclude boundary artifacts
                    r_peak = r_candidate
                    break

            all_peaks[sector_name].append(r_peak)

    # Compute resolution for each sector
    resolutions = {}
    for sector_name in ["xy", "z"]:
        r_peaks = np.array(all_peaks[sector_name])
        k_c_norm = np.max(r_peaks)

        # Get k_max from last computation
        _, _, k_max = sector_results[sector_name]

        if k_c_norm > 0 and k_max > 0:
            k_c_physical = k_c_norm * k_max
            resolutions[sector_name] = 1.0 / k_c_physical
        else:
            resolutions[sector_name] = float("inf")

    return resolutions


def _highpass_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian high-pass filter.

    Parameters
    ----------
    image : np.ndarray
        Input image (numpy or cupy array)
    sigma : float
        High-pass cutoff sigma in pixels (Gaussian blur radius)

    Returns
    -------
    filtered : np.ndarray
        High-pass filtered image (same device as input)
    """
    if sigma <= 0:
        return image.copy()

    # Compute low-pass (Gaussian blur) - cubic.skimage.filters.gaussian is device-agnostic
    # sigma is directly in pixels (following NanoPyx convention)
    lowpass = filters.gaussian(image, sigma=sigma, preserve_range=True)

    # High-pass = original - low-pass
    highpass = image - lowpass

    return highpass


def _generate_highpass_sigmas(
    image_shape: tuple[int, ...],
    num_highpass: int,
    sigma_min: float = 0.5,
    sigma_max: float | None = None,
) -> np.ndarray:
    """
    Generate logarithmically-spaced high-pass filter sigmas.

    Following NanoPyx/ImageJ DecorrAnalysis convention:
    - Sigmas are in pixel units (Gaussian blur radius)
    - Logarithmic spacing from sigma_min to sigma_max
    - Default sigma_max is half the smallest image dimension

    Parameters
    ----------
    image_shape : tuple
        Shape of the image (used to determine sigma_max)
    num_highpass : int
        Number of high-pass filters to generate
    sigma_min : float
        Minimum sigma in pixels (default: 0.5, very weak blur)
    sigma_max : float, optional
        Maximum sigma in pixels. If None, uses min(shape)/2.

    Returns
    -------
    sigmas : np.ndarray
        Array of sigma values in pixels, log-spaced
    """
    if sigma_max is None:
        # Default: half the smallest dimension (following NanoPyx g_max)
        sigma_max = min(image_shape) / 2.0

    if num_highpass <= 1:
        return np.array([sigma_min], dtype=np.float32)

    # Logarithmic spacing (like NanoPyx)
    # sig = exp(log(g_min) + (log(g_max) - log(g_min)) * k / (n_g - 1))
    log_min = np.log(max(sigma_min, 0.1))  # Avoid log(0)
    log_max = np.log(sigma_max)

    sigmas = np.exp(np.linspace(log_min, log_max, num_highpass)).astype(np.float32)

    return sigmas


def _tukey_window_1d(n: int, alpha: float, xp) -> np.ndarray:
    """
    Create 1D Tukey window on the specified device (CPU or GPU).

    Device-agnostic implementation matching scipy.signal.windows.tukey exactly.
    """
    if alpha <= 0:
        return xp.ones(n, dtype=np.float32)
    if alpha >= 1:
        # Hann window
        idx = xp.arange(n, dtype=np.float64)
        return (0.5 * (1 - xp.cos(2 * np.pi * idx / (n - 1)))).astype(np.float32)

    # Tukey window: flat in middle, cosine taper at edges
    # Following scipy's exact formula
    width = int(np.floor(alpha * (n - 1) / 2.0))

    # First taper: indices 0 to width (inclusive), width+1 elements
    n1 = xp.arange(0, width + 1, dtype=np.float64)
    w1 = 0.5 * (1 + xp.cos(np.pi * (-1 + 2.0 * n1 / alpha / (n - 1))))

    # Middle (flat) region
    n_middle = (n - width - 1) - (width + 1)
    w2 = xp.ones(max(n_middle, 0), dtype=np.float64)

    # Second taper: indices n-width-1 to n-1 (inclusive), width+1 elements
    n3 = xp.arange(n - width - 1, n, dtype=np.float64)
    w3 = 0.5 * (1 + xp.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (n - 1))))

    # Assemble
    window = xp.concatenate([w1, w2, w3])

    return window.astype(np.float32)


def _apply_tukey_window(image: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply Tukey (tapered cosine) window for edge apodization (in-place).

    Parameters
    ----------
    image : np.ndarray
        Input image (numpy or cupy array). Modified in-place.
    alpha : float
        Taper fraction (0=rectangular, 1=Hann)

    Returns
    -------
    windowed : np.ndarray
        Windowed image (same array as input, modified in-place)
    """
    xp = get_array_module(image)

    # Apply separable 1D windows along each axis (memory efficient)
    # Windows created directly on device - no CPU-GPU transfers
    for axis in range(image.ndim):
        w = _tukey_window_1d(image.shape[axis], alpha, xp)

        # Reshape for broadcasting: [1, 1, ..., N, ..., 1]
        shape = [1] * image.ndim
        shape[axis] = image.shape[axis]
        w = w.reshape(shape)

        # In-place multiply
        image *= w

    return image


def dcr_resolution(
    image: np.ndarray,
    *,
    spacing: float | Sequence[float] | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    resample_isotropic: bool = False,
    exclude_axis_angle: float = 0.0,
    use_sectioned: bool = True,
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
    resample_isotropic : bool
        If True, resample 3D images to isotropic voxel size before analysis.
        This matches the methodology in Koho et al. 2019 and is recommended
        for anisotropic volumes. Requires spacing to be provided. Default: False.
    exclude_axis_angle : float
        Exclude frequencies within this angle (degrees) from the Z axis.
        Following Koho et al. 2019 to avoid artifacts from piezo stage motion.
        Default: 0.0 (no exclusion). Typical value: 5.0. Only used for 3D.
    use_sectioned : bool
        If True (default), use full 3D angular sectioning for 3D images.
        If False, use legacy 2D slice-based approach.

    Returns
    -------
    resolution : float or dict
        For 2D: Estimated resolution in physical units (if spacing provided)
        For 3D: Dict with 'xy' and 'z' resolutions

    Examples
    --------
    >>> from cubic.metrics.frc import dcr_resolution
    >>> import numpy as np
    >>> # 2D image
    >>> image_2d = np.random.randn(64, 64)
    >>> res = dcr_resolution(image_2d, spacing=0.065)  # 0.065 μm/pixel
    >>> print(f"Resolution: {res:.2f} μm")
    >>> # 3D image with full sectioned analysis
    >>> image_3d = np.random.randn(30, 64, 64)
    >>> res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065],
    ...                      resample_isotropic=True, exclude_axis_angle=5.0)
    >>> print(f"XY: {res['xy']:.2f} μm, Z: {res['z']:.2f} μm")
    """
    if image.ndim == 2:
        # 2D: return single resolution
        resolution, _, _, _ = dcr_curve(
            image,
            spacing=spacing,
            num_radii=num_radii,
            num_highpass=num_highpass,
        )
        return resolution

    elif image.ndim == 3:
        # Normalize spacing
        if spacing is None:
            spacing_list = None
        elif isinstance(spacing, (int, float)):
            spacing_list = [float(spacing)] * 3
        else:
            spacing_list = list(spacing)

        # Resample to isotropic voxel size if requested
        if resample_isotropic:
            if spacing_list is None:
                raise ValueError(
                    "resample_isotropic=True requires spacing to be provided"
                )

            spacing_tuple = tuple(spacing_list)
            iso_spacing = spacing_tuple[1]  # Y spacing (assumes Y == X)

            # Calculate target Z size
            target_z_size = int(round(image.shape[0] * spacing_tuple[0] / iso_spacing))
            if target_z_size % 2 != 0:
                target_z_size -= 1  # Make even

            # Resample image to isotropic
            image = rescale_isotropic(
                image,
                spacing_tuple,
                downscale_xy=False,
                order=1,
                preserve_range=True,
                target_z_size=target_z_size,
            ).astype(image.dtype)

            # Update spacing to isotropic
            spacing_list = [iso_spacing] * 3

        if use_sectioned:
            # Full 3D analysis with angular sectioning
            return _dcr_curve_3d_sectioned(
                image,
                spacing=np.array(spacing_list) if spacing_list else None,
                num_radii=num_radii,
                num_highpass=num_highpass,
                angle_delta=45,
                exclude_axis_angle=exclude_axis_angle,
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
            )

            return {"xy": res_xy, "z": res_xz}

    else:
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")
