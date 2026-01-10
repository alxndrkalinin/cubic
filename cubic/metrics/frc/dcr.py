"""Decorrelation Analysis (DCR) - Descloux et al. 2019."""

from collections.abc import Sequence

import numpy as np
from scipy.signal import savgol_filter

from cubic.cuda import asnumpy, get_array_module
from cubic.skimage import filters
from cubic.image_utils import rescale_isotropic

from .radial import (
    _kmax_phys,
    radial_edges,
    radial_k_grid,
    _kmax_phys_max,
    sectioned_bin_id,
)


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
) -> tuple[float, float]:
    """
    Find peak in decorrelation curve (Descloux et al. 2019, Supplementary Note 1.1).

    Iteratively finds global maximum, excluding boundary artifacts and checking
    prominence. Rejects peaks near boundary if curve is monotonically increasing.
    """
    d_work = d_curve.copy()
    n = len(d_work)

    # Mask invalid frequency regions
    d_work[(radii < r_min) | (radii >= r_max)] = -np.inf

    for _ in range(20):  # Max iterations to prevent infinite loop
        peak_idx = int(np.argmax(d_work))
        if d_work[peak_idx] == -np.inf:
            return 0.0, 0.0

        r_peak = radii[peak_idx]
        a_peak = d_curve[peak_idx]

        # Reject if at boundary
        if peak_idx >= n - 1 or r_peak >= r_max - 0.01:
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

        # Prominence check: peak must exceed subsequent minimum
        if peak_idx < n - 1:
            if a_peak - np.min(d_curve[peak_idx + 1 :]) < min_prominence:
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
    smoothing: int | None = 11,
    windowing: bool = True,
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
        Default: 11 reduces noise while preserving peak shape.
        Set to None to disable smoothing.
    windowing : bool
        If True (default), apply internal Tukey window for edge apodization.
        Set to False when windowing is applied externally for consistent
        preprocessing across different methods.

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

    # Apply Tukey window for edge apodization (unless externally handled)
    if windowing:
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
            smoothing=smoothing,
        )

        # Find peak using algorithm from Descloux et al. 2019 Supplementary Note 1.1
        # Finds global maximum with iterative boundary exclusion and prominence check
        r_peak, a_peak = _find_peak_in_curve(radii, d_curve)

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
    smoothing: int | None = 11,
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
        Default: 11 reduces noise while preserving peak shape.
        Set to None to disable smoothing.

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

    k_radius_norm = k_radius / k_max

    # Build radial bins
    radii_cpu = np.linspace(0, 1, num_radii, dtype=np.float32)
    edges_cpu = np.concatenate([[0], radii_cpu]).astype(np.float32)
    edges = xp.asarray(edges_cpu) if xp is not np else edges_cpu

    # Assign pixels to radial bins
    k_flat = k_radius_norm.ravel()
    bin_id = np.clip(np.digitize(k_flat, edges) - 1, 0, num_radii - 1).astype(np.int32)
    absF_flat = absF.ravel()

    # Compute d(r) = cumsum(|F|) / sqrt(total_power * cumsum(count))
    cross_cumsum = np.cumsum(
        np.bincount(bin_id, weights=absF_flat, minlength=num_radii)
    )
    count_cumsum = np.cumsum(
        np.bincount(bin_id, minlength=num_radii).astype(np.float32)
    )
    power = np.sum(absF_flat**2)

    d_curve = cross_cumsum / (np.sqrt(power * count_cumsum) + 1e-10)
    d_curve = asnumpy(d_curve).astype(np.float32)

    return radii_cpu, _smooth_curve(d_curve, smoothing), k_max


def _compute_decorrelation_curve_sectioned(
    image: np.ndarray,
    num_radii: int = 100,
    angle_delta: int = 45,
    bin_delta: int = 1,
    spacing: np.ndarray | None = None,
    exclude_axis_angle: float = 0.0,
    smoothing: int | None = 11,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """
    Compute sectioned decorrelation curves for 3D images.

    Uses azimuthal angle in Y-Z plane (0-360°) following Koho et al. 2019:
    phi ≈ 0°/180° → Z resolution, phi ≈ 90°/270° → XY resolution.
    """
    if image.ndim != 3:
        raise ValueError("Sectioned DCR requires 3D images")

    xp = get_array_module(image)
    shape = image.shape

    F = np.fft.fftn(image)
    absF = np.abs(F)
    absF_flat = absF.ravel()

    # Separate radial edges for anisotropic data
    r_edges_z, radii_z = radial_edges(
        shape, bin_delta=bin_delta, spacing=spacing, use_max_nyquist=False
    )
    r_edges_xy, radii_xy = radial_edges(
        shape, bin_delta=bin_delta, spacing=spacing, use_max_nyquist=True
    )

    # Angular edges (azimuthal 0-360°)
    n_angle = 360 // angle_delta
    angle_edges_cpu = np.array(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )
    angle_edges = xp.asarray(angle_edges_cpu) if xp is not np else angle_edges_cpu

    # k_max for resolution conversion
    if spacing is not None:
        k_max_z, k_max_xy = _kmax_phys(shape, spacing), _kmax_phys_max(shape, spacing)
    else:
        k_max_z, k_max_xy = min(n // 2 for n in shape), max(n // 2 for n in shape)

    # Group angular sectors by direction
    sector_groups = {"z": [], "xy": []}
    for aid in range(n_angle):
        center = (aid + 0.5) * angle_delta
        dist_z = min(center, abs(center - 180), abs(center - 360))
        dist_xy = min(abs(center - 90), abs(center - 270))
        sector_groups["z" if dist_z <= dist_xy else "xy"].append(aid)

    results = {}
    for sector_name, sector_ids in sector_groups.items():
        r_edges_raw, radii_raw = (
            (r_edges_xy, radii_xy) if sector_name == "xy" else (r_edges_z, radii_z)
        )
        k_max = k_max_xy if sector_name == "xy" else k_max_z
        n_radial_raw = len(radii_raw)
        r_edges = xp.asarray(r_edges_raw) if xp is not np else r_edges_raw

        radial_id, angle_id = sectioned_bin_id(
            shape,
            r_edges,
            angle_edges,
            spacing=list(spacing) if spacing is not None else None,
            exclude_axis_angle=exclude_axis_angle,
        )

        # Mask for all angular sectors in this group
        sector_mask = np.isin(angle_id, sector_ids) & (radial_id >= 0)

        if not np.any(sector_mask):
            radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)
            results[sector_name] = (
                radii_norm,
                np.zeros(num_radii, dtype=np.float32),
                k_max,
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

        # Resample to num_radii points
        radii_norm = np.linspace(0, 1, num_radii, dtype=np.float32)
        d_curve = np.interp(
            radii_norm, radii_raw / k_max, asnumpy(d_curve_raw).astype(np.float32)
        )

        results[sector_name] = (radii_norm, _smooth_curve(d_curve, smoothing), k_max)

    return results


def _dcr_curve_3d_sectioned(
    image: np.ndarray,
    *,
    spacing: np.ndarray | None = None,
    num_radii: int = 100,
    num_highpass: int = 10,
    angle_delta: int = 45,
    bin_delta: int = 1,
    smoothing: int | None = 11,
    exclude_axis_angle: float = 0.0,
    windowing: bool = True,
) -> dict[str, float]:
    """Compute 3D DCR with angular sectoring for XY and Z directions."""
    if image.ndim != 3:
        raise ValueError("3D sectioned DCR requires 3D images")

    image = image.astype(np.float32)
    image -= np.mean(image)
    if windowing:
        image = _apply_tukey_window(image, alpha=0.1)

    spacing_arr = np.array(spacing, dtype=np.float32) if spacing is not None else None
    sigmas = _generate_highpass_sigmas(image.shape, num_highpass)
    all_peaks = {"xy": [], "z": []}

    for sigma_hp in sigmas:
        filtered_image = (
            _highpass_filter(image, sigma_hp) if sigma_hp > 0 else image.copy()
        )

        sector_results = _compute_decorrelation_curve_sectioned(
            filtered_image,
            num_radii=num_radii,
            angle_delta=angle_delta,
            bin_delta=bin_delta,
            spacing=spacing_arr,
            exclude_axis_angle=exclude_axis_angle,
            smoothing=smoothing,
        )

        for sector_name in ["xy", "z"]:
            radii, d_curve, _ = sector_results[sector_name]
            r_peak, _ = _find_peak_in_curve(radii, d_curve)
            all_peaks[sector_name].append(r_peak)

        del filtered_image

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
    """Apply Gaussian high-pass filter (device-agnostic)."""
    if sigma <= 0:
        return image.copy()
    return image - filters.gaussian(image, sigma=sigma, preserve_range=True)


def _generate_highpass_sigmas(
    image_shape: tuple[int, ...],
    num_highpass: int,
    sigma_min: float = 0.5,
    sigma_max: float | None = None,
) -> np.ndarray:
    """Generate log-spaced high-pass sigmas (NanoPyx convention)."""
    if sigma_max is None:
        sigma_max = min(image_shape) / 2.0
    if num_highpass <= 1:
        return np.array([sigma_min], dtype=np.float32)
    log_min = np.log(max(sigma_min, 0.1))
    log_max = np.log(sigma_max)
    return np.exp(np.linspace(log_min, log_max, num_highpass)).astype(np.float32)


def _tukey_window_1d(n: int, alpha: float, xp) -> np.ndarray:
    """Create 1D Tukey window (device-agnostic, matches scipy exactly)."""
    if alpha <= 0:
        return xp.ones(n, dtype=np.float32)
    if alpha >= 1:
        idx = xp.arange(n, dtype=np.float64)
        return (0.5 * (1 - xp.cos(2 * np.pi * idx / (n - 1)))).astype(np.float32)

    width = int(np.floor(alpha * (n - 1) / 2.0))
    n1 = xp.arange(0, width + 1, dtype=np.float64)
    w1 = 0.5 * (1 + xp.cos(np.pi * (-1 + 2.0 * n1 / alpha / (n - 1))))
    n_middle = (n - width - 1) - (width + 1)
    w2 = xp.ones(max(n_middle, 0), dtype=np.float64)
    n3 = xp.arange(n - width - 1, n, dtype=np.float64)
    w3 = 0.5 * (1 + xp.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (n - 1))))
    return xp.concatenate([w1, w2, w3]).astype(np.float32)


def _apply_tukey_window(image: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply separable Tukey window for edge apodization (in-place)."""
    xp = get_array_module(image)
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
    windowing: bool = True,
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
    windowing : bool
        If True (default), apply internal Tukey window for edge apodization.
        Set to False when windowing is applied externally for consistent
        preprocessing across different methods.

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
            windowing=windowing,
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
                windowing=windowing,
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
            )

            return {"xy": res_xy, "z": res_xz}

    else:
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")
