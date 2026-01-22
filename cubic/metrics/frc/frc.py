"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from collections.abc import Callable, Sequence

import numpy as np

from cubic.cuda import asnumpy
from cubic.image_utils import (
    crop_bl,
    crop_br,
    crop_tl,
    crop_tr,
    pad_image,
    crop_center,
    hamming_window,
    pad_image_to_cube,
    rescale_isotropic,
    checkerboard_split,
    get_xy_block_coords,
    reverse_checkerboard_split,
)

from .analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
)
from .iterators import FourierRingIterator, AxialExcludeSectionedFourierShellIterator


def _empty_aggregate(*args: np.ndarray, **kwargs) -> np.ndarray:
    """Return unchanged array."""
    return args[0]


def frc_checkerboard_split(
    image: np.ndarray,
    reverse: bool = False,
    disable_3d_sum: bool = False,
    preserve_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split image into two by checkerboard pattern."""
    if reverse:
        return reverse_checkerboard_split(
            image,
            disable_3d_sum=disable_3d_sum,
            preserve_range=preserve_range,
        )
    else:
        return checkerboard_split(
            image,
            disable_3d_sum=disable_3d_sum,
            preserve_range=preserve_range,
        )


def preprocess_images(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    zero_padding: bool = True,
    reverse_split: bool = False,
    disable_hamming: bool = False,
    disable_3d_sum: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess input images with all modifications (padding, windowing, splitting)."""
    single_image = image2 is None

    # Apply padding to first image
    if len(set(image1.shape)) > 1 and zero_padding:
        image1 = pad_image_to_cube(image1)

    if single_image:
        # Split single image using checkerboard pattern
        image1, image2 = frc_checkerboard_split(
            image1, reverse=reverse_split, disable_3d_sum=disable_3d_sum
        )
    else:
        # Apply padding to second image
        if len(set(image2.shape)) > 1 and zero_padding:
            image2 = pad_image_to_cube(image2)

    # Apply Hamming windowing to both images independently
    if not disable_hamming:
        image1 = hamming_window(image1)
        image2 = hamming_window(image2)

    return image1, image2


class FRC(object):
    """A class for calculating 2D Fourier ring correlation (unshifted FFT)."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, iterator, spacing=None):
        """Create new FRC executable object and perform FFT on input images."""
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.iterator = iterator
        self.spacing = spacing
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))
        self.shape = image1.shape

    def execute(self):
        """Calculate the FRC."""
        radii = self.iterator.radii
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for ind_ring, idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Calculate FRC
        # If spacing was provided, radii are in physical units; normalize to [0,1]
        # If spacing was None, radii are in index units; normalize by Nyquist
        if self.spacing is None:
            spatial_freq = asnumpy(radii.astype(np.float32) / self.freq_nyq)
        else:
            # radii are in physical units; normalize to max frequency
            max_freq = float(np.max(radii))
            spatial_freq = asnumpy(radii.astype(np.float32) / max_freq)
        c1 = asnumpy(c1)
        c2 = asnumpy(c2)
        c3 = asnumpy(c3)
        n_points = asnumpy(points)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eps = np.finfo(c1.dtype).tiny
            c1_safe = np.clip(np.abs(c1), eps, None)
            c2_safe = np.clip(c2, eps, None)
            c3_safe = np.clip(c3, eps, None)

            frc = np.exp(np.log(c1_safe) - 0.5 * (np.log(c2_safe) + np.log(c3_safe)))
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set


def _calculate_frc_core(
    image1: np.ndarray,
    image2: np.ndarray,
    bin_delta: int,
    *,
    backend: str = "mask",
    spacing: Sequence[float] | None = None,
) -> FourierCorrelationDataCollection:
    """
    Core FRC calculation logic.

    Args:
        image1: First input image
        image2: Second input image
        bin_delta: Bin width (step size between bins). Controls binning resolution
                   for both backends.
        backend: "mask" (existing iterators) or "hist" (radial histogram)
        spacing: Physical spacing per axis. If None, uses index units.
    """
    assert image1.shape == image2.shape
    frc_data = FourierCorrelationDataCollection()

    if backend == "hist":
        # Histogram backend using radial binning
        from cubic.cuda import get_array_module

        from .radial import radial_edges, reduce_cross, reduce_power, radial_bin_id

        # Compute unshifted FFT
        fft_image1 = np.fft.fftn(image1 - image1.mean())
        fft_image2 = np.fft.fftn(image2 - image2.mean())

        # Build radial bins
        shape = fft_image1.shape
        edges, radii = radial_edges(shape, bin_delta, spacing=spacing)
        xp = get_array_module(fft_image1)
        edges = xp.asarray(edges)

        bin_id = radial_bin_id(shape, edges, spacing=spacing)

        Sx2, Nx = reduce_power(fft_image1, bin_id)
        Sy2, Ny = reduce_power(fft_image2, bin_id)
        Sxy_re, _ = reduce_cross(fft_image1, fft_image2, bin_id, numerator="real")

        # Compute FRC: abs(Sxy) / sqrt(Sx2 * Sy2) in log domain for stability
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eps = np.finfo(np.float32).tiny
            Sx2_safe = np.clip(Sx2, eps, None)
            Sy2_safe = np.clip(Sy2, eps, None)
            Sxy_re_safe = np.clip(np.abs(Sxy_re), eps, None)

            frc = np.exp(
                np.log(Sxy_re_safe) - 0.5 * (np.log(Sx2_safe) + np.log(Sy2_safe))
            )
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        # Convert radii to normalized spatial frequency
        # If spacing was provided, radii are in physical units; normalize to [0,1]
        # If spacing was None, radii are in index units; normalize by Nyquist
        freq_nyq = int(np.floor(shape[0] / 2.0))
        if spacing is None:
            spatial_freq = radii.astype(np.float32) / freq_nyq
        else:
            # radii are in physical units; normalize to max frequency
            max_freq = float(np.max(radii))
            spatial_freq = radii.astype(np.float32) / max_freq
        frc = asnumpy(frc)
        n_points = asnumpy(Nx.astype(np.float32))

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points
        frc_data[0] = data_set
    else:
        # Default mask/iterator backend
        iterator = FourierRingIterator(image1.shape, bin_delta, spacing=spacing)
        frc_task = FRC(image1, image2, iterator, spacing=spacing)
        frc_data[0] = frc_task.execute()

    return frc_data


def _calibration_factor(freq_at_crossing: float) -> float:
    """Calculate calibration factor for one-image FRC/FSC.

    The checkerboard split creates a diagonal shift between subimage pairs that
    causes frequency compression in the FRC/FSC curve due to the Fourier shift
    theorem (Supplementary Note 1, Koho et al. 2019).

    This correction was empirically calibrated against two-image FRC at the 1/7
    threshold by imaging the same field of view at different pixel sizes
    (Supplementary Note 2, Supplementary Fig. 3, Koho et al. 2019).

    Parameters
    ----------
    freq_at_crossing : float
        Normalized frequency (0-1) at threshold crossing.

    Returns
    -------
    float
        Calibration factor. Divide raw resolution by this to get corrected value.
    """

    def calibration_func(x: float, a: float, b: float, c: float, d: float) -> float:
        return a * np.exp(c * (x - b)) + d

    # Parameters from miplib calibration (Koho et al. 2019)
    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]
    return calibration_func(freq_at_crossing, *params)


def _apply_cutoff_correction(result: FourierCorrelationData) -> None:
    """Apply cut-off correction for single image FRC."""
    point = result.resolution["resolution-point"][1]
    cut_off_correction = _calibration_factor(point)
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction


def calculate_frc(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    disable_hamming: bool = False,
    average: bool = True,
    z_correction: float = 1.0,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
    backend: str = "mask",
) -> FourierCorrelationData:
    """
    Calculate a regular FRC with single or two image inputs.

    Args:
        bin_delta: Bin width (step size between bins). Controls binning resolution
                   for both backends. Default: 1.
        backend: "mask" (existing iterators) or "hist" (radial histogram)
    """
    single_image = image2 is None
    reverse = average and single_image
    original_image1 = image1.copy() if reverse else None

    if spacing is None:
        spacing = None
    elif isinstance(spacing, (int, float)):
        spacing = [spacing] * image1.ndim
    else:
        spacing = list(spacing)

    image1, image2 = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        disable_hamming=disable_hamming,
    )

    # Adjust spacing to match preprocessed image shape (padding may have changed dimensions)
    # For padded dimensions, use the spacing from the first dimension
    if spacing is not None and len(spacing) != image1.ndim:
        if len(spacing) < image1.ndim:
            # Extend spacing if needed (use first spacing value for new dimensions)
            spacing = list(spacing) + [spacing[0]] * (image1.ndim - len(spacing))
        else:
            # Truncate if too long (shouldn't happen, but be safe)
            spacing = list(spacing[: image1.ndim])

    # Pass spacing to core calculation (for histogram backend)
    frc_data = _calculate_frc_core(
        image1, image2, bin_delta, backend=backend, spacing=spacing
    )

    # Average with reverse pattern (only for single image mode)
    if reverse:
        # Use original unprocessed image for reverse split
        image1, image2 = preprocess_images(
            original_image1,
            None,
            reverse_split=reverse,
            zero_padding=zero_padding,
            disable_hamming=disable_hamming,
        )

        frc_data_rev = _calculate_frc_core(
            image1, image2, bin_delta, backend=backend, spacing=spacing
        )

        # Average the two results
        frc_data[0].correlation["correlation"] = (
            0.5 * frc_data[0].correlation["correlation"]
            + 0.5 * frc_data_rev[0].correlation["correlation"]
        )

    # Analyze results
    analyzer = FourierCorrelationAnalysis(
        frc_data,
        spacing[0] if spacing is not None else 1.0,
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
    )
    result = analyzer.execute(z_correction=z_correction)[0]

    # Apply cut-off correction (only for single image case)
    if single_image:
        _apply_cutoff_correction(result)

    return result


def frc_resolution(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
    curve_fit_type: str = "smooth-spline",
    backend: str = "mask",
) -> float:
    """Calculate either single- or two-image FRC-based 2D image resolution."""
    frc_result = calculate_frc(
        image1,
        image2,
        bin_delta=bin_delta,
        curve_fit_type=curve_fit_type,
        spacing=spacing,
        zero_padding=zero_padding,
        backend=backend,
    )

    return frc_result.resolution["resolution"]


class DirectionalFSC(object):
    """Calculate the directional FSC between two images (unshifted FFT)."""

    def __init__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        iterator,
        normalize_power: bool = False,
    ):
        """Initialize the directional FSC."""
        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("Image must be 3D")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        self.iterator = iterator
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())
        if normalize_power:
            pixels = image1.shape[0] ** 3
            self.fft_image1 /= np.array(pixels * np.mean(image1))
            self.fft_image2 /= np.array(pixels * np.mean(image2))

        self._result = None

    @property
    def result(self):
        """Return the FRC results."""
        if self._result is None:
            return self.execute()
        else:
            return self._result

    def execute(self):
        """Calculate the FSC."""
        data_structure = FourierCorrelationDataCollection()
        radii, angles = self.iterator.steps
        freq_nyq = self.iterator.nyquist
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        # iterate through the sphere and calculate initial values
        for ind_ring, shell_idx, rotation_idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
            c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)

            points[rotation_idx, shell_idx] = len(subset1)

        # calculate FRC for every orientation
        for i in range(angles.size):
            spatial_freq = asnumpy(radii.astype(np.float32) / freq_nyq)
            c1_i = asnumpy(c1[i])
            c2_i = asnumpy(c2[i])
            c3_i = asnumpy(c3[i])
            n_points = asnumpy(points[i])

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                eps = np.finfo(c2_i.dtype).tiny
                c1_safe = np.clip(np.abs(c1_i), eps, None)
                c2_safe = np.clip(c2_i, eps, None)
                c3_safe = np.clip(c3_i, eps, None)

                frc = np.exp(
                    np.log(c1_safe) - 0.5 * (np.log(c2_safe) + np.log(c3_safe))
                )
                frc[frc == np.inf] = 0.0
                frc = np.nan_to_num(frc)

            result = FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            data_structure[angles[i]] = result

        return data_structure


def calculate_sectioned_fsc(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 10,
    angle_delta: int = 15,
    extract_angle_delta: float = 0.1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    disable_hamming: bool = False,
    z_correction: float = 1.0,
    disable_3d_sum: bool = False,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
) -> FourierCorrelationDataCollection:
    """Calculate sectioned FSC for one or two images."""
    single_image = image2 is None

    if spacing is None:
        spacing = None
    elif isinstance(spacing, (int, float)):
        spacing = [spacing] * image1.ndim
    else:
        spacing = list(spacing)

    image1, image2 = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        disable_hamming=disable_hamming,
        disable_3d_sum=disable_3d_sum,
    )

    iterator = AxialExcludeSectionedFourierShellIterator(
        image1.shape,
        bin_delta,
        angle_delta,
        extract_angle_delta,
        spacing=spacing,
    )
    fsc_task = DirectionalFSC(image1, image2, iterator)
    data = fsc_task.execute()

    analyzer = FourierCorrelationAnalysis(
        data,
        spacing[0],
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
    )
    result = analyzer.execute(z_correction=z_correction)

    # Apply cut-off correction (only for single image case)
    if single_image:
        for angle, dataset in result:
            _apply_cutoff_correction(dataset)

    return result


def _calculate_fsc_sectioned_hist(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    bin_delta: int = 10,
    angle_delta: int = 45,
    spacing: Sequence[float] | None = None,
    exclude_axis_angle: float = 0.0,
    use_max_nyquist: bool = False,
) -> dict[int, FourierCorrelationData]:
    """
    Calculate sectioned FSC using vectorized histogram approach.

    Uses polar angle from Z axis (0-90°):
    - theta ≈ 0° = Z-dominated frequencies → Z resolution
    - theta ≈ 90° = XY-dominated frequencies → XY resolution

    Parameters
    ----------
    angle_delta : int
        Angular bin width in degrees. Default 45 gives 2 bins.
        Use 15 to match mask backend's angular resolution.
    exclude_axis_angle : float
        Exclude frequencies within this angle (in degrees) from the Z axis.
        Following Koho et al. 2019 to avoid piezo/interpolation artifacts.
        Default: 0.0 (no exclusion).
    use_max_nyquist : bool
        If True, extend radial frequency range to maximum Nyquist (typically XY)
        instead of minimum Nyquist (typically Z). This allows XY-dominant sectors
        to measure higher frequencies for better XY resolution estimates.
        Default: False.
    """
    from cubic.cuda import asnumpy, get_array_module

    from .radial import (
        radial_edges,
        frc_from_sums,
        sectioned_bin_id,
        reduce_cross_sectioned,
        reduce_power_sectioned,
    )

    xp = get_array_module(image1)

    # Compute FFT
    fft_image1 = np.fft.fftn(image1 - image1.mean())
    fft_image2 = np.fft.fftn(image2 - image2.mean())

    shape = image1.shape

    # Build radial edges
    r_edges, radii = radial_edges(
        shape, bin_delta, spacing=spacing, use_max_nyquist=use_max_nyquist
    )
    r_edges = xp.asarray(r_edges)
    n_radial = len(radii)

    # Angular edges: polar angle from Z axis (0-90°)
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    n_angle = 90 // angle_delta
    angle_edges = xp.asarray(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )

    # Get bin IDs
    radial_id, angle_id = sectioned_bin_id(
        shape,
        r_edges,
        angle_edges,
        spacing=spacing,
        exclude_axis_angle=exclude_axis_angle,
    )

    # Compute per-bin sums
    Sx2, Nx = reduce_power_sectioned(fft_image1, radial_id, angle_id, n_radial, n_angle)
    Sy2, Ny = reduce_power_sectioned(fft_image2, radial_id, angle_id, n_radial, n_angle)
    Sxy = reduce_cross_sectioned(
        fft_image1, fft_image2, radial_id, angle_id, n_radial, n_angle
    )

    del fft_image1, fft_image2
    del radial_id, angle_id, r_edges, angle_edges

    # Compute FSC for each angle
    results = {}

    # Nyquist for normalization
    if spacing is not None:
        from .radial import _kmax_phys, _kmax_phys_max

        if use_max_nyquist:
            max_freq = _kmax_phys_max(shape, spacing)
        else:
            max_freq = _kmax_phys(shape, spacing)
    else:
        if use_max_nyquist:
            max_freq = max(n // 2 for n in shape)
        else:
            max_freq = min(n // 2 for n in shape)

    spatial_freq = asnumpy(radii.astype(np.float32) / max_freq)

    # Map bin index to output angle (polar angle from Z axis):
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    #
    # Output angle is the bin center in polar coordinates (0-90°)
    for aid in range(n_angle):
        # Bin center in polar coords
        output_angle = int(round((aid + 0.5) * angle_delta))

        fsc = frc_from_sums(
            asnumpy(Sx2[aid]),
            asnumpy(Sy2[aid]),
            asnumpy(Sxy[aid]),
        )
        n_points = asnumpy(Nx[aid].astype(np.float32))

        # Filter out bins with no points (can happen at low frequencies
        # where angular sectors may have no data, e.g., DC is purely Z-like)
        valid_mask = n_points > 0
        fsc_valid = fsc[valid_mask]
        freq_valid = spatial_freq[valid_mask]
        n_points_valid = n_points[valid_mask]

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = fsc_valid
        data_set.correlation["frequency"] = freq_valid
        data_set.correlation["points-x-bin"] = n_points_valid

        results[output_angle] = data_set

    return results


def fsc_resolution(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 10,
    angle_delta: int = 15,
    zero_padding: bool | None = None,
    spacing: float | Sequence[float] | None = None,
    resample_isotropic: bool = False,
    resample_order: int = 1,
    average: bool = True,
    exclude_axis_angle: float = 0.0,
    use_max_nyquist: bool = False,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    backend: str = "hist",
) -> dict[str, float]:
    """
    Calculate either single- or two-image FSC-based 3D image resolution.

    Args:
        image1: First 3D input image
        image2: Second 3D input image (optional, uses checkerboard split if None)
        bin_delta: Bin width for radial binning
        angle_delta: Angular bin width in degrees (default 15, same as mask backend)
        zero_padding: Whether to pad image to cube. Default: True for mask backend,
                      False for hist backend. The hist backend handles anisotropic
                      volumes correctly using physical frequency coordinates.
        spacing: Physical spacing per axis [z, y, x]. If None, uses index units.
        resample_isotropic: If True, resample images to isotropic voxel size before
                            FSC calculation. This matches the methodology in Koho et al.
                            2019 and is recommended for anisotropic volumes with limited
                            Z extent. Requires spacing to be provided. Default: False.
        resample_order: Interpolation order for isotropic resampling (0=nearest neighbor,
                        1=linear, 3=cubic). The miplib paper uses order=0 (nearest neighbor).
                        Default: 1 (linear interpolation).
        average: If True and single-image mode, average results from both diagonal
                 checkerboard splits (forward and reverse) to reduce variance.
                 Following Koho et al. 2019 methodology. Default: True.
        exclude_axis_angle: Exclude frequencies within this angle (in degrees) from
                            the Z axis. Following Koho et al. 2019 to avoid artifacts
                            from piezo stage motion and interpolation near the optical
                            axis. Default: 0.0 (no exclusion). Typical value: 5.0.
        use_max_nyquist: If True (hist backend only), extend frequency range to maximum
                         Nyquist (typically XY) instead of minimum (typically Z). This
                         allows XY-dominant sectors to measure higher frequencies for
                         better XY resolution estimates on anisotropic data.
                         Default: False.
        resolution_threshold: Threshold criterion for resolution calculation.
                              Options: "fixed", "one-bit", "half-bit", "three-sigma".
                              Default: "fixed" (uses threshold_value).
        threshold_value: Fixed threshold value when resolution_threshold="fixed".
                         Default: 0.143 (1/7 threshold). Note: The one-image calibration
                         correction was empirically calibrated for the 1/7 threshold
                         (Koho et al. 2019), so using other values may affect accuracy.
        backend: "hist" (vectorized, GPU-accelerated) or "mask" (deprecated)
    """
    import warnings

    # Set default zero_padding based on backend:
    # - mask backend requires padding to cube (iterator assumes isotropic shells)
    # - hist backend handles anisotropic volumes correctly using spacing
    if zero_padding is None:
        zero_padding = backend == "mask"

    # Resample to isotropic voxel size if requested
    # Save original Z spacing for Z-sector resolution calculation
    # (needed because isotropic resampling creates artificial high-frequency
    # correlation beyond the original Z Nyquist)
    original_spacing_z = None
    # z_factor for k(θ) correction (Koho et al. 2019, equation 5)
    # Will be set from original spacing before isotropic resampling overwrites it
    z_factor = 1.0

    if resample_isotropic:
        if spacing is None:
            raise ValueError("resample_isotropic=True requires spacing to be provided")

        # Convert spacing to list if needed
        if isinstance(spacing, (int, float)):
            spacing_tuple = (float(spacing),) * image1.ndim
        else:
            spacing_tuple = tuple(spacing)

        # Save original Z spacing before resampling
        original_spacing_z = spacing_tuple[0]

        # Calculate target Z size (must be even for checkerboard split)
        iso_spacing = spacing_tuple[1]  # Y spacing (assumes Y == X)

        # Calculate z_factor from ORIGINAL spacing for k(θ) correction.
        # Even after isotropic resampling, Z frequencies have lower information
        # density due to the original anisotropic acquisition.
        # miplib applies z_correction this way (see Koho et al. 2019 notebook).
        z_factor = original_spacing_z / iso_spacing
        target_z_size = int(round(image1.shape[0] * spacing_tuple[0] / iso_spacing))
        if target_z_size % 2 != 0:
            target_z_size -= 1  # Make even for checkerboard split

        # Resample image1 to isotropic (upscale Z to match XY resolution)
        # miplib paper uses order=0 (nearest neighbor), default here is order=1 (linear)
        image1 = rescale_isotropic(
            image1,
            spacing_tuple,
            downscale_xy=False,
            order=resample_order,
            preserve_range=True,
            target_z_size=target_z_size,
        ).astype(image1.dtype)

        # Resample image2 if provided
        if image2 is not None:
            image2 = rescale_isotropic(
                image2,
                spacing_tuple,
                downscale_xy=False,
                order=resample_order,
                preserve_range=True,
                target_z_size=target_z_size,
            ).astype(image2.dtype)

        # Crop to even dimensions (required for checkerboard split)
        # This is needed because rescale_isotropic may produce odd shapes
        even_shape = tuple(s - (s % 2) for s in image1.shape)
        if image1.shape != even_shape:
            slices = tuple(slice(0, es) for es in even_shape)
            image1 = image1[slices]
            if image2 is not None:
                image2 = image2[slices]

        # Update spacing to isotropic (use XY spacing for all axes)
        spacing = [iso_spacing] * image1.ndim

    if backend == "mask":
        warnings.warn(
            "backend='mask' is deprecated and will be removed in a future version. "
            "Use backend='hist' (default) for faster GPU-accelerated computation.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Original sectioned FSC with iterators (mask backend)
        # Use one-bit threshold for 3D FSC as recommended by Koho et al. 2019
        fsc_result = calculate_sectioned_fsc(
            image1,
            image2,
            bin_delta=bin_delta,
            angle_delta=angle_delta,
            resolution_threshold="one-bit",
            spacing=spacing,
            zero_padding=zero_padding,
        )

        angle_to_resolution = {
            int(angle): dataset.resolution["resolution"]
            for angle, dataset in fsc_result
        }

        # Mask backend uses phi=arctan2(y,z):
        #   phi=0°/180° → frequency along Z axis → Z resolution
        #   phi=90°/270° → frequency in XY plane → XY resolution
        # Average opposite angles for better statistics
        z_res = 0.5 * (
            angle_to_resolution.get(0, np.nan) + angle_to_resolution.get(180, np.nan)
        )
        xy_res = 0.5 * (
            angle_to_resolution.get(90, np.nan) + angle_to_resolution.get(270, np.nan)
        )
        return {"xy": xy_res, "z": z_res}

    # hist backend: vectorized sectioned FSC
    if spacing is None:
        spacing_list = None
    elif isinstance(spacing, (int, float)):
        spacing_list = [float(spacing)] * image1.ndim
    else:
        spacing_list = list(spacing)

    # Determine if we should average both splits (only for single-image mode)
    single_image = image2 is None
    do_average = average and single_image

    # Save original for reverse split if averaging
    original_image1 = image1.copy() if do_average else None

    # Preprocess images (forward split)
    image1_proc, image2_proc = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        disable_hamming=False,
        disable_3d_sum=False,
    )

    # Calculate sectioned FSC using hist backend (forward split)
    fsc_data = _calculate_fsc_sectioned_hist(
        image1_proc,
        image2_proc,
        bin_delta=bin_delta,
        angle_delta=angle_delta,
        spacing=spacing_list,
        exclude_axis_angle=exclude_axis_angle,
        use_max_nyquist=use_max_nyquist,
    )

    # Average with reverse split if enabled
    if do_average:
        # Preprocess with reverse split
        image1_rev, image2_rev = preprocess_images(
            original_image1,
            None,
            zero_padding=zero_padding,
            disable_hamming=False,
            disable_3d_sum=False,
            reverse_split=True,
        )

        # Calculate FSC with reverse split
        fsc_data_rev = _calculate_fsc_sectioned_hist(
            image1_rev,
            image2_rev,
            bin_delta=bin_delta,
            angle_delta=angle_delta,
            spacing=spacing_list,
            exclude_axis_angle=exclude_axis_angle,
            use_max_nyquist=use_max_nyquist,
        )

        # Average correlation arrays for each angle
        for angle in fsc_data.keys():
            if angle in fsc_data_rev:
                corr_fwd = np.asarray(fsc_data[angle].correlation["correlation"])
                corr_rev = np.asarray(fsc_data_rev[angle].correlation["correlation"])
                fsc_data[angle].correlation["correlation"] = (
                    0.5 * corr_fwd + 0.5 * corr_rev
                )

    # Analyze results to get resolution
    # spacing_list is [Z, Y, X] order
    #
    # Polar angle convention (0-90°):
    #   - angle ≈ 0° = Z-dominated → use Z spacing
    #   - angle ≈ 90° = XY-dominated → use XY spacing
    if spacing_list is not None:
        spacing_xy = spacing_list[1]  # Y spacing (assumes Y==X)
        spacing_z = spacing_list[0]  # Z spacing
    else:
        spacing_xy = 1.0
        spacing_z = 1.0

    # Calculate anisotropy factor for k(θ) correction (Koho et al. 2019, equation 5)
    # z_factor = spacing_z / spacing_xy (how many XY pixels fit in one Z step)
    # Note: If resample_isotropic=True, z_factor was already calculated from original
    # spacing before the spacing was overwritten to isotropic values.
    if not resample_isotropic and spacing_list is not None:
        z_factor = spacing_z / spacing_xy
    # If resample_isotropic=False and no spacing, z_factor remains 1.0 (no correction)

    # Select XY and Z sectors based on polar angle convention:
    # - theta ≈ 0° = Z-dominated (|kz| >> k_xy) → Z resolution
    # - theta ≈ 90° = XY-dominated (k_xy >> |kz|) → XY resolution
    #
    # With angle_delta=45: angles = [22, 67] (bin centers)
    # With angle_delta=15: angles = [7, 22, 37, 52, 67, 82]
    angles = sorted(fsc_data.keys())

    # For XY resolution: find sector closest to 90°
    angle_xy = max(angles)  # Highest angle is most XY-dominated

    # For Z resolution: find sector closest to 0°
    angle_z = min(angles)  # Lowest angle is most Z-dominated
    z_sectors = [angle_z]  # Single Z sector in polar convention

    results = {}

    # Process XY sector
    data_xy = fsc_data[angle_xy]
    data_collection = FourierCorrelationDataCollection()
    data_collection[0] = data_xy
    analyzer = FourierCorrelationAnalysis(
        data_collection,
        spacing_xy,
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        curve_fit_type="spline",
    )
    try:
        analyzed = analyzer.execute()[0]
        # Apply calibration correction for single-image mode
        if single_image:
            _apply_cutoff_correction(analyzed)
        results["xy"] = analyzed.resolution["resolution"]
    except Exception:
        results["xy"] = float("nan")

    # Process Z resolution by averaging multiple Z-relevant sectors
    if z_sectors and len(z_sectors) > 0:
        # Get reference frequency grid from first Z sector
        ref_data = fsc_data[z_sectors[0]]
        ref_freq = np.asarray(ref_data.correlation["frequency"])

        # Average correlations weighted by point counts
        sum_corr = np.zeros_like(ref_freq)
        sum_weights = np.zeros_like(ref_freq)
        for angle in z_sectors:
            data = fsc_data[angle]
            corr = np.asarray(data.correlation["correlation"])
            n_pts = np.asarray(data.correlation["points-x-bin"])
            sum_corr += corr * n_pts
            sum_weights += n_pts
        avg_corr = sum_corr / np.maximum(sum_weights, 1)

        # Create averaged Z data set
        z_data_set = FourierCorrelationData()
        z_data_set.correlation["correlation"] = avg_corr
        z_data_set.correlation["frequency"] = ref_freq
        z_data_set.correlation["points-x-bin"] = sum_weights

        # Determine spacing and frequency limit for Z
        # For isotropic resampling: frequency is normalized to isotropic Nyquist,
        # but Z only has valid data up to original Z Nyquist (z_freq_limit).
        # For non-isotropic: frequency is normalized to min(XY,Z) Nyquist = Z Nyquist,
        # so no filtering is needed.
        if original_spacing_z is not None:
            # Isotropic resampling: filter to valid Z frequencies
            z_freq_limit = spacing_xy / original_spacing_z
            analyzer_spacing = spacing_xy  # Frequencies in isotropic units
        else:
            # Non-isotropic: use all frequencies, Z spacing for resolution
            z_freq_limit = 1.0
            analyzer_spacing = spacing_z  # Frequencies normalized to Z Nyquist

        # Filter to valid Z frequencies (only applies for isotropic resampling)
        valid_z_mask = ref_freq <= z_freq_limit
        if np.any(valid_z_mask):
            z_freq_filtered = ref_freq[valid_z_mask]
            z_corr_filtered = avg_corr[valid_z_mask]
            z_weights_filtered = sum_weights[valid_z_mask]
        else:
            # No valid Z data (shouldn't happen, but handle gracefully)
            z_freq_filtered = ref_freq
            z_corr_filtered = avg_corr
            z_weights_filtered = sum_weights

        # Create filtered Z data set
        z_data_set_filtered = FourierCorrelationData()
        z_data_set_filtered.correlation["correlation"] = z_corr_filtered
        z_data_set_filtered.correlation["frequency"] = z_freq_filtered
        z_data_set_filtered.correlation["points-x-bin"] = z_weights_filtered

        # Try to find resolution with the filtered Z data
        data_collection = FourierCorrelationDataCollection()
        data_collection[0] = z_data_set_filtered
        analyzer = FourierCorrelationAnalysis(
            data_collection,
            analyzer_spacing,
            resolution_threshold=resolution_threshold,
            threshold_value=threshold_value,
            curve_fit_type="spline",
        )
        try:
            analyzed = analyzer.execute()[0]
            # Apply calibration correction for single-image mode
            if single_image:
                _apply_cutoff_correction(analyzed)
            z_resolution = analyzed.resolution["resolution"]
        except Exception:
            z_resolution = float("nan")

        # Fallback: simple threshold crossing if spline failed
        if np.isnan(z_resolution):
            below_threshold = z_corr_filtered < 0.5
            if np.any(below_threshold):
                first_below = int(np.argmax(below_threshold))
                freq_at_crossing = float(z_freq_filtered[first_below])
                if freq_at_crossing > 0:
                    z_resolution = analyzer_spacing / freq_at_crossing
                    # Apply calibration correction for single-image fallback
                    if single_image:
                        z_resolution /= _calibration_factor(freq_at_crossing)

        # Apply k(θ) correction to Z resolution (Koho et al. 2019, equation 5)
        # Using bin edge (0°) for Z sector to get maximum correction: k(0°) = z_factor
        # k(θ) = 1 + (z_factor-1) × |cos(θ)|, at θ=0°: k = z_factor
        if not np.isnan(z_resolution) and z_factor > 1.0:
            z_resolution = z_resolution * z_factor

        results["z"] = z_resolution
    else:
        results["z"] = float("nan")

    return results


def grid_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable | None = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by tiling and taking 2D slices along XY and XZ."""
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if isinstance(spacing, (int, float)):
        spacing = [spacing, spacing, spacing]

    assert len(image.shape) == 3 and len(spacing) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    spacing_xy = (spacing[1], spacing[2])
    spacing_xz = (spacing[0], spacing[2])

    locations = get_xy_block_coords(image.shape, crop_size)

    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for y1, y2, x1, x2 in locations:
        loc_image = image[:, y1:y2, x1:x2]
        max_projection_resolution = fsc_resolution(
            loc_image.max(0),
            bin_delta=bin_delta,
            spacing=spacing_xy,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                fsc_resolution(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    spacing=spacing_xy,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            pad_size = (xz_slice.shape[1] - xz_slice.shape[0]) // 2
            if (xz_slice.shape[1] - xz_slice.shape[0]) % 2 != 0:
                pad_size = (pad_size + 1, pad_size)

            padded_xz_slice = pad_image(xz_slice, pad_size, 0, pad_mode)

            xz_slice_resolutions.append(
                fsc_resolution(
                    padded_xz_slice,
                    bin_delta=bin_delta,
                    spacing=spacing_xz,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),
        "xy": aggregate_fn(xy_resolutions, axis=0),
        "xz": aggregate_fn(xz_resolutions, axis=0),
    }


def five_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by taking 2D slices along XY and XZ at 4 corners and the center."""
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if isinstance(spacing, (int, float)):
        spacing = [spacing, spacing, spacing]

    assert len(image.shape) == 3 and len(spacing) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    spacing_xy = (spacing[1], spacing[2])
    spacing_xz = (spacing[0], spacing[2])

    locations = [crop_tl, crop_bl, crop_tr, crop_br, crop_center]
    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for loc in locations:
        loc_image = loc(image, crop_size)  # type: ignore

        max_projection_resolution = fsc_resolution(
            loc_image.max(0),
            bin_delta=bin_delta,
            spacing=spacing_xy,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)
        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                fsc_resolution(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    spacing=spacing_xy,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            padded_xz_slice = pad_image(
                xz_slice, (xz_slice.shape[1] - xz_slice.shape[0]) // 2, 0, pad_mode
            )

            xz_slice_resolutions.append(
                fsc_resolution(
                    padded_xz_slice,
                    bin_delta=bin_delta,
                    spacing=spacing_xz,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),
        "xy": aggregate_fn(xy_resolutions, axis=0),
        "xz": aggregate_fn(xz_resolutions, axis=0),
    }


def frc_resolution_difference(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    bin_delta: int = 3,
    spacing: float | tuple[float, float] | None = None,
    backend: str = "mask",
) -> float:
    """Calculate difference between FRC-based resulutions of two images."""
    if isinstance(spacing, (int, float)):
        spacing = (spacing, spacing)

    image1_res = frc_resolution(
        image1, bin_delta=bin_delta, spacing=spacing, backend=backend
    )
    image2_res = frc_resolution(
        image2, bin_delta=bin_delta, spacing=spacing, backend=backend
    )
    return (image2_res - image1_res) * 1000  # return diff in nm
