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


def _apply_cutoff_correction(result: FourierCorrelationData) -> None:
    """Apply cut-off correction for single image FRC."""

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]
    point = result.resolution["resolution-point"][1]
    cut_off_correction = func(point, *params)
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

    Uses angular binning following Koho et al. 2019 convention:
    - angle=0° corresponds to XY plane (lateral resolution)
    - angle=90° corresponds to Z axis (axial resolution)

    Internally bins by polar angle from Z axis, then transforms to paper convention.

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

    # Angular edges: 0°=Z axis, 90°=XY plane
    # Build edges from 0 to 90 degrees with angle_delta step
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

    # Map bin index to output angle following Koho et al. 2019 convention:
    # - angle=0° = XY plane (lateral)
    # - angle=90° = Z axis (axial)
    #
    # Internally, bins are indexed by polar angle from Z axis:
    # - aid=0 covers polar 0° to angle_delta° (near Z axis)
    # - aid=n_angle-1 covers polar (90-angle_delta)° to 90° (near XY plane)
    #
    # Transform: output_angle = 90 - polar_angle
    # Using bin center: output_angle = 90 - (aid + 0.5) * angle_delta
    # Rounded to integer for dict key
    for aid in range(n_angle):
        # Transform polar angle to paper convention
        # Bin center in polar coords: (aid + 0.5) * angle_delta
        # Paper convention: 90 - polar_center
        polar_center = (aid + 0.5) * angle_delta
        output_angle = int(round(90 - polar_center))

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
    average: bool = True,
    exclude_axis_angle: float = 0.0,
    use_max_nyquist: bool = False,
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
        backend: "hist" (vectorized, GPU-accelerated) or "mask" (deprecated)
    """
    import warnings

    # Set default zero_padding based on backend:
    # - mask backend requires padding to cube (iterator assumes isotropic shells)
    # - hist backend handles anisotropic volumes correctly using spacing
    if zero_padding is None:
        zero_padding = backend == "mask"

    # Resample to isotropic voxel size if requested
    if resample_isotropic:
        if spacing is None:
            raise ValueError("resample_isotropic=True requires spacing to be provided")

        # Convert spacing to list if needed
        if isinstance(spacing, (int, float)):
            spacing_tuple = (float(spacing),) * image1.ndim
        else:
            spacing_tuple = tuple(spacing)

        # Calculate target Z size (must be even for checkerboard split)
        iso_spacing = spacing_tuple[1]  # Y spacing (assumes Y == X)
        target_z_size = int(round(image1.shape[0] * spacing_tuple[0] / iso_spacing))
        if target_z_size % 2 != 0:
            target_z_size -= 1  # Make even for checkerboard split

        # Resample image1 to isotropic (upscale Z to match XY resolution)
        # Use linear interpolation as in Koho et al. 2019
        image1 = rescale_isotropic(
            image1,
            spacing_tuple,
            downscale_xy=False,
            order=1,
            preserve_range=True,
            target_z_size=target_z_size,
        ).astype(image1.dtype)

        # Resample image2 if provided
        if image2 is not None:
            image2 = rescale_isotropic(
                image2,
                spacing_tuple,
                downscale_xy=False,
                order=1,
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
    # Hist backend uses Koho et al. 2019 convention:
    #   - angle=0° = XY plane (lateral) → use XY spacing
    #   - angle=90° = Z axis (axial) → use Z spacing
    if spacing_list is not None:
        spacing_xy = spacing_list[1]  # Y spacing (assumes Y==X)
        spacing_z = spacing_list[0]  # Z spacing
    else:
        spacing_xy = 1.0
        spacing_z = 1.0

    # Find min/max angles to determine XY and Z sectors
    # hist backend uses theta = arctan2(k_xy, |kz|):
    #   - theta ≈ 0° → frequency vector along Z axis → measures Z resolution
    #   - theta ≈ 90° → frequency vector in XY plane → measures XY resolution
    angles = sorted(fsc_data.keys())
    angle_xy = angles[
        -1
    ]  # High angle (near 90°) → XY plane frequencies → XY resolution
    angle_z = angles[0]  # Low angle (near 0°) → Z axis frequencies → Z resolution

    results = {}
    for angle, data_set in fsc_data.items():
        # Select appropriate spacing for this angle
        # XY resolution uses XY (lateral) spacing, Z resolution uses Z (axial) spacing
        analyzer_spacing = spacing_xy if angle == angle_xy else spacing_z

        # Create data collection properly
        data_collection = FourierCorrelationDataCollection()
        data_collection[0] = data_set

        # Use one-bit threshold for 3D FSC as recommended by Koho et al. 2019
        # (SNRe = 0.5, varies with number of voxels per bin)
        analyzer = FourierCorrelationAnalysis(
            data_collection,
            analyzer_spacing,
            resolution_threshold="one-bit",
            curve_fit_type="spline",
        )
        try:
            analyzed = analyzer.execute()[0]
            resolution = analyzed.resolution["resolution"]
        except Exception:
            # Fitting can fail with insufficient data points
            resolution = float("nan")

        if angle == angle_xy:
            results["xy"] = resolution
        elif angle == angle_z:
            results["z"] = resolution
        # Intermediate angles are computed but not returned

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
