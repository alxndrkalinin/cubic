"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from typing import Sequence, Callable, Any, cast

import numpy as np

from morphocell.cuda import asnumpy
from morphocell.image_utils import (
    crop_tl,
    crop_bl,
    crop_tr,
    crop_br,
    crop_center,
    pad_image,
    get_xy_block_coords,
    rescale_isotropic,
    pad_image_to_cube,
    hamming_window,
    checkerboard_split,
    reverse_checkerboard_split,
)

from .iterators import FourierRingIterator, AxialExcludeSectionedFourierShellIterator
from .analysis import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
    FourierCorrelationAnalysis,
)


def _empty_aggregate(*args: Any, **kwargs: Any) -> Any:
    """Return unchanged array."""
    return args[0]


def frc_checkerboard_split(
    image: np.ndarray, reverse: bool = False, disable_3d_sum: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Split image into two by checkerboard pattern."""
    if reverse:
        return reverse_checkerboard_split(image, disable_3d_sum=disable_3d_sum)
    else:
        return checkerboard_split(image, disable_3d_sum=disable_3d_sum)


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
        if image2 is not None and len(set(image2.shape)) > 1 and zero_padding:
            image2 = pad_image_to_cube(image2)

    assert image2 is not None

    # Apply Hamming windowing to both images independently
    if not disable_hamming:
        image1 = hamming_window(image1)
        image2 = hamming_window(image2)

    return image1, image2


class FRC(object):
    """A class for calculating 2D Fourier ring correlation."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, iterator):
        """Create new FRC executable object and perform FFT on input images."""
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.iterator = iterator
        # Calculate power spectra for the input images.
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))

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
        spatial_freq = asnumpy(radii.astype(np.float32) / self.freq_nyq)
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
    image1: np.ndarray, image2: np.ndarray, bin_delta: int
) -> FourierCorrelationDataCollection:
    """Core FRC calculation logic."""
    assert image1.shape == image2.shape
    frc_data = FourierCorrelationDataCollection()
    iterator = FourierRingIterator(image1.shape, bin_delta)
    frc_task = FRC(image1, image2, iterator)
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
    spacing: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
) -> FourierCorrelationData:
    """Calculate a regular FRC with single or two image inputs."""

    single_image = image2 is None
    reverse = average and single_image
    original_image1: np.ndarray | None = image1.copy() if reverse else None

    if isinstance(spacing, (int, float)):
        spacing = [spacing] * image1.ndim
    else:
        spacing = list(spacing)

    image1, image2 = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        disable_hamming=disable_hamming,
    )

    frc_data = _calculate_frc_core(image1, image2, bin_delta)

    # Average with reverse pattern (only for single image mode)
    if reverse:
        # Use original unprocessed image for reverse split
        assert original_image1 is not None
        image1, image2 = preprocess_images(
            original_image1,
            None,
            reverse_split=reverse,
            zero_padding=zero_padding,
            disable_hamming=disable_hamming,
        )

        frc_data_rev = _calculate_frc_core(image1, image2, bin_delta)

        # Average the two results
        frc_data[0].correlation["correlation"] = (
            0.5 * frc_data[0].correlation["correlation"]
            + 0.5 * frc_data_rev[0].correlation["correlation"]
        )

    # Analyze results
    analyzer = FourierCorrelationAnalysis(
        frc_data,
        spacing[0],
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
    spacing: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
    curve_fit_type: str = "smooth-spline",
) -> float:
    """Calculate either single- or two-image FRC-based 2D image resolution."""

    frc_result = calculate_frc(
        image1,
        image2,
        bin_delta=bin_delta,
        curve_fit_type=curve_fit_type,
        spacing=spacing,
        zero_padding=zero_padding,
    )

    return frc_result.resolution["resolution"]


class DirectionalFSC(object):
    """Calculate the directional FSC between two images."""

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
        self.fft_image1 = np.fft.fftshift(np.fft.fftn(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fftn(image2))
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
    spacing: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
) -> FourierCorrelationDataCollection:
    """Calculate sectioned FSC for one or two images."""

    single_image = image2 is None

    if isinstance(spacing, (int, float)):
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
        int(np.rad2deg(extract_angle_delta)),
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


def fsc_resolution(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 10,
    zero_padding: bool = True,
    spacing: float | Sequence[float] = 1.0,
) -> dict[str, float]:
    """Calculate either single- or two-image FSC-based 3D image resolution."""

    fsc_result = calculate_sectioned_fsc(
        image1,
        image2,
        bin_delta=bin_delta,
        spacing=spacing,
        zero_padding=zero_padding,
    )

    angle_to_resolution = {
        int(angle): dataset.resolution["resolution"] for angle, dataset in fsc_result
    }

    return {"xy": angle_to_resolution[0], "z": angle_to_resolution[90]}


def grid_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable | None = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by tiling and taking 2D slices along XY and XZ."""
    if not return_resolution or aggregate is None:
        aggregate_fn: Callable[..., Any] = _empty_aggregate
    else:
        aggregate_fn = cast(Callable[..., Any], aggregate)

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
    spacing: float | Sequence[float] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by taking 2D slices along XY and XZ at 4 corners and the center."""
    if not return_resolution or aggregate is None:
        aggregate_fn: Callable[..., Any] = _empty_aggregate
    else:
        aggregate_fn = cast(Callable[..., Any], aggregate)

    if isinstance(spacing, (int, float)):
        spacing = [spacing, spacing, spacing]

    assert len(image.shape) == 3 and len(spacing) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    spacing_xy = (spacing[1], spacing[2])
    spacing_xz = (spacing[0], spacing[2])

    locations: list[Callable[[np.ndarray, int], np.ndarray]] = [
        crop_tl,
        crop_bl,
        crop_tr,
        crop_br,
        lambda img, size: crop_center(img, size),
    ]
    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for loc in locations:
        loc_image = loc(image, crop_size)

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
    spacing: float | tuple[float, float, float] = 1.0,
    downscale_xy: bool = False,
    axis: str = "xy",
    frc_bin_delta: int = 3,
    aggregate: Callable = np.mean,
) -> float:
    """Calculate difference between FRC-based resulutions of two images."""
    if isinstance(spacing, (int, float)):
        spacing_tuple: tuple[float, float, float] = (
            float(spacing),
            float(spacing),
            float(spacing),
        )
    else:
        spacing_tuple = spacing
    if np.any(np.asarray(spacing_tuple) != 1.0):
        image1 = rescale_isotropic(
            image1, voxel_sizes=spacing_tuple, downscale_xy=downscale_xy
        )
        image2 = rescale_isotropic(
            image2, voxel_sizes=spacing_tuple, downscale_xy=downscale_xy
        )

    image1_res = grid_crop_resolution(
        image1,
        bin_delta=frc_bin_delta,
        spacing=spacing_tuple,
        aggregate=aggregate,
    )
    image2_res = grid_crop_resolution(
        image2,
        bin_delta=frc_bin_delta,
        spacing=spacing_tuple,
        aggregate=aggregate,
    )
    return (
        aggregate(image2_res[axis]) - aggregate(image1_res[axis])
    ) * 1000  # return diff in nm
