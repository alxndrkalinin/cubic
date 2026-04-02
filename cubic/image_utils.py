"""Implements utility functions that operate on 3D images."""

import warnings
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .cuda import asnumpy, to_same_device, get_array_module
from .skimage import measure, exposure, transform

# Thresholds for binomial_split calibration warnings
_FLOAT_NON_INTEGER_FRAC_THRESHOLD = 0.01  # pixel tolerance for integer check
_FLOAT_NON_INTEGER_WARN_FRACTION = 0.1  # warn if >10% of pixels are non-integer
_NEGATIVE_ELECTRON_WARN_FRACTION = 0.05  # warn if >5% of pixels go negative
_CLIPPED_READOUT_WARN_FRACTION = 0.1  # warn if >10% of pixels clipped to 0


# image operations assume ZYX channel order
def image_stats(
    img: np.ndarray,
    q: tuple[float, float] = (0.1, 99.9),
) -> dict[str, float]:
    """Compute intensity image statistics (min, max, mean, percentiles)."""
    q_min, q_max = np.percentile(img, q=q)
    return {
        "min": np.min(img),
        "max": np.max(img),
        "mean": np.mean(img),
        "percentile_min": q_min,
        "precentile_max": q_max,
    }


def rescale_xy(
    img: np.ndarray,
    scale: float = 1.0,
    order: int = 3,
    anti_aliasing: bool = True,
    preserve_range: bool = False,
) -> np.ndarray:
    """Rescale 2D image or 3D image in XY."""
    scale_by = scale if img.ndim == 2 else (1.0, scale, scale)
    return_dtype = img.dtype if preserve_range else np.float32
    return transform.rescale(
        img,
        scale_by,
        order=order,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
    ).astype(return_dtype)


def rescale_isotropic(
    img: np.ndarray,
    voxel_sizes: tuple[int, ...] | tuple[float, ...],
    downscale_xy: bool = False,
    order: int = 3,
    preserve_range: bool = True,
    target_z_size: int | None = None,
    target_z_voxel_size: float | None = None,
) -> np.ndarray:
    """Rescale image to isotropic voxels with arbitary Z (voxel) size."""
    if target_z_voxel_size is not None:
        target_z_size = int(
            round(img.shape[0] * (voxel_sizes[0] / target_z_voxel_size))
        )

    z_size_per_spacing = img.shape[0] * voxel_sizes[0] / np.asarray(voxel_sizes)
    if target_z_size is None:
        target_z_size = (
            img.shape[0] if downscale_xy else np.round(z_size_per_spacing[1])
        )
    factors = target_z_size / z_size_per_spacing
    return transform.rescale(
        img,
        factors,
        order=order,
        preserve_range=preserve_range,
        anti_aliasing=downscale_xy,
    )


def normalize_min_max(
    img: np.ndarray,
    q: tuple[float, float] = (0.1, 99.9),
) -> np.ndarray:
    """Normalize image intensities between percentiles."""
    vmin, vmax = np.percentile(img, q=q)
    return exposure.rescale_intensity(
        img, in_range=(float(vmin), float(vmax)), out_range=np.float32
    )


def img_mse(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Calculate pixel-wise MSE between two images."""
    assert len(a) == len(b)
    return np.square(a - b).mean()


def pad_image(
    img: np.ndarray,
    pad_size: int | Sequence[int],
    axes: int | Sequence[int] = 0,
    mode: str = "reflect",
    deps: dict | None = None,
) -> np.ndarray:
    """Pad an image."""
    npad = np.asarray([(0, 0)] * img.ndim)
    axes = [axes] if isinstance(axes, int) else axes
    for ax in axes:
        npad[ax] = [pad_size] * 2 if isinstance(pad_size, int) else [pad_size[ax]] * 2
    return np.pad(img, pad_width=npad, mode=mode)  # type: ignore[call-overload]


def pad_image_to_cube(
    img: np.ndarray,
    cube_size: int | None = None,
    mode: str = "constant",
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Pad all image axes up to cubic shape."""
    axes = list(range(img.ndim)) if axes is None else axes
    cube_size = cube_size if cube_size is not None else np.max(img.shape)

    pad_sizes = [(0, 0)] * img.ndim
    for ax in axes:
        dim = img.shape[ax]
        if dim < cube_size:
            pad_before = (cube_size - dim) // 2
            pad_after = cube_size - dim - pad_before
            pad_sizes[ax] = (pad_before, pad_after)

    img = np.pad(img, pad_sizes, mode=mode)  # type: ignore[call-overload]
    assert np.all([img.shape[i] == cube_size for i in axes])
    return img


def pad_image_to_shape(
    img: np.ndarray, new_shape: Sequence[int], mode: str = "constant"
) -> np.ndarray:
    """Pad all image axis up to specified shape."""
    for i, dim in enumerate(img.shape):
        if dim < new_shape[i]:
            pad_size = (new_shape[i] - dim) // 2
            img = pad_image(img, pad_size=pad_size, axes=i, mode=mode)

    assert np.all([dim == new_shape[i] for i, dim in enumerate(img.shape)])
    return img


def pad_to_matching_shape(
    img1: np.ndarray, img2: np.ndarray, mode: str = "constant"
) -> tuple[np.ndarray, np.ndarray]:
    """Apply zero padding to make the size of two Images match."""
    shape = tuple(max(x, y) for x, y in zip(img1.shape, img2.shape))

    if any(map(lambda x, y: x != y, img1.shape, shape)):
        img1 = pad_image_to_shape(img1, shape, mode=mode)
    if any(map(lambda x, y: x != y, img2.shape, shape)):
        img2 = pad_image_to_shape(img2, shape, mode=mode)

    return img1, img2


def crop_tl(
    img: np.ndarray,
    crop_size: int | Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop from the top-left corner."""
    return crop_corner(img, crop_size, axes, "tl")


def crop_bl(
    img: np.ndarray,
    crop_size: int | Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop from the bottom-left corner."""
    return crop_corner(img, crop_size, axes, "bl")


def crop_tr(
    img: np.ndarray,
    crop_size: int | Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop from the top-right corner."""
    return crop_corner(img, crop_size, axes, "tr")


def crop_br(
    img: np.ndarray,
    crop_size: int | Sequence[int],
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop from the bottom-right corner."""
    return crop_corner(img, crop_size, axes, "br")


def crop_corner(
    img: np.ndarray,
    crop_size: int | Sequence[int],
    axes: Sequence[int] | None = None,
    corner: str = "tl",
) -> np.ndarray:
    """Crop a corner from the image."""
    axes = [1, 2] if axes is None else axes
    crop_size = [crop_size] * len(axes) if isinstance(crop_size, int) else crop_size

    if len(crop_size) != len(axes):
        raise ValueError("Length of 'crop_sizes' must match the length of 'axes'.")

    slices = [slice(None)] * img.ndim

    for axis, size in zip(axes, crop_size):
        if axis >= img.ndim:
            raise ValueError("Axis index out of range for the image dimensions.")

        if "t" in corner or "l" in corner:
            start = 0
            end = size if size <= img.shape[axis] else img.shape[axis]
        else:
            start = -size if size <= img.shape[axis] else -img.shape[axis]
            end = None

        if "r" in corner and axis == axes[-1]:
            slices[axis] = slice(-end if end is not None else None, None)
        elif "b" in corner and axis == axes[-2]:
            slices[axis] = slice(-end if end is not None else None, None)
        else:
            slices[axis] = slice(start, end)

    return img[tuple(slices)]


def crop_center(
    img: np.ndarray,
    crop_size: int | Sequence[int] | None,
    axes: Sequence[int] | None = None,
) -> np.ndarray:
    """Crop from the center of the n-dimensional image."""
    axes = list(range(img.ndim)) if axes is None else axes
    if crop_size is None:
        crop_size = [min(img.shape[axis] for axis in axes)] * len(axes)
    elif isinstance(crop_size, int):
        crop_size = [crop_size] * len(axes)

    if len(axes) != len(crop_size):
        raise ValueError("The length of 'axes' should be the same as 'crop_size'")

    slices = []
    for axis in range(img.ndim):
        if axis in axes and img.shape[axis] > crop_size[axes.index(axis)]:
            idx = axes.index(axis)
            center = img.shape[axis] // 2
            half_crop = crop_size[idx] // 2
            start = center - half_crop
            end = center + half_crop + crop_size[idx] % 2
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))

    return img[tuple(slices)]


def get_random_crop_coords(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> tuple[int, int, int, int]:
    """Crop from a random location in the image."""
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(
    img: np.ndarray,
    crop_hw: int | tuple[int, int],
    return_coordinates: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop from a random location in the image."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = img.shape[1:]
    h_start, w_start = np.random.uniform(), np.random.uniform()
    x1, y1, x2, y2 = get_random_crop_coords(
        height, width, crop_h, crop_w, h_start, w_start
    )
    cropped = img[:, y1:y2, x1:x2] if img.ndim > 2 else img[y1:y2, x1:x2]
    if return_coordinates:
        return (cropped, (y1, y2, x1, x2))
    else:
        return cropped


def crop_to_divisor(
    img: np.ndarray,
    divisors: int | Sequence[int],
    axes: Sequence[int] | None = None,
    crop_type: str = "center",
) -> np.ndarray:
    """Crop image to be divisible by the given divisors along specified axes."""
    if axes is None:
        axes = [1, 2]  # default to xy axes in a 3d image

    if isinstance(divisors, int):
        divisors = [divisors] * len(axes)

    if len(axes) != len(divisors):
        raise ValueError("Length of 'axes' and 'divisors' must be the same")

    crop_size = [
        img.shape[axis] - (img.shape[axis] % divisor)
        for axis, divisor in zip(axes, divisors)
    ]

    if crop_type == "center":
        return crop_center(img, crop_size=crop_size, axes=axes)
    elif crop_type == "tl":
        return crop_tl(img, crop_size)
    elif crop_type == "bl":
        return crop_bl(img, crop_size)
    elif crop_type == "tr":
        return crop_tr(img, crop_size)
    elif crop_type == "br":
        return crop_br(img, crop_size)
    else:
        raise ValueError(
            "Invalid crop type specified. Choose from 'center', 'tl', 'bl', 'tr', 'br'."
        )


def rotate_image(
    image: np.ndarray, angle: float, interpolation: str = "nearest"
) -> np.ndarray:
    """Rotate 3D image around the Z axis by ``angle`` degrees."""
    xp = get_array_module(image)
    order = 1 if interpolation == "linear" else 0
    if xp.__name__ == np.__name__:
        from scipy.ndimage import rotate
    else:
        from cupyx.scipy.ndimage import rotate  # type: ignore
    return rotate(
        image, angle, axes=(1, 2), reshape=False, order=order, mode="constant"
    )


def get_xy_block_coords(
    image_shape: Sequence[int], crop_hw: int | tuple[int, int]
) -> np.ndarray:
    """Compute coordinates of non-overlapping image blocks of specified shape."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = image_shape[1:]

    block_coords = []  # type: list[tuple[int, ...]]
    for y in np.arange(0, height // crop_h) * crop_h:
        block_coords.extend(
            [
                (int(y), int(y + crop_h), int(x), int(x + crop_w))
                for x in np.arange(0, width // crop_w) * crop_w
            ]
        )

    return np.asarray(block_coords).astype(int)


def get_xy_block(img: np.ndarray, patch_coordinates: Sequence[int]) -> np.ndarray:
    """Slice subvolume of 3D image by XY coordinates."""
    return img[
        :,
        patch_coordinates[0] : patch_coordinates[1],
        patch_coordinates[2] : patch_coordinates[3],
    ]


def extract_patches(
    img: np.ndarray, patch_coordinates: Sequence[Sequence[int]]
) -> list[np.ndarray]:
    """Extract 3D patches from image given XY coordinates."""
    return [get_xy_block(img, patch_coords) for patch_coords in patch_coordinates]


def _nd_window(
    data: np.ndarray,
    filter_function: Callable[..., np.ndarray],
    power_function: Callable[..., np.ndarray],
    **kwargs: Any,
) -> np.ndarray:
    """Perform on N-dimensional spatial-domain data to mitigate boundary effects in the FFT."""
    result = data.copy().astype(np.float32)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [
            1,
        ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain array intensity
        power_function(window, (1.0 / data.ndim), out=window)
        result *= window
    return result


def hamming_window(data: np.ndarray) -> np.ndarray:
    """Apply Hamming window to data."""
    xp = get_array_module(data)
    return _nd_window(data, xp.hamming, xp.power)


def _tukey_window_1d(n: int, alpha: float, xp) -> np.ndarray:
    """Create 1D Tukey window (device-agnostic, matches scipy exactly).

    Parameters
    ----------
    n : int
        Window length.
    alpha : float
        Taper fraction (0 = rectangular, 1 = Hann).
    xp : module
        Array module (numpy or cupy).

    Returns
    -------
    np.ndarray
        1D Tukey window of length *n*, dtype float32.
    """
    if alpha <= 0:
        return xp.ones(n, dtype=np.float32)
    if alpha >= 1:
        if n <= 1:
            return xp.ones(max(n, 0), dtype=np.float32)
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


def tukey_window(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply separable Tukey window for edge apodization.

    Parameters
    ----------
    data : np.ndarray
        N-dimensional input array (numpy or cupy).
    alpha : float
        Taper fraction (0 = rectangular, 1 = Hann). Default 0.1.

    Returns
    -------
    np.ndarray
        Windowed copy of *data*.
    """
    xp = get_array_module(data)
    result = data.copy()
    for axis in range(result.ndim):
        w = _tukey_window_1d(result.shape[axis], alpha, xp)
        shape = [1] * result.ndim
        shape[axis] = result.shape[axis]
        w = w.reshape(shape)
        result *= w
    return result


def _checkerboard_split_impl(
    img: np.ndarray,
    disable_3d_sum: bool,
    preserve_range: bool,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Split image using checkerboard pattern.

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D image array
    disable_3d_sum : bool
        If True, use full 3D checkerboard without Z-summing.
    preserve_range : bool
        If True, keep the original range of values and data type.
    reverse : bool
        If True, use reverse checkerboard pattern (other diagonal).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two split images with half the size in each spatial dimension.
    """
    # Determine safe dtype and warn if needed
    needs_z_summing = img.ndim == 3 and not disable_3d_sum
    is_integer = np.issubdtype(img.dtype, np.integer)

    if preserve_range:
        safe_dtype = img.dtype
        if needs_z_summing and is_integer:
            warnings.warn(
                f"preserve_range=True with integer dtype {img.dtype} may cause "
                "overflow when summing consecutive Z pairs. Consider using "
                "preserve_range=False to convert to float32 for safe computation.",
                UserWarning,
                stacklevel=3,
            )
    else:
        safe_dtype = np.float32 if is_integer else img.dtype

    # Apply dtype conversion if needed
    img_safe = img if img.dtype == safe_dtype else img.astype(safe_dtype)

    # Define slicing patterns based on reverse flag
    # Pattern matches miplib implementation (Koho et al. 2019)
    # Both images sample from the same diagonal parity for proper FSC calculation
    if reverse:
        # Reverse pattern: (odd, even) vs (even, odd)
        pattern1_y, pattern1_x = 1, 0
        pattern2_y, pattern2_x = 0, 1
    else:
        # Regular pattern: (odd, odd) vs (even, even) - matches miplib
        pattern1_y, pattern1_x = 1, 1
        pattern2_y, pattern2_x = 0, 0

    if img.ndim == 2:
        # Truncate to even dimensions to ensure matching shapes
        h_even = img_safe.shape[0] // 2 * 2
        w_even = img_safe.shape[1] // 2 * 2
        img_even = img_safe[:h_even, :w_even]
        image1 = img_even[pattern1_y::2, pattern1_x::2]
        image2 = img_even[pattern2_y::2, pattern2_x::2]
    elif disable_3d_sum:
        # Truncate all dimensions to even to ensure matching shapes
        z_even = img_safe.shape[0] // 2 * 2
        h_even = img_safe.shape[1] // 2 * 2
        w_even = img_safe.shape[2] // 2 * 2
        img_even = img_safe[:z_even, :h_even, :w_even]
        image1 = img_even[1::2, pattern1_y::2, pattern1_x::2]
        image2 = img_even[0::2, pattern2_y::2, pattern2_x::2]
        if preserve_range:
            image1 = image1.astype(img.dtype)
            image2 = image2.astype(img.dtype)
    else:
        # Z-summing: sum consecutive Z pairs, then apply 2D checkerboard
        # Truncate all dimensions to even to ensure matching shapes
        z_even = img_safe.shape[0] // 2 * 2
        h_even = img_safe.shape[1] // 2 * 2
        w_even = img_safe.shape[2] // 2 * 2
        img_even = img_safe[:z_even, :h_even, :w_even]
        z_summed = img_even[0::2] + img_even[1::2]
        image1 = z_summed[:, pattern1_y::2, pattern1_x::2]
        image2 = z_summed[:, pattern2_y::2, pattern2_x::2]
        if preserve_range:
            image1 = image1.astype(img.dtype)
            image2 = image2.astype(img.dtype)

    return image1, image2


def checkerboard_split(
    img: np.ndarray,
    disable_3d_sum: bool = False,
    preserve_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image in two using checkerboard pattern.

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D image array
    disable_3d_sum : bool, optional
        If True, use full 3D checkerboard without Z-summing.
        Default False uses Koho et al. 2019 strategy.
    preserve_range : bool, optional
        If True, keep the original range of values and data type.
        If False (default), integer types are converted to float32 to avoid
        overflow when summing consecutive Z pairs. Float types are preserved.
        When True with integer types, overflow may occur if the sum exceeds
        the dtype range. Default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two split images with half the size in each spatial dimension.
    """
    return _checkerboard_split_impl(img, disable_3d_sum, preserve_range, reverse=False)


def reverse_checkerboard_split(
    img: np.ndarray,
    disable_3d_sum: bool = False,
    preserve_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an image using reverse checkerboard pattern (other diagonal).

    Parameters
    ----------
    img : np.ndarray
        Input 2D or 3D image array
    disable_3d_sum : bool, optional
        If True, use full 3D checkerboard without Z-summing.
        Default False uses Koho et al. 2019 strategy.
    preserve_range : bool, optional
        If True, keep the original range of values and data type.
        If False (default), integer types are converted to float32 to avoid
        overflow when summing consecutive Z pairs. Float types are preserved.
        When True with integer types, overflow may occur if the sum exceeds
        the dtype range. Default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two split images with half the size in each spatial dimension.
    """
    return _checkerboard_split_impl(img, disable_3d_sum, preserve_range, reverse=True)


def binomial_split(
    image: np.ndarray,
    p: float = 0.5,
    *,
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a single image into two noise-independent halves via binomial sampling.

    Implements the single-image FRC method from Rieger et al. (Optics Express 2024).
    Each pixel's photon count *n* is split into n1 ~ Binomial(n, p) and n2 = n - n1,
    producing two images with preserved Poisson statistics and no spatial subsampling.

    Parameters
    ----------
    image : np.ndarray
        Input 2D or 3D image (NumPy or CuPy).
    p : float
        Split probability. Default 0.5 (equal split).
    counts_mode : {"counts", "poisson_thinning"}
        ``"counts"`` — interpret pixels as photon counts (or convert via gain/offset).
        ``"poisson_thinning"`` — treat pixel values as Poisson rates λ and draw
        n1 ~ Poisson(p·λ), n2 ~ Poisson((1-p)·λ) independently. Does NOT conserve
        exact pixel sums. Useful as a fallback for float / deconvolved images, but
        note that it measures self-consistency, not physical resolution.
    gain : float
        Camera gain (ADU per electron). Used only in counts mode. Default 1.0.
        Conversion: ``electrons = (image - offset) / gain``.
    offset : float
        Camera offset (ADU). Subtracted before gain conversion. Default 0.0.
    readout_noise_rms : float
        Read-noise standard deviation in **electrons**. When > 0 in counts mode,
        applies the Rieger et al. Eq. 29 bias correction: σ² is added to the count
        estimate before splitting, then σ²/2 is subtracted from each half. This
        corrects the high-frequency FRC plateau caused by read noise. If gain/offset
        or readout_noise_rms are wrong, this correction can make results worse.
    rng : Generator, int, or None
        NumPy random number generator, seed, or None for unseeded.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two float32 images. In counts mode (without readout correction) their sum
        equals the integer count image exactly.

    Notes
    -----
    **Poisson thinning mode** does not split actual photon counts — it samples two
    independent Poisson draws. This is not physically interpretable as resolution on
    deconvolved or heavily processed images; it measures the effective reproducible
    bandwidth of a noise model applied to the data.

    References
    ----------
    Rieger, Droste, Gerritsma, ten Brink, Stallinga. "Single image Fourier ring
    correlation." Optics Express 32(12):21767, 2024.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    if gain <= 0:
        raise ValueError(f"gain must be > 0, got {gain}")
    if readout_noise_rms < 0:
        raise ValueError(f"readout_noise_rms must be >= 0, got {readout_noise_rms}")

    # Resolve RNG
    if isinstance(rng, int):
        np_rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        np_rng = rng
    else:
        np_rng = np.random.default_rng()

    if counts_mode == "counts":
        # --- counts mode ---
        # Warn if float input with default calibration (likely forgot gain/offset)
        if np.issubdtype(image.dtype, np.floating):
            frac_part = np.abs(image - np.rint(image))
            frac_fraction = float(
                np.mean(frac_part > _FLOAT_NON_INTEGER_FRAC_THRESHOLD)
            )
            if (
                frac_fraction > _FLOAT_NON_INTEGER_WARN_FRACTION
                and gain == 1.0
                and offset == 0.0
            ):
                warnings.warn(
                    f"counts_mode='counts' received float image with "
                    f"{frac_fraction:.0%} non-integer pixels and default "
                    f"gain/offset. Did you forget to set gain/offset?",
                    UserWarning,
                    stacklevel=2,
                )

        # Convert ADU → electrons
        electrons = (image.astype(np.float64) - offset) / gain

        # Warn if large negative region before clipping (offset not removed)
        neg_fraction = float(np.mean(electrons < -0.5))
        if neg_fraction > _NEGATIVE_ELECTRON_WARN_FRACTION:
            warnings.warn(
                f"{neg_fraction:.0%} of pixels are negative after offset "
                f"subtraction — check that offset={offset} is correct.",
                UserWarning,
                stacklevel=2,
            )

        electrons = np.clip(electrons, 0, None)

        # Readout noise bias correction (Rieger Eq. 29): add σ² before splitting
        if readout_noise_rms > 0:
            electrons = electrons + readout_noise_rms**2

        # Round to integer counts
        n = np.rint(electrons).astype(np.int64)

        # Draw binomial split on CPU (transfer if needed)
        n_cpu = asnumpy(n)
        n1_cpu = np_rng.binomial(n_cpu, p)
        n2_cpu = n_cpu - n1_cpu

        img1 = n1_cpu.astype(np.float32)
        img2 = n2_cpu.astype(np.float32)

        # Readout noise bias correction: subtract σ²/2 from each half
        if readout_noise_rms > 0:
            half_var = np.float32(readout_noise_rms**2 / 2.0)
            img1 = np.clip(img1 - half_var, 0, None)
            img2 = np.clip(img2 - half_var, 0, None)
            clipped_fraction = float(np.mean(n1_cpu.astype(np.float32) < half_var))
            if clipped_fraction > _CLIPPED_READOUT_WARN_FRACTION:
                warnings.warn(
                    f"Readout noise correction clipped {clipped_fraction:.0%} of "
                    f"pixels to zero — readout_noise_rms={readout_noise_rms} may "
                    f"be too large for these photon counts.",
                    UserWarning,
                    stacklevel=2,
                )

        # Transfer back to original device if needed
        img1 = to_same_device(img1, image)
        img2 = to_same_device(img2, image)

    elif counts_mode == "poisson_thinning":
        # --- poisson thinning mode ---
        rate = np.clip(image.astype(np.float64), 0, None)

        rate_cpu = asnumpy(rate)
        n1_cpu = np_rng.poisson(p * rate_cpu).astype(np.float32)
        n2_cpu = np_rng.poisson((1.0 - p) * rate_cpu).astype(np.float32)

        img1 = to_same_device(n1_cpu, image)
        img2 = to_same_device(n2_cpu, image)

    else:
        raise ValueError(
            f"counts_mode must be 'counts' or 'poisson_thinning', got {counts_mode!r}"
        )

    return img1, img2


def label(img: npt.ArrayLike, **kwargs: Any) -> np.ndarray:
    """Label image using skimage.measure.label."""
    return measure.label(img, **kwargs)


def select_max_contrast_slices(
    img: np.ndarray, num_slices: int = 128, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, slice]:
    """Select num_slices consecutive Z slices with maximum contrast from a 3D volume."""
    assert img.ndim > 2, "Image should have more than 2 dimensions."
    std_devs = asnumpy(img.std(tuple(range(1, img.ndim))))
    # calculate rolling sum of standard deviations for num_slices
    rolling_sum = np.convolve(std_devs, np.ones(num_slices), "valid")
    max_contrast_idx = np.argmax(rolling_sum)
    indices = slice(max_contrast_idx, max_contrast_idx + num_slices)
    if return_indices:
        return img[indices], indices
    return img[indices]


def distance_transform_edt(
    img: npt.ArrayLike,
    sampling: Sequence[float] | None = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: npt.ArrayLike | None = None,
    indices: npt.ArrayLike | None = None,
    block_params: tuple[int, int, int] | None = None,
    float64_distances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Euclidean distance transform of a binary image."""
    if isinstance(img, np.ndarray):
        if block_params is not None or float64_distances:
            raise ValueError(
                "NumPy array found. 'block_params' and 'float64_distances' can only be used with CuPy arrays."
            )
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(
            img,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
            distances=distances,
            indices=indices,
        )
    else:
        # cuCIM access interface is different from scipy.ndimage
        from cucim.core.operations.morphology import distance_transform_edt

        return distance_transform_edt(
            img,
            sampling=sampling,
            return_distances=return_distances,
            return_indices=return_indices,
            distances=distances,
            indices=indices,
            block_params=None,
            float64_distances=False,
        )


def clahe(
    img: np.ndarray,
    kernel_size: np.ndarray | tuple[int, int, int] = (2, 3, 5),
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> np.ndarray:
    """Apply CLAHE to the image."""
    assert len(img.shape) == len(kernel_size)
    kernel_size = np.asarray(img.shape) // kernel_size
    img = exposure.equalize_adapthist(
        img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins
    )
    return img
