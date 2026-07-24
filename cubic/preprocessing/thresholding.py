"""Implement 3D image thresholding."""

import numpy as np
import numpy.typing as npt

from ..cuda import asnumpy
from ..skimage import util, filters
from ..image_utils import get_xy_block, get_xy_block_coords


def get_threshold_otsu(
    image: npt.ArrayLike,
    blur_sigma: int = 5,
    preserve_range: bool = False,
    nbins: int = 256,
) -> float:
    """Perform Otsu's thresholding with Gaussian blur."""
    image = filters.gaussian(image, sigma=blur_sigma, preserve_range=preserve_range)
    return filters.threshold_otsu(image, nbins=nbins)


def select_nonempty_patches(
    image: np.ndarray,
    patch_size: int = 512,
    min_nonzeros: float = 0.02,
    threshold: float | None = None,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Select XY patches from 3D image by percent of nonzero voxels.

    Returns the selected rows of :func:`get_xy_block_coords`, i.e. NumPy int
    arrays of ``(y_start, y_end, x_start, x_end)``. CuPy input is moved to the
    host first, since the returned coordinates are host values either way.
    """
    verboseprint = print if verbose else lambda *a, **k: None

    selected_patches: list[np.ndarray] = []

    image_np = asnumpy(image)

    if threshold is None:
        threshold = float(get_threshold_otsu(image_np))

    binary_image = (util.img_as_float32(image_np) > threshold).astype(np.uint8)

    patch_coordinates = get_xy_block_coords(image_np.shape, patch_size)
    verboseprint(f"Nonzero pixels in the image: {binary_image.mean()}")

    for single_patch_coords in patch_coordinates:
        binary_tile = get_xy_block(binary_image, single_patch_coords)
        patch_nonzero = np.count_nonzero(binary_tile) / binary_tile.size

        if patch_nonzero >= min_nonzeros:
            selected_patches.append(single_patch_coords)

    return selected_patches
