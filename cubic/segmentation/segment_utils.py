"""Implement pre- and post-processing for segmentation."""

import warnings
from collections.abc import Sequence

import numpy as np
from skimage.segmentation import watershed

from ..cuda import asnumpy, to_device, get_device
from ..skimage import feature, filters, transform, morphology
from ..image_utils import label, pad_image, distance_transform_edt
from ._clear_border import clear_border


def downscale_and_filter(
    image: np.ndarray,
    downscale_factor: float = 0.5,
    downscale_order: int = 3,
    downscale_anti_aliasing: bool = True,
    filter_size: int = 3,
    filter_shape: str = "square",
    *,
    downscale_xy_only: bool = True,
    filter_mode: str = "nearest",
) -> np.ndarray:
    """Subsample and filter image prior to segmentation.

    Parameters
    ----------
    image : np.ndarray
        Image to be downsampled and filtered.
    downscale_factor : float, optional
        Factor by which to downscale the image, by default 0.5.
    downscale_order : int, optional
        Interpolation order for downscaling, by default 3.
    downscale_anti_aliasing : bool, optional
        Whether to apply anti-aliasing during downscaling, by default True.
    downscale_xy_only : bool, optional
        If True (default), only downscale XY dimensions, preserving Z for
        3D images. If False, downscale all dimensions uniformly.
    filter_size : int, optional
        Size of median filter kernel, by default 3.
    filter_shape : str, optional
        Shape of the filter kernel: ``"square"`` (cube in 3D) uses
        ``scipy.ndimage.median_filter(size=filter_size)`` which supports
        boundary modes; ``"circular"`` (ball in 3D) uses
        ``skimage.filters.median`` with a shaped footprint.
    filter_mode : str, optional
        Boundary mode for the median filter, by default ``"nearest"``.
        Only used when ``filter_shape="square"``. Common values:
        ``"constant"`` (zero-padding), ``"nearest"``, ``"reflect"``.

    Returns
    -------
    np.ndarray
        Filtered and downsampled image.

    """
    from ..scipy import ndimage as _ndimage

    if filter_shape not in ("square", "circular"):
        raise ValueError("filter_shape must be 'square' or 'circular'.")

    if downscale_factor < 1.0:
        if downscale_xy_only:
            from ..image_utils import rescale_xy

            image = rescale_xy(
                image,
                scale=downscale_factor,
                order=downscale_order,
                anti_aliasing=downscale_anti_aliasing,
            )
        else:
            image = transform.rescale(
                image,
                downscale_factor,
                order=downscale_order,
                anti_aliasing=downscale_anti_aliasing,
            )

    if filter_shape == "square":
        return _ndimage.median_filter(image, size=filter_size, mode=filter_mode)

    if image.ndim == 2:
        footprint = morphology.disk(filter_size)
    elif image.ndim == 3:
        footprint = morphology.ball(filter_size)
    else:
        raise ValueError("Image must be 2D or 3D.")

    return filters.median(image, footprint=footprint)


def check_labeled_binary(image):
    """Check if the given image is a labeled image.

    Parameters
    ----------
    image : ndarray
        The image to be checked.

    Returns
    -------
    None

    """
    assert np.issubdtype(image.dtype, np.integer), "Image must be of integer type."

    unique_values = np.unique(image)
    assert len(unique_values) > 1, "Image is constant."
    if len(unique_values) == 2:
        warnings.warn(
            "Only one label was provided in the image. Make sure to label components first."
        )


def cleanup_segmentation(
    label_img: np.ndarray,
    min_obj_size: int | None = None,
    max_obj_size: int | None = None,
    border_buffer_size: int | None = None,
    max_hole_size: int | None = None,
) -> np.ndarray:
    """Clean up segmented image by removing small objects, clearing borders, and closing holes."""
    check_labeled_binary(label_img)

    # first 3 transforms preserve labels
    if min_obj_size is not None:
        # min_obj_size = to_device(min_obj_size, get_device(label_image))
        label_img = morphology.remove_small_objects(label_img, min_size=min_obj_size)

    if max_obj_size is not None:
        label_img = remove_large_objects(label_img, max_size=max_obj_size)

    if border_buffer_size is not None:
        label_img = clear_xy_borders(label_img, buffer_size=border_buffer_size)

    # returns boolean array
    if max_hole_size is not None:
        for label_id in np.unique(label_img)[1:]:
            mask = label_img == label_id
            filled_mask = morphology.remove_small_holes(
                mask, area_threshold=max_hole_size
            )
            label_img[filled_mask] = label_id

    return label(label_img).astype(np.uint16)


def find_objects(label_image, max_label=None):
    """Find objects in a labeled nD array.

    Parameters
    ----------
    label_image : cupy.ndarray
        nD array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        specified, the positions of all objects up to the highest label are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label `l` corresponds to
        the index `l-1` in the returned list.

    """
    if max_label is None:
        max_label = int(np.max(label_image))

    object_slices = [None] * max_label

    for label_idx in range(1, max_label + 1):
        mask = label_image == label_idx
        if not mask.any():
            continue

        slices = []
        for dim in range(mask.ndim):
            axis_indices = np.any(
                mask,
                axis=tuple(range(mask.ndim))[:dim] + tuple(range(mask.ndim))[dim + 1 :],
            )
            if not axis_indices.any():
                slices.append(None)
                continue
            min_idx = int(np.where(axis_indices)[0].min())
            max_idx = int(np.where(axis_indices)[0].max()) + 1
            slices.append(slice(min_idx, max_idx))

        object_slices[label_idx - 1] = tuple(slices)

    return object_slices


def remove_large_objects(label_image: np.ndarray, max_size: int = 100000) -> np.ndarray:
    """Remove objects with volume above specified threshold."""
    check_labeled_binary(label_image)
    label_volumes = np.bincount(label_image.ravel())
    too_large = label_volumes > max_size
    too_large_mask = too_large[label_image]
    label_image[too_large_mask] = 0
    return label_image


def remove_small_objects(label_image: np.ndarray, min_size: int = 500) -> np.ndarray:
    """Remove objects with volume below specified threshold."""
    check_labeled_binary(label_image)
    label_image = morphology.remove_small_objects(label_image, min_size=min_size)
    return label_image


def clear_xy_borders(label_image: np.ndarray, buffer_size: int = 0) -> np.ndarray:
    """Remove masks that touch XY borders."""
    check_labeled_binary(label_image)
    if label_image.ndim == 2:
        return clear_border(label_image, buffer_size=buffer_size)
    label_image = pad_image(
        label_image,
        (buffer_size + 1, buffer_size + 1),
        mode="constant",
    )
    label_image = clear_border(label_image, buffer_size=buffer_size)
    return label(label_image[buffer_size + 1 : -(buffer_size + 1), :, :])


def remove_touching_objects(
    label_image: np.ndarray, border_value: int = 100
) -> np.ndarray:
    """Find labelled masks that overlap and remove from the image."""
    check_labeled_binary(label_image)

    exclude_masks = []
    for mask_idx in np.unique(label_image)[1:]:
        if mask_idx not in exclude_masks:
            binary_mask = label_image == mask_idx
            dilated_mask = morphology.binary_dilation(binary_mask, morphology.cube(3))
            mask_outline = dilated_mask & ~binary_mask

            masks_copy = label_image.copy()
            masks_copy[mask_outline] += border_value

            if masks_copy[masks_copy > border_value].sum() > 0:
                overlap_masks = (
                    np.unique(masks_copy[masks_copy > border_value]) - border_value
                )
                exclude_masks += [mask_idx] + list(overlap_masks)

    for exclude_mask in exclude_masks:
        label_image[label_image == exclude_mask] = 0

    return label_image


def remove_thin_objects(label_image, min_z=2):
    """Remove objects thinner than a specified minimum value in Z."""
    unique_labels = [
        regionlabel for regionlabel in np.unique(label_image) if regionlabel != 0
    ]
    for regionlabel in unique_labels:
        mask = label_image == regionlabel

        maskz = np.any(mask, axis=(1, 2))
        z1 = np.argmax(maskz)
        z2 = len(maskz) - np.argmax(maskz[::-1])
        size_z = abs(z2 - z1)

        if size_z <= min_z:
            label_image[mask] = 0

    return label_image


def segment_watershed(
    image: np.ndarray,
    markers: np.ndarray | None = None,
    ball_size: int = 15,
    *,
    mask: np.ndarray | None = None,
    dilate_seeds: bool = False,
) -> np.ndarray:
    """Segment image using watershed algorithm.

    When ``markers`` is None, computes a distance-based watershed:
    EDT of the binary image is used to find peaks, which become markers,
    and the watershed floods the negated distance.

    When ``markers`` is provided, runs a marker-based watershed. By default
    the image is used as both the landscape and mask. If ``mask`` is also
    provided, the watershed uses the negated EDT of the mask as the
    landscape (shape-based partitioning) and restricts flooding to the mask.

    Parameters
    ----------
    image : np.ndarray
        Binary image to segment (distance-based) or intensity image
        (marker-based when no mask is given).
    markers : np.ndarray or None, optional
        Pre-computed markers for marker-based watershed. If None,
        markers are generated from distance-transform peaks.
    mask : np.ndarray or None, optional
        Binary mask restricting the watershed. When provided with markers,
        the watershed landscape is the negated EDT of the mask (shape-based
        partitioning). Only used when ``markers`` is not None.
    ball_size : int, optional
        Radius of the ball footprint for ``peak_local_max``, by default 15.
        Only used when ``markers`` is None.
    dilate_seeds : bool, optional
        If True, dilate seed points with ``ball(1)`` before labeling.
        This merges nearby peaks and reduces over-segmentation.
        Only used when ``markers`` is None.

    Returns
    -------
    np.ndarray
        Label image on the same device as the input.

    """
    from ..cuda import to_same_device

    device = get_device(image)

    # Distance-based watershed (no markers provided)
    if markers is None:
        distance = distance_transform_edt(image)
        footprint = morphology.ball(ball_size)
        footprint = to_same_device(footprint, distance)  # type: ignore[arg-type]
        coords = feature.peak_local_max(distance, footprint=footprint, labels=image)

        seed_mask = np.zeros(distance.shape, dtype=bool)  # type: ignore[union-attr]
        seed_mask[tuple(asnumpy(coords).T)] = True
        seed_mask = to_device(seed_mask, device)
        if dilate_seeds:
            seed_mask = morphology.binary_dilation(
                seed_mask, to_same_device(morphology.ball(1), seed_mask)
            )
        markers = label(seed_mask)
        # watershed is not in cucim — run on CPU, return to original device
        labels = watershed(-asnumpy(distance), asnumpy(markers), mask=asnumpy(image))  # type: ignore[arg-type]
        return to_device(labels, device)

    # Marker-based watershed with explicit mask (shape-based partitioning)
    if mask is not None:
        distance = distance_transform_edt(asnumpy(mask))
        assert isinstance(distance, np.ndarray)
        ws_image = -distance
        ws_image = ws_image - ws_image.min()
        labels = watershed(ws_image, markers=asnumpy(markers), mask=asnumpy(mask))
        return to_device(labels, device)

    # Marker-based watershed without mask (image as landscape and mask)
    img_cpu = asnumpy(image)
    labels = watershed(img_cpu, markers=asnumpy(markers), mask=img_cpu)
    return to_device(labels, device)


def _binary_fill_holes(image):
    """Fill holes in binary objects."""
    if get_device(image) == "GPU":
        from cupyx.scipy.ndimage import binary_fill_holes
    elif get_device(image) == "CPU":
        from scipy.ndimage import binary_fill_holes
    else:
        raise ValueError("Unknown device.")

    return binary_fill_holes(image)


def fill_label_holes(lbl_img, **binary_fill_holes_kwargs):
    """Fill small holes in label image.

    Inspired by: https://github.com/stardist/stardist/blob/master/stardist/utils.py
    """

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = _binary_fill_holes(grown_mask, **binary_fill_holes_kwargs)[
            shrink_slice
        ]
        lbl_img_filled[sl][mask_filled] = i

    return lbl_img_filled


def fill_holes_slicer(
    image: np.ndarray,
    area_threshold: int = 1000,
    num_iterations: int = 1,
    axes: Sequence[int] | None = None,
):
    """Fill holes in slices of binary or labeled objects.

    Inspired by: https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/morphology/fill_holes.py
    """
    img = np.asarray(image)
    axes = range(img.ndim) if axes is None else axes

    for label_id in np.unique(img)[1:]:
        binary = img == label_id

        for _ in range(num_iterations):
            for axis in axes:
                slicers = [slice(None)] * img.ndim
                for i in range(binary.shape[axis]):
                    slicers[axis] = slice(i, i + 1)
                    binary_slice = binary[tuple(slicers)]
                    filled_slice = morphology.remove_small_holes(
                        binary_slice, area_threshold
                    )
                    if filled_slice.shape != binary_slice.shape:
                        raise ValueError(
                            "Shape mismatch between filled_slice and binary_slice"
                        )
                    binary[tuple(slicers)] = filled_slice

        img[binary] = label_id

    return img
