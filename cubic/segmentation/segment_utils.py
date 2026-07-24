"""Implement pre- and post-processing for segmentation."""

import inspect
import warnings
import itertools
from collections.abc import Iterator, Sequence

import numpy as np
from skimage import morphology as _sk_morphology
from skimage.segmentation import watershed

try:
    from cucim.skimage import morphology as _cu_morphology
except ImportError:  # cucim is an optional GPU-only dep
    _cu_morphology = None  # type: ignore[assignment]

from ..cuda import asnumpy, to_device, get_device, to_same_device
from ..scipy import ndimage as _ndimage
from ..skimage import feature, filters, transform, morphology
from ..image_utils import label, pad_image, rescale_xy, distance_transform_edt
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
    if filter_shape not in ("square", "circular"):
        raise ValueError("filter_shape must be 'square' or 'circular'.")

    if downscale_factor < 1.0:
        if downscale_xy_only:
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

    Raises ``TypeError`` if the input is not integer-typed. Empty FOVs
    (constant images, e.g. all-zero label masks) are accepted silently —
    downstream cleanup steps are no-ops on them and crashing here would
    require every caller to pre-check.

    Parameters
    ----------
    image : ndarray
        The image to be checked.

    """
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError(f"Image must be of integer type, got {image.dtype}.")

    unique_values = np.unique(image)
    if unique_values.size and int(unique_values[0]) < 0:
        # Negative ids are not labels, and downstream helpers index arrays by
        # label value (``remove_touching_objects``) or call ``np.bincount``
        # (``remove_large_objects``), both of which fail obscurely on them.
        raise ValueError(
            f"Label image must not contain negative values; got "
            f"{int(unique_values[0])}."
        )
    if len(unique_values) == 2:
        warnings.warn(
            "Only one label was provided in the image. Make sure to label components first."
        )


# skimage 0.26 deprecated `min_size` / `area_threshold` on remove_small_objects
# / remove_small_holes in favor of `max_size`. cucim 26.02 still uses the old
# names. Detect once at import which kwarg the local skimage accepts; the
# helpers below dispatch per device.
_SKIMAGE_USES_MAX_SIZE: bool = (
    "max_size" in inspect.signature(_sk_morphology.remove_small_objects).parameters
)


def _remove_small_objects(label_img: np.ndarray, min_size: int) -> np.ndarray:
    """Remove objects smaller than ``min_size`` across skimage/cucim API drift.

    skimage 0.26 ``max_size=N`` removes objects of size ≤ N; old ``min_size=N``
    removed size < N. Pass ``max_size=min_size - 1`` to preserve the old
    "keep size ≥ min_size" semantics. cucim still uses ``min_size``.
    """
    if get_device(label_img) == "GPU":
        if _cu_morphology is None:
            raise ImportError(
                "cucim is required to process GPU arrays but is not installed; "
                "install cubic with the GPU extras or move the input to CPU."
            )
        return _cu_morphology.remove_small_objects(label_img, min_size=min_size)
    if _SKIMAGE_USES_MAX_SIZE:
        return _sk_morphology.remove_small_objects(label_img, max_size=min_size - 1)  # type: ignore[call-arg]
    return _sk_morphology.remove_small_objects(label_img, min_size=min_size)


def _remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    """Fill holes smaller than ``area_threshold`` across skimage/cucim API drift.

    skimage 0.26 ``max_size=N`` fills holes of area ≤ N; old ``area_threshold=N``
    filled area < N. Pass ``max_size=area_threshold - 1`` to preserve the old
    semantics, exactly as ``_remove_small_objects`` does — without the shift the
    CPU and GPU branches disagree on holes of area exactly ``area_threshold``.
    cucim still uses ``area_threshold``.
    """
    if get_device(mask) == "GPU":
        if _cu_morphology is None:
            raise ImportError(
                "cucim is required to process GPU arrays but is not installed; "
                "install cubic with the GPU extras or move the input to CPU."
            )
        return _cu_morphology.remove_small_holes(mask, area_threshold=area_threshold)
    if _SKIMAGE_USES_MAX_SIZE:
        return _sk_morphology.remove_small_holes(mask, max_size=area_threshold - 1)  # type: ignore[call-arg]
    return _sk_morphology.remove_small_holes(mask, area_threshold=area_threshold)


def cleanup_segmentation(
    label_img: np.ndarray,
    min_obj_size: int | None = None,
    max_obj_size: int | None = None,
    border_buffer_size: int | None = None,
    max_hole_size: int | None = None,
) -> np.ndarray:
    """Clean up segmented image by removing small objects, clearing borders, and closing holes."""
    check_labeled_binary(label_img)

    # the min/max-size filters and border clearing preserve label ids
    if min_obj_size is not None:
        label_img = _remove_small_objects(label_img, min_size=min_obj_size)

    if max_obj_size is not None:
        label_img = remove_large_objects(label_img, max_size=max_obj_size)

    if border_buffer_size is not None:
        label_img = clear_xy_borders(label_img, buffer_size=border_buffer_size)

    # per-label hole filling runs on the full-size mask, not a bbox crop: a crop
    # would break up the exterior background, whose area then falls below
    # ``max_hole_size`` for small objects and gets filled in as label
    if max_hole_size is not None:
        label_img = label_img.copy()
        for label_id in _label_ids(label_img):
            mask = label_img == label_id
            filled_mask = _remove_small_holes(mask, area_threshold=max_hole_size)
            label_img[filled_mask] = label_id

    label_img = label(label_img)
    # uint16 keeps the historical dtype; widen when the label count would wrap
    n_labels = int(label_img.max())
    dtype = (
        np.uint16
        if n_labels <= np.iinfo(np.uint16).max
        else np.min_scalar_type(n_labels)
    )
    return label_img.astype(dtype)


def _label_ids(label_image: np.ndarray) -> np.ndarray:
    """Return the sorted non-zero label ids of ``label_image``.

    ``np.unique(...)[1:]`` is wrong when the image has no background pixel at
    all — index 0 is then a real label and gets silently skipped.
    """
    ids = np.unique(label_image)
    return ids[ids != 0]


def find_objects(label_image, max_label=None):
    """Find the bounding-box slices of every object in a labeled nD array.

    Delegates to :mod:`scipy.ndimage` / :mod:`cupyx.scipy.ndimage` through the
    :mod:`cubic.scipy` proxy, which is device-agnostic and orders of magnitude
    faster than a per-label scan of the whole image.

    Parameters
    ----------
    label_image : np.ndarray
        nD array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `label_image`. If max_label is not
        specified, the positions of all objects up to the highest label are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. Labels absent from the image
        get ``None`` instead of a tuple. The label `l` corresponds to the index
        `l-1` in the returned list.

    """
    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError(
            f"label_image must be of integer type, got {label_image.dtype}."
        )
    if max_label is not None and max_label < 0:
        raise ValueError(f"max_label must be >= 0, got {max_label}.")
    # scipy overloads 0 as "all labels", but here it means "search up to label
    # 0", i.e. none. Answer that directly rather than letting it be silently
    # reinterpreted as the opposite.
    if max_label == 0:
        return []
    return _ndimage.find_objects(label_image, 0 if max_label is None else max_label)


def _iter_label_boxes(
    label_image: np.ndarray,
) -> Iterator[tuple[int, tuple[slice, ...]]]:
    """Yield ``(label_id, bbox_slices)`` for every label present in the image.

    Shared by the post-processing filters so they crop to each object's bounding
    box instead of scanning the whole image once per label.
    """
    for label_id, slices in enumerate(find_objects(label_image), start=1):
        if slices is not None:
            yield label_id, slices


def remove_large_objects(label_image: np.ndarray, max_size: int = 100000) -> np.ndarray:
    """Remove objects with volume above specified threshold.

    The input is left untouched; the filtered labels are returned as a copy.
    """
    check_labeled_binary(label_image)
    label_image = label_image.copy()
    label_volumes = np.bincount(label_image.ravel())
    too_large = label_volumes > max_size
    too_large_mask = too_large[label_image]
    label_image[too_large_mask] = 0
    return label_image


def remove_small_objects(label_image: np.ndarray, min_size: int = 500) -> np.ndarray:
    """Remove objects with volume below specified threshold."""
    check_labeled_binary(label_image)
    return _remove_small_objects(label_image, min_size=min_size)


def clear_xy_borders(label_image: np.ndarray, buffer_size: int = 0) -> np.ndarray:
    """Remove masks that touch XY borders.

    Label ids are preserved (border clearing only zeroes pixels, so no relabel
    is needed) for both 2D and 3D inputs.
    """
    check_labeled_binary(label_image)
    if label_image.ndim == 2:
        return clear_border(label_image, buffer_size=buffer_size)
    # pad Z so the top/bottom planes are not treated as borders
    label_image = pad_image(label_image, buffer_size + 1, axes=0, mode="constant")
    label_image = clear_border(label_image, buffer_size=buffer_size)
    return label_image[buffer_size + 1 : -(buffer_size + 1), :, :]


def remove_touching_objects(
    label_image: np.ndarray, border_value: int = 100
) -> np.ndarray:
    """Find labelled masks that touch each other and remove them.

    Two labels count as touching when they are within a ``(3,) * ndim``
    neighbourhood of each other (8-connectivity in 2D, 26-connectivity in 3D).
    All adjacent pairs are derived in a single pass from shifted comparisons of
    the label image, instead of dilating each object over the whole volume; the
    input is left untouched and the filtered labels are returned as a copy.

    ``border_value`` is accepted for backwards compatibility but no longer
    used: neighbouring labels are read directly from the label image. The
    previous additive-offset trick misclassified any label id greater than
    ``border_value`` (Cellpose routinely emits hundreds of labels), which
    silently deleted non-touching objects.
    """
    check_labeled_binary(label_image)
    label_image = label_image.copy()

    max_label = int(label_image.max())
    if max_label == 0:
        return label_image

    touching = np.zeros(max_label + 1, dtype=bool)
    touching = to_same_device(touching, label_image)  # type: ignore[arg-type]
    shape = label_image.shape
    # half of the (3,) * ndim offsets suffices: offset d and -d yield the same
    # pairs, and both members of every pair are flagged
    for offset in itertools.product((-1, 0, 1), repeat=label_image.ndim):
        if offset <= (0,) * label_image.ndim:
            continue
        here = tuple(
            slice(max(o, 0), s + min(o, 0)) for o, s in zip(offset, shape, strict=True)
        )
        there = tuple(
            slice(max(-o, 0), s + min(-o, 0))
            for o, s in zip(offset, shape, strict=True)
        )
        a = label_image[here]
        b = label_image[there]
        adjacent = (a != b) & (a != 0) & (b != 0)
        touching[a[adjacent]] = True
        touching[b[adjacent]] = True

    label_image[touching[label_image]] = 0
    return label_image


def remove_thin_objects(label_image: np.ndarray, min_z: int = 2) -> np.ndarray:
    """Remove objects thinner than a specified minimum value in Z.

    Objects spanning fewer than ``min_z`` Z planes are removed; one spanning
    exactly ``min_z`` planes is kept. The input is left untouched; the filtered
    labels are returned as a copy.
    """
    if label_image.ndim != 3:
        raise ValueError(
            f"remove_thin_objects operates on Z extents and needs a 3D (ZYX) "
            f"label image, got {label_image.ndim}D."
        )
    label_image = label_image.copy()

    for label_id, slices in _iter_label_boxes(label_image):
        if slices[0].stop - slices[0].start < min_z:
            region = label_image[slices]
            region[region == label_id] = 0

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
            # ``dilation`` replaces the deprecated ``binary_dilation`` (removed in
            # skimage 0.28); identical here because ``ball(1)`` is symmetric
            seed_mask = morphology.dilation(
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
        labels = watershed(-distance, markers=asnumpy(markers), mask=asnumpy(mask))
        return to_device(labels, device)

    # Marker-based watershed without mask (image as landscape and mask)
    img_cpu = asnumpy(image)
    labels = watershed(img_cpu, markers=asnumpy(markers), mask=img_cpu)
    return to_device(labels, device)


def _binary_fill_holes(image, **kwargs):
    """Fill holes in binary objects."""
    # the device-conditional import must stay local: cupyx is a GPU-only dep
    if get_device(image) == "GPU":
        from cupyx.scipy.ndimage import binary_fill_holes
    else:
        from scipy.ndimage import binary_fill_holes

    return binary_fill_holes(image, **kwargs)


def fill_label_holes(lbl_img, **binary_fill_holes_kwargs):
    """Fill small holes in label image.

    Extra keyword arguments (e.g. ``structure``) are forwarded to
    ``scipy.ndimage.binary_fill_holes`` / its cupyx counterpart.

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

    Runs on CPU or GPU arrays; the input is left untouched and the filled labels
    are returned as a copy. Like :func:`cleanup_segmentation`'s hole filling,
    each label is processed on the full-size mask rather than a bbox crop, which
    would break up the exterior background and fill it in as label.

    Inspired by: https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/morphology/fill_holes.py
    """
    img = image.copy()
    axes = range(img.ndim) if axes is None else axes

    for label_id in _label_ids(img):
        binary = img == label_id

        for _ in range(num_iterations):
            for axis in axes:
                slicers = [slice(None)] * img.ndim
                for i in range(binary.shape[axis]):
                    slicers[axis] = slice(i, i + 1)
                    binary_slice = binary[tuple(slicers)]
                    filled_slice = _remove_small_holes(
                        binary_slice, area_threshold=area_threshold
                    )
                    if filled_slice.shape != binary_slice.shape:
                        raise ValueError(
                            "Shape mismatch between filled_slice and binary_slice"
                        )
                    binary[tuple(slicers)] = filled_slice

        img[binary] = label_id

    return img
