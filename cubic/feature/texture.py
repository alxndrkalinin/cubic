"""Device-agnostic gray-level co-occurrence matrix (GLCM) texture features.

cuCIM has no ``graycomatrix`` (rapidsai/cucim#530), and scikit-image's is
2D-only (``check_nD(image, 2)``). This module provides Haralick texture
features for both 2D ``(H, W)`` and 3D ``(D, H, W)`` images that run on
NumPy (CPU) or CuPy (GPU) arrays through duck typing.

The co-occurrence matrix is accumulated with ``np.bincount`` (which works on
both NumPy and CuPy arrays), and the Haralick properties are computed with
closed-form reductions that reproduce ``skimage.feature.graycoprops``
exactly, including its ``correlation`` guard (set to ``1.0`` when a marginal
standard deviation is below ``1e-15`` rather than yielding NaN).

Quantization is performed over a per-image ``value_range`` by default, which
makes the features scale-invariant: ``glcm_features(x) == glcm_features(2x + 5)``
because an affine intensity change leaves the relative bin assignments
unchanged. Pass an explicit ``value_range`` to quantize several regions over a
shared set of levels (e.g. cells of one already-normalized image).
"""

import itertools

import numpy as np

from ..cuda import asnumpy

# Haralick properties emitted by :func:`glcm_features`, in a fixed order.
_PROPS: tuple[str, ...] = (
    "ASM",
    "energy",
    "entropy",
    "contrast",
    "correlation",
    "homogeneity",
    "dissimilarity",
)


def _unit_offsets(ndim: int) -> list[tuple[int, ...]]:
    """Return the unique half-space neighbor offsets for ``ndim`` dimensions.

    Each offset and its negation produce the same symmetric co-occurrence
    matrix, so only one representative per antipodal pair is kept: the one
    whose first non-zero coordinate is positive. This yields the 4 unique
    2D directions (skimage angles 0/45/90/135 deg) and the 13 unique 3D
    directions of the 26-neighborhood.
    """
    offsets = []
    for off in itertools.product((-1, 0, 1), repeat=ndim):
        first_nonzero = next((o for o in off if o != 0), 0)
        if first_nonzero > 0:
            offsets.append(off)
    return offsets


def _slices_for_offset(
    off: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Return (center, neighbor) slice tuples for the overlapping pair region.

    Matches scikit-image's GLCM valid-region convention: for a positive
    offset the center is ``[0:n-o]`` and the neighbor ``[o:n]``; for a
    negative offset the roles are mirrored.
    """
    center, neighbor = [], []
    for o, n in zip(off, shape):
        if o >= 0:
            center.append(slice(0, n - o))
            neighbor.append(slice(o, n))
        else:
            center.append(slice(-o, n))
            neighbor.append(slice(0, n + o))
    return tuple(center), tuple(neighbor)


def _quantize(image: np.ndarray, levels: int, lo: float, hi: float) -> np.ndarray:
    """Quantize ``image`` into ``[0, levels - 1]`` over ``[lo, hi]``.

    Stays on the input array's device (NumPy or CuPy). A constant range
    (``hi <= lo``) maps every voxel to bin 0.
    """
    span = hi - lo
    if span <= 0:
        return (image * 0).astype(np.int64)
    scaled = (image - lo) / span * levels
    return np.clip(np.floor(scaled), 0, levels - 1).astype(np.int64)


def _direction_matrix(
    quant: np.ndarray,
    off: tuple[int, ...],
    levels: int,
    mask: np.ndarray | None,
    symmetric: bool,
    normed: bool,
) -> np.ndarray:
    """Accumulate the GLCM for a single offset and return it as a NumPy array.

    Pairs are counted with ``np.bincount`` on the device, then the small
    ``(levels, levels)`` matrix is moved to the host for the property
    reductions.
    """
    center_sl, neighbor_sl = _slices_for_offset(off, quant.shape)
    center = quant[center_sl]
    neighbor = quant[neighbor_sl]
    if mask is not None:
        valid = mask[center_sl] & mask[neighbor_sl]
        center = center[valid]
        neighbor = neighbor[valid]
    index = (center * levels + neighbor).ravel()
    if index.size == 0:
        # No valid pairs for this direction: an empty overlap (e.g. a
        # single-row image's vertical offsets) or a fully masked-out
        # direction. ``cupy.bincount`` raises on empty input (it reduces
        # ``x.max()`` to size the output, which has no identity), so build
        # the zero matrix directly. The resulting all-zero GLCM yields the
        # same properties skimage gives for an empty co-occurrence matrix
        # (contrast/entropy 0, correlation 1.0 via the std guard).
        counts = np.zeros((levels, levels), dtype=np.float64)
    else:
        counts = (
            asnumpy(np.bincount(index, minlength=levels * levels))
            .reshape(levels, levels)
            .astype(np.float64)
        )
    if symmetric:
        counts = counts + counts.T
    if normed:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


def _haralick_props(glcm: np.ndarray, levels: int) -> dict[str, float]:
    """Compute Haralick properties of a normalized GLCM (host NumPy array).

    Reproduces ``skimage.feature.graycoprops`` exactly for ``contrast``,
    ``dissimilarity``, ``homogeneity``, ``ASM``, ``energy``, ``correlation``
    (including the near-zero-std guard), plus ``entropy`` (natural log).
    """
    levels_idx = np.arange(levels)
    i = levels_idx[:, None]
    j = levels_idx[None, :]
    diff = i - j

    asm = float(np.sum(glcm**2))
    nonzero = glcm > 0
    entropy = float(-np.sum(glcm[nonzero] * np.log(glcm[nonzero])))

    mean_i = float(np.sum(i * glcm))
    mean_j = float(np.sum(j * glcm))
    diff_i = i - mean_i
    diff_j = j - mean_j
    std_i = float(np.sqrt(np.sum(glcm * diff_i**2)))
    std_j = float(np.sqrt(np.sum(glcm * diff_j**2)))
    cov = float(np.sum(glcm * diff_i * diff_j))
    if std_i < 1e-15 or std_j < 1e-15:
        correlation = 1.0
    else:
        correlation = cov / (std_i * std_j)

    return {
        "ASM": asm,
        "energy": float(np.sqrt(asm)),
        "entropy": entropy,
        "contrast": float(np.sum(glcm * diff**2)),
        "correlation": correlation,
        "homogeneity": float(np.sum(glcm / (1.0 + diff**2))),
        "dissimilarity": float(np.sum(glcm * np.abs(diff))),
    }


def glcm_features(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    levels: int = 32,
    distances: tuple[int, ...] = (1,),
    symmetric: bool = True,
    normed: bool = True,
    value_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Extract Haralick GLCM texture features, device-agnostic for 2D and 3D.

    The image is quantized into ``levels`` gray levels over ``value_range``,
    a co-occurrence matrix is accumulated for every half-space direction
    (4 in 2D, 13 in 3D) and distance, the Haralick properties are computed
    per direction, and the results are averaged over all directions for
    rotation invariance.

    Parameters
    ----------
    image : np.ndarray
        2D ``(H, W)`` or 3D ``(D, H, W)`` image (NumPy or CuPy array).
    mask : np.ndarray, optional
        Boolean foreground mask of the same shape as ``image``. When given,
        a pair contributes to the GLCM only if both voxels are inside the
        mask, so background-to-foreground pairs are excluded.
    levels : int, default 32
        Number of gray levels to quantize into. Must be >= 2.
    distances : tuple of int, default (1,)
        Pixel-pair offset distances. Each unit direction is scaled by the
        distance (integer offsets); for ``distance == 1`` the directions
        match scikit-image's GLCM offsets exactly.
    symmetric : bool, default True
        If True, accumulate ``(i, j)`` and ``(j, i)`` together so each
        direction's matrix is symmetric.
    normed : bool, default True
        If True, normalize each direction's matrix to sum to 1. Keep True
        (the default) for properties that match ``graycoprops``.
    value_range : tuple of float, optional
        ``(low, high)`` intensity range used for quantization. Defaults to
        the per-image (or per-mask) ``(min, max)``, which makes the features
        scale-invariant. Pass a shared range to quantize several regions
        over the same levels.

    Returns
    -------
    dict of str to float
        ``contrast``, ``dissimilarity``, ``homogeneity``, ``ASM``,
        ``energy``, ``correlation`` and ``entropy``, each averaged over all
        directions.

    Raises
    ------
    ValueError
        If ``image`` is not 2D or 3D, ``mask`` shape mismatches, ``levels``
        is below 2, or the masked region is empty when ``value_range`` is
        derived from the image.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D (H, W) or 3D (D, H, W) images are supported; got ndim={image.ndim}"
        )
    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"mask shape {mask.shape} must match image shape {image.shape}"
        )
    if levels < 2:
        raise ValueError(f"levels must be >= 2; got {levels}")

    if value_range is None:
        region = image[mask] if mask is not None else image
        if region.size == 0:
            raise ValueError("Cannot derive value_range from an empty masked region.")
        lo = float(region.min())
        hi = float(region.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])

    quant = _quantize(image, levels, lo, hi)
    offsets = [
        tuple(distance * axis for axis in unit)
        for distance in distances
        for unit in _unit_offsets(image.ndim)
    ]

    totals = {prop: 0.0 for prop in _PROPS}
    for off in offsets:
        glcm = _direction_matrix(quant, off, levels, mask, symmetric, normed)
        props = _haralick_props(glcm, levels)
        for prop in _PROPS:
            totals[prop] += props[prop]

    n = len(offsets)
    return {prop: totals[prop] / n for prop in _PROPS}
