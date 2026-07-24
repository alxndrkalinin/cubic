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
        return np.zeros_like(image, dtype=np.int64)
    scaled = (image - lo) / span * levels
    return np.clip(np.floor(scaled), 0, levels - 1).astype(np.int64)


def _direction_matrices(
    quant: np.ndarray,
    offsets: list[tuple[int, ...]],
    levels: int,
    mask: np.ndarray | None,
    symmetric: bool,
    normed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate one GLCM per offset and return them as host NumPy arrays.

    Every direction's pair indices are shifted by ``direction * levels ** 2``
    into one flat histogram, which is accumulated on the device and moved to the
    host in a single transfer, rather than one transfer (and device
    synchronization) per direction.

    Returns
    -------
    matrices : np.ndarray
        ``(n_offsets, levels, levels)`` co-occurrence matrices.
    nonempty : np.ndarray
        Boolean mask of the directions that contributed at least one pair. A
        direction can be empty because its overlap region is empty (a
        single-slice volume has no z-neighbors, and neither does any direction
        whose distance exceeds the extent) or because the mask excludes every
        pair. Their matrices are all-zero and must be dropped rather than
        averaged in as zeros, which would scale every property down by the
        fraction of non-empty directions.
    """
    n_pairs = levels * levels
    n_bins = len(offsets) * n_pairs
    counts = None
    sizes = []
    for direction, off in enumerate(offsets):
        center_sl, neighbor_sl = _slices_for_offset(off, quant.shape)
        center = quant[center_sl]
        neighbor = quant[neighbor_sl]
        if mask is not None:
            valid = mask[center_sl] & mask[neighbor_sl]
            center = center[valid]
            neighbor = neighbor[valid]
        index = (center * levels + neighbor).ravel()
        sizes.append(index.size)
        # Skip empty directions: ``cupy.bincount`` raises on empty input (it
        # reduces ``x.max()`` to size the output, which has no identity).
        if index.size:
            # One ``bincount`` per direction, summed into the same histogram.
            # Deliberately NOT the alternative of concatenating every direction's
            # indices for a single ``bincount``: that peaks at
            # ``n_directions * volume`` (13x a 3D volume as int64, ~4 GB for a
            # 256^3 volume against 0.55 GB here), which is an OOM risk on a shared
            # GPU, and it measured no faster on large volumes. The cost this fixes
            # was never the launch count but the per-direction host round-trip,
            # which the single transfer after the loop removes.
            index += direction * n_pairs  # in-place: index is our own temporary
            partial = np.bincount(index, minlength=n_bins)
            counts = partial if counts is None else counts + partial

    nonempty = np.asarray(sizes) > 0
    host_counts = np.zeros(n_bins) if counts is None else asnumpy(counts)
    matrices = host_counts.reshape(len(offsets), levels, levels).astype(np.float64)

    if symmetric:
        matrices = matrices + matrices.transpose(0, 2, 1)
    if normed:
        totals = matrices.sum(axis=(1, 2), keepdims=True)
        matrices = np.divide(matrices, totals, out=matrices, where=totals > 0)
    return matrices, nonempty


def _haralick_props(
    glcm: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    diff_sq: np.ndarray,
    abs_diff: np.ndarray,
) -> dict[str, float]:
    """Compute Haralick properties of a normalized GLCM (host NumPy array).

    ``i``/``j`` are the level-index ramps and ``diff_sq``/``abs_diff`` the
    squared and absolute level differences; they are direction-independent,
    so the caller computes them once and reuses them across directions.

    Reproduces ``skimage.feature.graycoprops`` exactly for ``contrast``,
    ``dissimilarity``, ``homogeneity``, ``ASM``, ``energy``, ``correlation``
    (including the near-zero-std guard), plus ``entropy`` (natural log).
    """
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
        "contrast": float(np.sum(glcm * diff_sq)),
        "correlation": correlation,
        "homogeneity": float(np.sum(glcm / (1.0 + diff_sq))),
        "dissimilarity": float(np.sum(glcm * abs_diff)),
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
    per direction, and the results are averaged over all directions that
    contain at least one voxel pair, for rotation invariance.

    Parameters
    ----------
    image : np.ndarray
        2D ``(H, W)`` or 3D ``(D, H, W)`` image (NumPy or CuPy array).
    mask : np.ndarray, optional
        Boolean foreground mask of the same shape as ``image``. When given,
        a pair contributes to the GLCM only if both voxels are inside the
        mask, so background-to-foreground pairs are excluded. Must have
        ``dtype == bool``.
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
        ``energy``, ``correlation`` and ``entropy``, each averaged over the
        directions that contain at least one voxel pair.

    Raises
    ------
    ValueError
        If ``image`` is not 2D or 3D, ``mask`` shape mismatches or is not
        boolean, ``levels`` is below 2, ``distances`` is empty or contains a
        non-positive value, the masked region is empty when ``value_range`` is
        derived from the image, or no direction contains a voxel pair.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D (H, W) or 3D (D, H, W) images are supported; got ndim={image.ndim}"
        )
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match image shape {image.shape}"
            )
        if mask.dtype != bool:
            # An integer mask would silently turn ``mask & mask`` into a bitwise
            # AND and the subsequent ``center[valid]`` into integer fancy
            # indexing, which returns wrong-but-plausible features, not an error.
            raise ValueError(f"mask must be a boolean array; got dtype {mask.dtype}")
    if levels < 2:
        raise ValueError(f"levels must be >= 2; got {levels}")
    if not distances or any(distance < 1 for distance in distances):
        raise ValueError(
            f"distances must be a non-empty sequence of positive integers; got {distances}"
        )

    if value_range is None:
        region = image[mask] if mask is not None else image
        if region.size == 0:
            raise ValueError("Cannot derive value_range from an empty masked region.")
        lo = float(region.min())
        hi = float(region.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])

    quant = _quantize(image, levels, lo, hi)
    unit_offsets = _unit_offsets(image.ndim)
    offsets = [
        tuple(distance * axis for axis in unit)
        for distance in distances
        for unit in unit_offsets
    ]

    # Direction-independent level-index ramps, computed once and reused for
    # every direction's property reduction.
    levels_idx = np.arange(levels)
    i = levels_idx[:, None]
    j = levels_idx[None, :]
    diff = i - j
    diff_sq = diff * diff
    abs_diff = np.abs(diff)

    matrices, nonempty = _direction_matrices(
        quant, offsets, levels, mask, symmetric, normed
    )
    if not nonempty.any():
        raise ValueError(
            "No voxel pair falls inside the image for any direction; the mask or "
            f"the distances {distances} leave every direction of shape "
            f"{image.shape} empty."
        )

    # Directions without a single valid pair have an all-zero matrix; averaging
    # them in would dilute every property towards zero (and flip the sign of
    # correlation) by a factor that depends only on the object's shape.
    per_direction = [
        _haralick_props(matrices[direction], i, j, diff_sq, abs_diff)
        for direction in np.flatnonzero(nonempty)
    ]
    n = len(per_direction)
    return {prop: sum(d[prop] for d in per_direction) / n for prop in per_direction[0]}
