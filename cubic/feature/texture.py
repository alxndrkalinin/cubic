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

from ..cuda import asnumpy, get_array_module, check_same_device


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


def _quantize(
    image: np.ndarray,
    levels: int,
    lo: float | np.ndarray,
    hi: float | np.ndarray,
) -> np.ndarray:
    """Quantize ``image`` into ``[0, levels - 1]`` over ``[lo, hi]``.

    Stays on the input array's device (NumPy or CuPy). ``lo`` / ``hi`` are
    scalars or per-voxel arrays. A constant range (``hi <= lo``) maps its
    voxels to bin 0.
    """
    span = hi - lo
    if np.ndim(span) == 0:
        if span <= 0:
            return np.zeros_like(image, dtype=np.int64)
        scaled = (image - lo) / span * levels
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(span > 0, (image - lo) / span * levels, 0.0)
    return np.clip(np.floor(scaled), 0, levels - 1).astype(np.int64)


def _validate_glcm_args(
    image: np.ndarray, levels: int, distances: tuple[int, ...]
) -> None:
    """Raise unless ``image`` is 2D/3D, ``levels >= 2`` and ``distances`` are positive."""
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D (H, W) or 3D (D, H, W) images are supported; got ndim={image.ndim}"
        )
    if levels < 2:
        raise ValueError(f"levels must be >= 2; got {levels}")
    if not distances or any(distance < 1 for distance in distances):
        raise ValueError(
            f"distances must be a non-empty sequence of positive integers; got {distances}"
        )


def _glcm_offsets(ndim: int, distances: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Every half-space unit direction scaled by every distance."""
    return [
        tuple(distance * axis for axis in unit)
        for distance in distances
        for unit in _unit_offsets(ndim)
    ]


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
    _validate_glcm_args(image, levels, distances)
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

    if value_range is None:
        region = image[mask] if mask is not None else image
        if region.size == 0:
            raise ValueError("Cannot derive value_range from an empty masked region.")
        lo = float(region.min())
        hi = float(region.max())
    else:
        lo, hi = float(value_range[0]), float(value_range[1])

    quant = _quantize(image, levels, lo, hi)
    offsets = _glcm_offsets(image.ndim, distances)

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


# Property names in the order :func:`_haralick_props` returns them.
_PROP_NAMES = (
    "ASM",
    "energy",
    "entropy",
    "contrast",
    "correlation",
    "homogeneity",
    "dissimilarity",
)


def _haralick_props_batched(glcm: np.ndarray) -> dict[str, np.ndarray]:
    """:func:`_haralick_props` over the leading axes of ``(..., levels, levels)``.

    Same formulas and the same ``correlation`` guard. Entropy sums
    ``g * log(g)`` with zeros masked in place rather than compacted, so the
    values agree with the per-matrix form to rounding (~1e-16), not bitwise.
    """
    levels = glcm.shape[-1]
    levels_idx = np.arange(levels, dtype=np.float64)
    i = levels_idx[:, None]
    j = levels_idx[None, :]
    diff = i - j
    diff_sq = diff * diff
    axes = (-2, -1)
    asm = np.sum(glcm**2, axis=axes)
    nonzero = glcm > 0
    log_glcm = np.log(np.where(nonzero, glcm, 1.0))
    entropy = -np.sum(np.where(nonzero, glcm * log_glcm, 0.0), axis=axes)
    mean_i = np.sum(i * glcm, axis=axes)[..., None, None]
    mean_j = np.sum(j * glcm, axis=axes)[..., None, None]
    diff_i = i - mean_i
    diff_j = j - mean_j
    std_i = np.sqrt(np.sum(glcm * diff_i**2, axis=axes))
    std_j = np.sqrt(np.sum(glcm * diff_j**2, axis=axes))
    cov = np.sum(glcm * diff_i * diff_j, axis=axes)
    flat = (std_i < 1e-15) | (std_j < 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(flat, 1.0, cov / (std_i * std_j))
    return {
        "ASM": asm,
        "energy": np.sqrt(asm),
        "entropy": entropy,
        "contrast": np.sum(glcm * diff_sq, axis=axes),
        "correlation": correlation,
        "homogeneity": np.sum(glcm / (1.0 + diff_sq), axis=axes),
        "dissimilarity": np.sum(glcm * np.abs(diff), axis=axes),
    }


def glcm_features_by_label(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    levels: int = 32,
    distances: tuple[int, ...] = (1,),
    symmetric: bool = True,
    normed: bool = True,
    value_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """:func:`glcm_features` for every labeled region of one image, in one pass.

    Equivalent to calling :func:`glcm_features` on each region with
    ``mask=labels == label``: a voxel pair counts towards a region when both
    voxels carry its label, and each region averages its properties over its
    own non-empty directions. Instead of one call per region, every direction
    accumulates the co-occurrences of all regions in a single ``bincount``,
    so the image is traversed once per direction for each bounded chunk of
    labels (one chunk for a typical FOV), not once per region. Per-region
    ranges quantize in float64, while :func:`glcm_features` quantizes a
    float32 image in float32, so a voxel on a level boundary can land one level
    apart there.

    Parameters
    ----------
    image : np.ndarray
        2D ``(H, W)`` or 3D ``(D, H, W)`` image (NumPy or CuPy array).
    labels : np.ndarray
        Integer label image of the same shape; ``0`` is background.
    levels, distances, symmetric, normed
        As in :func:`glcm_features`.
    value_range : tuple of float, optional
        ``(low, high)`` quantization range shared by all regions. Defaults to
        each region's own ``(min, max)``, as :func:`glcm_features` with a mask.

    Returns
    -------
    dict of str to np.ndarray
        ``label`` (the sorted non-zero labels present) and one host float64
        array per property, aligned with ``label``. A region without a voxel
        pair in any direction, where :func:`glcm_features` raises, is NaN.

    Raises
    ------
    ValueError
        If ``image`` is not 2D or 3D, ``labels`` mismatches its shape or is not
        an integer array, ``levels`` is below 2, or ``distances`` is empty or
        contains a non-positive value.
    """
    _validate_glcm_args(image, levels, distances)
    check_same_device(image, labels)
    if labels.shape != image.shape:
        raise ValueError(
            f"labels shape {labels.shape} must match image shape {image.shape}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be an integer array; got dtype {labels.dtype}")

    present = np.unique(labels)
    present = present[present != 0]
    label_ids = asnumpy(present)
    n_labels = int(label_ids.size)
    if n_labels == 0:
        return {"label": label_ids, **{p: np.empty(0) for p in _PROP_NAMES}}
    # Compact labels 1..n_labels in ascending label order; 0 stays background.
    # A sorted search keeps memory proportional to the image, whatever the
    # label values (sparse or negative ids included).
    xp = get_array_module(labels)
    compact = np.searchsorted(present, labels) + 1
    compact[labels == 0] = 0

    if value_range is None:
        # Per-region (min, max) by scatter, in float64 like glcm_features' Python
        # floats: integer ranges cannot overflow, every dtype is one CuPy
        # ``ufunc.at`` supports, and a NaN only poisons its own region. Slot 0
        # collects the background and is never read for a labeled voxel.
        values = image.astype(np.float64, copy=False)
        lo = xp.full(n_labels + 1, np.inf)
        hi = xp.full(n_labels + 1, -np.inf)
        np.minimum.at(lo, compact, values)
        np.maximum.at(hi, compact, values)
        quant = _quantize(values, levels, lo[compact], hi[compact])
        del values, lo, hi
    else:
        quant = _quantize(image, levels, float(value_range[0]), float(value_range[1]))

    offsets = _glcm_offsets(image.ndim, distances)
    n_dir = len(offsets)
    n_pairs = levels * levels
    columns: dict[str, list[np.ndarray]] = {name: [] for name in _PROP_NAMES}
    # Labels go in chunks whose (direction, label, i, j) histogram stays under
    # _LABEL_CHUNK_BINS, on the device and in the float64 property stage on the
    # host; each chunk scans the image once per direction. A typical FOV
    # (hundreds of cells, 3D, 32 levels) is one chunk.
    step = max(1, _LABEL_CHUNK_BINS // (n_dir * n_pairs))
    for start in range(0, n_labels, step):
        stop = min(start + step, n_labels)
        n_chunk_bins = (stop - start) * n_pairs
        # Each direction counts into its own slice of the chunk's histogram.
        counts = xp.zeros((n_dir, n_chunk_bins), dtype=np.int64)
        for direction, off in enumerate(offsets):
            center_sl, neighbor_sl = _slices_for_offset(off, image.shape)
            lab_c = compact[center_sl]
            # Compact labels start at 1. Compacting to the in-region pairs first
            # measured faster than binning every pair with a sentinel bin
            # (0.131 vs 0.174 s on 13 cells of 48x640x960, A40): most voxels
            # are background.
            same = (lab_c > start) & (lab_c <= stop) & (lab_c == compact[neighbor_sl])
            bins = (lab_c[same] - 1 - start) * n_pairs
            if bins.size == 0:  # cupy.bincount rejects empty input
                continue
            bins += quant[center_sl][same] * levels + quant[neighbor_sl][same]
            counts[direction] += np.bincount(bins, minlength=n_chunk_bins)
        host = asnumpy(counts).reshape(n_dir, stop - start, levels, levels)
        del counts
        chunk = host.transpose(1, 0, 2, 3)
        for name, values in _direction_mean_props(chunk, symmetric, normed).items():
            columns[name].append(values)
    out: dict[str, np.ndarray] = {"label": label_ids}
    for name in _PROP_NAMES:
        out[name] = np.concatenate(columns[name])
    return out


# Histogram bins per label chunk of glcm_features_by_label: 64 MB as int64 on the
# device, and ~10 float64 temporaries of that size in the property stage.
_LABEL_CHUNK_BINS = 1 << 23


def _direction_mean_props(
    counts: np.ndarray, symmetric: bool, normed: bool
) -> dict[str, np.ndarray]:
    """Haralick properties of ``(labels, directions, L, L)`` co-occurrence counts.

    Averaged over each label's non-empty directions; NaN for a label with none.
    """
    matrices = counts.astype(np.float64)
    nonempty = matrices.sum(axis=(2, 3)) > 0
    if symmetric:
        matrices = matrices + matrices.transpose(0, 1, 3, 2)
    if normed:
        totals = matrices.sum(axis=(2, 3), keepdims=True)
        matrices = np.divide(matrices, totals, out=matrices, where=totals > 0)
    props = _haralick_props_batched(matrices)
    n_nonempty = nonempty.sum(axis=1)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in _PROP_NAMES:
            total = np.where(nonempty, props[name], 0.0).sum(axis=1)
            out[name] = np.where(n_nonempty > 0, total / n_nonempty, np.nan)
    return out
