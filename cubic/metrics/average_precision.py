"""Implements device-agnostic average precision metric for segmentation.

Modified from StarDist/Cellpose with added support for CUDA GPUs by Alexandr Kalinin.

Copyright (c) 2024 Alexandr Kalinin unless stated otherwise.

For functions from Cellpose/Stardist, the original copyright is retained.
Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L45
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
https://github.com/MouseLand/cellpose/blob/509ffca33737058b0b4e2e96d506514e10620eb3/cellpose/metrics.py
"""

import warnings
from typing import Literal, overload
from functools import cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..cuda import asnumpy, get_device, get_array_module

# Hoist numba JIT outside function body so the decorated helper is compiled
# once and cached across calls, not re-wrapped on every invocation.
try:
    from numba import jit as _numba_jit

    @_numba_jit(nopython=True)
    def _numba_label_overlap(
        x: np.ndarray, y: np.ndarray, overlap: np.ndarray
    ) -> np.ndarray:
        for i in range(len(x)):
            overlap[x[i], y[i]] += 1
        return overlap

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _check_has_background(mask: np.ndarray) -> bool:
    """Return True if *mask* contains background (label 0) pixels."""
    return bool((mask == 0).any())


def _check_sequential_labels(mask: np.ndarray) -> bool:
    """Return True if the distinct labels of *mask* are consecutive.

    Paired with :func:`_check_has_background`, which pins the first label
    at 0, this makes the foreground labels run 1..n with no gaps.
    """
    labels = np.unique(mask)
    return bool(np.all(np.diff(labels) == 1))


def _as_label_index(a: np.ndarray) -> np.ndarray:
    """Cast a label image to ``uint32`` indices, rejecting invalid values.

    Both the CPU and the GPU overlap kernels index an accumulator with the
    label values, so the input must be a non-negative integer image. The
    cast is done here rather than per-backend so a float label image
    (common after a lossy I/O round-trip) behaves identically on both
    devices instead of raising only on CPU.
    """
    if not (np.issubdtype(a.dtype, np.integer) or a.dtype == bool):
        if not bool((a == np.floor(a)).all()):
            raise ValueError(
                f"Label image must hold integer values; got dtype {a.dtype} "
                "with non-integral entries."
            )
    if bool((a < 0).any()):
        raise ValueError("Label image must not contain negative labels.")
    # ``copy=False`` is safe: both backends only read the label arrays.
    return a.astype(np.uint32, copy=False)


@cache
def _label_overlap_kernel():
    """Compile the CuPy raw kernel for label overlap once, then reuse it.

    ``jit.rawkernel`` wraps the Python function in a new kernel object on
    every call, which defeats CuPy's per-object compilation cache, so the
    wrapping must not happen inside the hot path.
    """
    from cupyx import jit

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="cupyx.jit._interface"
        )

        @jit.rawkernel()
        def kernel(x, y, overlap, N):
            idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
            if idx < N:
                jit.atomic_add(overlap, (x[idx], y[idx]), 1)

    return kernel


def _label_overlap_gpu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Measure label overlap on GPU using CuPy.

    Copyright (c) 2024 Alexandr Kalinin
    """
    x = x.ravel()
    y = y.ravel()
    # Allocate on device directly: a host allocation plus ``ascupy`` copies a
    # multi-MB accumulator across the bus on every call for no benefit.
    xp = get_array_module(x)
    overlap = xp.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint32)

    N = np.uint32(x.size)
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    _label_overlap_kernel()[blocks_per_grid, threads_per_block](x, y, overlap, N)
    return overlap


def _label_overlap_cpu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Measure label overlap on CPU using either NumPy or Numba if available.

    Modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L45
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint)

    if _HAS_NUMBA:
        return _numba_label_overlap(x, y, overlap)
    np.add.at(overlap, (x, y), 1)
    return overlap


def _label_overlap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Route label overlap calculation based on the device."""
    device_x = get_device(x)
    device_y = get_device(y)

    if device_x != device_y:
        raise ValueError("x and y should be on the same device.")

    if x.shape != y.shape:
        raise ValueError(
            f"x and y should have the same shape. Got {x.shape} and {y.shape} instead."
        )

    x = _as_label_index(x)
    y = _as_label_index(y)

    if device_x == "GPU":
        return _label_overlap_gpu(x, y)
    else:
        return _label_overlap_cpu(x, y)


def _intersection_over_union(
    masks_true: np.ndarray, masks_pred: np.ndarray
) -> np.ndarray:
    """Calculate the intersection over union of all object pairs, device agnostic.

    Returns
    -------
    np.ndarray
        The ``(n_true, n_pred)`` object-to-object matrix. **The background
        row and column are EXCLUDED**: they are consumed by the per-label
        pixel counts and dropped before the division, so a mask with no
        background pixels does not evaluate 0/0 at the background entry.

        Callers must therefore *not* apply ``[1:, 1:]`` themselves. Three
        call sites depend on this: ``compute_matches`` and
        ``average_precision`` in this module, and
        ``cubic.segmentation.cellpose_dynamics._stitch3D``. Re-adding the
        slice — or reverting this function to return the full
        ``(1 + max_true, 1 + max_pred)`` matrix without updating all three
        — drops the first true and first predicted label and makes
        ``_stitch3D`` mis-stitch **silently, with no exception**. Change
        both sides together.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L168

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L65
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = overlap.sum(axis=0, keepdims=True)
    n_pixels_true = overlap.sum(axis=1, keepdims=True)
    objects = overlap[1:, 1:]
    denom = n_pixels_pred[:, 1:] + n_pixels_true[1:, :] - objects
    return objects / denom


def _iou_host(masks_true: np.ndarray, masks_pred: np.ndarray) -> np.ndarray:
    """Object IoU matrix as a host array with NaN (absent labels) → 0."""
    return np.nan_to_num(asnumpy(_intersection_over_union(masks_true, masks_pred)))


def _matches_at_threshold(iou: np.ndarray, th: float) -> tuple[np.ndarray, np.ndarray]:
    """Identify matches based on IoU and threshold.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L201

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L172
    """
    n_min = min(iou.shape[0], iou.shape[1])
    if n_min <= 0:
        raise ValueError("No masks to match")
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return true_ind[match_ok], pred_ind[match_ok]


@overload
def compute_matches(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    thresholds: list[float] | np.ndarray,
    return_iou: Literal[False] = ...,
) -> dict[float, tuple[np.ndarray, np.ndarray]]: ...


@overload
def compute_matches(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    thresholds: list[float] | np.ndarray,
    return_iou: Literal[True],
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], np.ndarray]: ...


def compute_matches(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    thresholds: list[float] | np.ndarray,
    return_iou: bool = False,
) -> (
    dict[float, tuple[np.ndarray, np.ndarray]]
    | tuple[dict[float, tuple[np.ndarray, np.ndarray]], np.ndarray]
):
    """Compute and store IoU and matching indices for various thresholds.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L82

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L109

    Raises
    ------
    ValueError
        If either mask has no background (label 0) pixels, or if its
        foreground labels are not contiguous. Also, via
        ``_matches_at_threshold``, ``"No masks to match"`` when a mask is
        entirely background so there is nothing to assign.
    """
    for name, mask in (("mask_true", mask_true), ("mask_pred", mask_pred)):
        # Reported separately: "sequential labels" is a confusing complaint
        # about a fully-labelled mask whose ids are in fact contiguous.
        if not _check_has_background(mask):
            raise ValueError(
                f"{name} has no background pixels; label 0 must be present."
            )
        if not _check_sequential_labels(mask):
            raise ValueError(f"{name} should have sequential labels.")
    iou = _iou_host(mask_true, mask_pred)
    matches = {}
    for th in thresholds:
        th_matches = _matches_at_threshold(iou, th)
        matches[th] = (th_matches[0] + 1, th_matches[1] + 1)
    return (matches, iou) if return_iou else matches


def average_precision(
    masks_true: np.ndarray,
    masks_pred: np.ndarray,
    thresholds: list[float] | np.ndarray,
    matches_per_threshold: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    return_iou: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Calculate average precision and other metrics for a single pair of mask images with pre-computed matches.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L82

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L109

    Parameters
    ----------
    return_iou : bool, default False
        If True, also return the object-overlap IoU matrix (shape
        ``(n_true, n_pred)``, background dropped, NaN replaced with 0) as a
        fifth element so a caller that also needs a Dice/overlap metric can
        reuse the single overlap pass instead of recomputing it.
    """
    iou: np.ndarray | None = None
    if matches_per_threshold is None:
        # ``compute_matches`` builds the IoU matrix either way, so asking for it
        # is free and its shape gives the object counts: the labels are known to
        # be contiguous from 1 because ``compute_matches`` just verified it.
        matches_per_threshold, iou = compute_matches(
            masks_true, masks_pred, thresholds, return_iou=True
        )

    tp = np.asarray([len(matches_per_threshold[th][0]) for th in thresholds])
    if iou is not None:
        n_true, n_pred = int(iou.shape[0]), int(iou.shape[1])
    else:
        # Caller-supplied matches bypass the label checks in ``compute_matches``,
        # so count distinct foreground labels rather than using ``.max()``, which
        # would over-count across a gap in the label ids and corrupt FP/FN/AP.
        n_true = int(np.count_nonzero(np.unique(masks_true)))
        n_pred = int(np.count_nonzero(np.unique(masks_pred)))
    fp = n_pred - tp
    fn = n_true - tp

    sum_counts = tp + fp + fn
    ap = np.where(sum_counts > 0, tp / sum_counts, 0)

    if return_iou:
        if iou is None:
            # Caller supplied matches_per_threshold, so compute_matches never
            # ran; compute the overlap once here (matching its normalization).
            iou = _iou_host(masks_true, masks_pred)
        return ap, tp, fp, fn, iou
    return ap, tp, fp, fn
