"""Implements device-agnostic average precision metric for segmentation.

Modified from StarDist/Cellpose with added support for CUDA GPUs by Alexandr Kalinin.

Copyright (c) 2024 Alexandr Kalinin unless stated otherwise.

For functions from Cellpose/Stardist, the original copyright is retained.
Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L45
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
https://github.com/MouseLand/cellpose/blob/509ffca33737058b0b4e2e96d506514e10620eb3/cellpose/metrics.py
"""

import logging
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..cuda import ascupy, asnumpy, get_device

logger = logging.getLogger(__name__)

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


def _check_sequential_labels(mask: np.ndarray) -> bool:
    labels = np.unique(mask)
    return bool((labels[0] == 0) and np.all(np.diff(labels) == 1))


def _label_overlap_gpu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Measure label overlap on GPU using CuPy.

    Copyright (c) 2024 Alexandr Kalinin
    """
    from cupyx import jit

    x = x.ravel().astype(np.uint32)
    y = y.ravel().astype(np.uint32)
    overlap = np.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint32)
    overlap = ascupy(overlap)  # type: ignore[assignment]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="cupyx.jit._interface"
        )

        @jit.rawkernel()
        def label_overlap_kernel(x, y, overlap, N):
            idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
            if idx < N:
                jit.atomic_add(overlap, (x[idx], y[idx]), 1)

    N = np.uint32(x.size)
    threads_per_block = 128
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    label_overlap_kernel[blocks_per_grid, threads_per_block](x, y, overlap, N)
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

    if device_x == "GPU":
        return _label_overlap_gpu(x, y)
    else:
        return _label_overlap_cpu(x, y)


def _intersection_over_union(
    masks_true: np.ndarray, masks_pred: np.ndarray
) -> np.ndarray:
    """Calculate the intersection over union of all mask pairs, device agnostic.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L168

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L65
    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = overlap.sum(axis=0, keepdims=True)
    n_pixels_true = overlap.sum(axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    return iou


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
    """
    if not _check_sequential_labels(mask_true):
        raise ValueError("mask_true should have sequential labels.")
    if not _check_sequential_labels(mask_pred):
        raise ValueError("mask_pred should have sequential labels.")
    iou = _intersection_over_union(mask_true, mask_pred)[1:, 1:]
    iou = np.nan_to_num(asnumpy(iou))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate average precision and other metrics for a single pair of mask images with pre-computed matches.

    Modified from: Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    https://github.com/MouseLand/cellpose/blob/0ce365352c9d43ce7a15ebff6955f24f2035a303/cellpose/metrics.py#L82

    , which was modified from: Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
    https://github.com/stardist/stardist/blob/586f8ca76d063bf3443f7a9a66fe94658bc155b8/stardist/matching.py#L109
    """
    if matches_per_threshold is None:
        matches_per_threshold = compute_matches(masks_true, masks_pred, thresholds)  # type: ignore[assignment]

    if matches_per_threshold is None:
        raise ValueError("No matches found.")
    tp = np.asarray([len(matches_per_threshold[th][0]) for th in thresholds])
    fp = asnumpy(masks_pred.max()) - tp
    fn = asnumpy(masks_true.max()) - tp

    sum_counts = tp + fp + fn
    ap = np.where(sum_counts > 0, tp / sum_counts, 0)

    return ap, tp, fp, fn
