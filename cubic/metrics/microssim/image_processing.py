"""Image normalization helpers for MicroSSIM (device-agnostic).

Ported from juglab/microssim@8bccb17d (``image_processing/linearize.py`` and
``image_processing/micro_ssim_normalization.py``). All operations rely on
NumPy's array API and therefore work transparently with NumPy or CuPy arrays
through duck typing — no ``cubic.cuda.ascupy`` conversions are performed, so
the device of the input arrays is preserved.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray


def linearize_list(images: list[NDArray] | NDArray) -> NDArray:
    """Flatten and concatenate a list of arrays into a single 1-D array.

    If ``images`` is already an array, it is returned unchanged. Ragged
    shapes are supported because each element is independently raveled
    before concatenation.

    Parameters
    ----------
    images : list of ndarray or ndarray
        Input image(s). May be a list of arrays with possibly different
        shapes, or a single ndarray (in which case the input is returned
        unchanged).

    Returns
    -------
    ndarray
        Concatenation of ``np.ravel(x)`` for every ``x`` in the input
        list, or ``images`` itself when ``images`` is not a list.

    Notes
    -----
    Matches upstream ``microssim.image_processing.linearize.linearize_list``.
    """
    if not isinstance(images, list):
        return images
    return np.concatenate([np.ravel(x) for x in images])


def compute_norm_parameters(
    gt: list[NDArray] | NDArray,
    pred: list[NDArray] | NDArray,
    bg_percentile: float = 3,
    offset_gt: float | None = None,
    offset_pred: float | None = None,
    max_val: float | None = None,
) -> tuple[float, float, float]:
    """Compute normalization parameters for MicroSSIM.

    Any parameter that is not ``None`` is returned unchanged. Missing
    offsets are estimated as the ``bg_percentile``-th percentile of the
    corresponding (flattened) image data; the missing maximum is the
    maximum of the ground-truth data after subtracting ``offset_gt``.

    Parameters
    ----------
    gt : ndarray or list of ndarray
        Reference image(s).
    pred : ndarray or list of ndarray
        Image(s) being compared to the reference.
    bg_percentile : float, default=3
        Percentile of the image intensities considered as background.
    offset_gt : float or None, default=None
        Pre-computed background estimate for ``gt``. If ``None`` it is
        derived from the data.
    offset_pred : float or None, default=None
        Pre-computed background estimate for ``pred``. If ``None`` it is
        derived from the data.
    max_val : float or None, default=None
        Pre-computed maximum used in normalization. If ``None`` it is
        computed as ``(gt_linearized - offset_gt).max()``.

    Returns
    -------
    tuple of float
        ``(offset_gt, offset_pred, max_val)``.

    Notes
    -----
    Matches upstream
    ``microssim.image_processing.micro_ssim_normalization.compute_norm_parameters``.
    ``np.percentile`` operates on both NumPy and CuPy arrays via duck typing,
    so this function is device-agnostic.
    """
    gt_lin = linearize_list(gt)
    pred_lin = linearize_list(pred)
    if offset_gt is None:
        offset_gt = float(np.percentile(gt_lin, bg_percentile))
    if offset_pred is None:
        offset_pred = float(np.percentile(pred_lin, bg_percentile))
    if max_val is None:
        max_val = float((gt_lin - offset_gt).max())
    return offset_gt, offset_pred, max_val


def normalize_min_max(
    images: list[NDArray] | NDArray,
    offset: float,
    max_val: float,
) -> list[NDArray] | NDArray:
    """Normalize images by ``(x - offset) / max_val``.

    The function is list-recursive: when ``images`` is a list, the
    transformation is applied to each element and the result is returned
    as a list.

    Parameters
    ----------
    images : ndarray or list of ndarray
        Image or list of images to normalize.
    offset : float
        Value subtracted from every pixel before scaling.
    max_val : float
        Divisor applied to the offset-subtracted images. Note that this
        is *not* ``max_val - offset`` — the divisor matches upstream
        microssim's behavior exactly.

    Returns
    -------
    ndarray or list of ndarray
        Normalized image(s). The container type mirrors the input.

    Notes
    -----
    Matches upstream
    ``microssim.image_processing.micro_ssim_normalization.normalize_min_max``.
    """
    if isinstance(images, list):
        return [cast(NDArray, normalize_min_max(x, offset, max_val)) for x in images]
    return (images - offset) / max_val
