"""Implements GPU-compatible metrics from scikit-image."""

from typing import Any
from functools import wraps
from collections.abc import Callable

import numpy as np

from ..cuda import check_same_device
from ..skimage import metrics, morphology


def scale_invariant(fn: Callable) -> Callable:
    """Decorate a function to make it scale invariant."""

    @wraps(fn)
    def wrapped(
        image_true: np.ndarray,
        image_test: np.ndarray,
        *args: Any,
        scale_invariant: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Transform input images to be scale invariant."""
        check_same_device(image_true, image_test)
        if not scale_invariant:
            return fn(image_true, image_test, *args, **kwargs)

        mask = kwargs.get("mask", None)

        if mask is None:
            gt_zero = image_true - image_true.mean()
            gt_norm = gt_zero / image_true.std()

            pred_zero = image_test - image_test.mean()
            alpha = (gt_norm * pred_zero).sum() / (pred_zero * pred_zero).sum()

            pred_scaled = pred_zero * alpha
            range_param = (image_true.max() - image_true.min()) / image_true.std()
        else:
            m = mask.astype(bool)
            gt_mean = image_true[m].mean()
            gt_std = image_true[m].std()
            gt_zero = image_true - gt_mean
            gt_norm = gt_zero / gt_std

            pred_mean = image_test[m].mean()
            pred_zero = image_test - pred_mean
            alpha = (gt_norm[m] * pred_zero[m]).sum() / (
                pred_zero[m] * pred_zero[m]
            ).sum()

            pred_scaled = pred_zero * alpha
            range_param = (image_true[m].max() - image_true[m].min()) / gt_std

        return fn(gt_norm, pred_scaled, *args, **{**kwargs, "data_range": range_param})

    return wrapped


@scale_invariant
def nrmse(
    image_true: np.ndarray,
    image_test: np.ndarray,
    normalization: str | None = None,
    data_range: float | None = None,
    mask: np.ndarray | None = None,
):
    """Compute the normalized root mean squared error (NRMSE) between two images."""
    x = image_true[mask] if mask is not None else image_true
    y = image_test[mask] if mask is not None else image_test

    if data_range is not None:
        mse = metrics.mean_squared_error(x, y)
        return (mse**0.5) / data_range
    elif normalization is not None:
        return metrics.normalized_root_mse(x, y, normalization=normalization)
    else:
        return metrics.normalized_root_mse(x, y)


@scale_invariant
def psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: int | None = None,
    mask: np.ndarray | None = None,
):
    """Compute the peak signal to noise ratio (PSNR) between two images."""
    x = image_true[mask] if mask is not None else image_true
    y = image_test[mask] if mask is not None else image_test
    return metrics.peak_signal_noise_ratio(x, y, data_range=data_range)


@scale_invariant
def ssim(
    im1: np.ndarray,
    im2: np.ndarray,
    win_size: int | None = None,
    gradient: bool | None = False,
    data_range: float | None = None,
    channel_axis: int | None = None,
    gaussian_weights: bool | None = False,
    full: bool | None = False,
    mask: np.ndarray | None = None,
    **kwargs,
):
    """Compute the mean structural similarity index between two images."""
    if mask is None:
        return metrics.structural_similarity(
            im1,
            im2,
            win_size=win_size,
            gradient=gradient,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            full=full,
            **kwargs,
        )
    else:
        # Compute SSIM map on the full image, then average over valid centers
        # whose SSIM window fits entirely inside the foreground mask.
        _, ssim_map = metrics.structural_similarity(
            im1,
            im2,
            win_size=win_size,
            gradient=gradient,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            full=True,
            **kwargs,
        )

        effective_win = win_size if win_size is not None else 7
        r = effective_win // 2
        valid = morphology.erosion(mask.astype(bool), morphology.square(2 * r + 1))
        mssim_masked = float(np.mean(ssim_map[valid]))

        if full:
            return mssim_masked, ssim_map
        else:
            return mssim_masked
