"""Implements GPU-compatible metrics from scikit-image."""

from typing import Any
from functools import wraps
from collections.abc import Callable

import numpy as np

from ..cuda import asnumpy, _is_torch_tensor, check_same_device
from ..skimage import metrics, morphology


def _canonicalize_torch(*arrays: Any) -> tuple[Any, ...]:
    """Materialize torch.Tensor inputs as host NumPy arrays.

    Non-torch inputs (NumPy/CuPy) are passed through unchanged so GPU
    acceleration via CuPy/cuCIM is preserved. Torch CUDA tensors are
    moved to host because the downstream skimage primitives do not
    accept torch tensors directly.
    """
    return tuple(asnumpy(a) if _is_torch_tensor(a) else a for a in arrays)


def _min_max_to_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize ``x`` to [0, 1] (per-input, device-agnostic).

    Each input is independently rescaled to [0, 1] using its own min/max.
    The output dtype is float64 so downstream MSE/PSNR math is precise.
    """
    x = x.astype(np.float64)
    lo = x.min()
    rng = x.max() - lo
    # Plain Python max() works on numpy/cupy 0-d scalars via __float__.
    rng = rng if float(rng) > eps else type(rng)(eps)
    return (x - lo) / rng


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
        image_true, image_test = _canonicalize_torch(image_true, image_test)
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
    normalize: str | None = None,
    data_range: float | None = None,
    mask: np.ndarray | None = None,
):
    """Compute the normalized root mean squared error (NRMSE) between two images.

    Parameters
    ----------
    image_true, image_test : np.ndarray
        Images to compare. Must have the same shape.
    normalization : str, optional
        Forwarded to skimage's ``normalized_root_mse`` (``"euclidean"``,
        ``"min-max"``, or ``"mean"``).
    normalize : str, optional
        Per-input pre-normalization applied before NRMSE. Currently
        ``"min_max"`` is supported, which independently rescales each
        input to [0, 1] using its own min/max. When set, ``data_range``
        defaults to 1.0.
    data_range : float, optional
        Explicit dynamic range used to scale the RMSE.
    mask : np.ndarray, optional
        Boolean mask restricting the comparison region.
    """
    if normalize is not None:
        if normalize != "min_max":
            raise ValueError(
                f"normalize={normalize!r} not supported; use 'min_max' or None"
            )
        image_true = _min_max_to_unit(image_true)
        image_test = _min_max_to_unit(image_test)
        if data_range is None:
            data_range = 1.0

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
    data_range: float | None = None,
    normalize: str | None = None,
    mask: np.ndarray | None = None,
):
    """Compute the peak signal to noise ratio (PSNR) between two images.

    Parameters
    ----------
    image_true, image_test : np.ndarray
        Images to compare. Must have the same shape.
    data_range : float, optional
        Dynamic range of the input. Forwarded to skimage. When
        ``normalize="min_max"``, defaults to 1.0.
    normalize : str, optional
        Per-input pre-normalization applied before PSNR. Currently
        ``"min_max"`` is supported (each input independently rescaled
        to [0, 1] using its own min/max).
    mask : np.ndarray, optional
        Boolean mask restricting the comparison region.
    """
    if normalize is not None:
        if normalize != "min_max":
            raise ValueError(
                f"normalize={normalize!r} not supported; use 'min_max' or None"
            )
        image_true = _min_max_to_unit(image_true)
        image_test = _min_max_to_unit(image_test)
        if data_range is None:
            data_range = 1.0

    x = image_true[mask] if mask is not None else image_true
    y = image_test[mask] if mask is not None else image_test
    return metrics.peak_signal_noise_ratio(x, y, data_range=data_range)


def _ssim_single(
    im1: np.ndarray,
    im2: np.ndarray,
    *,
    win_size: int | None,
    gradient: bool | None,
    data_range: float | None,
    channel_axis: int | None,
    gaussian_weights: bool | None,
    full: bool | None,
    **kwargs: Any,
) -> Any:
    """Compute a single 2-D / 3-D SSIM (no batched dispatch)."""
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
    spatial_dims: int | None = None,
    **kwargs,
):
    """Compute the mean structural similarity index between two images.

    Parameters
    ----------
    im1, im2 : np.ndarray
        Images to compare. Must have the same shape.
    win_size, gradient, data_range, channel_axis, gaussian_weights, full
        Forwarded to ``skimage.metrics.structural_similarity``.
    mask : np.ndarray, optional
        Boolean foreground mask. Only valid for 2-D and 3-D inputs; the
        returned mean averages SSIM over voxels whose window fits
        entirely inside the mask.
    spatial_dims : int, optional
        When set, enables batched dispatch over inputs of shape
        ``[N, C, (D,) H, W]``. ``spatial_dims=2`` expects a 4-D input,
        ``spatial_dims=3`` expects a 5-D input. The mean SSIM is
        averaged across the ``N*C`` slabs. Required by callers that
        want the same call signature as ``torch_ssim`` / torchmetrics.
        ``mask`` is not supported in the batched path.
    """
    if spatial_dims is not None:
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3; got {spatial_dims}")
        expected_ndim = spatial_dims + 2
        if im1.ndim != expected_ndim:
            raise ValueError(
                f"spatial_dims={spatial_dims} expects ndim={expected_ndim}; "
                f"got ndim={im1.ndim}"
            )
        if im1.shape != im2.shape:
            raise ValueError(f"Shape mismatch: im1 {im1.shape} vs im2 {im2.shape}")
        if mask is not None:
            raise ValueError(
                "mask is not supported with spatial_dims (batched dispatch)"
            )
        n_batch, n_channel = im1.shape[:2]
        accum = 0.0
        for n in range(n_batch):
            for c in range(n_channel):
                accum += float(
                    _ssim_single(
                        im1[n, c],
                        im2[n, c],
                        win_size=win_size,
                        gradient=gradient,
                        data_range=data_range,
                        channel_axis=channel_axis,
                        gaussian_weights=gaussian_weights,
                        full=full,
                        **kwargs,
                    )
                )
        return accum / (n_batch * n_channel)

    if mask is None:
        return _ssim_single(
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
        if mask.ndim == 2:
            footprint = morphology.square(2 * r + 1)
        elif mask.ndim == 3:
            footprint = morphology.cube(2 * r + 1)
        else:
            raise ValueError(f"Unsupported mask dimensions: {mask.ndim}")
        valid = morphology.erosion(mask.astype(bool), footprint)
        mssim_masked = float(np.mean(ssim_map[valid]))

        if full:
            return mssim_masked, ssim_map
        else:
            return mssim_masked
