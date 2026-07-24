"""Implements GPU-compatible metrics from scikit-image."""

from typing import Any
from functools import wraps
from collections.abc import Callable

import numpy as np

from ..cuda import (
    CUDAManager,
    asnumpy,
    to_same_device,
    _is_torch_tensor,
    check_same_device,
)
from ..skimage import metrics, morphology

#: Denominators below this magnitude are treated as zero (matches ``pcc``).
_ZERO_TOL = 1e-12

#: skimage hard-codes ``truncate = 3.5`` on the Gaussian path (identical in
#: 0.22 through 0.26), so its window depends only on ``sigma``.
_GAUSSIAN_TRUNCATE = 3.5
_DEFAULT_SIGMA = 1.5
#: skimage's ``win_size`` default for ``gaussian_weights=False``.
_DEFAULT_WIN_UNIFORM = 7


def _gaussian_win_size(sigma: float) -> int:
    """Return the ``win_size`` skimage derives from *sigma*.

    Mirrors ``_structural_similarity``: ``2 * int(truncate * sigma + 0.5) + 1``.
    Hard-coding the ``sigma=1.5`` result (11) silently under-erodes whenever a
    caller passes a larger ``sigma`` — at ``sigma=3.0`` skimage's window is 23,
    so out-of-mask pixels leak back into the "masked" mean.
    """
    return 2 * int(_GAUSSIAN_TRUNCATE * float(sigma) + 0.5) + 1


def _canonicalize_torch(*arrays: Any) -> tuple[Any, ...]:
    """Convert torch.Tensor inputs to the appropriate array type.

    Non-torch inputs (NumPy/CuPy) are passed through unchanged so GPU
    acceleration via CuPy/cuCIM is preserved.

    For torch tensors:
    - CUDA tensor + CuPy available → ``cupy.asarray`` zero-copy via
      ``__cuda_array_interface__``. Computation stays on GPU.
    - CPU tensor or no CuPy → ``asnumpy`` (host NumPy).
    """
    cp = CUDAManager().get_cp()
    result = []
    for a in arrays:
        if _is_torch_tensor(a):
            if cp is not None and a.device.type == "cuda":  # type: ignore[attr-defined]
                result.append(cp.asarray(a))
            else:
                result.append(asnumpy(a))
        else:
            result.append(a)
    return tuple(result)


def _min_max_to_unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize ``x`` to [0, 1] (per-input, device-agnostic).

    Each input is independently rescaled to [0, 1] using its own min/max.
    The output dtype is float64 so downstream MSE/PSNR math is precise.
    """
    x = x.astype(np.float64)
    lo = x.min()
    # Plain Python max() works on numpy/cupy 0-d scalars via __float__.
    # Rebuilding the scalar instead (``type(rng)(eps)``) breaks on CuPy, whose
    # ndarray constructor takes a *shape* as its first argument.
    rng = max(float(x.max() - lo), eps)
    return (x - lo) / rng


def _nonzero_or_raise(value: Any, what: str) -> float:
    """Return ``float(value)`` or raise if it is (numerically) zero.

    Guards the divisions in the scale-invariant path so a constant image —
    or a mask selecting a constant region — raises instead of propagating
    ``inf``/``nan`` into the metric.
    """
    scalar = float(value)
    if abs(scalar) < _ZERO_TOL:
        raise ValueError(
            f"scale_invariant=True requires non-zero {what}; got {scalar!r}. "
            "The input (or the masked region) is constant."
        )
    return scalar


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

        # ``normalization`` (``nrmse`` only) picks skimage's RMSE denominator,
        # so it conflicts with the ``data_range`` injected at the end of this
        # branch. Reject it here to give a message that names both options.
        if kwargs.get("normalization") is not None:
            raise ValueError(
                "scale_invariant=True is incompatible with normalization="
                f"{kwargs['normalization']!r}: the scale-invariant path derives "
                "its own data_range. Pass exactly one of the two."
            )

        # ``mask`` is keyword-only in every decorated function, so a caller
        # cannot hide it in *args and bypass the masked branch below.
        mask = kwargs.get("mask", None)

        if mask is None:
            gt_std = _nonzero_or_raise(image_true.std(), "image_true std")
            gt_zero = image_true - image_true.mean()
            gt_norm = gt_zero / gt_std

            pred_zero = image_test - image_test.mean()
            pred_energy = _nonzero_or_raise(
                (pred_zero * pred_zero).sum(), "image_test variance"
            )
            alpha = (gt_norm * pred_zero).sum() / pred_energy

            pred_scaled = pred_zero * alpha
            range_param = (image_true.max() - image_true.min()) / gt_std
        else:
            m = mask.astype(bool)
            gt_mean = image_true[m].mean()
            gt_std = _nonzero_or_raise(image_true[m].std(), "masked image_true std")
            gt_zero = image_true - gt_mean
            gt_norm = gt_zero / gt_std

            pred_mean = image_test[m].mean()
            pred_zero = image_test - pred_mean
            pred_energy = _nonzero_or_raise(
                (pred_zero[m] * pred_zero[m]).sum(), "masked image_test variance"
            )
            alpha = (gt_norm[m] * pred_zero[m]).sum() / pred_energy

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
    *,
    mask: np.ndarray | None = None,
):
    """Compute the normalized root mean squared error (NRMSE) between two images.

    Parameters
    ----------
    image_true, image_test : np.ndarray
        Images to compare. Must have the same shape.
    normalization : str, optional
        Forwarded to skimage's ``normalized_root_mse`` (``"euclidean"``,
        ``"min-max"``, or ``"mean"``). Mutually exclusive with both
        ``data_range`` and ``normalize``: all three choose the RMSE
        denominator, so combining them raises rather than silently
        preferring one.
    normalize : str, optional
        Per-input pre-normalization applied before NRMSE. Currently
        ``"min_max"`` is supported, which independently rescales each
        input to [0, 1] using its own min/max. When set, ``data_range``
        defaults to 1.0 — and therefore excludes ``normalization``.
    data_range : float, optional
        Explicit dynamic range used to scale the RMSE.
    mask : np.ndarray, optional
        Boolean mask restricting the comparison region. Keyword-only.

    Notes
    -----
    ``scale_invariant=True`` (added by the decorator) supplies its own
    ``data_range``, so it is likewise incompatible with
    ``normalization``; that combination raises in the decorator.
    """
    if normalization is not None:
        # ``normalize`` counts too: it sets ``data_range=1.0`` below, which
        # would then short-circuit past ``normalization`` just as an explicit
        # ``data_range`` does.
        for name, value in (("data_range", data_range), ("normalize", normalize)):
            if value is not None:
                raise ValueError(
                    f"normalization={normalization!r} and {name}={value!r} are "
                    "mutually exclusive: both determine the RMSE denominator. "
                    "Pass exactly one."
                )

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
    *,
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
        Boolean mask restricting the comparison region. Keyword-only.
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
    *,
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
        Boolean foreground mask. Keyword-only. Only valid for 2-D and 3-D
        inputs; the returned mean averages SSIM over voxels whose window
        fits entirely inside the mask.
    spatial_dims : int, optional
        Keyword-only. When set, enables batched dispatch over inputs of
        shape ``[N, C, (D,) H, W]``. ``spatial_dims=2`` expects a 4-D
        input, ``spatial_dims=3`` expects a 5-D input. The mean SSIM is
        averaged across the ``N*C`` slabs. Required by callers that
        want the same call signature as ``torch_ssim`` / torchmetrics.
        ``mask``, ``full``, and ``gradient`` are not supported in the
        batched path.
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
        # Either flag makes ``structural_similarity`` return a tuple, which the
        # per-slab mean below cannot accumulate; there is also no sensible way
        # to average an SSIM map or gradient across slabs.
        for flag, name in ((full, "full"), (gradient, "gradient")):
            if flag:
                raise ValueError(
                    f"{name}=True is not supported with spatial_dims "
                    "(batched dispatch returns only the mean SSIM)"
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
        # The masked branch forces ``full=True`` to get the SSIM map, so a
        # ``gradient=True`` request would make skimage return a 3-tuple into a
        # 2-target unpack. There is also no sensible way to reduce a gradient
        # over the valid-centre mask.
        if gradient:
            raise ValueError(
                "gradient=True is not supported with mask (the masked path "
                "reduces the SSIM map over valid centers only)"
            )
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

        # The erosion footprint must match the window skimage actually used,
        # otherwise out-of-mask pixels leak into the "masked" mean through
        # windows that straddle the mask boundary.
        if win_size is not None:
            effective_win = win_size
        elif gaussian_weights:
            effective_win = _gaussian_win_size(kwargs.get("sigma", _DEFAULT_SIGMA))
        else:
            effective_win = _DEFAULT_WIN_UNIFORM
        if mask.ndim not in (2, 3):
            raise ValueError(f"Unsupported mask dimensions: {mask.ndim}")
        r = effective_win // 2
        # ``footprint_rectangle`` replaces ``square``/``cube``, which are
        # deprecated in scikit-image 0.25 and removed in 0.27. One call covers
        # both the 2-D and 3-D cases and is byte-identical to the pair it
        # replaces for these symmetric footprints.
        footprint = morphology.footprint_rectangle((2 * r + 1,) * mask.ndim)
        # It receives only a shape tuple, so the proxy sees no array argument,
        # cannot detect the device, and returns a NumPy footprint. cuCIM's
        # ``erosion`` rejects a host footprint paired with a GPU mask
        # (``_footprint_is_sequence`` requires a CuPy ndarray), so move the
        # footprint onto the mask's device first.
        footprint = to_same_device(footprint, mask)
        valid = morphology.erosion(mask.astype(bool), footprint)
        mssim_masked = float(np.mean(ssim_map[valid]))

        if full:
            return mssim_masked, ssim_map
        else:
            return mssim_masked
