"""Utility functions for 3D image deconvolution."""

from typing import Any
from functools import partial
from collections.abc import Callable

import numpy as np

from cubic.cuda import (
    asnumpy,
    to_same_device,
    check_same_device,
)
from cubic.skimage import util, restoration  # noqa: F401
from cubic.image_utils import pad_image

from .richardson_lucy_xp import richardson_lucy_xp


def richardson_lucy_skimage(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution using cubic.skimage."""
    rl_partial = partial(
        restoration.richardson_lucy,
        psf=psf,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    if observer_fn is None:
        return rl_partial(image, num_iter=n_iter)

    estimate = image
    for i in range(1, n_iter + 1):
        estimate = rl_partial(estimate, num_iter=1)
        observer_fn(estimate, i)

    return estimate


def decon_skimage(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 1,
    pad_psf: bool = False,
    pad_size_z: int = 0,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
) -> np.ndarray:
    """Perform scikit-image deconvolution with image padding."""
    check_same_device(image, psf)

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf

    wrapper_observer = None
    if observer_fn is not None:

        def wrapper_observer(restored_image, i, *args):
            processed_image = restored_image[
                pad_size_z : image.shape[0] + pad_size_z, :, :
            ]
            observer_fn(processed_image, i, *args)

    decon_image = richardson_lucy_skimage(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        observer_fn=wrapper_observer,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    decon_image = decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
    return decon_image


def decon_xpy(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 1,
    pad_psf: bool = False,
    pad_size_z: int = 0,
    *,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
    observer_fn: Callable | None = None,
) -> np.ndarray:
    """Perform NumPy-based deconvolution with optional non-circulant edges."""
    check_same_device(image, psf)

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf

    wrapper_observer = None
    if observer_fn is not None:

        def wrapper_observer(restored_image, i, *args):
            processed_image = restored_image[
                pad_size_z : image.shape[0] + pad_size_z, :, :
            ]
            observer_fn(processed_image, i, *args)

    decon_image = richardson_lucy_xp(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        noncirc=noncirc,
        mask=mask,
        observer_fn=wrapper_observer,
    )

    decon_image = decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
    return decon_image


def richardson_lucy_iter(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    implementation: str = "xp",
    pad_psf: bool = False,
    pad_size_z: int = 0,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Unified Richardson-Lucy deconvolution function with iteration observer function.

    Parameters
    ----------
    image : np.ndarray
        Input image to deconvolve.
    psf : np.ndarray
        Point spread function.
    n_iter : int, default=10
        Number of iterations.
    implementation : str, default="xp"
        Implementation to use: "skimage" or "xp".
    pad_psf : bool, default=False
        Whether to pad the PSF along with the image.
    pad_size_z : int, default=0
        Number of slices to pad in z-dimension.
    observer_fn : Callable | None, default=None
        Function to call after each iteration.
    clip : bool, default=True
        Whether to clip values (skimage only).
    filter_epsilon : float | None, default=None
        Filter epsilon parameter (skimage only).
    noncirc : bool, default=False
        Enable non-circulant edge handling (xp only).
    mask : np.ndarray | None, default=None
        Mask array (xp only).

    Returns
    -------
    np.ndarray
        Deconvolved image.

    Raises
    ------
    ValueError
        If implementation is not "skimage" or "xp".

    """
    if implementation == "skimage":
        return decon_skimage(
            image=image,
            psf=psf,
            n_iter=n_iter,
            pad_psf=pad_psf,
            pad_size_z=pad_size_z,
            observer_fn=observer_fn,
            clip=clip,
            filter_epsilon=filter_epsilon,
        )
    elif implementation == "xp":
        return decon_xpy(
            image=image,
            psf=psf,
            n_iter=n_iter,
            pad_psf=pad_psf,
            pad_size_z=pad_size_z,
            noncirc=noncirc,
            mask=mask,
            observer_fn=observer_fn,
        )
    else:
        raise ValueError(
            f"Unknown implementation: {implementation}. Use 'skimage' or 'xp'."
        )


def deconv_iter_num_finder(
    image: np.ndarray,
    psf: np.ndarray,
    metric_fn: Callable,
    metric_threshold: int | float,
    metric_kwargs: dict[str, Any] | None = None,
    max_iter: int = 25,
    pad_size_z: int = 1,
    verbose: bool = False,
    implementation: str = "xpy",
    noncirc: bool = False,
) -> tuple[int, list[dict[str, int | float | np.ndarray]]]:
    """Find number of LR deconvolution iterations using an image similarity metric.

    Parameters
    ----------
    implementation : str, optional
        Which LR implementation to use, ``"skimage"`` (default) or ``"xpy"``.
    noncirc : bool, optional
        When ``implementation='xpy'``, enable non-circulant edge handling.

    """
    verboseprint = print if verbose else lambda *a, **k: None

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if metric_kwargs is None:
        metric_kwargs = {}

    thresh_iter = 0
    results = [{"metric_gain": metric_threshold, "iter_image": asnumpy(image)}]

    def get_decon_observer(metric_fn, metric_kwargs):
        nonlocal thresh_iter

        def decon_observer(restored_image, i, *args):
            nonlocal thresh_iter

            if thresh_iter > 0:
                return

            prev_iter_image = to_same_device(results[-1]["iter_image"], restored_image)
            metric_result = metric_fn(prev_iter_image, restored_image, **metric_kwargs)

            metric_gain = (
                metric_result[0] if isinstance(metric_result, tuple) else metric_result
            )

            results.append(
                {
                    "metric_gain": float(metric_gain),
                    "iter_image": asnumpy(restored_image),
                    "metric_result": metric_result,
                }
            )
            verboseprint(f"Iteration {i}: improvement {metric_gain:.8f}")

            if (i > 1) and (metric_gain > metric_threshold):
                thresh_iter = i
                metric_gain_total = metric_fn(
                    to_same_device(results[0]["iter_image"], restored_image),
                    restored_image,
                    **metric_kwargs,
                )
                if isinstance(metric_gain_total, tuple):
                    metric_gain_total = metric_gain_total[0]

                verboseprint(
                    f"\nThreshold {metric_threshold} reached at iteration {i}"
                    f" with improvement: {metric_gain:.8f}.\n"
                    f"Metric between original and restored images: {metric_gain_total:.8f}.\n"
                )

        return decon_observer

    observer = get_decon_observer(metric_fn=metric_fn, metric_kwargs=metric_kwargs)
    if implementation == "skimage":
        _ = decon_skimage(
            image,
            psf,
            n_iter=max_iter,
            observer_fn=observer,
            pad_size_z=pad_size_z,
        )
    elif implementation == "xpy":
        _ = decon_xpy(
            image,
            psf,
            n_iter=max_iter,
            noncirc=noncirc,
            observer_fn=observer,
            pad_size_z=pad_size_z,
        )
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    for i, result in enumerate(results):
        results[i]["iter_image"] = asnumpy(result["iter_image"])

    return (thresh_iter, results)
