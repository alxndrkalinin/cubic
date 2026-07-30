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
from cubic.skimage import util, restoration
from cubic.image_utils import pad_image

from .richardson_lucy_xp import richardson_lucy_xp

_XP_ALIASES = ("xp", "xpy")
_BACKPROJECTOR_SKIMAGE_ERROR = (
    "backprojector is only supported with the NumPy/CuPy implementation "
    "('xp', alias 'xpy'), not 'skimage'."
)


def _normalize_implementation(implementation: str) -> str:
    """Map an implementation name onto ``"skimage"`` or ``"xp"``.

    ``"xp"`` and ``"xpy"`` are interchangeable: the entry points in this module
    historically spelled the NumPy/CuPy backend differently, so both are
    accepted everywhere.
    """
    if implementation == "skimage":
        return "skimage"
    if implementation in _XP_ALIASES:
        return "xp"
    raise ValueError(
        f"Unknown implementation: {implementation}. Use 'skimage', 'xp' or 'xpy'."
    )


def _unpad_z(image: np.ndarray, pad_size_z: int, orig_size_z: int) -> np.ndarray:
    """Undo the z-padding that :func:`pad_image` added along axis 0."""
    return image[pad_size_z : orig_size_z + pad_size_z]


def _make_unpad_observer(
    observer_fn: Callable | None, pad_size_z: int, orig_size_z: int
) -> Callable | None:
    """Wrap ``observer_fn`` so it observes z-unpadded estimates."""
    if observer_fn is None:
        return None

    def wrapper_observer(restored_image, i, *args):
        observer_fn(_unpad_z(restored_image, pad_size_z, orig_size_z), i, *args)

    return wrapper_observer


def richardson_lucy_skimage(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution using cubic.skimage.

    With an ``observer_fn`` the deconvolution is recomputed from the original
    image at every depth, so each observed estimate is the true ``i``-iteration
    RL result and the returned value matches the no-observer path. Looping with
    ``num_iter=1`` on the previous output instead both re-clips every iteration
    (skimage clips per call) and feeds the deconvolution back as the image,
    diverging from a single call.

    The cost of that guarantee is quadratic: observing ``n_iter`` iterations
    runs ``n_iter * (n_iter + 1) / 2`` RL iterations in total (325 for
    ``n_iter=25``). Callers that pay this indirectly -- notably
    :func:`deconv_iter_num_finder` with ``implementation="skimage"`` -- should
    keep ``n_iter``/``max_iter`` modest.
    """
    rl_partial = partial(
        restoration.richardson_lucy,
        psf=psf,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    # With nothing to observe (n_iter < 1) defer to the single call, whose result
    # is skimage's constant initial estimate rather than ``image`` itself.
    if observer_fn is None or n_iter < 1:
        return rl_partial(image, num_iter=n_iter)

    for i in range(1, n_iter + 1):
        estimate = rl_partial(image, num_iter=i)
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

    decon_image = richardson_lucy_skimage(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        observer_fn=_make_unpad_observer(observer_fn, pad_size_z, image.shape[0]),
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    return _unpad_z(decon_image, pad_size_z, image.shape[0])


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
    backprojector: np.ndarray | None = None,
) -> np.ndarray:
    """Perform NumPy-based deconvolution with optional non-circulant edges.

    When ``backprojector`` is given, an unmatched Richardson-Lucy update is
    used (see :func:`richardson_lucy_xp`). The back projector is not padded; it
    must stay PSF-shaped, so ``pad_psf`` must be ``False`` in that case.

    ``mask`` is z-padded exactly like the image (``mode="reflect"``), since
    :func:`richardson_lucy_xp` requires a mask of the image's shape. Reflecting
    it keeps the mirrored border valid; zero-padding it instead would drive the
    padded region to zero and place a hard edge at the boundary of the returned
    crop, defeating the purpose of ``pad_size_z``.
    """
    arrays = (image, psf) if backprojector is None else (image, psf, backprojector)
    check_same_device(*arrays)
    if backprojector is not None and pad_psf:
        raise ValueError(
            "pad_psf=True is not supported with a backprojector; the back "
            "projector must stay PSF-shaped (pass pad_psf=False)."
        )

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf
    padded_mask = (
        pad_image(mask, pad_size_z, mode="reflect") if mask is not None else None
    )

    decon_image = richardson_lucy_xp(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        noncirc=noncirc,
        mask=padded_mask,
        observer_fn=_make_unpad_observer(observer_fn, pad_size_z, image.shape[0]),
        backprojector=backprojector,
    )

    return _unpad_z(decon_image, pad_size_z, image.shape[0])


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
    backprojector: np.ndarray | None = None,
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
        Implementation to use: ``"skimage"`` or ``"xp"`` (alias ``"xpy"``).
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
    backprojector : np.ndarray | None, default=None
        Optional unmatched back projector PSF (xp only). See
        :func:`richardson_lucy_xp`.

    Returns
    -------
    np.ndarray
        Deconvolved image.

    Raises
    ------
    ValueError
        If implementation is not "skimage", "xp" or "xpy", or if
        ``backprojector`` is given with ``implementation='skimage'``.

    """
    if _normalize_implementation(implementation) == "skimage":
        if backprojector is not None:
            raise ValueError(_BACKPROJECTOR_SKIMAGE_ERROR)
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
    return decon_xpy(
        image=image,
        psf=psf,
        n_iter=n_iter,
        pad_psf=pad_psf,
        pad_size_z=pad_size_z,
        noncirc=noncirc,
        mask=mask,
        observer_fn=observer_fn,
        backprojector=backprojector,
    )


class _MetricThresholdReached(Exception):
    """Internal signal that the metric threshold was hit; aborts the RL loop."""


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
    backprojector: np.ndarray | None = None,
) -> tuple[int, list[dict[str, int | float | np.ndarray]]]:
    """Find number of LR deconvolution iterations using an image similarity metric.

    The deconvolution runs with an observer that compares each iteration against
    the previous one. The first time the gain exceeds ``metric_threshold`` (from
    iteration 2 onwards) the deconvolution is aborted, so no iterations past the
    threshold are computed.

    Parameters
    ----------
    image : np.ndarray
        Input image to deconvolve.
    psf : np.ndarray
        Point spread function.
    metric_fn : Callable
        Called as ``metric_fn(previous, current, **metric_kwargs)``. It may
        return a scalar or a tuple whose first element is the scalar gain.
    metric_threshold : int | float
        Consecutive-iteration gain above which the search stops.
    metric_kwargs : dict[str, Any] | None, default=None
        Extra keyword arguments forwarded to ``metric_fn``.
    max_iter : int, default=25
        Maximum number of deconvolution iterations to try.
    pad_size_z : int, default=1
        Number of slices to reflect-pad in z before deconvolving.
    verbose : bool, default=False
        Print the per-iteration gain and the threshold-crossing summary.
    implementation : str, default="xpy"
        Which LR implementation to use: ``"xpy"`` (alias ``"xp"``) or
        ``"skimage"``. The ``"skimage"`` path recomputes from scratch at every
        depth, so it costs ``max_iter * (max_iter + 1) / 2`` RL iterations (see
        :func:`richardson_lucy_skimage`).
    noncirc : bool, default=False
        With the NumPy/CuPy implementation, enable non-circulant edge handling.
    backprojector : np.ndarray | None, default=None
        Optional unmatched back projector PSF (NumPy/CuPy implementation only).
        See :func:`richardson_lucy_xp`.

    Returns
    -------
    tuple[int, list[dict[str, int | float | np.ndarray]]]
        ``(thresh_iter, results)``. ``thresh_iter`` is the 1-based iteration at
        which the gain first exceeded ``metric_threshold``, or ``0`` if it never
        did. ``results[k]`` describes iteration ``k``, with every image on the
        host (NumPy):

        - ``"metric_gain"``: ``float`` gain against iteration ``k - 1``.
        - ``"iter_image"``: the z-unpadded estimate after iteration ``k``.
        - ``"metric_result"``: the raw ``metric_fn`` return value (scalar or
          tuple).

        Entry ``0`` holds the input image: it has no ``"metric_result"`` key and
        carries ``metric_gain=metric_threshold`` as a sentinel.

    Raises
    ------
    ValueError
        If ``implementation`` is unknown, or ``backprojector`` is given with
        ``implementation='skimage'``.

    """
    impl = _normalize_implementation(implementation)
    if backprojector is not None and impl == "skimage":
        raise ValueError(_BACKPROJECTOR_SKIMAGE_ERROR)
    verboseprint = print if verbose else lambda *a, **k: None

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if metric_kwargs is None:
        metric_kwargs = {}

    thresh_iter = 0
    results: list[dict[str, int | float | np.ndarray]] = [
        {"metric_gain": metric_threshold, "iter_image": asnumpy(image)}
    ]

    def get_decon_observer(metric_fn, metric_kwargs):
        def decon_observer(restored_image, i, *args):
            nonlocal thresh_iter

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
                # Stop the deconvolution itself: every further iteration would be
                # computed, observed and then discarded.
                raise _MetricThresholdReached

        return decon_observer

    observer = get_decon_observer(metric_fn=metric_fn, metric_kwargs=metric_kwargs)
    try:
        if impl == "skimage":
            decon_skimage(
                image,
                psf,
                n_iter=max_iter,
                observer_fn=observer,
                pad_size_z=pad_size_z,
            )
        else:
            decon_xpy(
                image,
                psf,
                n_iter=max_iter,
                noncirc=noncirc,
                observer_fn=observer,
                pad_size_z=pad_size_z,
                backprojector=backprojector,
            )
    except _MetricThresholdReached:
        pass

    return (thresh_iter, results)
