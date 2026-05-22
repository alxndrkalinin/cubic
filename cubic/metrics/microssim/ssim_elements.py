"""Per-pixel SSIM elements (means, variances, covariance) for MicroSSIM.

Ported from juglab/microssim@8bccb17d ``ssim_utils.py`` with device-agnostic
NumPy/CuPy semantics. The dataclass field order matches the upstream
``SSIMElements`` exactly so result tuples remain interchangeable.

Behavior contract
-----------------
- NaN input propagates through the Gaussian / uniform filter convolution;
  callers must scrub NaNs if they are not desired.
- Integer / bool / complex dtypes are promoted to ``float64``;
  ``float16`` is promoted to ``float32``; ``float32`` and ``float64`` are
  preserved. Mirrors upstream's ``_supported_float_type``
  (``ssim_utils.py:131``).
- The variance estimates ``vx``, ``vy`` are NOT clamped to be non-negative
  (matches upstream ``ssim_utils.py:235-237``). Float round-off can
  produce tiny negative values for nearly-constant inputs; this is
  intentional for parity with upstream microssim.
- Input layout is always 2-D spatial: ``(H, W)`` or ``(N, H, W)`` with a
  leading batch axis that the filter does not mix.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import numpy as np

from ...cuda import check_same_device


@dataclass(frozen=True)
class SSIMElements:
    """Per-pixel SSIM elements after local mean / variance estimation.

    Field order matches upstream microssim's ``SSIMElements``
    (``ssim_utils.py:23-42``); construct with keyword arguments only.

    Attributes
    ----------
    ux : numpy.ndarray
        Local mean of ``image1`` after the filter.
    uy : numpy.ndarray
        Local mean of ``image2`` after the filter.
    vxy : numpy.ndarray
        ``cov_norm * (uxy - ux*uy)``. Order matches upstream
        (``vxy`` precedes ``vx`` / ``vy``).
    vx : numpy.ndarray
        ``cov_norm * (uxx - ux*ux)``. NOT clamped to be non-negative;
        matches upstream ``ssim_utils.py:235-237``.
    vy : numpy.ndarray
        ``cov_norm * (uyy - uy*uy)``. NOT clamped to be non-negative;
        matches upstream ``ssim_utils.py:235-237``.
    C1 : float
        ``(K1 * data_range) ** 2``.
    C2 : float
        ``(K2 * data_range) ** 2``.
    """

    ux: np.ndarray
    uy: np.ndarray
    vxy: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    C1: float
    C2: float


def _promoted_float_dtype(dtype: np.dtype) -> np.dtype:
    """Return the float dtype used for SSIM computation.

    Mirrors upstream ``_supported_float_type`` (``ssim_utils.py:131``):
    ``float32`` / ``float64`` are preserved; ``float16`` is promoted to
    ``float32``; integer / bool / complex / anything else is promoted to
    ``float64``.

    Parameters
    ----------
    dtype : numpy.dtype
        Input dtype.

    Returns
    -------
    numpy.dtype
        Output dtype to use for SSIM intermediates.
    """
    if dtype.kind != "f":
        return np.dtype(np.float64)
    if dtype == np.float16:
        return np.dtype(np.float32)
    return dtype


def compute_ssim_elements(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    data_range: float,
    gaussian_weights: bool = False,
    win_size: int | None = None,
    sigma: float = 1.5,
    truncate: float = 3.5,
    use_sample_covariance: bool = True,
    K1: float = 0.01,
    K2: float = 0.03,
    crop: bool = True,
) -> SSIMElements:
    """Compute SSIM elements (means, variances, covariance) on a 2-D image pair.

    Two filter modes are supported, matching upstream's ``gaussian_weights``
    flag (``ssim_utils.py:90, 208-213``):

    * ``gaussian_weights=False`` (upstream default, the path the RI-factor
      fit takes via ``ri_factor.py:89``): use ``uniform_filter`` from
      ``cubic.scipy.ndimage`` with ``size`` ``(1,)*(ndim-2) + (win_size,
      win_size)``. Default ``win_size=7``.
    * ``gaussian_weights=True`` (the path ``MicroSSIM.score`` takes per
      ``micro_ssim.py:31``): use ``gaussian`` from ``cubic.skimage.filters``
      with ``sigma=(0,)*(ndim-2) + (sigma, sigma)``, ``truncate=truncate``,
      ``mode="reflect"``. Default ``win_size`` is derived as
      ``2 * int(truncate*sigma + 0.5) + 1`` (= 11 for sigma=1.5,
      truncate=3.5) and is used only to crop the result.

    Parameters
    ----------
    image1 : numpy.ndarray
        First image. Shape ``(H, W)`` or ``(N, H, W)`` with leading batch
        axis. Dtype is promoted per :func:`_promoted_float_dtype`.
    image2 : numpy.ndarray
        Second image; must match ``image1`` shape and device.
    data_range : float
        Dynamic range of the input (``max - min``). Must be finite-positive.
    gaussian_weights : bool, default=False
        If True, use Gaussian filter with ``sigma`` / ``truncate``
        (MicroSSIM scoring path). If False, use ``uniform_filter`` with
        ``win_size`` (upstream's default; RI-factor fit path).
    win_size : int, optional
        Filter window side. Defaults to 7 when ``gaussian_weights=False``
        and to ``2*int(truncate*sigma+0.5)+1`` when ``gaussian_weights=True``
        (= 11 for the defaults). Must be odd and positive.
    sigma : float, default=1.5
        Gaussian standard deviation; used only when ``gaussian_weights=True``.
    truncate : float, default=3.5
        Gaussian truncation radius in units of ``sigma``; used only when
        ``gaussian_weights=True``.
    use_sample_covariance : bool, default=True
        If True, scale the covariance and variance by
        ``cov_norm = NP / (NP - 1)`` with ``NP = win_size**2``; otherwise
        use ``cov_norm = 1.0``. (Spatial filter is always 2-D; the leading
        batch axis is never aggregated.)
    K1 : float, default=0.01
        SSIM stability constant for the luminance term.
    K2 : float, default=0.03
        SSIM stability constant for the contrast / structure term.
    crop : bool, default=True
        If True, trim ``(win_size-1)//2`` from each spatial edge so the
        filter response uses only fully-supported windows. Matches upstream
        microssim's cropping convention.

    Returns
    -------
    SSIMElements
        Bundle of ``ux, uy, vxy, vx, vy, C1, C2``.

    Raises
    ------
    ValueError
        If ``image1.ndim`` is not 2 or 3, ``image1.shape != image2.shape``,
        ``win_size`` is even or non-positive, the smaller spatial extent
        is less than ``win_size``, or ``data_range`` is not finite-positive.

    Notes
    -----
    The invariant "always 2-D spatial, optional leading batch axis" is
    enforced by ``ValueError`` (not ``assert``), because asserts are
    stripped under ``python -O``.

    ``vx``, ``vy`` are NOT clamped to be non-negative — matches upstream
    ``ssim_utils.py:235-237``. Tiny negative values can appear from
    floating-point round-off when ``uxx ≈ ux**2``; callers that need a
    non-negative variance must clamp themselves.
    """
    check_same_device(image1, image2)

    if image1.ndim not in (2, 3):
        raise ValueError(
            f"Only (H, W) or (N, H, W) input is supported; got ndim={image1.ndim}"
        )
    if image1.shape != image2.shape:
        raise ValueError(
            f"Shape mismatch between image1 and image2: "
            f"{image1.shape} vs {image2.shape}"
        )
    if not (np.isfinite(data_range) and data_range > 0):
        raise ValueError(f"data_range must be finite and positive; got {data_range}")

    if win_size is None:
        if gaussian_weights:
            # Mirror upstream (ssim_utils.py:157-163): radius as in ndimage.
            r = int(truncate * sigma + 0.5)
            win_size = 2 * r + 1
        else:
            win_size = 7
    if win_size < 1 or win_size % 2 == 0:
        raise ValueError(f"win_size must be odd and positive; got {win_size}")
    if min(image1.shape[-2:]) < win_size:
        raise ValueError(
            f"Spatial dims must be >= win_size={win_size}; "
            f"got spatial shape {image1.shape[-2:]}"
        )

    promoted = _promoted_float_dtype(image1.dtype)
    if image1.dtype != promoted:
        image1 = image1.astype(promoted, copy=False)
    if image2.dtype != promoted:
        image2 = image2.astype(promoted, copy=False)

    # Build the per-axis filter. The leading axis (if any) is a batch
    # dimension; pass sigma=0 (gaussian) or size=1 (uniform) on it so the
    # filter is a no-op there. Routing is via cubic.scipy / cubic.skimage
    # proxies so CPU vs GPU dispatch is automatic.
    ndim = image1.ndim
    if gaussian_weights:
        import cubic.skimage as _sk

        sigma_axes = (0.0,) * (ndim - 2) + (float(sigma), float(sigma))

        def _filter(arr: np.ndarray) -> np.ndarray:
            return _sk.filters.gaussian(
                arr, sigma=sigma_axes, truncate=truncate, mode="reflect"
            )
    else:
        import cubic.scipy as _sp

        size_axes = (1,) * (ndim - 2) + (int(win_size), int(win_size))

        def _filter(arr: np.ndarray) -> np.ndarray:
            return _sp.ndimage.uniform_filter(arr, size=size_axes, mode="reflect")

    ux = _filter(image1)
    uy = _filter(image2)
    uxx = _filter(image1 * image1)
    uyy = _filter(image2 * image2)
    uxy = _filter(image1 * image2)

    # Spatial filter is always 2-D — NP = win_size**2 regardless of batch axis.
    NP = win_size * win_size
    cov_norm = NP / (NP - 1) if use_sample_covariance else 1.0

    vxy = cov_norm * (uxy - ux * uy)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    # NO clamp on vx / vy — matches upstream microssim (ssim_utils.py:235-237).

    if crop:
        pad = (win_size - 1) // 2
        sl: tuple[Any, ...] = (slice(None),) * (ndim - 2) + (
            slice(pad, -pad),
            slice(pad, -pad),
        )
        ux = ux[sl]
        uy = uy[sl]
        vxy = vxy[sl]
        vx = vx[sl]
        vy = vy[sl]

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    return SSIMElements(
        ux=ux,
        uy=uy,
        vxy=vxy,
        vx=vx,
        vy=vy,
        C1=float(C1),
        C2=float(C2),
    )
