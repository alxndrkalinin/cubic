"""General-purpose multi-scale SSIM (Wang et al. 2003), torchmetrics-faithful.

This module is **standalone** — it does NOT import from
``cubic.metrics.microssim``. The torchmetrics-faithful MS-SSIM path differs
from ``compute_ssim_elements`` along three axes (population vs sample
variance, explicit variance clamp, and pad-then-valid-conv boundary
handling), so the two implementations are kept independent.

Reference: ``torchmetrics/functional/image/ssim.py`` ``_ssim_update`` and
``multiscale_structural_similarity_index_measure``. Defaults match
``MultiScaleStructuralSimilarityIndexMeasure``: ``kernel_size=11``,
``sigma=1.5``, ``K1=0.01``, ``K2=0.03``,
``betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)`` (finest → coarsest),
``normalize="relu"`` semantics, average-pool 2x2 downsample.
"""

from __future__ import annotations

import numpy as np

import cubic.skimage as _sk

from ..cuda import check_same_device

DEFAULT_BETAS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
"""Default MS-SSIM scale weights, finest → coarsest (torchmetrics order)."""


def _avgpool2(x: np.ndarray) -> np.ndarray:
    """Apply 2x2 strided mean over the last two axes.

    Matches ``torch.nn.functional.avg_pool2d(x, (2, 2))`` with default
    stride and ``ceil_mode=False``: odd-sized trailing rows / columns are
    discarded by trimming to the largest even extent before averaging.

    Parameters
    ----------
    x : numpy.ndarray
        Input with at least two spatial axes (last two).

    Returns
    -------
    numpy.ndarray
        Downsampled array; spatial extent floor-divided by 2.
    """
    h = (x.shape[-2] // 2) * 2
    w = (x.shape[-1] // 2) * 2
    x = x[..., :h, :w]
    return 0.25 * (
        x[..., 0::2, 0::2]
        + x[..., 1::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 1::2]
    )


def _torchmetrics_ssim_update(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    c1: float,
    c2: float,
    kernel_size: int,
    sigma: float,
    truncate: float,
) -> tuple[float, float]:
    """Compute ``(ssim_full_mean, cs_cropped_mean)`` per torchmetrics' ``_ssim_update``.

    Torchmetrics' boundary handling is **explicit pad + valid conv**
    (``ssim.py:142-154``): ``F.pad(x, mode="reflect", pad=(pad, ...))``
    followed by ``F.conv2d(...)`` with no further padding. scipy /
    skimage's ``gaussian(mode="reflect")`` is a separable filter that
    handles the boundary per-axis as part of the convolution pipeline,
    which is *not* bit-identical to pad-then-valid-conv. To match
    torchmetrics within ``< 1e-3`` we therefore:

    1. ``np.pad(x, ..., mode="reflect")`` on the spatial axes with
       ``pad = (kernel_size - 1) // 2``, mirroring PyTorch's
       reflection_pad convention.
    2. Gaussian-filter the padded arrays with ``mode="constant", cval=0``
       so the kernel boundary contributes nothing past the reflected
       region — i.e. equivalent to a "valid" convolution on padded input.
    3. Slice back to the original spatial extent.

    Variance uses the **population** estimator (no ``NP / (NP - 1)``
    factor). ``vx`` and ``vy`` are clamped to ``>= 0`` to absorb
    round-off noise (matches torchmetrics ``ssim.py:163-164``); ``vxy``
    is signed and is not clamped.

    Parameters
    ----------
    image1, image2 : numpy.ndarray
        Same-shape, same-device images. Shape ``(H, W)`` or ``(N, H, W)``.
    c1, c2 : float
        SSIM stability constants ``(K1 * data_range) ** 2`` and
        ``(K2 * data_range) ** 2``.
    kernel_size : int
        Gaussian kernel side; must be odd-positive. Pad width is
        ``(kernel_size - 1) // 2``.
    sigma : float
        Gaussian standard deviation.
    truncate : float
        Gaussian truncation radius in units of ``sigma``, sized so the
        kernel exactly fits the pad width.

    Returns
    -------
    tuple of float
        ``(ssim_full_mean, cs_cropped_mean)``. The SSIM map is averaged
        over the full (un-further-cropped) slice; the contrast-sensitivity
        map is averaged over an additional ``[pad:-pad, pad:-pad]``
        crop — matches ``torchmetrics ssim.py:177``.
    """
    pad = (kernel_size - 1) // 2
    # Pad only the spatial axes (last two); leave the batch axis alone.
    pad_widths = [(0, 0)] * (image1.ndim - 2) + [(pad, pad), (pad, pad)]
    i1p = np.pad(image1, pad_widths, mode="reflect")
    i2p = np.pad(image2, pad_widths, mode="reflect")

    sigma_axes = (0.0,) * (i1p.ndim - 2) + (float(sigma), float(sigma))

    def _filter(arr: np.ndarray) -> np.ndarray:
        return _sk.filters.gaussian(
            arr, sigma=sigma_axes, truncate=truncate, mode="constant", cval=0
        )

    ux_p = _filter(i1p)
    uy_p = _filter(i2p)
    uxx_p = _filter(i1p * i1p)
    uyy_p = _filter(i2p * i2p)
    uxy_p = _filter(i1p * i2p)

    # Slice back to the original spatial extent.
    sl = (slice(None),) * (image1.ndim - 2) + (slice(pad, -pad), slice(pad, -pad))
    ux = ux_p[sl]
    uy = uy_p[sl]
    uxx = uxx_p[sl]
    uyy = uyy_p[sl]
    uxy = uxy_p[sl]

    # Population variance with clamp on vx, vy; vxy is signed (no clamp).
    vx = np.maximum(uxx - ux * ux, 0.0)
    vy = np.maximum(uyy - uy * uy, 0.0)
    vxy = uxy - ux * uy

    upper = 2.0 * vxy + c2
    lower = vx + vy + c2

    ssim_full = ((2.0 * ux * uy + c1) * upper) / ((ux * ux + uy * uy + c1) * lower)

    # CS is further cropped by `pad` to match torchmetrics ssim.py:177.
    cs_full = upper / lower
    sl_cs = (slice(None),) * (image1.ndim - 2) + (
        slice(pad, -pad),
        slice(pad, -pad),
    )
    cs_cropped = cs_full[sl_cs]

    return float(ssim_full.mean()), float(cs_cropped.mean())


def ms_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    data_range: float,
    betas: tuple[float, ...] = DEFAULT_BETAS,
    kernel_size: int = 11,
    sigma: float = 1.5,
    truncate: float = 3.5,
    K1: float = 0.01,
    K2: float = 0.03,
    normalize: str = "relu",
) -> float:
    """Compute multi-scale SSIM (Wang et al. 2003) compatible with torchmetrics.

    Mirrors ``torchmetrics.MultiScaleStructuralSimilarityIndexMeasure``
    with default settings. At each of ``len(betas)`` scales the
    contrast-sensitivity map (cropped) is exponentiated by ``betas[j]``
    and multiplied; at the coarsest scale the full SSIM map is used
    instead. Downsampling between scales is 2x2 average pooling.

    Parameters
    ----------
    image1, image2 : numpy.ndarray
        Same-shape, same-device images. Shape ``(H, W)`` or ``(N, H, W)``.
    data_range : float
        Dynamic range of the input (``max - min``). Must be finite-positive.
    betas : tuple of float, default=DEFAULT_BETAS
        Per-scale weights, finest → coarsest. The number of scales is
        ``len(betas)``.
    kernel_size : int, default=11
        Gaussian kernel side; must be odd-positive.
    sigma : float, default=1.5
        Gaussian standard deviation.
    truncate : float, default=3.5
        Gaussian truncation radius in units of ``sigma``.
    K1, K2 : float, default=0.01, 0.03
        SSIM stability constants.
    normalize : str, default="relu"
        If ``"relu"``, clamp each per-scale CS and the final SSIM
        component to ``>= 0`` before exponentiation. Any other value
        disables clamping (mirrors torchmetrics' ``normalize=None``).

    Returns
    -------
    float
        Scalar MS-SSIM value.

    Raises
    ------
    ValueError
        If ``image1.shape != image2.shape``, ``image1.ndim`` is not 2 or
        3, ``kernel_size`` is not odd-positive, ``data_range`` is not
        finite-positive, or any spatial dim is below
        ``2 ** (n_scales - 1) * kernel_size`` (176 for the defaults).
    """
    check_same_device(image1, image2)

    if image1.shape != image2.shape:
        raise ValueError(
            f"Shape mismatch between image1 and image2: "
            f"{image1.shape} vs {image2.shape}"
        )
    if image1.ndim not in (2, 3):
        raise ValueError(
            f"Only (H, W) or (N, H, W) input is supported; got ndim={image1.ndim}"
        )
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and positive; got {kernel_size}")
    if not (np.isfinite(data_range) and data_range > 0):
        raise ValueError(f"data_range must be finite and positive; got {data_range}")

    n_scales = len(betas)
    min_spatial = 2 ** (n_scales - 1) * kernel_size
    if min(image1.shape[-2:]) < min_spatial:
        raise ValueError(
            f"Spatial dims must be >= {min_spatial} for {n_scales}-scale "
            f"MS-SSIM with kernel_size={kernel_size}; "
            f"got spatial shape {image1.shape[-2:]}"
        )

    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2

    cs_powers: list[float] = []
    final: float = 1.0
    for j in range(n_scales):
        ssim_full, cs_cropped = _torchmetrics_ssim_update(
            image1,
            image2,
            c1=c1,
            c2=c2,
            kernel_size=kernel_size,
            sigma=sigma,
            truncate=truncate,
        )
        if j < n_scales - 1:
            if normalize == "relu":
                cs_cropped = max(cs_cropped, 0.0)
            cs_powers.append(cs_cropped ** betas[j])
            image1 = _avgpool2(image1)
            image2 = _avgpool2(image2)
        else:
            if normalize == "relu":
                ssim_full = max(ssim_full, 0.0)
            final = ssim_full ** betas[j]

    return float(final * float(np.prod(cs_powers)))
